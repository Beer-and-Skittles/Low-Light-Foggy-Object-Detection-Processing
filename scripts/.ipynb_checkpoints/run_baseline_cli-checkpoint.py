import argparse, subprocess, shutil, os, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

from preprocessing.baselines_pil import to_np, to_pil, clahe, gamma, retinex, hist_match


# ------------------ Enhancement dispatch ------------------
def enhance(img_np, method, ref_np=None):
    if method == "clahe":
        return clahe(img_np)
    if method == "gamma":
        return gamma(img_np)
    if method == "retinex":
        return retinex(img_np)
    if method == "hist_match":
        return hist_match(img_np, ref_np)
    raise ValueError(method)


# ------------------ Split list ------------------
def load_pairs(fpath):
    pairs = []
    with open(fpath) as f:
        for line in f:
            img, anno = line.strip().split("\t")
            pairs.append((Path(img), Path(anno) if anno else None))
    return pairs


# ------------------ ExDark bbGt parser ------------------
def parse_exdark_bbgt(txt_path, class2id, W, H):
    """
    ExDark bbGt format example:
      % bbGt version=3
      Car 740 918 1389 1150 0 0 0 0 0 0 0

    We handle both possible conventions:
      (xmin, ymin, xmax, ymax)  OR  (x, y, w, h)
    via a simple heuristic.
    """
    labels = []
    if txt_path is None or not txt_path.exists():
        return labels

    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        if ln.startswith("%"):  # header
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue

        cls_name = parts[0].lower()
        if cls_name not in class2id:
            continue
        cls_id = class2id[cls_name]

        a, b, c, d = map(float, parts[1:5])

        # heuristic: if c>d? no. if c>a and d>b and within bounds -> treat as xmax/ymax
        if c > a and d > b and c <= W * 1.5 and d <= H * 1.5:
            xmin, ymin, xmax, ymax = a, b, c, d
        else:
            # treat as x,y,w,h
            xmin, ymin = a, b
            xmax, ymax = a + c, b + d

        # clamp
        xmin = max(0, min(xmin, W - 1))
        ymin = max(0, min(ymin, H - 1))
        xmax = max(0, min(xmax, W - 1))
        ymax = max(0, min(ymax, H - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        # YOLO normalize
        x_c = (xmin + xmax) / 2.0 / W
        y_c = (ymin + ymax) / 2.0 / H
        w_n = (xmax - xmin) / W
        h_n = (ymax - ymin) / H

        labels.append((cls_id, x_c, y_c, w_n, h_n))

    return labels


# ------------------ RTTS VOC XML parser ------------------
def parse_rtts_voc(xml_path, class2id, W, H):
    """
    RTTS VOC xml:
      <object><name>bus</name><bndbox>...</bndbox></object>
    """
    labels = []
    if xml_path is None or not xml_path.exists():
        return labels

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        cls_name = name.lower()
        if cls_name not in class2id:
            continue
        cls_id = class2id[cls_name]

        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = float(box.findtext("xmin", 0))
        ymin = float(box.findtext("ymin", 0))
        xmax = float(box.findtext("xmax", 0))
        ymax = float(box.findtext("ymax", 0))

        xmin = max(0, min(xmin, W - 1))
        ymin = max(0, min(ymin, H - 1))
        xmax = max(0, min(xmax, W - 1))
        ymax = max(0, min(ymax, H - 1))
        if xmax <= xmin or ymax <= ymin:
            continue

        x_c = (xmin + xmax) / 2.0 / W
        y_c = (ymin + ymax) / 2.0 / H
        w_n = (xmax - xmin) / W
        h_n = (ymax - ymin) / H

        labels.append((cls_id, x_c, y_c, w_n, h_n))

    return labels


def write_yolo_label_file(label_path, labels):
    """
    labels: list of (cls_id, x_c, y_c, w_n, h_n)
    """
    if not labels:
        # YOLO allows missing/empty label files. Create empty file for clarity.
        label_path.write_text("")
        return
    with open(label_path, "w") as f:
        for cls_id, x_c, y_c, w_n, h_n in labels:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--method", required=True,
                    choices=["clahe", "gamma", "retinex", "hist_match"])
    ap.add_argument("--class_names_json", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--temp_dir", default="outputs/tmp")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out_root = Path(args.temp_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    images_dir = out_root / "images" / "test"
    labels_dir = out_root / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # class mapping
    class_names = json.load(open(args.class_names_json))
    class2id = {n.lower(): i for i, n in enumerate(class_names)}

    # reference image (for hist-match only)
    ref_np = None
    if args.method == "hist_match":
        if args.reference is None:
            raise ValueError("hist_match needs --reference")
        ref_np = to_np(Image.open(args.reference))

    pairs = load_pairs(args.test_list)

    # -------- main loop: enhance + generate YOLO labels on the fly --------
    for img_path, anno_path in tqdm(pairs, desc=f"Enhancing {args.method}"):
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size

        img_np = to_np(pil_img)
        out_np = enhance(img_np, args.method, ref_np)
        out_pil = to_pil(out_np)

        # save enhanced image
        out_img_path = images_dir / img_path.name
        out_pil.save(out_img_path)

        # parse original annotation into YOLO labels
        labels = []
        if anno_path is not None:
            suf = anno_path.suffix.lower()
            if suf == ".xml":
                labels = parse_rtts_voc(anno_path, class2id, W, H)
            elif suf == ".txt":
                labels = parse_exdark_bbgt(anno_path, class2id, W, H)

        # YOLO label filename must match image stem
        out_label_path = labels_dir / f"{img_path.stem}.txt"
        write_yolo_label_file(out_label_path, labels)

    # -------- write valid YOLO dataset YAML --------
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {class_names}\n"
    )

    # -------- run YOLO val --------
    cmd = [
        "yolo", "val",
        "model=yolov8n.pt",
        f"data={yaml_path}",
        "split=test",
        "workers=0",
        "batch=1",
        f"device={args.device}",
    ]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU safety

    subprocess.run(cmd, env=env, check=True)
    print("Done.")


if __name__ == "__main__":
    main()
