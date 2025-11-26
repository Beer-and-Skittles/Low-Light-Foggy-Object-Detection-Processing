import argparse, shutil, os, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO

from preprocessing.baselines_pil import to_np, to_pil, clahe, gamma, retinex, hist_match
from datasets.alias_mapping import ALIAS_TO_YOLO
from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc


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
    if method == "clahe_gamma":
        return gamma(clahe(img_np))
    if method == "none":
        return img_np
    raise ValueError(method)


# ------------------ Split list ------------------
def load_pairs(fpath):
    pairs = []
    with open(fpath) as f:
        for line in f:
            img, anno = line.strip().split("\t")
            pairs.append((Path(img), Path(anno) if anno else None))
    return pairs


def write_yolo_label_file(label_path, labels):
    if not labels:
        label_path.write_text("")
        return
    with open(label_path, "w") as f:
        for cls_id, x_c, y_c, w_n, h_n in labels:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list", required=True)
    ap.add_argument("--method", required=True,
                    choices=["none", "clahe", "gamma", "retinex", "hist_match", "clahe_gamma"])
    ap.add_argument("--class_names_json", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--temp_dir", default="outputs/tmp")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # fresh temp dir
    out_root = Path(args.temp_dir)
    if out_root.exists():
       shutil.rmtree(out_root)
    images_dir = out_root / "images" / "test"
    labels_dir = out_root / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # class mapping
    from datasets.class_mapping import build_class2yolo_id

    class2id, class_names = build_class2yolo_id(
        args.class_names_json,
        yolo_weights="yolov8n.pt"
    )
    print("class2id", class2id)
    print("class_names", class_names)

    # reference for hist-match
    ref_np = None
    if args.method == "hist_match":
        if args.reference is None:
            raise ValueError("hist_match needs --reference")
        ref_np = to_np(Image.open(args.reference))

    pairs = load_pairs(args.test_list)

    # -------- enhancement + YOLO-label generation --------
    for img_path, anno_path in tqdm(pairs, desc=f"Enhancing {args.method}"):
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size

        img_np = to_np(pil_img)
        out_np = enhance(img_np, args.method, ref_np)
        out_pil = to_pil(out_np)

        out_img_path = images_dir / img_path.name
        out_pil.save(out_img_path)

        labels = []
        if anno_path is not None:
            suf = anno_path.suffix.lower()
            if suf == ".xml":
                labels = parse_rtts_voc(anno_path, class2id, W, H)
            elif suf == ".txt":
                labels = parse_exdark_bbgt(anno_path, class2id, W, H)

        out_label_path = labels_dir / f"{img_path.stem}.txt"
        write_yolo_label_file(out_label_path, labels)

    

    # -------- Run YOLO eval IN PYTHON (no CLI needed) --------
    det = YOLO("yolov8n.pt")


    # YOLO dataset YAML
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {class_names}\n"
    )
    
    metrics = det.val(
        data=str(yaml_path),
        split="test",
        device=args.device,
        workers=0,
        batch=1,
        verbose=True,
        # classes=[1, 8, 39, 5, 2, 15, 56, 41, 16, 3, 0, 60]
    )

    print("\n=== YOLO Evaluation Complete ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("Results saved to:", metrics.save_dir)


if __name__ == "__main__":
    main()
