import argparse, shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
from ultralytics import YOLO

from preprocessing.baselines_pil import to_np, to_pil, clahe, gamma, retinex, hist_match
from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc

import sys
from pathlib import Path

# Add project root (parent of scripts/) to sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


# ------------------ Stage 1: prepare enhanced YOLO dataset ------------------
def prepare_dataset(
    test_list,
    method,
    class_names_json,
    temp_dir,
    reference=None,
):
    """Create enhanced images, YOLO labels, and data.yaml under temp_dir."""
    out_root = Path(temp_dir)

    # fresh temp dir
    if out_root.exists():
        shutil.rmtree(out_root)
    images_dir = out_root / "images" / "test"
    labels_dir = out_root / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # class mapping
    from datasets.class_mapping import build_class2yolo_id

    class2id, class_names = build_class2yolo_id(
        class_names_json,
        yolo_weights="yolov8n.pt"
    )
    print("class2id", class2id)
    print("class_names", class_names)

    # reference for hist-match
    ref_np = None
    if method == "hist_match":
        if reference is None:
            raise ValueError("hist_match needs --reference")
        ref_np = to_np(Image.open(reference))

    pairs = load_pairs(test_list)

    # -------- enhancement + YOLO-label generation --------
    for img_path, anno_path in tqdm(pairs, desc=f"Enhancing {method}"):
        pil_img = Image.open(img_path).convert("RGB")
        W, H = pil_img.size

        img_np = to_np(pil_img)
        out_np = enhance(img_np, method, ref_np)
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

    # -------- Write YOLO dataset YAML --------
    model = YOLO("yolov8n.pt")  # or use args.weights if you prefer
    model_names = model.model.names  # dict or list, usually 0..79 COCO
    
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {list(model_names.values()) if isinstance(model_names, dict) else model_names}\n"
    )
    
    print(f"[prepare] Wrote data.yaml with {len(model_names)} classes (COCO).")
    return out_root, yaml_path


# ------------------ Stage 2: run YOLO eval on existing dataset ------------------
def run_yolo_eval(dataset_root, device="cpu", weights="yolov8n.pt"):
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"No data.yaml found at {yaml_path} (did you run stage=prepare?)")

    print(f"[eval] Loading YOLO model: {weights}")
    det = YOLO(weights)

    print(f"[eval] Using dataset YAML: {yaml_path.resolve()}")
    metrics = det.val(
        data=str(yaml_path),
        split="test",
        device=device,
        workers=0,
        batch=1,
        verbose=True,
        name=str(dataset_root).split('/')[-1]
        # classes=[...]
    )

    print("\n=== YOLO Evaluation Complete ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("Results saved to:", metrics.save_dir)
    return metrics


# ------------------ MAIN ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "prepare", "eval"], default="all",
                    help="all = prepare + eval (original behavior); "
                         "prepare = only build enhanced dataset; "
                         "eval = only run YOLO on existing dataset_root")

    # common
    ap.add_argument("--temp_dir",
                    help="Root directory for enhanced dataset (stage=prepare/all) "
                         "or existing dataset root (stage=eval)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--weights", default="yolov8n.pt",
                    help="YOLO weights path or model name (for eval)")

    # needed for stage=prepare/all
    ap.add_argument("--test_list", help="TSV: img_path<TAB>anno_path")
    ap.add_argument("--method",
                    choices=["none", "clahe", "gamma", "retinex", "hist_match", "clahe_gamma"],
                    help="Enhancement method")
    ap.add_argument("--class_names_json", default="data/class_names.json",
                    help="JSON for building dataset->YOLO class mapping")
    ap.add_argument("--reference", default=None,
                    help="Reference image for hist_match")

    args = ap.parse_args()

    # ---------- validate arguments per stage ----------
    if args.stage in ("all", "prepare"):
        missing = []
        if args.test_list is None:
            missing.append("--test_list")
        if args.method is None:
            missing.append("--method")
        if args.class_names_json is None:
            missing.append("--class_names_json")
        if missing:
            ap.error(f"Stage '{args.stage}' requires: {', '.join(missing)}")
        if args.method == "hist_match" and args.reference is None:
            ap.error("Method 'hist_match' requires --reference for stage 'prepare' or 'all'")
        if args.temp_dir is None:
            dataset = (args.test_list.split('/')[-1]).split('.')[0]
            args.temp_dir = 'outputs/' + dataset + '_' + args.method
            

    # ---------- run stages ----------
    dataset_root = Path(args.temp_dir)

    if args.stage in ("all", "prepare"):
        dataset_root, _ = prepare_dataset(
            test_list=args.test_list,
            method=args.method,
            class_names_json=args.class_names_json,
            temp_dir=args.temp_dir,
            reference=args.reference,
        )

    if args.stage in ("all", "eval"):
        run_yolo_eval(
            dataset_root=dataset_root,
            device=args.device,
            weights=args.weights,
        )


if __name__ == "__main__":
    main()
