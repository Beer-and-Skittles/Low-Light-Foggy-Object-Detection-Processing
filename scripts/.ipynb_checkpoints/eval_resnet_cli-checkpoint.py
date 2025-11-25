# <Project>/scripts/eval_resnet_cli.py
import argparse, shutil, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

from ultralytics import YOLO
from models.resnet_enhancer import ResNetEnhancer
from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc
from datasets.class_mapping import build_class2yolo_id


def load_pairs(fpath):
    pairs = []
    with open(fpath) as f:
        for line in f:
            img, anno = line.strip().split("\t")
            anno = anno if anno and anno != "None" else None
            pairs.append((Path(img), Path(anno) if anno else None))
    return pairs


def write_yolo_label_file(label_path, labels):
    if not labels:
        label_path.write_text("")
        return
    with open(label_path, "w") as f:
        for cls_id, x_c, y_c, w_n, h_n in labels:
            f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_list", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--class_names_json", required=True)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--temp_dir", default="outputs/tmp_resnet_eval")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save_samples", action="store_true")
    ap.add_argument("--num_samples", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------------- Load ResNet enhancer ----------------
    net = ResNetEnhancer(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["net"])
    net.eval()

    # ---------------- Build alias-aware class mapping ----------------
    class2id, class_names = build_class2yolo_id(
        args.class_names_json,
        yolo_weights="yolov8n.pt"
    )

    print("class2id mapping:")
    for k, v in class2id.items():
        print(f"  {k} -> {v}")

    # ---------------- Build temp dataset ----------------
    out_root = Path(args.temp_dir)
    if out_root.exists():
        shutil.rmtree(out_root)

    images_dir = out_root / "images" / "test"
    labels_dir = out_root / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = out_root / "samples"
    if args.save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.split_list)
    saved = 0

    # ---------------- Enhance images & write YOLO labels ----------------
    with torch.no_grad():
        for img_path, anno_path in tqdm(pairs, desc="Enhancing with ResNet"):
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size

            pil_r = pil.resize((args.img_size, args.img_size))
            arr = np.array(pil_r).astype(np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            enh = net(img)
            enh_np = (
                enh.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
            enh_pil = Image.fromarray(enh_np)

            enh_pil.save(images_dir / img_path.name)

            if args.save_samples and saved < args.num_samples:
                enh_pil.save(sample_dir / img_path.name)
                saved += 1

            labels = []
            if anno_path is not None:
                suf = anno_path.suffix.lower()
                if suf == ".xml":
                    labels = parse_rtts_voc(anno_path, class2id, W, H)
                elif suf == ".txt":
                    labels = parse_exdark_bbgt(anno_path, class2id, W, H)

            write_yolo_label_file(labels_dir / f"{img_path.stem}.txt", labels)

    # ---------------- Write valid YOLO dataset YAML ----------------
    det = YOLO("yolov8n.pt")
    yolo_names = list(det.model.names.values())  # full COCO list

    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {yolo_names}\n"
    )

    # ---------------- YOLO eval in Python ----------------
    metrics = det.val(
        data=str(yaml_path),
        split="test",
        device=args.device,
        workers=0,
        batch=1,
        verbose=True,
    )

    print("\n=== YOLO Evaluation Complete (ResNet) ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("Results saved to:", metrics.save_dir)
    print("Temp dataset at:", out_root)


if __name__ == "__main__":
    main()
