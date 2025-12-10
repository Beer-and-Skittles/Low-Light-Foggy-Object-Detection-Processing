# <Project>/scripts/eval_aod_cli.py
import argparse, shutil, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

from ultralytics import YOLO

from models.aod_net import AODNet
from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc
from datasets.class_mapping import build_class2yolo_id


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split_list", required=True, help="train/test list txt")
    ap.add_argument("--ckpt", required=True, help="path to aod_epoch*.pt")
    ap.add_argument("--class_names_json", default="data/class_names.json")
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--temp_dir", default="outputs/tmp_aod_eval")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_samples", action="store_true",
                    help="save a few enhanced images for report")
    ap.add_argument("--num_samples", type=int, default=20)
    args = ap.parse_args()

    device = torch.device(args.device)

    # ---------------- load AOD-Net ----------------
    aod = AODNet().to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    aod.load_state_dict(ckpt["aod"])
    aod.eval()

    # -------------- class mapping (alias-aware) --------------
    class2id, class_names = build_class2yolo_id(
        args.class_names_json,
        yolo_weights="yolov8n.pt",
    )

    # optional: debug print
    print("class2id (dataset label -> YOLO id):")
    for k, v in class2id.items():
        print(f"  {k} -> {v}")

    # -------------- temp dataset ------------------
    out_root = Path(args.temp_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    images_dir = out_root / "images" / "test"
    labels_dir = out_root / "labels" / "test"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # optional sample dump folder
    sample_dir = out_root / "samples"
    if args.save_samples:
        sample_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- enhance + write labels ----------------
    pairs = load_pairs(args.split_list)
    saved = 0

    with torch.no_grad():
        for img_path, anno_path in tqdm(pairs, desc="Enhancing with AOD-Net"):
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size

            pil_r = pil.resize((args.img_size, args.img_size))
            img = torch.from_numpy(np.array(pil_r)).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0).to(device)

            enh, _ = aod(img)  # (1,3,S,S)
            enh_np = (
                enh.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
            ).clip(0, 255).astype(np.uint8)
            enh_pil = Image.fromarray(enh_np)

            # save enhanced image for YOLO eval
            out_img_path = images_dir / img_path.name
            enh_pil.save(out_img_path)

            # save some report samples if asked
            if args.save_samples and saved < args.num_samples:
                enh_pil.save(sample_dir / img_path.name)
                saved += 1

            # parse original anno -> YOLO labels
            labels = []
            if anno_path is not None:
                suf = anno_path.suffix.lower()
                if suf == ".xml":
                    labels = parse_rtts_voc(anno_path, class2id, W, H)
                elif suf == ".txt":
                    labels = parse_exdark_bbgt(anno_path, class2id, W, H)

            out_label_path = labels_dir / f"{img_path.stem}.txt"
            write_yolo_label_file(out_label_path, labels)

    # ---------------- write YOLO yaml ----------------
    # Use full COCO names to match global YOLO indices in labels
    det = YOLO("yolov8n.pt")
    yolo_names = list(det.model.names.values())

    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {yolo_names}\n"
    )

    # ---------------- run YOLO val directly in Python ----------------
    metrics = det.val(
        data=str(yaml_path),
        split="test",
        device=args.device,
        workers=0,
        batch=1,
        verbose=True,
    )

    print("\n=== YOLO Evaluation Complete (AOD) ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print("Results saved to:", metrics.save_dir)
    print("Temp dataset at:", out_root)


if __name__ == "__main__":
    main()
