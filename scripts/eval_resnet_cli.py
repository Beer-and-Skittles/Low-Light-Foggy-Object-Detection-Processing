# <Project>/scripts/eval_resnet_cli.py
import argparse, subprocess, shutil, os, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

from models.resnet_enhancer import ResNetEnhancer
from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc


def load_pairs(fpath):
    pairs = []
    with open(fpath) as f:
        for line in f:
            img, anno = line.strip().split("\t")
            anno = anno if anno != "None" else None
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

    # ---------------- load ResNet enhancer ----------------
    net = ResNetEnhancer(pretrained=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    net.load_state_dict(ckpt["net"])
    net.eval()

    # ---------------- class mapping ----------------
    class_names = json.load(open(args.class_names_json))
    class2id = {n.lower(): i for i, n in enumerate(class_names)}

    # ---------------- build temp dataset ----------------
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

    # ---------------- enhance & write YOLO labels ----------------
    with torch.no_grad():
        for img_path, anno_path in tqdm(pairs, desc="Enhancing with ResNet"):
            pil = Image.open(img_path).convert("RGB")
            W, H = pil.size

            # resize -> tensor
            pil_r = pil.resize((args.img_size, args.img_size))
            arr = np.array(pil_r).astype(np.float32) / 255.0
            img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

            enh = net(img)  # (1,3,S,S)
            enh_np = (enh.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0)
            enh_np = enh_np.clip(0, 255).astype(np.uint8)
            enh_pil = Image.fromarray(enh_np)

            out_img_path = images_dir / img_path.name
            enh_pil.save(out_img_path)

            # save samples
            if args.save_samples and saved < args.num_samples:
                enh_pil.save(sample_dir / img_path.name)
                saved += 1

            # parse annotation and write YOLO label file
            labels = []
            if anno_path is not None:
                if anno_path.suffix.lower() == ".xml":
                    labels = parse_rtts_voc(anno_path, class2id, W, H)
                elif anno_path.suffix.lower() == ".txt":
                    labels = parse_exdark_bbgt(anno_path, class2id, W, H)

            out_label_path = labels_dir / f"{img_path.stem}.txt"
            write_yolo_label_file(out_label_path, labels)

    # ---------------- create YOLO dataset yaml ----------------
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(
        f"path: {out_root.resolve()}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"test: images/test\n"
        f"names: {class_names}\n"
    )

    # ---------------- run YOLO eval ----------------
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
    env["CUDA_VISIBLE_DEVICES"] = ""

    subprocess.run(cmd, env=env, check=True)

    print("\nResults saved under runs/detect/val*/")
    print("Temp dataset at:", out_root)


if __name__ == "__main__":
    main()
