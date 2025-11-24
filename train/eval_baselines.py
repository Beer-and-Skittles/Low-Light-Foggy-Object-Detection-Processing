import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from preprocessing.clahe import apply_clahe
from preprocessing.gamma import apply_gamma
from preprocessing.retinex import apply_retinex
# from preprocessing.hist_match import apply_hist_match

from yolo.detector import YoloDetector
from yolo.metrics import compute_map

import yaml


def read_rgb_3ch(img_path):
    # read unchanged so we can see alpha/grayscale
    bgr = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if bgr is None:
        return None

    # grayscale -> BGR
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    # RGBA/BGRA -> BGR (drop alpha)
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = bgr[:, :, :3]

    # make sure dtype + contiguity are safe for OpenCV
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_split_list(txt_path):
    pairs = []
    with open(txt_path, "r") as f:
        for line in f:
            img_path, anno_path = line.strip().split("\t")
            pairs.append((Path(img_path), Path(anno_path) if anno_path else None))
    return pairs


def enhance_image(img, method, hist_ref=None):
    if method == "clahe":
        return apply_clahe(img)
    elif method == "gamma":
        return apply_gamma(img)
    elif method == "retinex":
        return apply_retinex(img)
    # elif method == "hist_match":
    #     return apply_hist_match(img, hist_ref)
    else:
        raise ValueError(f"Unknown method: {method}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_list", required=True,
                    help="Path to exdark_test.txt or rtts_test.txt or combined_test.txt")
    ap.add_argument("--data_yaml", required=True,
                    help="Path to YOLO dataset YAML file (labels + images)")
    ap.add_argument("--method", required=True,
                    choices=["clahe", "gamma", "retinex", "hist_match"])
    ap.add_argument("--reference", default=None,
                    help="Reference image for histogram matching")
    args = ap.parse_args()

    # prepare YOLO
    detector = YoloDetector("yolov8n.pt")

    # optional reference image for histogram matching
    hist_ref = None
    if args.method == "hist_match":
        if args.reference is None:
            raise ValueError("Histogram matching requires --reference <img.jpg>")
        hist_ref = cv2.cvtColor(cv2.imread(args.reference), cv2.COLOR_BGR2RGB)

    # load test list
    pairs = load_split_list(args.test_list)

    print(f"Running baseline: {args.method}")
    print(f"Total test images: {len(pairs)}")

    # We do NOT save enhanced images; we pass enhanced array directly to YOLO
    # YOLO mAP will be computed by temporarily evaluating detection.
    # For this demo, we keep it simple: we accumulate predictions and run YOLO val mode.

    # NOTE:
    # To truly compute mAP for enhanced images, you need a YOLO dataset folder
    # matching the enhanced images. But to avoid saving images, we instead run YOLO detection
    # and compute mAP manually or dump results. Here we show a simplified approach:
    
    preds = []  # store YOLO result objects

    for img_path, anno_path in tqdm(pairs):
        img = read_rgb_3ch(img_path)
        if img is None:
            print(f"[WARN] failed to read {img_path}")
            continue

        enhanced = enhance_image(img, args.method, hist_ref)
        res = detector.infer_img(enhanced)
        preds.append(res)


    # Placeholder: prints detection boxes count
    # Later you can integrate with a custom mAP evaluator without needing saved images
    num_boxes = sum(len(p.boxes) for p in preds)
    print(f"[INFO] Completed {args.method} baseline. Total predicted boxes: {num_boxes}")


if __name__ == "__main__":
    main()
