import argparse, shutil, os, json
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
from ultralytics import YOLO

from datasets.alias_mapping import ALIAS_TO_YOLO

# ------------------ ExDark bbGt parser ------------------
def parse_exdark_bbgt(txt_path, class2id, W, H):
    labels = []
    if txt_path is None or not txt_path.exists():
        return labels

    with open(txt_path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for ln in lines:
        if ln.startswith("%"):
            continue

        parts = ln.split()
        if len(parts) < 5:
            continue

        cls_name = ALIAS_TO_YOLO[parts[0].lower()]
        if cls_name not in class2id:
            print("Warning, invalid cls_name from exdark:", cls_name)
            continue
        cls_id = class2id[cls_name]

        a, b, c, d = map(float, parts[1:5])

        if c > a and d > b:
            xmin, ymin, xmax, ymax = a, b, c, d
        else:
            xmin, ymin = a, b
            xmax, ymax = a + c, b + d

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


# ------------------ RTTS VOC XML parser ------------------
def parse_rtts_voc(xml_path, class2id, W, H):
    labels = []
    if xml_path is None or not xml_path.exists():
        return labels

    root = ET.parse(xml_path).getroot()

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue

        cls_name = ALIAS_TO_YOLO[name.lower()]
        if cls_name not in class2id:
            print("Warning, invalid cls_name from rtts:", cls_name)
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