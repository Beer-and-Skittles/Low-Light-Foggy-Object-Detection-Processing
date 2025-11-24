# <Project>/datasets/anno_parsers.py
from pathlib import Path
import xml.etree.ElementTree as ET


def parse_exdark_bbgt(txt_path: Path, class2id, W, H):
    """
    ExDark bbGt:
      % bbGt version=3
      Car 740 918 1389 1150 0 0 0 0 0 0 0

    Returns list of (cls_id, x_c, y_c, w, h) normalized [0,1].
    We assume (xmin, ymin, xmax, ymax). If your files are (x,y,w,h),
    the heuristic handles it.
    """
    labels = []
    if txt_path is None or not txt_path.exists():
        return labels

    lines = txt_path.read_text().splitlines()
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.startswith("%"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue

        cls = parts[0].lower()
        if cls not in class2id:
            continue
        cls_id = class2id[cls]

        a, b, c, d = map(float, parts[1:5])

        # heuristic: treat as xyxy if c>a and d>b, else xywh
        if c > a and d > b and c <= W * 1.5 and d <= H * 1.5:
            xmin, ymin, xmax, ymax = a, b, c, d
        else:
            xmin, ymin, xmax, ymax = a, b, a + c, b + d

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


def parse_rtts_voc(xml_path: Path, class2id, W, H):
    """
    RTTS VOC XML -> YOLO normalized.
    """
    labels = []
    if xml_path is None or not xml_path.exists():
        return labels

    root = ET.parse(xml_path).getroot()
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name is None:
            continue
        cls = name.lower()
        if cls not in class2id:
            continue
        cls_id = class2id[cls]

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
