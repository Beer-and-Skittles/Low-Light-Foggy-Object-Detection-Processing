# <Project>/datasets/rtts.py
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

def list_rtts_pairs(rtts_img_root, rtts_anno_root):
    """
    RTTS images are flat:
      RTTS/*.png
    Annotations are VOC xml:
      RTTS_Anno/*.xml
    We return a list of (img_path, anno_path_or_None).
    """
    rtts_img_root = Path(rtts_img_root)
    rtts_anno_root = Path(rtts_anno_root)

    pairs = []
    for img_path in sorted(rtts_img_root.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        anno_path = rtts_anno_root / (img_path.stem + ".xml")
        if not anno_path.exists():
            anno_path = None

        pairs.append((img_path, anno_path))

    return pairs
