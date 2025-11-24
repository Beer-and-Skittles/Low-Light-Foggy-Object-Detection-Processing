# <Project>/datasets/exdark.py
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

def list_exdark_pairs(exdark_img_root, exdark_anno_root):
    """
    ExDark images are in class subfolders:
      ExDark/Bicycle/*.png
    Annotations mirror the structure:
      ExDark_Anno/Bicycle/*.txt
    We return a list of (img_path, anno_path_or_None).
    """
    exdark_img_root = Path(exdark_img_root)
    exdark_anno_root = Path(exdark_anno_root)

    pairs = []
    for cls_dir in sorted(exdark_img_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        anno_cls_dir = exdark_anno_root / cls_dir.name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue

            anno_path = anno_cls_dir / f"{img_path.name}.txt" 
            if not anno_path.exists():
                anno_path = None  # allow missing labels

            pairs.append((img_path, anno_path))
    return pairs
