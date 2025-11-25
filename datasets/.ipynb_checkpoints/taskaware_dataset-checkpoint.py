# <Project>/datasets/taskaware_dataset.py
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from datasets.anno_parsers import parse_exdark_bbgt, parse_rtts_voc


class TaskAwareDetectionDataset(Dataset):
    def __init__(self, class_names_json, yolo_weights, split_txt, img_size=640):
        self.items = []
        with open(split_txt, "r") as f:
            for line in f:
                img_p, anno_p = line.strip().split("\t")
                self.items.append((Path(img_p), Path(anno_p) if anno_p else None))


        from datasets.class_mapping import build_class2yolo_id

        self.class2id, self.class_names = build_class2yolo_id(
            class_names_json,
            yolo_weights=yolo_weights
        )
        
        self.img_size = img_size

        self.tf = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),  # -> float32 [0,1]
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, anno_path = self.items[idx]

        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        img = self.tf(pil)  # 3xSxS

        # parse original anno to YOLO labels (normalized to original W,H)
        labels = []
        if anno_path is not None:
            suf = anno_path.suffix.lower()
            if suf == ".xml":
                labels = parse_rtts_voc(anno_path, self.class2id, W, H)
            elif suf == ".txt":
                labels = parse_exdark_bbgt(anno_path, self.class2id, W, H)

        # convert to tensors YOLO wants: cls, xywh
        if len(labels) == 0:
            cls = torch.zeros((0, 1), dtype=torch.float32)
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            cls = torch.tensor([[l[0]] for l in labels], dtype=torch.float32)
            bboxes = torch.tensor([[l[1], l[2], l[3], l[4]] for l in labels], dtype=torch.float32)

        return {
            "img": img,
            "cls": cls,        # (n,1)
            "bboxes": bboxes,  # (n,4) normalized xywh
            "img_path": str(img_path),
        }


def collate_fn(batch):
    """
    Build Ultralytics-style batch dict:
      batch['img']: (B,3,S,S)
      batch['cls']: (N,1)
      batch['bboxes']: (N,4)
      batch['batch_idx']: (N,)
    """
    imgs = torch.stack([b["img"] for b in batch], 0)

    cls_all = []
    box_all = []
    batch_idx = []
    for i, b in enumerate(batch):
        n = b["cls"].shape[0]
        if n > 0:
            cls_all.append(b["cls"])
            box_all.append(b["bboxes"])
            batch_idx.append(torch.full((n,), i, dtype=torch.int64))

    if len(cls_all) == 0:
        cls_all = torch.zeros((0, 1), dtype=torch.float32)
        box_all = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.int64)
    else:
        cls_all = torch.cat(cls_all, 0)
        box_all = torch.cat(box_all, 0)
        batch_idx = torch.cat(batch_idx, 0)

    return {
        "img": imgs,
        "cls": cls_all,
        "bboxes": box_all,
        "batch_idx": batch_idx,
    }
