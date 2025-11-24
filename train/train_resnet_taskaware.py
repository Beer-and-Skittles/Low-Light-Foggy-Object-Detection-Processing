# <Project>/train/train_resnet_taskaware.py
import argparse, yaml
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
from types import SimpleNamespace

from datasets.taskaware_dataset import TaskAwareDetectionDataset, collate_fn
from models.resnet_enhancer import ResNetEnhancer


def build_frozen_yolo_teacher(weights, device):
    yolo_wrapper = YOLO(weights)
    yolo = yolo_wrapper.model.to(device)
    yolo.eval()
    for p in yolo.parameters():
        p.requires_grad = False

    # dict -> namespace
    if hasattr(yolo, "args") and isinstance(yolo.args, dict):
        yolo.args = SimpleNamespace(**yolo.args)
    if hasattr(yolo, "hyp") and isinstance(yolo.hyp, dict):
        yolo.hyp = SimpleNamespace(**yolo.hyp)

    # default gains
    default_hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
    if not hasattr(yolo, "hyp") or yolo.hyp is None:
        yolo.hyp = SimpleNamespace(**default_hyp)
    else:
        for k, v in default_hyp.items():
            if not hasattr(yolo.hyp, k):
                setattr(yolo.hyp, k, v)

    if hasattr(yolo, "init_criterion"):
        yolo.criterion = yolo.init_criterion()

    # criterion hyp copy fix
    crit = yolo.criterion
    if hasattr(crit, "hyp") and isinstance(crit.hyp, dict):
        crit.hyp = SimpleNamespace(**crit.hyp)
    if not hasattr(crit, "hyp") or crit.hyp is None:
        crit.hyp = SimpleNamespace(**default_hyp)
    else:
        for k, v in default_hyp.items():
            if not hasattr(crit.hyp, k):
                setattr(crit.hyp, k, v)

    return yolo


def main(cfg):
    device = torch.device(cfg["device"])

    # dataset
    ds = TaskAwareDetectionDataset(
        split_txt=cfg["train_list"],
        class_names=cfg["class_names"],
        img_size=cfg["img_size"],
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ResNet enhancer
    net = ResNetEnhancer(pretrained=cfg.get("pretrained", True)).to(device)

    # YOLO teacher
    yolo = build_frozen_yolo_teacher(cfg["yolo_weights"], device)

    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_det = cfg.get("lambda_det", 1.0)
    lam_id  = cfg.get("lambda_identity", 0.05)

    for epoch in range(cfg["epochs"]):
        net.train()
        running = 0.0

        for batch in dl:
            imgs = batch["img"].to(device)
            cls = batch["cls"].to(device)
            bboxes = batch["bboxes"].to(device)
            batch_idx = batch["batch_idx"].to(device)

            enh = net(imgs)

            yolo_batch = {
                "img": enh,
                "cls": cls,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
            }

            out = yolo.loss(yolo_batch)
            if isinstance(out, tuple):
                det_loss, _ = out
            else:
                det_loss = out
            if det_loss.ndim > 0:
                det_loss = det_loss.mean()

            id_loss = torch.mean(torch.abs(enh - imgs))
            loss = lam_det * det_loss + lam_id * id_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch+1}/{cfg['epochs']} | loss={avg:.4f}  det={det_loss.item():.4f}  id={id_loss.item():.4f}")

        ckpt_path = out_dir / f"resnet_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "cfg": cfg,
        }, ckpt_path)

    print("Training done. Checkpoints saved to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.cfg))
    main(cfg)
