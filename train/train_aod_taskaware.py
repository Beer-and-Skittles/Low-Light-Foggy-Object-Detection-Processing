# <Project>/train/train_aod_taskaware.py
import argparse, yaml, os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from datasets.taskaware_dataset import TaskAwareDetectionDataset, collate_fn
from models.aod_net import AODNet


def main(cfg):
    device = torch.device(cfg["device"])

    # -------------------- Dataset --------------------
    ds = TaskAwareDetectionDataset(
        split_txt=cfg["train_list"],
        class_names=cfg["class_names"],
        img_size=cfg["img_size"],
    )
    dl = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,   # WSL-safe
        collate_fn=collate_fn,
    )

    # -------------------- AOD-Net --------------------
    aod = AODNet().to(device)

    # -------------------- YOLO Teacher (Frozen) --------------------
    from types import SimpleNamespace
    from ultralytics import YOLO

    yolo_wrapper = YOLO(cfg["yolo_weights"])
    yolo = yolo_wrapper.model.to(device)
    yolo.eval()
    for p in yolo.parameters():
        p.requires_grad = False

    # Convert dict hyp/args → namespace if needed
    if hasattr(yolo, "args") and isinstance(yolo.args, dict):
        yolo.args = SimpleNamespace(**yolo.args)
    if hasattr(yolo, "hyp") and isinstance(yolo.hyp, dict):
        yolo.hyp = SimpleNamespace(**yolo.hyp)

    # Default gains YOLO loss needs
    default_hyp = {"box": 7.5, "cls": 0.5, "dfl": 1.5}

    # Ensure model.hyp has gains
    if not hasattr(yolo, "hyp") or yolo.hyp is None:
        yolo.hyp = SimpleNamespace(**default_hyp)
    else:
        for k, v in default_hyp.items():
            if not hasattr(yolo.hyp, k):
                setattr(yolo.hyp, k, v)

    # Build criterion
    if hasattr(yolo, "init_criterion"):
        yolo.criterion = yolo.init_criterion()

    # 🔥 CRITICAL FIX: criterion has its own hyp copy
    crit = yolo.criterion
    if hasattr(crit, "hyp") and isinstance(crit.hyp, dict):
        crit.hyp = SimpleNamespace(**crit.hyp)
    if not hasattr(crit, "hyp") or crit.hyp is None:
        crit.hyp = SimpleNamespace(**default_hyp)
    else:
        for k, v in default_hyp.items():
            if not hasattr(crit.hyp, k):
                setattr(crit.hyp, k, v)

    # (optional debug once)
    print("criterion hyp:", {k: getattr(crit.hyp, k) for k in ["box","cls","dfl"]})


    # -------------------- Add missing YOLO loss hyperparameters --------------------
    # YOLO loss expects: hyp.box, hyp.cls, hyp.dfl
    default_hyp = {
        "box": 7.5,  # bbox regression gain
        "cls": 0.5,  # classification gain
        "dfl": 1.5,  # distribution focal loss gain
    }

    if not hasattr(yolo, "hyp") or yolo.hyp is None:
        yolo.hyp = SimpleNamespace(**default_hyp)
    else:
        for k, v in default_hyp.items():
            if not hasattr(yolo.hyp, k):
                setattr(yolo.hyp, k, v)

    # Rebuild YOLO criterion AFTER fixing hyp
    if hasattr(yolo, "init_criterion"):
        yolo.criterion = yolo.init_criterion()

    # -------------------- Optimizer --------------------
    opt = torch.optim.Adam(aod.parameters(), lr=cfg["lr"])

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    lam_det = cfg.get("lambda_det", 1.0)
    lam_id = cfg.get("lambda_identity", 0.05)

    global_step = 0

    # =====================================================
    #                     Training Loop
    # =====================================================
    for epoch in range(cfg["epochs"]):
        aod.train()
        running = 0.0

        for batch in dl:
            imgs = batch["img"].to(device)  # (B,3,S,S)
            cls = batch["cls"].to(device)
            bboxes = batch["bboxes"].to(device)
            batch_idx = batch["batch_idx"].to(device)

            # ---- AOD enhancement ----
            enh, _ = aod(imgs)  # enh is the enhanced image

            # ---- YOLO-style batch dict ----
            yolo_batch = {
                "img": enh,
                "cls": cls,
                "bboxes": bboxes,
                "batch_idx": batch_idx,
            }

            # ---- YOLO detection loss ----
            out = yolo.loss(yolo_batch)
            if isinstance(out, tuple):
                det_loss, _ = out
            else:
                det_loss = out

            if det_loss.ndim > 0:
                det_loss = det_loss.mean()


            # ---- Identity loss (stability) ----
            id_loss = torch.mean(torch.abs(enh - imgs))

            # ---- Total loss ----
            loss = lam_det * det_loss + lam_id * id_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()
            global_step += 1

        avg = running / max(1, len(dl))
        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"loss={avg:.4f}  det={det_loss.item():.4f}  id={id_loss.item():.4f}")

        # ---- Save checkpoint ----
        ckpt_path = out_dir / f"aod_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "aod": aod.state_dict(),
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
