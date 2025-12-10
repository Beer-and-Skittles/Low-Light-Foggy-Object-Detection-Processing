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
    log_path = cfg["log_path"]
    ckpt_path = cfg["ckpt_path"]

    # -------------------- Dataset --------------------
    ds = TaskAwareDetectionDataset(
        class_names_json=cfg["class_names_json"],
        yolo_weights=cfg["yolo_weights"],
        split_txt=cfg["train_list"],
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
        # ----- Sanity check: is AODNet initially identity? -----
    aod.eval()
    with torch.no_grad():
        batch = next(iter(dl))
        imgs = batch["img"].to(device)  # assumed in [0,1]
        enh, k = aod(imgs)

        diff = (enh - imgs).abs().mean().item()
        k_mean = k.mean().item()
        k_min = k.min().item()
        k_max = k.max().item()

        print(f"[AOD Identity Check] mean |enh - img| = {diff:.8f}")
        print(f"[AOD Identity Check] k stats: mean={k_mean:.6f}, min={k_min:.6f}, max={k_max:.6f}")

    # -------------------- YOLO Teacher (Frozen) --------------------
    from types import SimpleNamespace
    from ultralytics import YOLO as YOLOWrapper

    yolo_wrapper = YOLOWrapper(cfg["yolo_weights"])
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
    print("criterion hyp:", {k: getattr(crit.hyp, k) for k in ["box", "cls", "dfl"]})

    # -------------------- Add missing YOLO loss hyperparameters --------------------
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

    # -------------------- Resume from checkpoint (optional) --------------------
    prev_epoch_in_ckpt = 0
    best_loss = float("inf")  # track best avg_total loss

    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt_path = Path(ckpt_path)
        print(f"[resume] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        aod.load_state_dict(ckpt["aod"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        prev_epoch_in_ckpt = ckpt.get("epoch", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"[resume] Checkpoint epoch: {prev_epoch_in_ckpt}")
        print(f"[resume] Previous best_loss: {best_loss}")
        start_epoch = prev_epoch_in_ckpt + 1
        print(f"[resume] Continuing from epoch {start_epoch}")
    else:
        start_epoch = 1

    # -------------------- Output directory & logging --------------------
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # If resuming and log exists, append; otherwise, create with header
    if ckpt_path is None or not Path(log_path).exists():
        with open(log_path, "w") as f:
            f.write("epoch,total_loss,det_loss,id_loss\n")

    lam_det = cfg.get("lambda_det", 1.0)
    lam_id = cfg.get("lambda_identity", 0.05)

    global_step = 0

    # =====================================================
    #                     Training Loop
    # =====================================================
    end_epoch = start_epoch + cfg["epochs"] - 1
    for epoch in range(start_epoch, end_epoch + 1):
        aod.train()
        running_total = 0.0
        running_det   = 0.0
        running_id    = 0.0

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

            running_total += loss.item()
            running_det   += det_loss.item()
            running_id    += id_loss.item()
            global_step += 1

        avg_total = running_total / max(1, len(dl))
        avg_det   = running_det   / max(1, len(dl))
        avg_id    = running_id    / max(1, len(dl))

        print(f"Epoch {epoch}/{end_epoch} | "
              f"loss={avg_total:.4f}  det={avg_det:.4f}  id={avg_id:.4f}")

        # ---- Append loss to log file ----
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_total:.6f},{avg_det:.6f},{avg_id:.6f}\n")

        # ---- Save *best* checkpoint only (by avg_total) ----
        if epoch % 10 == 0 or epoch <= 5:
            ckpt_path = out_dir / str("checkpoint_" + str(epoch) + ".pt")
            torch.save({
                "epoch": epoch,
                "aod": aod.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg,
                "best_loss": best_loss,
            }, ckpt_path)
            
        if avg_total < best_loss:
            best_loss = avg_total
            best_ckpt_path = out_dir / "best.pt"
            torch.save({
                "epoch": epoch,
                "aod": aod.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg,
                "best_loss": best_loss,
            }, best_ckpt_path)
            print(f"[ckpt] New best loss {best_loss:.4f} at epoch {epoch}. "
                  f"Saved to {best_ckpt_path}")

    print("Training done. Checkpoints saved to:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    main(
        cfg=cfg,
    )