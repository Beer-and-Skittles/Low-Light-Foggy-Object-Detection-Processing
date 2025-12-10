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
        num_workers=0,
        collate_fn=collate_fn,
    )

    # -------------------- ResNet enhancer --------------------
    net = ResNetEnhancer(pretrained=cfg.get("pretrained", True)).to(device)

    # -------------------- YOLO teacher (frozen) --------------------
    yolo = build_frozen_yolo_teacher(cfg["yolo_weights"], device)

    # -------------------- Optimizer --------------------
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    # -------------------- Resume from checkpoint (optional) --------------------
    prev_epoch_in_ckpt = 0
    best_loss = float("inf")

    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt_path = Path(ckpt_path)
        print(f"[resume] Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["net"])
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

    # If starting fresh or log doesn't exist, create header
    if ckpt_path is None or not Path(log_path).exists():
        with open(log_path, "w") as f:
            f.write("epoch,total_loss,det_loss,id_loss\n")

    lam_det = cfg.get("lambda_det", 1.0)
    lam_id  = cfg.get("lambda_identity", 0.05)

    # =====================================================
    #                     Training Loop
    # =====================================================
    end_epoch = start_epoch + cfg["epochs"] - 1

    for epoch in range(start_epoch, end_epoch + 1):
        net.train()
        running_total = 0.0
        running_det   = 0.0
        running_id    = 0.0

        for batch in dl:
            imgs = batch["img"].to(device)
            cls = batch["cls"].to(device)
            bboxes = batch["bboxes"].to(device)
            batch_idx = batch["batch_idx"].to(device)

            # ---- ResNet enhancement ----
            enh = net(imgs)

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

            # ---- Identity loss ----
            id_loss = torch.mean(torch.abs(enh - imgs))

            # ---- Total loss ----
            loss = lam_det * det_loss + lam_id * id_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_total += loss.item()
            running_det   += det_loss.item()
            running_id    += id_loss.item()

        avg_total = running_total / max(1, len(dl))
        avg_det   = running_det   / max(1, len(dl))
        avg_id    = running_id    / max(1, len(dl))

        print(f"Epoch {epoch}/{end_epoch} | "
              f"loss={avg_total:.4f}  det={avg_det:.4f}  id={avg_id:.4f}")

        # ---- Append epoch losses to log ----
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_total:.6f},{avg_det:.6f},{avg_id:.6f}\n")

        # ---- Save *best* checkpoint only (by avg_total) ----
        if epoch % 10 == 0 or epoch <= 5:
            ckpt_path = out_dir / str("checkpoint_" + str(epoch) + ".pt")
            torch.save({
                "epoch": epoch,
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg,
                "best_loss": best_loss,
            }, ckpt_path)
            
        if avg_total < best_loss:
            best_loss = avg_total
            best_ckpt = out_dir / "resnet_best.pt"
            torch.save({
                "epoch": epoch,
                "net": net.state_dict(),
                "opt": opt.state_dict(),
                "cfg": cfg,
                "best_loss": best_loss,
            }, best_ckpt)
            print(f"[ckpt] New best loss {best_loss:.4f} at epoch {epoch}. "
                  f"Saved to {best_ckpt}")

    print("Training done. Best checkpoint stored in:", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg))
    main(
        cfg=cfg,
    )