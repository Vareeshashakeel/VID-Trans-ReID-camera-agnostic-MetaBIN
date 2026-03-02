# VID_Trans_ReID.py  (camera-agnostic, stable, PyTorch>=2.0 compatible)

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import amp

from Dataloader import dataloader
from VID_Trans_model import VID_Trans
from Loss_fun import make_loss
from utility import AverageMeter, optimizer as build_optimizer, scheduler as build_scheduler


# -------------------- Reproducibility --------------------
def set_seed(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------- Evaluation --------------------
def evaluate(distmat: np.ndarray,
             q_pids: np.ndarray,
             g_pids: np.ndarray,
             q_camids: np.ndarray,
             g_camids: np.ndarray,
             max_rank: int = 21):
    num_q, num_g = distmat.shape
    max_rank = min(max_rank, num_g)

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

    all_cmc, all_AP = [], []
    num_valid_q = 0

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = ~remove

        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = np.asarray([x / (i + 1.0) for i, x in enumerate(tmp_cmc)]) * orig_cmc
        all_AP.append(tmp_cmc.sum() / num_rel)

    if num_valid_q == 0:
        raise RuntimeError("No valid queries for evaluation (all query IDs missing in gallery).")

    all_cmc = np.asarray(all_cmc, dtype=np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    return all_cmc, mAP


# -------------------- Feature Extraction --------------------
@torch.inference_mode()
def extract_features(model, loader, device: torch.device):
    """
    Handles both:
      - 5D: [B, T, C, H, W]
      - 6D: [B, K, T, C, H, W] (K clips per tracklet)
    Returns:
      feats: torch.Tensor [N, D]
      pids, camids: np.ndarray [N]
    """
    model.eval()
    feats, pids, camids = [], [], []

    for imgs, pid, camid, _paths in loader:
        # Move to device + correct shape
        if imgs.dim() == 6:
            # [B, K, T, C, H, W] -> [B*K, T, C, H, W]
            B, K, T, C, H, W = imgs.shape
            imgs = imgs.view(B * K, T, C, H, W).to(device, non_blocking=True)

            f = model(imgs).detach()          # [B*K, D]
            f = f.view(B, K, -1).mean(dim=1)  # [B, D]
        else:
            # [B, T, C, H, W]
            imgs = imgs.to(device, non_blocking=True)
            f = model(imgs).detach()          # [B, D]

        # Move to CPU for distance computation
        f = f.cpu()

        # Keep per-tracklet features (B is usually 1 in query/gallery loaders)
        # If B>1, we append all of them.
        if f.dim() == 1:
            f = f.unsqueeze(0)

        feats.append(f)

        # pid/camid can be tensor; normalize to python ints per element
        if torch.is_tensor(pid):
            pid_list = pid.detach().cpu().tolist()
        else:
            pid_list = list(pid) if isinstance(pid, (list, tuple)) else [int(pid)]

        if torch.is_tensor(camid):
            cam_list = camid.detach().cpu().tolist()
        else:
            cam_list = list(camid) if isinstance(camid, (list, tuple)) else [int(camid)]

        pids.extend([int(x) for x in pid_list])
        camids.extend([int(x) for x in cam_list])

    feats = torch.cat(feats, dim=0)  # [N, D]
    return feats, np.asarray(pids), np.asarray(camids)


@torch.inference_mode()
def test(model, query_loader, gallery_loader, device: torch.device):
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)

    # Euclidean distance matrix
    distmat = torch.cdist(qf, gf, p=2).numpy()

    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    return float(cmc[0]), float(mAP)


# -------------------- Checkpoint --------------------
def save_checkpoint(path: Path, model, epoch: int, best_r1: float, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_r1": best_r1,
            "args": vars(args),
        },
        str(path),
    )


# -------------------- Main Training --------------------
def main():
    parser = argparse.ArgumentParser("VID-Trans-ReID (camera-agnostic, stable, PyTorch2 AMP OK)")
    parser.add_argument("--Dataset_name", required=True, choices=["Mars", "iLIDSVID", "PRID"])
    parser.add_argument("--pretrain_path", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--do_eval", action="store_true")

    # stability knobs
    parser.add_argument("--attn_w", type=float, default=0.1)
    parser.add_argument("--center_w", type=float, default=0.0005)
    parser.add_argument("--center_lr", type=float, default=0.05)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # loaders
    train_loader, _num_query, num_classes, camera_num, _view_num, query_loader, gallery_loader = dataloader(
        args.Dataset_name,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_workers=args.num_workers,
    )

    # model
    # Keep args names exactly as your VID_Trans expects:
    # In your previous working run you used: VID_Trans(num_classes=..., camera_num=..., pretrainpath=...)
    model = VID_Trans(
        num_classes=num_classes,
        camera_num=camera_num,
        pretrainpath=args.pretrain_path
    ).to(device)

    # losses
    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    center_criterion = center_criterion.to(device)

    # optimizers
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=args.center_lr)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    # AMP scaler (PyTorch2 compatible)
    scaler = amp.GradScaler(enabled=(device.type == "cuda"))

    loss_meter, acc_meter = AverageMeter(), AverageMeter()

    best_r1 = -1.0
    best_path = outdir / f"{args.Dataset_name}_camera_agnostic_best.pth"
    latest_path = outdir / "latest.pth"
    last_path = outdir / "last_epoch.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()

        # scheduler step (robust)
        if hasattr(scheduler, "step"):
            try:
                scheduler.step(epoch)
            except TypeError:
                scheduler.step()

        loss_meter.reset()
        acc_meter.reset()

        for it, (img, pid, _camid, erase_mask) in enumerate(train_loader, start=1):
            img = img.to(device, non_blocking=True)
            pid = pid.to(device, non_blocking=True)
            erase_mask = erase_mask.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            optimizer_center.zero_grad(set_to_none=True)

            # ✅ FIX: PyTorch>=2.0 requires device_type
            with amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Your model signature (from your working logs): model(img, pid)
                scores, feats, a_vals = model(img, pid)

                loss_id, center = loss_fun(scores, feats, pid)

                # attention loss (bounded, stable)
                # If shapes broadcast, this is fine; otherwise adjust in your model output.
                attn_loss = (a_vals * erase_mask).sum(dim=1).mean() if a_vals.dim() >= 2 else (a_vals * erase_mask).mean()

                loss = loss_id + args.center_w * center + args.attn_w * attn_loss

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"NaN/Inf loss detected at epoch={epoch} iter={it} | "
                    f"id={float(loss_id.detach()):.4f} center={float(center.detach()):.4f} attn={float(attn_loss.detach()):.4f}"
                )

            scaler.scale(loss).backward()

            # gradient clipping (after unscale)
            if args.clip_grad and args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            # center step (no grad-mult hack; already weighted by center_w)
            optimizer_center.step()

            # accuracy
            if isinstance(scores, list):
                pred = scores[0].argmax(dim=1)
            else:
                pred = scores.argmax(dim=1)
            acc = (pred == pid).float().mean()

            loss_meter.update(float(loss.detach().item()), img.size(0))
            acc_meter.update(float(acc.detach().item()), 1)

            if it % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch[{epoch}/{args.epochs}] Iter[{it}/{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} Acc: {acc_meter.avg:.4f} LR: {lr:.2e} | "
                    f"id={float(loss_id.detach()):.3f} center={float(center.detach()):.3f} attn={float(attn_loss.detach()):.3f}"
                )

        # checkpoints
        save_checkpoint(latest_path, model, epoch, best_r1, args)
        save_checkpoint(last_path, model, epoch, best_r1, args)
        print(f"[OK] Saved checkpoint: {latest_path}")
        print(f"[OK] Saved checkpoint: {last_path}")

        # evaluation
        if args.do_eval and (epoch % args.eval_every == 0):
            r1, mAP = test(model, query_loader, gallery_loader, device)
            print(f"[Eval @ epoch {epoch}] Rank-1: {r1:.4f} mAP: {mAP:.4f}")
            if r1 > best_r1:
                best_r1 = r1
                save_checkpoint(best_path, model, epoch, best_r1, args)
                print(f"[OK] Saved BEST checkpoint: {best_path} (Rank-1={best_r1:.4f})")

    print("Training finished.")
    print(f"Best Rank-1: {best_r1:.4f}")
    print(f"Latest checkpoint: {latest_path}")
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
