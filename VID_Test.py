import argparse
import numpy as np
import torch

from Dataloader import dataloader
from VID_Trans_model import VID_Trans


def evaluate(distmat: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
             q_camids: np.ndarray, g_camids: np.ndarray, max_rank: int = 21):
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
        raise RuntimeError("All query identities do not appear in gallery.")

    all_cmc = np.asarray(all_cmc, dtype=np.float32).sum(0) / num_valid_q
    mAP = float(np.mean(all_AP))
    return all_cmc, mAP


def _to_tracklet_feature(model_output: torch.Tensor) -> torch.Tensor:
    """
    Convert model output to a single feature vector per clip.
    Handles common shapes:
      - [N, D]
      - [N, tokens, D] -> take cls token [:, 0, :]
    """
    x = model_output
    if isinstance(x, (list, tuple)):
        x = x[0]

    # If output is token sequence, take CLS token
    if x.dim() == 3:
        x = x[:, 0, :]  # [N, D]

    # Now expect [N, D]
    if x.dim() != 2:
        raise RuntimeError(f"Unexpected model output shape: {tuple(x.shape)}")

    return x


@torch.inference_mode()
def extract_features(model, loader, device: torch.device):
    model.eval()
    feats, pids, camids = [], [], []

    for imgs, pid, camid, _paths in loader:
        # imgs: [1, Nclips, T, C, H, W] from dense sampling
        # convert to [Nclips, T, C, H, W] (5D) so model forward works
        imgs = imgs.squeeze(0).to(device, non_blocking=True)

        # Forward (camera-agnostic)
        out = model(imgs)

        clip_feats = _to_tracklet_feature(out)  # [Nclips, D]
        tracklet_feat = clip_feats.mean(dim=0)  # [D]
        tracklet_feat = tracklet_feat.detach().cpu()

        feats.append(tracklet_feat)

        # pid is list[int] from val_collate_fn
        pids.append(int(pid[0]) if isinstance(pid, (list, tuple)) else int(pid))

        # camid is tensor [1] from val_collate_fn
        camids.append(int(camid[0].item()) if hasattr(camid[0], "item") else int(camid[0]))

    feats = torch.stack(feats, dim=0)  # [num_tracklets, D]
    return feats, np.asarray(pids), np.asarray(camids)


@torch.inference_mode()
def test(model, query_loader, gallery_loader, device: torch.device):
    qf, q_pids, q_camids = extract_features(model, query_loader, device)
    gf, g_pids, g_camids = extract_features(model, gallery_loader, device)

    distmat = torch.cdist(qf, gf, p=2).numpy()
    cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)
    return float(cmc[0]), float(mAP)


def load_checkpoint_to_model(model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys ignored: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys ignored: {len(unexpected)}")


def main():
    parser = argparse.ArgumentParser("VID-Trans-ReID Test (camera-agnostic)")
    parser.add_argument("--Dataset_name", required=True, choices=["Mars", "iLIDSVID", "PRID"])
    parser.add_argument("--pretrain_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build loaders (we only need query/gallery loaders + num_classes)
    _train_loader, _num_query, num_classes, camera_num, _view_num, query_loader, gallery_loader = dataloader(
        args.Dataset_name, batch_size=64, seq_len=args.seq_len, num_workers=args.num_workers
    )

    model = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.pretrain_path).to(device)
    load_checkpoint_to_model(model, args.model_path)

    r1, mAP = test(model, query_loader, gallery_loader, device)
    print(f"✅ Test Results | Rank-1: {r1:.4f} | mAP: {mAP:.4f}")


if __name__ == "__main__":
    main()
