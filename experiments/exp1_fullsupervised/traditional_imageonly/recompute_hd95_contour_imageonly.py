"""Recompute image-only U-Net / TransUNet HD95 with the CONTOUR definition.

Why this exists
---------------
The same two-implementation split that affected the box-conditioned block is
present in the image-only block of Table 1:

* ``run_nnunet.py:25`` erodes each mask to its contour first — the conventional
  definition — so the image-only nnU-Net rows are already contour-based.
* U-Net and TransUNet go through ``Trainer.evaluate`` ->
  ``thyroidbench.metrics.compute_hd95``, which takes distances from **every**
  mask pixel. Interior pixels contribute 0, so the 95th percentile falls inside
  that zero mass and the value is several times smaller than a true HD95.

So the published image-only HD95 column compares nnU-Net under a stricter metric
than the other two, exactly as in the box-conditioned block. This script
re-scores the image-only U-Net and TransUNet checkpoints under the contour
definition. DSC is written alongside as a provenance check: it must reproduce
the published value, since no other change is applied here.

This is the image-only twin of
``experiments/01v2_fullsup_boxcond/recompute_hd95_contour.py`` and uses the same
contour function, the same empty-prediction cap and the same output schema. The
only differences are the checkpoint layout, the 3-channel network and the
absence of a box channel.

Outputs
-------
``results/hd95_contour/{model}_{dataset}_per_image.csv`` and a single
``results/hd95_contour/summary.csv``.

Usage
-----
    python experiments/01_fullsup_traditional/recompute_hd95_contour_imageonly.py
    python experiments/01_fullsup_traditional/recompute_hd95_contour_imageonly.py --datasets ddti
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from thyroidbench.data import get_dataloader  # noqa: E402
from thyroidbench.models.transunet import build_transunet  # noqa: E402
from thyroidbench.models.unet import build_unet  # noqa: E402

logger = logging.getLogger(__name__)

_BUILD = {"unet": build_unet, "transunet": build_transunet}
_NAME = {"unet": "unet_resnet50", "transunet": "transunet_r50_vitb16"}
_EXP_DIR = REPO / "experiments/01_fullsup_traditional"
OUT_DIR = _EXP_DIR / "results/hd95_contour"
HD95_INF_CAP = float(np.sqrt(2) * 512)  # image diagonal; matches compute_stats.py policy.


def hd95_contour(pred: np.ndarray, gt: np.ndarray) -> float:
    """Contour-based HD95 in pixels — identical to eval_nnunet_boxcond.py:hd95."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return HD95_INF_CAP
    pb = pred ^ binary_erosion(pred)
    gb = gt ^ binary_erosion(gt)
    dt_p = distance_transform_edt(~pb.astype(bool))
    dt_g = distance_transform_edt(~gb.astype(bool))
    return float(np.percentile(np.concatenate([dt_g[pb > 0], dt_p[gb > 0]]), 95))


def hd95_maskwise(pred: np.ndarray, gt: np.ndarray) -> float:
    """The superseded definition, kept so the delta stays auditable."""
    pb, gb = pred.astype(bool), gt.astype(bool)
    if not pb.any() or not gb.any():
        return float("inf")
    return float(np.percentile(np.concatenate(
        [distance_transform_edt(~gb)[pb], distance_transform_edt(~pb)[gb]]), 95))


def _find_checkpoint(model: str, dataset: str) -> Optional[Path]:
    """Checkpoint of the BEST LR for this model/dataset (from the results CSV).

    The LR search saves one *_best.pt per LR; picking by mtime would grab the
    last LR trained, NOT the best. Read the LR from the results CSV — its last
    row is the configuration reported in the paper — and build the matching
    checkpoint name. U-Net files are ``{dataset}_lr{lr}_best.pt``; TransUNet
    files carry an extra ``transunet_`` prefix.
    """
    ck_dir = _EXP_DIR / "results" / model / dataset / "checkpoints"
    res_csv = (_EXP_DIR / "results" / model / dataset / "results"
               / f"{dataset}_{model}_results.csv")
    prefix = "transunet_" if model == "transunet" else ""
    if res_csv.exists():
        best_lr = float(pd.read_csv(res_csv).iloc[-1]["lr"])
        ck = ck_dir / f"{prefix}{dataset}_lr{best_lr:.0e}_best.pt"
        if ck.exists():
            return ck
    # Fallback: most recent if CSV/name lookup fails.
    cands = sorted(ck_dir.glob(f"{prefix}{dataset}_lr*_best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _published_dice(model: str, dataset: str) -> Optional[float]:
    """test_dice of the reported run, for the provenance check."""
    res_csv = (_EXP_DIR / "results" / model / dataset / "results"
               / f"{dataset}_{model}_results.csv")
    if not res_csv.exists():
        return None
    return float(pd.read_csv(res_csv).iloc[-1]["test_dice"])


@torch.no_grad()
def evaluate(model_key: str, dataset: str, device: str, seed: int) -> pd.DataFrame:
    """Per-image DSC and both HD95 flavours for one (model, dataset) cell."""
    ckpt = _find_checkpoint(model_key, dataset)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(f"no image-only checkpoint for {model_key}/{dataset}")
    logger.info("%s/%s <- %s", model_key, dataset, ckpt.name)

    net = _BUILD[model_key](pretrained=False, device=device, in_channels=3)
    net.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"])
    net.eval()

    loader = get_dataloader(dataset, "test", REPO / "data", "full_supervised",
                            batch_size=16, num_workers=4, seed=seed)

    rows: List[Dict[str, object]] = []
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = net(images)
        preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(np.uint8)
        gts = masks.cpu().numpy().astype(np.uint8)
        names = batch.get("filename", [""] * preds.shape[0])
        for i in range(preds.shape[0]):
            p, g = preds[i, 0], gts[i, 0]
            rows.append({
                "filename": names[i],
                "model": _NAME[model_key],
                "dataset": dataset,
                "dice": 2 * float((p & g).sum()) / float(p.sum() + g.sum() + 1e-10),
                "pred_empty": int(p.sum() == 0),
                "hd95_contour": hd95_contour(p, g),
                "hd95_maskwise": hd95_maskwise(p, g),
            })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["unet", "transunet"], choices=list(_BUILD))
    ap.add_argument("--datasets", nargs="+",
                    default=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, object]] = []
    for ds in args.datasets:
        for mk in args.models:
            df = evaluate(mk, ds, device, args.seed)
            df.to_csv(OUT_DIR / f"{_NAME[mk]}_{ds}_per_image.csv", index=False)
            # Both flavours reported impute-aware: an empty prediction is a
            # failure, not a missing value, so it must not vanish from the mean.
            mw = df["hd95_maskwise"].replace([np.inf, -np.inf], HD95_INF_CAP)
            summary.append({
                "model": _NAME[mk], "dataset": ds, "n_images": len(df),
                "n_pred_empty": int(df["pred_empty"].sum()),
                "dice_mean": round(float(df["dice"].mean()), 6),
                "dice_published": _published_dice(mk, ds),
                "hd95_contour_mean": round(float(df["hd95_contour"].mean()), 4),
                "hd95_maskwise_mean": round(float(mw.mean()), 4),
                "seed": args.seed,
            })
            logger.info("  DSC %.4f (published %s) | HD95 contour %.3f | maskwise %.3f",
                        summary[-1]["dice_mean"], summary[-1]["dice_published"],
                        summary[-1]["hd95_contour_mean"], summary[-1]["hd95_maskwise_mean"])

    out = pd.DataFrame(summary)
    out.to_csv(OUT_DIR / "summary.csv", index=False)
    logger.info("wrote %s", OUT_DIR / "summary.csv")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
