#!/usr/bin/env python3
"""
Box-sensitivity evaluation + box-fill NULL floor for box-conditioned CNNs.

For a trained box-conditioned U-Net / TransUNet checkpoint, this evaluates the
model on the test split under each eval-box mode:

    tight | jitter10 | jitter25 | shifted | adversarial

and, for each mode, also computes the **box-fill NULL floor** — the Dice you get
by painting the (mode-specific) box itself as the prediction, with NO network.
The headline contribution of the model is then ``model_dice - floor_dice`` (A5 /
D4): a small delta means the model is just filling the box (box-collapse); a
large delta means it refines using the image. The adversarial floor (~0) is the
yardstick that makes the box-collapse argument interpretable (C2).

Design notes:
  * The box channel and the floor box are rasterised with the SAME
    ``boxes.boxes_to_channel`` the model was trained with (SF-3), so the
    floor box is identical to the box the model sees.
  * Model Dice and floor Dice use the identical ``compute_metrics`` path; the
    binary floor box is fed as saturated pseudo-logits so the same sigmoid+
    threshold pipeline applies.
  * jitter modes are seeded per (mode, batch_idx) → reproducible (D5).
  * Full-image fallbacks (empty mask → all-ones box) are counted and reported
    (MIN-2).

Usage:
    python experiments/exp1_fullsupervised/traditional_boxcond/box_eval.py \
        --model unet --dataset ddti \
        --checkpoint experiments/exp1_fullsupervised/traditional_boxcond/results/unet/checkpoints/box_unet_ddti_lr1e-03_best.pt
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thyroidbench.data import get_dataloader
from thyroidbench.metrics import compute_metrics
from thyroidbench.models.transunet import build_transunet
from thyroidbench.models.unet import build_unet
from thyroidbench.boxes import (
    EVAL_BOX_MODES,
    boxes_to_channel,
    build_eval_boxes,
    count_fullimage_fallbacks,
)

logger = logging.getLogger(__name__)

_BUILD = {"unet": build_unet, "transunet": build_transunet}
_NAME = {"unet": "unet_resnet50", "transunet": "transunet_r50_vitb16"}
_EXP_DIR = "experiments/exp1_fullsupervised/traditional_boxcond"

# Dice here is batch-weighted micro-Dice (thyroidbench.metrics), so model_dice
# matches run.py's test_dice ONLY at the training batch size. Pin it (SF-1).
TRAINING_BATCH = 16


def _box_generator(seed: int, mode: str, batch_idx: int) -> torch.Generator:
    """Deterministic CPU RNG for a (mode, batch) so jitter modes are reproducible."""
    mode_id = EVAL_BOX_MODES.index(mode)
    s = (seed * 1_000_003 + mode_id * 7_919 + batch_idx) % (2**31 - 1)
    g = torch.Generator()
    g.manual_seed(int(s))
    return g


def _find_checkpoint(model: str, dataset: str) -> Optional[Path]:
    """Checkpoint of the BEST LR for this model/dataset (from the results CSV).

    The LR search saves one *_best.pt per LR; picking by mtime would grab the
    last LR trained, NOT the best. Read the best LR from the results CSV and
    build the matching checkpoint name (box_{model}_{dataset}_lr{lr:.0e}_best.pt).
    """
    ck_dir = Path(_EXP_DIR) / "results" / model / "checkpoints"
    res_csv = Path(_EXP_DIR) / "results" / model / f"{dataset}_{model}_boxcond_results.csv"
    if res_csv.exists():
        best_lr = float(pd.read_csv(res_csv).iloc[-1]["lr"])
        ck = ck_dir / f"box_{model}_{dataset}_lr{best_lr:.0e}_best.pt"
        if ck.exists():
            return ck
    # Fallback: most recent if CSV/name lookup fails.
    cands = sorted(ck_dir.glob(f"box_{model}_{dataset}_*_best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


@torch.no_grad()
def evaluate_modes(model: torch.nn.Module, loader, device: str, seed: int,
                   modes: List[str]) -> List[Dict[str, float]]:
    """Return one metrics row per eval-box mode (model + floor + delta)."""
    model.eval()
    rows: List[Dict[str, float]] = []

    for mode in modes:
        # Per-mode running sums (sample-weighted means over the test set).
        m_dice = m_iou = f_dice = f_iou = 0.0
        n_img = 0
        n_fallback = 0

        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            H, W = images.shape[-2:]
            # SF-2: tight-box identity across regimes assumes hard-binary masks
            # (foundation get_gt_box thresholds >0, ours >0.5 — identical only on
            # {0,1} masks). Assert no antialiased values leaked through resize.
            if batch_idx == 0 and mode == modes[0]:
                uniq = torch.unique(masks)
                assert bool(((uniq == 0) | (uniq == 1)).all()), \
                    f"masks not hard-binary (values {uniq[:8].tolist()}) — box derivation would differ across regimes"
            n_fallback += count_fullimage_fallbacks(masks)

            gen = _box_generator(seed, mode, batch_idx)
            boxes = build_eval_boxes(masks, mode, H=H, W=W, generator=gen)
            box_chan = boxes_to_channel(boxes, H=H, W=W)        # (B,1,H,W) {0,1}

            # --- model under this box mode ---
            net_in = torch.cat([images, box_chan], dim=1)
            logits = model(net_in)
            mm = compute_metrics(logits, masks)

            # --- box-fill NULL floor: the box itself as the prediction ---
            floor_logits = (box_chan - 0.5) * 20.0   # saturate → sigmoid≈{0,1}
            fm = compute_metrics(floor_logits, masks)

            b = images.shape[0]
            m_dice += mm["dice"] * b; m_iou += mm["iou"] * b
            f_dice += fm["dice"] * b; f_iou += fm["iou"] * b
            n_img += b

        rows.append({
            "eval_box_mode": mode,
            "model_dice": round(m_dice / n_img, 6),
            "model_iou":  round(m_iou / n_img, 6),
            "floor_dice": round(f_dice / n_img, 6),
            "floor_iou":  round(f_iou / n_img, 6),
            "delta_dice": round((m_dice - f_dice) / n_img, 6),
            "n_images":   n_img,
            "n_fullimage_fallback": n_fallback,
        })
        logger.info("mode=%-11s model_dice=%.4f floor_dice=%.4f Δ=%.4f (fallback=%d)",
                    mode, rows[-1]["model_dice"], rows[-1]["floor_dice"],
                    rows[-1]["delta_dice"], n_fallback)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-sensitivity + box-fill floor eval")
    p.add_argument("--model",   required=True, choices=list(_BUILD))
    p.add_argument("--dataset", required=True,
                   choices=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    p.add_argument("--checkpoint", default=None,
                   help="path to *_best.pt (default: most recent for model/dataset)")
    p.add_argument("--modes", nargs="+", default=list(EVAL_BOX_MODES),
                   choices=list(EVAL_BOX_MODES))
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        device = "cpu"

    ckpt = Path(args.checkpoint) if args.checkpoint else _find_checkpoint(args.model, args.dataset)
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint for {args.model}/{args.dataset} "
            f"(looked in {_EXP_DIR}/results/{args.model}/checkpoints). "
            f"Pass --checkpoint explicitly."
        )
    logger.info("Checkpoint: %s", ckpt)
    if args.batch_size != TRAINING_BATCH:
        logger.warning("--batch_size %d != training batch %d: batch-weighted micro-Dice "
                       "will drift from run.py test_dice (SF-1). Delta-over-floor stays valid.",
                       args.batch_size, TRAINING_BATCH)

    model = _BUILD[args.model](pretrained=False, device=device, in_channels=4)
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    logger.info("Loaded %s (epoch %d, val_dice=%.4f)",
                args.model, state.get("epoch", -1), state.get("best_val_dice", float("nan")))

    loader = get_dataloader(args.dataset, "test", args.data_root, "full_supervised",
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            seed=args.seed)

    rows = evaluate_modes(model, loader, device, args.seed, args.modes)
    for r in rows:
        r.update({"dataset": args.dataset, "model": _NAME[args.model],
                  "box_conditioned": True,
                  "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                  "seed": args.seed})

    out_csv = Path(args.out_csv or
                   Path(_EXP_DIR) / "results" / args.model /
                   f"{args.dataset}_{args.model}_box_sensitivity.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["dataset", "model", "box_conditioned", "eval_box_mode",
            "model_dice", "model_iou", "floor_dice", "floor_iou", "delta_dice",
            "n_images", "n_fullimage_fallback", "seed", "timestamp"]
    pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)


if __name__ == "__main__":
    main()
