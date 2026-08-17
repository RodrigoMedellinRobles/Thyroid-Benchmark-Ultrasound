#!/usr/bin/env python3
"""
Cross-dataset transfer for BOX-CONDITIONED CNN baselines (U-Net, TransUNet).

Stream A of the box-conditioned cross-dataset experiment. EVAL-ONLY: the
box-conditioned checkpoints already exist (Experiment 1 box-conditioned, one best-LR
checkpoint per model/dataset). For every (model, source dataset) we load the
checkpoint trained on the source and evaluate it on every target dataset's
test split, feeding the SAME oracle tight box (4th input channel) the model
was trained with. This is the box-conditioned analogue of the foundation
cross-dataset transfer in the foundation cross-dataset evaluator and produces rows with the
identical per-image / summary schema so the two merge into one table.

Diagonal cells (src == tgt) are kept on purpose: they must reproduce the Exp 1
in-domain box-conditioned tight-box numbers and therefore act as a built-in
sanity check on the loader / box / checkpoint plumbing.

nnU-Net box-conditioned cross-dataset is a separate pipeline (Stream B,
``eval_nnunet_cross.py``) because nnU-Net evaluates through its own predictor.

Usage (smoke, one cell):
    python experiments/exp4_crossdataset/traditional/run.py \
        --model unet --src ddti --tgt tn3k

Usage (full Stream A: unet+transunet, all 16 src x tgt):
    python experiments/exp4_crossdataset/traditional/run.py --model all
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thyroidbench.data import get_dataloader
from thyroidbench.metrics import compute_hd95, compute_metrics
from thyroidbench.models.transunet import build_transunet
from thyroidbench.models.unet import build_unet
from thyroidbench.boxes import boxes_to_channel, build_eval_boxes

logger = logging.getLogger(__name__)

_BUILD = {"unet": build_unet, "transunet": build_transunet}
_NAME = {"unet": "unet_resnet50", "transunet": "transunet_r50_vitb16"}
_DATASETS = ["ddti", "tn3k", "thyroidxl", "stanford_aimi"]
_BOXCOND_DIR = Path("experiments/exp1_fullsupervised/traditional_boxcond")
_OUT_DIR = Path("experiments/exp4_crossdataset/traditional/results")


def _find_checkpoint(model: str, dataset: str) -> Optional[Path]:
    """Best-LR box-conditioned checkpoint for (model, dataset), from Exp 1.

    Mirrors ``box_eval._find_checkpoint``: the LR search saves one *_best.pt per
    LR, so we read the winning LR from the per-dataset results CSV rather than
    guessing by mtime. Falls back to most-recent if the CSV/name lookup fails.
    """
    ck_dir = _BOXCOND_DIR / "results" / model / "checkpoints"
    res_csv = _BOXCOND_DIR / "results" / model / f"{dataset}_{model}_boxcond_results.csv"
    if res_csv.exists():
        best_lr = float(pd.read_csv(res_csv).iloc[-1]["lr"])
        ck = ck_dir / f"box_{model}_{dataset}_lr{best_lr:.0e}_best.pt"
        if ck.exists():
            return ck
    cands = sorted(ck_dir.glob(f"box_{model}_{dataset}_*_best.pt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


@torch.no_grad()
def evaluate_test(model: torch.nn.Module, loader, device: str, seed: int) -> List[Dict]:
    """Per-image tight-box evaluation, schema identical to 05/run.py evaluate_test."""
    model.eval()
    results: List[Dict] = []
    # tight mode ignores jitter, but build_eval_boxes still wants a generator.
    gen = torch.Generator().manual_seed(seed)

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        filenames = batch.get("filename", [f"img_{len(results)+i}" for i in range(images.shape[0])])
        patient_ids = batch.get("patient_id", ["NA"] * images.shape[0])
        H, W = images.shape[-2:]

        # Hard-binary mask guard (box derivation is only regime-consistent on {0,1}).
        if not results:
            uniq = torch.unique(masks)
            assert bool(((uniq == 0) | (uniq == 1)).all()), \
                f"masks not hard-binary (values {uniq[:8].tolist()})"

        boxes = build_eval_boxes(masks, "tight", H=H, W=W, generator=gen)
        box_chan = boxes_to_channel(boxes, H=H, W=W)            # (B,1,H,W) {0,1}
        logits = model(torch.cat([images, box_chan], dim=1)).float()

        B = images.shape[0]
        for i in range(B):
            li = logits[i:i+1]
            mi = masks[i:i+1]
            m = compute_metrics(li, mi)
            pred_bin = (torch.sigmoid(li) >= 0.5).cpu().numpy()[0, 0].astype(np.uint8)
            mask_bin = mi[0, 0].cpu().numpy().astype(np.uint8)
            hd95 = compute_hd95(pred_bin, mask_bin)
            results.append({
                "filename":   filenames[i],
                "patient_id": patient_ids[i],
                "dice":       round(m["dice"],      6),
                "iou":        round(m["iou"],       6),
                "precision":  round(m["precision"], 6),
                "recall":     round(m["recall"],    6),
                "hd95":       round(hd95, 4) if np.isfinite(hd95) else "",
                "skipped":    False,
            })
    return results


def compute_summary(model: str, src: str, tgt: str, results: List[Dict]) -> Dict:
    """Aggregate per-image rows; schema mirrors 05/run.py compute_summary."""
    valid = [r for r in results if not r["skipped"]]
    n_total = len(results)
    dice = np.array([r["dice"] for r in valid], dtype=float)
    iou = np.array([r["iou"] for r in valid], dtype=float)
    prec = np.array([r["precision"] for r in valid], dtype=float)
    rec = np.array([r["recall"] for r in valid], dtype=float)
    # Impute-aware HD95 (paper-wide policy). Empty / non-finite predictions are
    # penalised at the image diagonal (sqrt(2)*512 = 724.077 px) BEFORE the
    # per-image mean, rather than being silently dropped. Dropping empties
    # understates HD95 and was the defect that corrupted the cross-dataset
    # figure and Supplementary S4/S5 tables. Ground truth:
    # the impute-aware HD95 recomputation described below
    HD95_IMPUTE_PX = float(np.sqrt(2) * 512)  # 724.077
    hd95_finite = np.array([float(r["hd95"]) for r in valid
                            if r["hd95"] != "" and np.isfinite(float(r["hd95"]))],
                           dtype=float)
    hd95_impute = np.array(
        [float(r["hd95"]) if (r["hd95"] != "" and np.isfinite(float(r["hd95"])))
         else HD95_IMPUTE_PX for r in valid], dtype=float)
    return dict(
        model=_NAME[model], src_dataset=src, tgt_dataset=tgt,
        # fraction is numeric 1.0 (full-supervised) so it survives np.isclose /
        # float filters when concatenated into the foundation cross master.
        fraction=1.0, status="ok", box_conditioned=True, diagonal=(src == tgt),
        n_images=n_total, n_hd95_finite=int(len(hd95_finite)),
        n_hd95_empty=int(len(hd95_impute) - len(hd95_finite)),
        dice_mean=float(np.mean(dice)) if len(dice) else 0.0,
        dice_std=float(np.std(dice)) if len(dice) else 0.0,
        iou_mean=float(np.mean(iou)) if len(iou) else 0.0,
        precision_mean=float(np.mean(prec)) if len(prec) else 0.0,
        recall_mean=float(np.mean(rec)) if len(rec) else 0.0,
        hd95_mean=float(np.mean(hd95_impute)) if len(hd95_impute) else float("nan"),
        hd95_std=float(np.std(hd95_impute)) if len(hd95_impute) else float("nan"),
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), seed=42,
    )


def run_cell(model_key: str, src: str, tgt: str, args) -> Optional[Dict]:
    ckpt = _find_checkpoint(model_key, src)
    if ckpt is None:
        logger.error("No box-cond checkpoint for %s/%s — skipping", model_key, src)
        return None
    logger.info("[%s] src=%s -> tgt=%s | ckpt=%s", model_key, src, tgt, ckpt.name)

    model = _BUILD[model_key](pretrained=False, device=args.device, in_channels=4)
    state = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(state["model_state_dict"])

    loader = get_dataloader(tgt, "test", args.data_root, "full_supervised",
                            batch_size=args.batch_size, num_workers=args.num_workers,
                            seed=args.seed)
    results = evaluate_test(model, loader, args.device, args.seed)
    summary = compute_summary(model_key, src, tgt, results)
    logger.info("    DSC=%.4f IoU=%.4f HD95=%.2f (n=%d)%s",
                summary["dice_mean"], summary["iou_mean"], summary["hd95_mean"],
                summary["n_images"], "  [diagonal/in-domain sanity]" if src == tgt else "")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{_NAME[model_key]}_src{src}_tgt{tgt}"
    pd.DataFrame(results).to_csv(_OUT_DIR / f"{tag}_per_image.csv", index=False)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-conditioned CNN cross-dataset transfer (Stream A)")
    p.add_argument("--model", default="all", choices=["unet", "transunet", "all"])
    p.add_argument("--src", default=None, choices=_DATASETS, help="source (omit = all)")
    p.add_argument("--tgt", default=None, choices=_DATASETS, help="target (omit = all)")
    p.add_argument("--skip_diagonal", action="store_true",
                   help="skip src==tgt (default keeps them as in-domain sanity checks)")
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — using CPU")
        args.device = "cpu"
    # Determinism parity with the foundation cross harness (eval is already
    # deterministic under no_grad + tight box; this is for reproducibility hygiene).
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    models = ["unet", "transunet"] if args.model == "all" else [args.model]
    srcs = [args.src] if args.src else _DATASETS
    tgts = [args.tgt] if args.tgt else _DATASETS

    summaries: List[Dict] = []
    for mk in models:
        for s in srcs:
            for t in tgts:
                if args.skip_diagonal and s == t:
                    continue
                row = run_cell(mk, s, t, args)
                if row is not None:
                    summaries.append(row)

    if summaries:
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
        master = _OUT_DIR / "boxcond_cnn_crossdataset_master.csv"
        df = pd.DataFrame(summaries)
        if master.exists():
            prev = pd.read_csv(master)
            df = pd.concat([prev, df], ignore_index=True)
            df = df.drop_duplicates(subset=["model", "src_dataset", "tgt_dataset"], keep="last")
        df.to_csv(master, index=False)
        logger.info("Wrote %s (%d cells)", master, len(summaries))


if __name__ == "__main__":
    main()
