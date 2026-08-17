#!/usr/bin/env python3
"""
Section 5.1 — Box-conditioned full-supervised traditional baselines.

Full-supervised training protocol: Adam optimiser, CosineAnnealingLR,
per-dataset LR search, DiceBCE loss, 100 epochs, early-stopping patience 15,
batch 16, seed 42. The distinguishing input is a 4th channel holding the
rasterised oracle bounding box (binary {0,1}), so the CNN receives the same
spatial prior as the box-prompted foundation models.

Key design points (all enforced by the shared thyroidbench code):
  * 4th-channel conv is ZERO-initialised -> the box channel contributes nothing
    at initialisation (thyroidbench/models/{unet,transunet}.py).
  * Train-time box jitter is +/-10% fractional, SEEDED and epoch-varying
    (thyroidbench/trainer.py + boxes.jitter_boxes).
  * Evaluation uses the TIGHT box (no jitter), matching the few-shot LoRA eval
    so Figure 1 is apples-to-apples (D1/D3).

nnU-Net box-conditioning has its own path (see convert_to_nnunet_boxcond.py).

Usage:
    python experiments/exp1_fullsupervised/traditional_boxcond/run.py --model unet      --dataset ddti
    python experiments/exp1_fullsupervised/traditional_boxcond/run.py --model transunet --dataset tn3k
"""

import argparse
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thyroidbench.data import get_all_dataloaders
from thyroidbench.models.transunet import TRANSUNET_CONFIG, build_transunet
from thyroidbench.models.unet import UNET_CONFIG, build_unet
from thyroidbench.losses import get_loss
from thyroidbench.trainer import Trainer

_MODELS = {
    "unet":      {"build": build_unet,      "config": UNET_CONFIG,      "name": "unet_resnet50"},
    "transunet": {"build": build_transunet, "config": TRANSUNET_CONFIG, "name": "transunet_r50_vitb16"},
}

_EXP_DIR = "experiments/exp1_fullsupervised/traditional_boxcond"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(logs_dir: Path, model: str, dataset: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"box_{model}_{dataset}_{ts}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    return log_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sec 5.1 box-conditioned traditional baselines")
    p.add_argument("--model",   required=True, choices=list(_MODELS))
    p.add_argument("--dataset", required=True,
                   choices=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    p.add_argument("--lr", type=float, nargs="+", default=None,
                   help="LR override (default: model's lr_search list)")
    p.add_argument("--max_epochs",  type=int, default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--box_jitter_frac", type=float, default=0.1,
                   help="Train-time box jitter fraction (eval is always tight)")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--wandb_project", default="thyroidbench")
    p.add_argument("--wandb_entity",  default=None)
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument("--data_root", default="data")
    p.add_argument("--results_dir",    default=None,
                   help=f"default: {_EXP_DIR}/results/{{model}}")
    p.add_argument("--checkpoint_dir", default=None,
                   help=f"default: {_EXP_DIR}/results/{{model}}/checkpoints")
    p.add_argument("--logs_dir",       default=f"{_EXP_DIR}/logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = _MODELS[args.model]

    base = Path(_EXP_DIR) / "results" / args.model
    results_dir    = Path(args.results_dir    or base)
    checkpoint_dir = Path(args.checkpoint_dir or base / "checkpoints")
    logs_dir       = Path(args.logs_dir)
    for d in (results_dir, checkpoint_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log_path = setup_logging(logs_dir, args.model, args.dataset)
    logger = logging.getLogger(__name__)
    set_seed(args.seed)

    config = spec["config"].copy()
    if args.max_epochs is not None:
        config["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    lr_list = args.lr if args.lr else config["lr_search"]

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU")
        device = "cpu"

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.info("BOX-CONDITIONED | Model=%s | Dataset=%s | LRs=%s | epochs=%d | batch=%d | jitter=%.2f | device=%s (%s)",
                args.model, args.dataset, lr_list, config["max_epochs"],
                config["batch_size"], args.box_jitter_frac, device, gpu_name)
    logger.info("Log → %s", log_path)

    loss_fn = get_loss(config["loss"])
    dataloaders = get_all_dataloaders(
        dataset_name=args.dataset,
        data_root=str(args.data_root),
        regime="full_supervised",
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # --- LR search ---
    best_model, best_val_dice, best_lr = None, 0.0, None
    weight_decay = config.get("weight_decay", 1e-4)

    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb

    for lr in lr_list:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

        run_name = f"box_{args.model}_{args.dataset}_lr{lr:.0e}"
        wandb_run = None
        if use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config={
                    "model":     args.model,
                    "dataset":   args.dataset,
                    "lr":        lr,
                    "batch":     config["batch_size"],
                    "epochs":    config["max_epochs"],
                    "patience":  config["early_stop_patience"],
                    "seed":      args.seed,
                    "box_conditioned": True,
                    "box_jitter_frac": args.box_jitter_frac,
                    "version":   "v2_boxcond",
                },
                tags=["sec5.1", "v2", "box_conditioned", args.model, args.dataset],
                reinit=True,
            )
        model = spec["build"](pretrained=True, device=device, in_channels=4)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

        trainer = Trainer(
            model=model, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler,
            device=device, wandb_run=wandb_run, checkpoint_dir=str(checkpoint_dir),
            box_conditioned=True, box_jitter_frac=args.box_jitter_frac,
            box_seed=args.seed,
        )
        trainer.run_name = run_name
        history = trainer.fit(
            dataloaders["train"], dataloaders["val"],
            config["max_epochs"], config["early_stop_patience"],
        )
        if wandb_run is not None:
            wandb_run.summary["best_val_dice"] = max(history["val_dice"]) if history["val_dice"] else 0.0
            wandb_run.finish()
        trial_best = max(history["val_dice"])
        logger.info("lr=%.1e → best val_dice=%.4f", lr, trial_best)

        if trial_best > best_val_dice:
            best_val_dice = trial_best
            best_model = model
            best_lr = lr

    logger.info("Best lr=%.1e | val_dice=%.4f", best_lr, best_val_dice)

    # --- test evaluation (TIGHT box) ---
    eval_trainer = Trainer(
        model=best_model, loss_fn=loss_fn,
        optimizer=optim.Adam(best_model.parameters(), lr=1e-4),
        scheduler=None, device=device, checkpoint_dir=str(checkpoint_dir),
        box_conditioned=True, box_jitter_frac=args.box_jitter_frac,
        box_seed=args.seed,
    )
    test_metrics = eval_trainer.evaluate(dataloaders["test"], compute_hd95=True)

    # --- save CSV ---
    row = {
        "dataset":   args.dataset,
        "model":     spec["name"],
        "box_conditioned": True,
        "lr":        best_lr,
        "val_dice":  round(best_val_dice, 6),
        "test_dice": round(test_metrics["test_dice"], 6),
        "test_iou":  round(test_metrics["test_iou"], 6),
        "test_precision": round(test_metrics["test_precision"], 6),
        "test_recall":    round(test_metrics["test_recall"], 6),
        "test_hd95":      round(test_metrics.get("test_hd95", float("nan")), 4),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed":      args.seed,
    }
    csv_path = results_dir / f"{args.dataset}_{args.model}_boxcond_results.csv"
    if csv_path.exists():
        df = pd.concat([pd.read_csv(csv_path), pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)

    sep = "=" * 60
    logger.info(sep)
    logger.info("DONE (box-conditioned) — %s / %s", args.dataset, spec["name"])
    logger.info("  Best lr:    %.1e", best_lr)
    logger.info("  Val Dice:   %.4f", best_val_dice)
    logger.info("  Test Dice:  %.4f", test_metrics["test_dice"])
    logger.info("  Test IoU:   %.4f", test_metrics["test_iou"])
    logger.info("  Test HD95:  %.2f px", test_metrics.get("test_hd95", float("nan")))
    logger.info("  Results CSV: %s", csv_path)
    logger.info(sep)


if __name__ == "__main__":
    main()
