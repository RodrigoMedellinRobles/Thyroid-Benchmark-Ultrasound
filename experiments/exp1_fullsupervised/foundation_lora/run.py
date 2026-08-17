#!/usr/bin/env python3
"""
Full-supervised foundation models via LoRA (100% labels) — fills the missing
"full-supervised foundation" cell of the regime matrix (Exp 1 = full-sup
traditional CNNs; this = full-sup foundation, the fair full-vs-full counterpart).

This trains the LoRA adapters on the ENTIRE training split. It deliberately
keeps the few_shot regime at fraction=1.0 (every patient) rather than the
full_supervised regime, so the "lora" augmentation pipeline matches the
f=5/10/25/50% curve exactly; the only difference from
experiments/exp3_fewshot_lora/run.py is the training-set size (100% vs a
sampled fraction). EVERY other knob — LoRA config (r=16, alpha=32), loss
(dice_bce_50_50), optimiser (Adam lr=1e-4, wd=1e-4), cosine schedule, 100
epochs, patience 15, batch_size 4, bfloat16 autocast, seed 42, and the val/test
splits — is identical to the few-shot protocol, so the f=100% point is a valid
extension of the data-efficiency curve and a like-for-like comparison against
the box-conditioned CNN ceiling.

SAM (original ViT-H) is excluded from LoRA adaptation, matching Exp 3.

Usage:
  python experiments/exp1_fullsupervised/foundation_lora/run.py --model sam2 --dataset tn3k
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from thyroidbench.data import get_all_dataloaders, DATASET_STATS
from thyroidbench.metrics import MetricsAccumulator, compute_hd95, compute_metrics
from thyroidbench.models.foundation_base import get_gt_box
from thyroidbench.lora import build_lora_wrapper
from thyroidbench.losses import get_loss

# Recorded in outputs so the f=100% point is unambiguous on the curve.
FRACTION = 1.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Full-supervised (100% labels) LoRA training — foundation models")
    p.add_argument("--model",    required=True,
                   choices=["sam2", "medsam", "medsam2", "sam3"])
    p.add_argument("--dataset",  required=True,
                   choices=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    p.add_argument("--device",           default="cuda")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--max_epochs",       type=int,   default=100)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--min_delta",        type=float, default=1e-3,
                   help="Minimum val_dice improvement to count as 'better' and reset the "
                        "early-stop patience counter. Without it, sub-noise improvements "
                        "(~1e-4) on a converged plateau reset patience forever and the run "
                        "never early-stops. LR schedule (T_max=max_epochs) is unchanged, so "
                        "this is consistent with runs trained before the fix.")
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--data_root",        default="data")
    p.add_argument("--output_dir",       default="experiments/exp1_fullsupervised/foundation_lora/results")
    p.add_argument("--wandb_project",    default="thyroidbench")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--resume_eval_only", action="store_true",
                   help="Skip training, load existing checkpoint and run test eval only")
    p.add_argument("--detector_csv", default=None,
                   help="End-to-end mode: CSV with columns filename,x1,y1,x2,y2 of a "
                        "learned detector's box. When set, the LoRA checkpoint is loaded "
                        "(forces --resume_eval_only) and prompted with the detector box "
                        "instead of the tight GT box; images with no detection score "
                        "zero Dice. Outputs land in a suffixed run dir (see --output_suffix).")
    p.add_argument("--output_suffix", default="",
                   help="Suffix appended to the run dir for eval outputs only (e.g. '_yolo'); "
                        "the checkpoint is still read from the un-suffixed dir.")
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_detector_boxes(csv_path: str) -> dict:
    """Map filename -> [x1, y1, x2, y2] from a learned detector's box CSV.

    Files absent from the CSV (the detector fired nothing for them) are simply
    not in the dict; the caller scores them as zero Dice.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    out = {}
    for _, row in df.iterrows():
        out[str(row["filename"])] = np.array(
            [row["x1"], row["y1"], row["x2"], row["y2"]], dtype=np.float32)
    return out


def extract_boxes(masks: torch.Tensor, device: torch.device, img_size: int = 512) -> torch.Tensor:
    boxes = []
    for i in range(masks.shape[0]):
        mask_np = masks[i, 0].cpu().numpy().astype(np.uint8)
        box = get_gt_box(mask_np)
        if box is None:
            box = np.array([0, 0, img_size - 1, img_size - 1], dtype=np.float32)
        boxes.append(box)
    return torch.tensor(np.stack(boxes), dtype=torch.float32, device=device)


def train_epoch(wrapper, loader, loss_fn, optimizer, device):
    wrapper.set_train_mode()
    acc = MetricsAccumulator()
    total_loss, n = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        boxes  = extract_boxes(masks, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapper(images, boxes)
        loss = loss_fn(logits.float(), masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc.update(logits.detach().float(), masks)
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), acc.compute()


@torch.no_grad()
def val_epoch(wrapper, loader, loss_fn, device):
    wrapper.model.eval()
    acc = MetricsAccumulator()
    total_loss, n = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)
        boxes  = extract_boxes(masks, device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapper(images, boxes)
        loss = loss_fn(logits.float(), masks)

        acc.update(logits.float(), masks)
        total_loss += loss.item()
        n += 1

    return total_loss / max(n, 1), acc.compute()


@torch.no_grad()
def evaluate_test(wrapper, loader, device, detector_boxes=None):
    """Test-set eval. With detector_boxes (filename -> xyxy box) the model is
    prompted with the learned detector's box instead of the tight GT box, and
    images the detector missed (absent from the dict) score zero Dice."""
    wrapper.model.eval()
    results = []

    for batch in loader:
        images      = batch["image"].to(device)
        masks       = batch["mask"].to(device)
        filenames   = batch["filename"]
        patient_ids = batch["patient_id"]
        B = images.shape[0]

        for i in range(B):
            img_i  = images[i:i+1]
            mask_i = masks[i:i+1]

            if detector_boxes is not None:
                det = detector_boxes.get(str(filenames[i]))
                if det is None:
                    # Detector found nothing -> no prompt -> zero-Dice row.
                    results.append({
                        "filename":   filenames[i],
                        "patient_id": patient_ids[i],
                        "dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0,
                        "hd95": "", "skipped": True,
                    })
                    continue
                boxes = torch.tensor(det[None], dtype=torch.float32, device=device)
            else:
                boxes = extract_boxes(mask_i, device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = wrapper(img_i, boxes)
            logits = logits.float()

            m = compute_metrics(logits, mask_i)
            pred_bin = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0, 0].astype(np.uint8)
            mask_bin = mask_i[0, 0].cpu().numpy().astype(np.uint8)
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


def compute_summary(model, dataset, fraction, results):
    valid   = [r for r in results if not r["skipped"]]
    n_total = len(results)
    n_skip  = sum(1 for r in results if r["skipped"])
    if not valid:
        return dict(model=model, dataset=dataset, fraction=fraction,
                    n_images=n_total, n_skipped=n_skip,
                    dice_mean=0, dice_std=0, iou_mean=0, iou_std=0,
                    precision_mean=0, precision_std=0, recall_mean=0, recall_std=0,
                    hd95_mean=0, hd95_std=0)

    dice  = np.array([r["dice"]      for r in valid], dtype=float)
    iou   = np.array([r["iou"]       for r in valid], dtype=float)
    prec  = np.array([r["precision"] for r in valid], dtype=float)
    rec   = np.array([r["recall"]    for r in valid], dtype=float)
    hd95s = np.array(
        [r["hd95"] for r in valid if r["hd95"] != "" and np.isfinite(float(r["hd95"]))],
        dtype=float)

    return dict(
        model=model, dataset=dataset, fraction=fraction,
        n_images=n_total, n_skipped=n_skip,
        dice_mean=float(np.mean(dice)),         dice_std=float(np.std(dice)),
        iou_mean=float(np.mean(iou)),           iou_std=float(np.std(iou)),
        precision_mean=float(np.mean(prec)),    precision_std=float(np.std(prec)),
        recall_mean=float(np.mean(rec)),        recall_std=float(np.std(rec)),
        hd95_mean=float(np.mean(hd95s)) if len(hd95s) else 0.0,
        hd95_std=float(np.std(hd95s))  if len(hd95s) else 0.0,
    )


def save_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    # Pin output_dir to project root so relative paths never land at $cwd.
    project_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (project_root / out_dir).resolve()
    args.output_dir = str(out_dir)

    run_name  = f"fullsup_{args.model}_{args.dataset}"
    run_dir   = out_dir / run_name                      # checkpoint always lives here
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / f"{run_name}_best.pt"

    # End-to-end mode: load the trained checkpoint and re-evaluate it with a
    # learned detector's box. Outputs go to a suffixed dir so the oracle-box
    # results are never overwritten.
    detector_boxes = None
    if args.detector_csv:
        args.resume_eval_only = True
        detector_boxes = load_detector_boxes(args.detector_csv)
        print(f"[run] detector mode: {len(detector_boxes)} boxes from {args.detector_csv}",
              flush=True)
    eval_dir = out_dir / f"{run_name}{args.output_suffix}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] output_dir={out_dir}  run_dir={run_dir}  eval_dir={eval_dir}", flush=True)

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={**vars(args), "fraction": FRACTION, "lora_r": 16, "lora_alpha": 32,
                "lora_target_modules": ["attn.qkv", "attn.proj"],
                "precision": "bfloat16", "gradient_checkpointing": "pending"},
        tags=["fullsup_foundation", args.model, args.dataset, "f1.0", "bf16"],
    )

    if args.resume_eval_only and not ckpt_path.exists():
        raise RuntimeError(f"--resume_eval_only set but no checkpoint at {ckpt_path}")

    ds_stats = DATASET_STATS[args.dataset]
    wrapper  = build_lora_wrapper(
        model_name=args.model,
        device=str(device),
        ds_mean=ds_stats["mean"],
        ds_std=ds_stats["std"],
    )
    wandb.config.update({"gradient_checkpointing": wrapper.gc_applied}, allow_val_change=True)

    # Full data via the few_shot regime at fraction=1.0: this selects EVERY
    # patient (= the entire training split) while keeping the "lora"
    # augmentation pipeline used by the f=5/10/25/50% curve. Using the
    # full_supervised regime instead would silently switch to the heavier
    # "train" augmentation and break protocol fidelity. Val/test splits are
    # unchanged, so the comparison is on the same test set.
    loaders      = get_all_dataloaders(
        dataset_name=args.dataset,
        data_root=Path(args.data_root),
        regime="few_shot",
        few_shot_fraction=FRACTION,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train_loader = loaders["train"]
    val_loader   = loaders["val"]
    test_loader  = loaders["test"]

    if train_loader is None:
        raise RuntimeError("full_supervised train_loader is None")

    print(f"Train={len(train_loader.dataset)} | Val={len(val_loader.dataset)} | Test={len(test_loader.dataset)}")
    batch0 = next(iter(train_loader))
    print(f"Batch shapes: image={batch0['image'].shape}, mask={batch0['mask'].shape}")

    loss_fn   = get_loss("dice_bce_50_50").to(device)
    optimizer = Adam(wrapper.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    best_val_dice    = -1.0
    best_epoch       = -1
    patience_counter = 0

    # Per-epoch training log on disk (val_dice progression for plots / sanity)
    train_log_path = run_dir / "training_log.csv"
    train_log_fields = ["epoch", "lr", "train_loss", "train_dice", "train_iou",
                        "val_loss", "val_dice", "val_iou"]
    if not args.resume_eval_only:
        with open(train_log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=train_log_fields).writeheader()

    if not args.resume_eval_only:
        for epoch in range(1, args.max_epochs + 1):
            train_loss, train_m = train_epoch(wrapper, train_loader, loss_fn, optimizer, device)
            val_loss,   val_m   = val_epoch(wrapper, val_loader, loss_fn, device)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            wandb.log({
                "epoch": epoch, "lr": lr,
                "train_loss": train_loss, "train_dice": train_m["dice"], "train_iou": train_m["iou"],
                "val_loss":   val_loss,   "val_dice":   val_m["dice"],   "val_iou":   val_m["iou"],
            }, step=epoch)

            with open(train_log_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=train_log_fields).writerow({
                    "epoch": epoch, "lr": lr,
                    "train_loss": train_loss, "train_dice": train_m["dice"], "train_iou": train_m["iou"],
                    "val_loss":   val_loss,   "val_dice":   val_m["dice"],   "val_iou":   val_m["iou"],
                })

            print(f"ep{epoch:3d} | train_loss={train_loss:.4f} dice={train_m['dice']:.4f} | "
                  f"val_loss={val_loss:.4f} dice={val_m['dice']:.4f} | lr={lr:.2e}")

            if val_m["dice"] > best_val_dice + args.min_delta:
                best_val_dice    = val_m["dice"]
                best_epoch       = epoch
                patience_counter = 0
                torch.save({"model_state": wrapper.state_dict(),
                            "epoch": epoch, "val_dice": best_val_dice}, ckpt_path)
                print(f"  → best val_dice={best_val_dice:.4f} → {ckpt_path}")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_dice={best_val_dice:.4f})")
    ckpt = torch.load(ckpt_path, map_location=device)
    wrapper.load_state_dict(ckpt["model_state"])
    del ckpt
    torch.cuda.empty_cache()

    print("Running test evaluation...")
    results = evaluate_test(wrapper, test_loader, device, detector_boxes=detector_boxes)
    summary = compute_summary(args.model, args.dataset, FRACTION, results)

    wandb.run.summary.update({
        "test_dice":      summary["dice_mean"],
        "test_iou":       summary["iou_mean"],
        "test_precision": summary["precision_mean"],
        "test_recall":    summary["recall_mean"],
        "test_hd95":      summary["hd95_mean"],
    })

    per_img_path = str(eval_dir / "per_image_results.csv")
    sum_path     = str(eval_dir / "summary_metrics.csv")
    save_csv(results, per_img_path,
             fieldnames=["filename", "patient_id", "dice", "iou", "precision", "recall", "hd95", "skipped"])
    save_csv([summary], sum_path, fieldnames=list(summary.keys()))

    print(f"\n=== SUMMARY ===")
    print(f"  DSC:  {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"  IoU:  {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
    print(f"  HD95: {summary['hd95_mean']:.2f} ± {summary['hd95_std']:.2f}")
    print(f"  n={summary['n_images']}, skipped={summary['n_skipped']}")
    print(f"Saved: {per_img_path}, {sum_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
