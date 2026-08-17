#!/usr/bin/env python3
"""
Experiment 3: Few-shot training for traditional models (U-Net, TransUNet, nnU-Net).
Usage: python experiments/exp3_fewshot_lora/run_traditional.py --model unet --dataset tn3k --fraction 0.10
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from thyroidbench.data import get_all_dataloaders
from thyroidbench.metrics import compute_hd95, compute_metrics
from thyroidbench.models.transunet import build_transunet
from thyroidbench.models.unet import build_unet
from thyroidbench.losses import get_loss
from thyroidbench.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Experiment 3 few-shot — traditional models")
    p.add_argument("--model",   required=True, choices=["unet", "transunet", "nnunet"])
    p.add_argument("--dataset", required=True, choices=["thyroidxl", "tn3k", "ddti", "stanford_aimi"])
    p.add_argument("--fraction", required=True, type=float, choices=[0.05, 0.10, 0.25, 0.50])
    p.add_argument("--device",           default="cuda")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--max_epochs",       type=int, default=100)
    p.add_argument("--patience",         type=int, default=15)
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--data_root",        default="data")
    p.add_argument("--nnunet_results_root", default="/tmp/nnunet_results")
    p.add_argument("--output_dir",       default="experiments/exp3_fewshot_lora/results")
    p.add_argument("--wandb_project",    default="thyroidbench")
    p.add_argument("--num_workers",      type=int, default=4)
    return p.parse_args()


def save_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def compute_summary(model, dataset, fraction, results):
    valid = [r for r in results if not r["skipped"]]
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
    hd95s = np.array([r["hd95"] for r in valid if r["hd95"] != "" and np.isfinite(float(r["hd95"]))],
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


# ---------------------------------------------------------------------------
# U-Net / TransUNet
# ---------------------------------------------------------------------------

def run_unet_transunet(args, run):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    loaders = get_all_dataloaders(
        dataset_name=args.dataset,
        data_root=Path(args.data_root),
        regime="few_shot",
        few_shot_fraction=args.fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train={len(loaders['train'].dataset)} Val={len(loaders['val'].dataset)} "
          f"Test={len(loaders['test'].dataset)}")

    if args.model == "unet":
        model = build_unet(pretrained=True, device=args.device)
    else:
        model = build_transunet(pretrained=True, device=args.device)

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    loss_fn   = get_loss("dice_bce_50_50")

    run_name  = f"fewshot_{args.model}_{args.dataset}_f{args.fraction}"
    run_dir   = Path(args.output_dir) / run_name
    ckpt_dir  = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=args.device,
        wandb_run=run,
        checkpoint_dir=str(ckpt_dir),
    )
    trainer.run_name = run_name

    trainer.fit(
        loaders["train"],
        loaders["val"],
        max_epochs=args.max_epochs,
        early_stop_patience=args.patience,
    )

    # Per-image test evaluation with manual loop (Trainer.evaluate gives only aggregate)
    model.eval()
    results = []
    with torch.no_grad():
        for batch in loaders["test"]:
            images     = batch["image"].to(args.device)
            masks      = batch["mask"].to(args.device)
            filenames  = batch["filename"]
            patient_ids= batch["patient_id"]

            logits = model(images)
            B = images.shape[0]

            for i in range(B):
                logit_i = logits[i:i+1]
                mask_i  = masks[i:i+1]

                m = compute_metrics(logit_i, mask_i)
                pred_bin = (torch.sigmoid(logit_i) >= 0.5).cpu().numpy()[0, 0].astype(np.uint8)
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


# ---------------------------------------------------------------------------
# nnU-Net
# ---------------------------------------------------------------------------

def run_nnunet_fewshot(args, run):
    data_root  = Path(args.data_root)
    splits_dir = data_root / "splits"

    # Patient-level few-shot sampling — must match dataloader logic exactly (seed=42)
    train_csv = splits_dir / f"{args.dataset}_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {train_csv}")
    df_train = pd.read_csv(train_csv)

    unique_patients = df_train["patient_id"].astype(str).unique()
    n_total  = len(unique_patients)
    n_sample = max(1, int(n_total * args.fraction))
    rng = np.random.default_rng(42)
    sampled = set(rng.choice(unique_patients, size=n_sample, replace=False).tolist())
    df_few   = df_train[df_train["patient_id"].astype(str).isin(sampled)]
    print(f"nnU-Net few-shot: {n_total} patients → {n_sample} sampled ({len(df_few)} images)")

    # Build temp nnU-Net dataset
    dataset_id      = 900
    dataset_name_nn = f"Dataset{dataset_id:03d}_ThyroidFewShot"
    nnunet_base = Path(args.nnunet_results_root)
    nnunet_raw  = nnunet_base / "nnUNet_raw" / dataset_name_nn
    (nnunet_raw / "imagesTr").mkdir(parents=True, exist_ok=True)
    (nnunet_raw / "labelsTr").mkdir(parents=True, exist_ok=True)

    processed = data_root / "processed" / args.dataset
    img_dir  = processed / "images"
    mask_dir = processed / "masks"

    case_ids = []
    for _, row in df_few.iterrows():
        fn   = row["filename"]
        stem = Path(fn).stem
        src_img  = img_dir  / fn
        src_mask = mask_dir / fn
        if not src_img.exists() or not src_mask.exists():
            print(f"  Warning: missing {fn}, skip")
            continue
        shutil.copy(src_img,  nnunet_raw / "imagesTr" / f"{stem}_0000.png")
        shutil.copy(src_mask, nnunet_raw / "labelsTr" / f"{stem}.png")
        case_ids.append(stem)

    if not case_ids:
        raise RuntimeError("No images copied for nnU-Net — check processed/ paths")
    print(f"Copied {len(case_ids)} images to {nnunet_raw}")

    (nnunet_raw / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "US"},
        "labels": {"background": 0, "nodule": 1},
        "numTraining": len(case_ids),
        "file_ending": ".png",
    }, indent=2))

    env = {
        **os.environ,
        "nnUNet_raw":          str(nnunet_base / "nnUNet_raw"),
        "nnUNet_preprocessed": str(nnunet_base / "nnUNet_preprocessed"),
        "nnUNet_results":      str(nnunet_base / "nnUNet_trained_models"),
    }
    for d in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_trained_models"]:
        (nnunet_base / d).mkdir(parents=True, exist_ok=True)

    print("nnU-Net: plan + preprocess...")
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(dataset_id),
                    "--verify_dataset_integrity"], env=env, check=True)

    print("nnU-Net: training (2d, fold 0)...")
    subprocess.run(["nnUNetv2_train", str(dataset_id), "2d", "0", "--npz"],
                   env=env, check=True)

    # Test set evaluation
    test_csv = splits_dir / f"{args.dataset}_test.csv"
    df_test  = pd.read_csv(test_csv)

    test_in  = nnunet_base / "nnunet_test_input"  / args.dataset
    test_out = nnunet_base / "nnunet_test_output" / args.dataset
    for d in [test_in, test_out]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    for _, row in df_test.iterrows():
        fn = row["filename"]
        src = img_dir / fn
        if src.exists():
            shutil.copy(src, test_in / f"{Path(fn).stem}_0000.png")

    print("nnU-Net: predict on test set...")
    subprocess.run(["nnUNetv2_predict",
                    "-i", str(test_in), "-o", str(test_out),
                    "-d", str(dataset_id), "-c", "2d", "-f", "0"],
                   env=env, check=True)

    results = []
    for _, row in df_test.iterrows():
        fn   = row["filename"]
        stem = Path(fn).stem
        pid  = str(row["patient_id"])
        pred_path = test_out / f"{stem}.png"
        gt_path   = mask_dir / fn

        if not pred_path.exists():
            results.append({"filename": fn, "patient_id": pid, "skipped": True,
                             "dice": "", "iou": "", "precision": "", "recall": "", "hd95": ""})
            continue

        pred_np = (np.array(Image.open(pred_path).convert("L")) > 0).astype(np.uint8)
        gt_np   = (np.array(Image.open(gt_path).convert("L")) > 0).astype(np.uint8)

        # Resize pred to gt size if needed (nnU-Net may resize)
        if pred_np.shape != gt_np.shape:
            pred_np = np.array(
                Image.fromarray(pred_np * 255).resize(gt_np.shape[::-1], Image.NEAREST) > 0,
                dtype=np.uint8,
            )

        pred_t = torch.from_numpy(pred_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        gt_t   = torch.from_numpy(gt_np.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        # Binary pred → high-magnitude logits so sigmoid gives 0/1
        pred_logits = pred_t * 20.0 - 10.0

        try:
            m    = compute_metrics(pred_logits, gt_t)
            hd95 = compute_hd95(pred_np, gt_np)
            results.append({
                "filename": fn, "patient_id": pid, "skipped": False,
                "dice":      round(m["dice"],      6),
                "iou":       round(m["iou"],       6),
                "precision": round(m["precision"], 6),
                "recall":    round(m["recall"],    6),
                "hd95":      round(hd95, 4) if np.isfinite(hd95) else "",
            })
        except Exception as exc:
            print(f"  Error metrics {fn}: {exc}")
            results.append({"filename": fn, "patient_id": pid, "skipped": True,
                             "dice": "", "iou": "", "precision": "", "recall": "", "hd95": ""})

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_name = f"fewshot_{args.model}_{args.dataset}_f{args.fraction}"
    run_dir  = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
        tags=["fewshot", args.model, args.dataset, f"frac{args.fraction}"],
    )

    try:
        if args.model == "nnunet":
            results = run_nnunet_fewshot(args, run)
        else:
            results = run_unet_transunet(args, run)

        summary = compute_summary(args.model, args.dataset, args.fraction, results)

        # Log to wandb
        run.summary.update({
            "test_dice":      summary["dice_mean"],
            "test_iou":       summary["iou_mean"],
            "test_precision": summary["precision_mean"],
            "test_recall":    summary["recall_mean"],
            "test_hd95":      summary["hd95_mean"],
        })

        per_img_path = str(run_dir / "per_image_results.csv")
        save_csv(results, per_img_path, fieldnames=[
            "filename", "patient_id", "dice", "iou", "precision", "recall", "hd95", "skipped",
        ])
        sum_path = str(run_dir / "summary_metrics.csv")
        save_csv([summary], sum_path, fieldnames=list(summary.keys()))

        print(f"\n=== SUMMARY ===")
        print(f"  DSC:  {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
        print(f"  IoU:  {summary['iou_mean']:.4f} ± {summary['iou_std']:.4f}")
        print(f"  HD95: {summary['hd95_mean']:.2f} ± {summary['hd95_std']:.2f}")
        print(f"  n={summary['n_images']}, skipped={summary['n_skipped']}")
        print(f"Saved: {per_img_path}, {sum_path}")

    except Exception as exc:
        print(f"ERROR: {exc}")
        import traceback; traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()
