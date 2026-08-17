"""Cross-dataset eval-only for LoRA-adapted foundation models (Exp 4.5).

Loads a LoRA checkpoint trained on `src_dataset` (Exp 3) and evaluates it on the
test split of `tgt_dataset`. No training. Reuses build_lora_wrapper, dataloaders,
and metrics from thyroidbench/.

Usage:
    python experiments/exp4_crossdataset/foundation/run_crosseval.py \
        --model sam2 --src_dataset ddti --tgt_dataset tn3k --fraction 0.25 \
        --data_root data --output_dir experiments/exp4_crossdataset/foundation/results \
        --src_ckpt_root experiments/exp3_fewshot_lora/results

Source checkpoint is resolved as
    <src_ckpt_root>/fewshot_<model>_<src_dataset>_f<fraction>/fewshot_<model>_<src_dataset>_f<fraction>_best.pt
Outputs land in
    <output_dir>/exp4_5_<model>_src<src>_tgt<tgt>_f<fraction>/
        - summary_metrics.csv
        - per_image_results.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from thyroidbench.data import get_all_dataloaders
from thyroidbench.data import DATASET_STATS
from thyroidbench.metrics import compute_hd95, compute_metrics
from thyroidbench.models.foundation_base import get_gt_box  # noqa: F401  (kept for parity)
from thyroidbench.lora import build_lora_wrapper

VALID_MODELS = ("sam2", "medsam", "medsam2", "sam3")
VALID_DATASETS = ("ddti", "tn3k", "thyroidxl", "stanford_aimi")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=VALID_MODELS)
    p.add_argument("--src_dataset", required=True, choices=VALID_DATASETS,
                   help="Dataset the LoRA checkpoint was fine-tuned on (Exp 3).")
    p.add_argument("--tgt_dataset", required=True, choices=VALID_DATASETS,
                   help="Dataset whose test split the checkpoint is evaluated on.")
    p.add_argument("--fraction", required=True, type=float, choices=[0.05, 0.1, 0.25, 0.5])
    p.add_argument("--data_root", default="data")
    p.add_argument("--src_ckpt_root", default="experiments/exp3_fewshot_lora/results")
    p.add_argument("--output_dir", default="experiments/exp4_crossdataset/foundation/results")
    p.add_argument("--norm_stats", choices=["src", "tgt"], default="src",
                   help="Use source-dataset (training) or target-dataset normalization at inference.")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="thyroidbench")
    return p.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_boxes(masks: torch.Tensor, device: torch.device, img_size: int = 512) -> torch.Tensor:
    boxes = []
    for i in range(masks.shape[0]):
        m = masks[i, 0].cpu().numpy()
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            boxes.append([0.0, 0.0, float(img_size), float(img_size)])
        else:
            x0, y0 = float(xs.min()), float(ys.min())
            x1, y1 = float(xs.max()), float(ys.max())
            boxes.append([x0, y0, x1, y1])
    return torch.tensor(boxes, dtype=torch.float32, device=device)


def evaluate_test(wrapper, loader, device: torch.device) -> list[dict]:
    wrapper.model.eval()
    results: list[dict] = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            filenames = batch["filename"]
            patient_ids = batch["patient_id"]
            for i in range(images.shape[0]):
                img_i = images[i:i + 1]
                mask_i = masks[i:i + 1]
                boxes = extract_boxes(mask_i, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = wrapper(img_i, boxes)
                logits = logits.float()
                m = compute_metrics(logits, mask_i)
                pred_bin = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0, 0].astype(np.uint8)
                mask_bin = mask_i[0, 0].cpu().numpy().astype(np.uint8)
                hd95 = compute_hd95(pred_bin, mask_bin)
                results.append({
                    "filename": filenames[i],
                    "patient_id": patient_ids[i],
                    "dice": round(m["dice"], 6),
                    "iou": round(m["iou"], 6),
                    "precision": round(m["precision"], 6),
                    "recall": round(m["recall"], 6),
                    "hd95": round(hd95, 4) if np.isfinite(hd95) else "",
                    "skipped": False,
                })
    return results


def compute_summary(model: str, src: str, tgt: str, fraction: float, results: list[dict]) -> dict:
    valid = [r for r in results if not r["skipped"]]
    n_total = len(results)
    n_skip = sum(1 for r in results if r["skipped"])
    if not valid:
        zeros = {k: 0.0 for k in ("dice_mean", "dice_std", "iou_mean", "iou_std",
                                  "precision_mean", "precision_std", "recall_mean",
                                  "recall_std", "hd95_mean", "hd95_std")}
        return {"model": model, "src_dataset": src, "tgt_dataset": tgt,
                "fraction": fraction, "n_images": n_total, "n_skipped": n_skip, **zeros}

    arr = lambda key: np.array([r[key] for r in valid], dtype=float)
    dice, iou = arr("dice"), arr("iou")
    prec, rec = arr("precision"), arr("recall")
    hd95s = np.array(
        [r["hd95"] for r in valid if r["hd95"] != "" and np.isfinite(float(r["hd95"]))],
        dtype=float)
    return dict(
        model=model, src_dataset=src, tgt_dataset=tgt, fraction=fraction,
        n_images=n_total, n_skipped=n_skip,
        dice_mean=float(np.mean(dice)), dice_std=float(np.std(dice)),
        iou_mean=float(np.mean(iou)), iou_std=float(np.std(iou)),
        precision_mean=float(np.mean(prec)), precision_std=float(np.std(prec)),
        recall_mean=float(np.mean(rec)), recall_std=float(np.std(rec)),
        hd95_mean=float(np.mean(hd95s)) if len(hd95s) else 0.0,
        hd95_std=float(np.std(hd95s)) if len(hd95s) else 0.0,
    )


def save_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.src_dataset == args.tgt_dataset:
        print(f"[skip] src == tgt ({args.src_dataset}); diagonal is in Exp 3 already.", flush=True)
        return

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    src_ckpt_root = Path(args.src_ckpt_root)
    if not src_ckpt_root.is_absolute():
        src_ckpt_root = (PROJECT_ROOT / src_ckpt_root).resolve()

    src_run_name = f"fewshot_{args.model}_{args.src_dataset}_f{args.fraction}"
    src_ckpt_path = src_ckpt_root / src_run_name / f"{src_run_name}_best.pt"
    if not src_ckpt_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {src_ckpt_path}")

    run_name = f"exp4_5_{args.model}_src{args.src_dataset}_tgt{args.tgt_dataset}_f{args.fraction}"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[crosseval] src_ckpt={src_ckpt_path}", flush=True)
    print(f"[crosseval] run_dir={run_dir}", flush=True)
    print(f"[crosseval] norm_stats={args.norm_stats}", flush=True)

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={**vars(args), "lora_r": 16, "lora_alpha": 32,
                "lora_target_modules": ["qkv", "proj"], "precision": "bfloat16",
                "regime": "exp4_5_crosseval"},
        tags=["exp4_5_crosseval", args.model, f"src_{args.src_dataset}",
              f"tgt_{args.tgt_dataset}", f"frac{args.fraction}"],
    )

    norm_dataset = args.src_dataset if args.norm_stats == "src" else args.tgt_dataset
    ds_stats = DATASET_STATS[norm_dataset]
    wrapper = build_lora_wrapper(
        model_name=args.model,
        device=str(device),
        ds_mean=ds_stats["mean"],
        ds_std=ds_stats["std"],
    )

    print(f"[crosseval] loading state from {src_ckpt_path}", flush=True)
    ckpt = torch.load(src_ckpt_path, map_location=device)
    wrapper.load_state_dict(ckpt["model_state"])
    print(f"[crosseval] loaded ckpt epoch={ckpt.get('epoch', '?')} val_dice={ckpt.get('val_dice', '?')}", flush=True)
    del ckpt
    torch.cuda.empty_cache()

    loaders = get_all_dataloaders(
        dataset_name=args.tgt_dataset,
        data_root=Path(args.data_root),
        regime="few_shot",
        few_shot_fraction=args.fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    test_loader = loaders["test"]
    print(f"[crosseval] target test set size = {len(test_loader.dataset)}", flush=True)

    print("[crosseval] running test evaluation...", flush=True)
    results = evaluate_test(wrapper, test_loader, device)
    summary = compute_summary(args.model, args.src_dataset, args.tgt_dataset,
                              args.fraction, results)

    wandb.run.summary.update({
        "test_dice": summary["dice_mean"],
        "test_iou": summary["iou_mean"],
        "test_precision": summary["precision_mean"],
        "test_recall": summary["recall_mean"],
        "test_hd95": summary["hd95_mean"],
    })

    save_csv(results, run_dir / "per_image_results.csv",
             ["filename", "patient_id", "dice", "iou", "precision", "recall", "hd95", "skipped"])
    save_csv([summary], run_dir / "summary_metrics.csv", list(summary.keys()))
    print(f"[crosseval] DSC={summary['dice_mean']:.4f}  HD95={summary['hd95_mean']:.2f}  "
          f"n={summary['n_images']}", flush=True)
    print(f"[crosseval] saved: {run_dir}/summary_metrics.csv", flush=True)

    wandb.finish()


if __name__ == "__main__":
    main()
