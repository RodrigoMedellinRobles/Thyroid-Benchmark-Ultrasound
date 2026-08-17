"""
Trainer for thyroid nodule segmentation (U-Net, TransUNet).
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import get_all_dataloaders
from .boxes import build_box_input
from .losses import get_loss
from .metrics import compute_metrics, compute_batch_hd95, MetricsAccumulator

logger = logging.getLogger(__name__)


class Trainer:
    """Generic trainer for binary segmentation models."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[object],
        device: str,
        wandb_run: Optional[object] = None,
        checkpoint_dir: str = "checkpoints",
        box_conditioned: Optional[bool] = None,
        box_jitter_frac: float = 0.1,
        box_seed: int = 42,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.wandb_run = wandb_run
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch = 0
        self.best_val_dice = 0.0
        self.run_name = "model"
        # Auto-detect box-conditioned models if the caller did not specify.
        if box_conditioned is None:
            inner = getattr(self.model, "model", self.model)
            in_channels = (
                getattr(self.model, "in_channels", None)
                or getattr(inner, "in_channels", None)
                or 3
            )
            box_conditioned = in_channels == 4
        self.box_conditioned = bool(box_conditioned)
        self.box_jitter_frac = box_jitter_frac
        self.box_seed = box_seed

    def _box_generator(self, batch_idx: int) -> torch.Generator:
        """Deterministic CPU RNG for train-time box jitter (D5).

        Seeded from (box_seed, epoch, batch_idx) so the jitter is epoch-varying
        yet fully reproducible across runs. Lives on CPU; box_utils moves the
        sampled noise to the box device.
        """
        seed = (self.box_seed * 1_000_003 + self.current_epoch * 9_973 + batch_idx) % (2**31 - 1)
        gen = torch.Generator()  # CPU
        gen.manual_seed(int(seed))
        return gen

    def _prepare_input(self, images: torch.Tensor, masks: torch.Tensor,
                       training: bool, batch_idx: int = 0) -> torch.Tensor:
        if not self.box_conditioned:
            return images
        generator = self._box_generator(batch_idx) if training else None
        return build_box_input(images, masks, training=training,
                               jitter_frac=self.box_jitter_frac,
                               generator=generator)

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """One training epoch. Returns {'train_loss', 'train_dice', 'train_iou'}."""
        self.model.train()
        acc = MetricsAccumulator()
        total_loss, n_batches = 0.0, 0

        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"Train ep{self.current_epoch+1}", leave=False)
        ):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            net_in = self._prepare_input(images, masks, training=True,
                                         batch_idx=batch_idx)

            self.optimizer.zero_grad()
            logits = self.model(net_in)
            loss = self.loss_fn(logits, masks)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                acc.update(logits.detach(), masks)
                total_loss += loss.item()
                n_batches += 1

        metrics = acc.compute()
        return {
            "train_loss": total_loss / max(n_batches, 1),
            "train_dice": metrics["dice"],
            "train_iou": metrics["iou"],
        }

    @torch.no_grad()
    def val_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """One validation epoch. Returns {'val_loss', 'val_dice', 'val_iou'}."""
        self.model.eval()
        acc = MetricsAccumulator()
        total_loss, n_batches = 0.0, 0

        for batch in tqdm(dataloader, desc=f"Val   ep{self.current_epoch+1}", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            net_in = self._prepare_input(images, masks, training=False)
            logits = self.model(net_in)
            loss = self.loss_fn(logits, masks)
            acc.update(logits, masks)
            total_loss += loss.item()
            n_batches += 1

        metrics = acc.compute()
        return {
            "val_loss": total_loss / max(n_batches, 1),
            "val_dice": metrics["dice"],
            "val_iou": metrics["iou"],
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        early_stop_patience: int = 15,
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping on val_dice."""
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_dice": [], "train_iou": [],
            "val_loss": [], "val_dice": [], "val_iou": [],
        }
        epochs_no_improve = 0

        for epoch in range(max_epochs):
            self.current_epoch = epoch
            train_m = self.train_epoch(train_loader)
            val_m = self.val_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            for k, v in {**train_m, **val_m}.items():
                history[k].append(v)

            if self.wandb_run is not None:
                self.wandb_run.log(
                    {**train_m, **val_m,
                     "lr": self.optimizer.param_groups[0]["lr"],
                     "epoch": epoch},
                    step=epoch,
                )

            logger.info(
                "ep%d | train_loss=%.4f train_dice=%.4f | val_loss=%.4f val_dice=%.4f | lr=%.2e",
                epoch + 1,
                train_m["train_loss"], train_m["train_dice"],
                val_m["val_loss"], val_m["val_dice"],
                self.optimizer.param_groups[0]["lr"],
            )

            if val_m["val_dice"] > self.best_val_dice:
                self.best_val_dice = val_m["val_dice"]
                epochs_no_improve = 0
                best_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
                self.save_checkpoint(str(best_path), val_m)
                logger.info("  → new best val_dice=%.4f saved to %s", self.best_val_dice, best_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop_patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        best_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
        if best_path.exists():
            self.load_checkpoint(str(best_path))

        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, compute_hd95: bool = False) -> Dict[str, float]:
        """Test-set evaluation. Returns test_dice, iou, precision, recall (+ hd95 optional)."""
        self.model.eval()
        acc = MetricsAccumulator(track_hd95=compute_hd95)

        for batch in tqdm(dataloader, desc="Test eval", leave=False):
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            net_in = self._prepare_input(images, masks, training=False)
            logits = self.model(net_in)
            acc.update(logits, masks)

        metrics = acc.compute()
        return {
            "test_dice": metrics["dice"],
            "test_iou": metrics["iou"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            **( {"test_hd95": metrics["hd95"]} if compute_hd95 else {} ),
        }

    def save_checkpoint(self, path: str, metrics: Dict[str, float]) -> None:
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_dice": self.best_val_dice,
            "run_name": self.run_name,
            "metrics": metrics,
        }, path)

    def load_checkpoint(self, path: str) -> Dict[str, float]:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler and ckpt.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"]
        self.best_val_dice = ckpt["best_val_dice"]
        self.run_name = ckpt.get("run_name", "model")
        logger.info("Loaded checkpoint from %s (epoch %d, val_dice=%.4f)",
                    path, self.current_epoch, self.best_val_dice)
        return ckpt.get("metrics", {})


def hyperparameter_search(
    model_fn: Callable[[], nn.Module],
    loss_fn_name: str,
    data_root: str,
    dataset_name: str,
    regime: str,
    lr_list: List[float],
    batch_size: int,
    max_epochs: int,
    patience: int,
    device: str,
    wandb_project: str,
    wandb_entity: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> Tuple[nn.Module, float, Dict[str, Any]]:
    """
    Random search over lr_list. Returns (best_model, best_val_dice, best_config).
    Each trial: fresh model, Adam(lr, weight_decay=1e-4), CosineAnnealingLR.
    """
    import wandb

    dataloaders = get_all_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        regime=regime,
        batch_size=batch_size,
        num_workers=4,
        seed=seed,
    )

    best_model: Optional[nn.Module] = None
    best_val_dice = 0.0
    best_config: Dict[str, Any] = {}

    for lr in lr_list:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        run_name = f"{dataset_name}_lr{lr:.0e}"
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=run_name,
            config={"lr": lr, "batch_size": batch_size, "max_epochs": max_epochs,
                    "patience": patience, "loss_fn": loss_fn_name,
                    "dataset": dataset_name, "seed": seed},
            reinit=True,
        )

        try:
            model = model_fn()
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

            trainer = Trainer(
                model=model,
                loss_fn=get_loss(loss_fn_name),
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                wandb_run=run,
            )
            trainer.run_name = run_name

            history = trainer.fit(
                train_loader=dataloaders["train"],
                val_loader=dataloaders["val"],
                max_epochs=max_epochs,
                early_stop_patience=patience,
            )

            trial_best = max(history["val_dice"])
            run.summary["best_val_dice"] = trial_best
            run.finish()

            logger.info("lr=%.1e → val_dice=%.4f", lr, trial_best)

            if trial_best > best_val_dice:
                best_val_dice = trial_best
                best_model = model
                best_config = {"lr": lr, "val_dice": trial_best,
                               "loss_fn": loss_fn_name, "dataset": dataset_name}

        except Exception as exc:
            logger.error("Trial lr=%.1e failed: %s", lr, exc)
            run.finish(exit_code=1)

    if best_model is None:
        raise RuntimeError("All hyperparameter search trials failed")

    return best_model, best_val_dice, best_config
