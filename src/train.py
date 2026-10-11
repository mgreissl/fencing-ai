import src.compat  # noqa: F401 — must be first to patch Hydra for Python 3.14+

import logging
import os
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.inference import autocast_context, predict_with_flip_tta
from src.metrics import ambiguous_macro_f1, macro_f1, per_class_recall
from src.seed import make_generator, seed_everything, seed_worker

log = logging.getLogger(__name__)


def _setup_wandb(cfg: DictConfig) -> bool:
    """Initialize Weights & Biases if enabled. Returns True if active."""
    if not cfg.wandb.enabled:
        log.info("W&B logging disabled")
        return False

    import wandb

    try:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.run_name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        log.info(f"W&B run initialized: {getattr(wandb.run, 'url', 'local')}")
    except Exception as e:
        log.warning(f"W&B online init failed ({e}); falling back to offline mode")
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            tags=list(cfg.wandb.tags),
            config=OmegaConf.to_container(cfg, resolve=True),
            mode="offline",
        )
        log.info("W&B running in offline mode (run data saved locally)")
    return True


SELECTION_METRICS = {
    "macro_f1": "val/macro_f1",
    "accuracy": "val/accuracy",
    # macro-F1 on the clips the lights do not decide (On-On, foil On-Off/Off-On)
    "on_on_macro_f1": "val/on_on_macro_f1",
}
FALLBACK_SELECTION_METRIC = "macro_f1"


def _selection_key(name: str) -> str:
    """Validation metric key for `training.selection_metric`."""
    if name not in SELECTION_METRICS:
        raise ValueError(
            f"training.selection_metric must be one of {sorted(SELECTION_METRICS)}, "
            f"got {name!r}"
        )
    return SELECTION_METRICS[name]


def _selection_score(
    val_metrics: dict, name: str, warned: set
) -> tuple[float, str]:
    """Score for checkpoint selection and the metric name actually used.

    ``on_on_macro_f1`` needs ambiguous clips in the validation split; without
    them it falls back to ``macro_f1`` (warning logged once per run via
    ``warned``).
    """
    key = _selection_key(name)
    value = val_metrics.get(key)
    if value is None or value != value:
        if name not in warned:
            warned.add(name)
            log.warning(
                f"selection_metric={name} is undefined (no ambiguous clips in the "
                f"validation split); falling back to {FALLBACK_SELECTION_METRIC}"
            )
        name = FALLBACK_SELECTION_METRIC
        value = val_metrics[_selection_key(name)]
    return float(value), name


def _clip_metadata(dataloader: DataLoader) -> dict:
    """{abs clip path: (hit_type, weapon)} from the loader's dataset records."""
    samples = getattr(dataloader.dataset, "samples", None) or []
    return {rec.abs_path(): (rec.hit_type, rec.weapon) for rec in samples}


def _build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Build learning rate scheduler from config."""
    sched_cfg = cfg.training.scheduler

    if sched_cfg.type == "none":
        return None

    warmup_steps = sched_cfg.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.epochs * steps_per_epoch

    if sched_cfg.type == "cosine":
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR,
            LinearLR,
            SequentialLR,
        )

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=max(warmup_steps, 1),
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=sched_cfg.min_lr,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[max(warmup_steps, 1)],
        )
        return scheduler

    elif sched_cfg.type == "step":
        from torch.optim.lr_scheduler import StepLR

        return StepLR(optimizer, step_size=10 * steps_per_epoch, gamma=0.1)

    return None


def _train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    use_amp: bool,
    grad_clip_norm: float,
    epoch: int,
    wandb_active: bool,
) -> dict:
    """Run one training epoch. Returns metrics dict."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    # AMP scaler — only for CUDA; MPS doesn't support GradScaler
    use_scaler = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler() if use_scaler else None

    pbar = tqdm(dataloader, desc=f"Train E{epoch}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long).to(device)
        weapon_ids = torch.as_tensor(batch["weapon_id"], dtype=torch.long).to(device) if "weapon_id" in batch else None
        kinematics = batch["kinematics"].to(device) if "kinematics" in batch else None

        optimizer.zero_grad()

        # Mixed precision forward (CUDA and Apple Silicon MPS)
        with autocast_context(device, use_amp):
            logits = model(pixel_values, weapon_id=weapon_ids, kinematics=kinematics)
            loss = criterion(logits, labels)

        # Backward
        if use_scaler:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Metrics
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100 * correct / total:.1f}%",
        )

    metrics = {
        "train/loss": total_loss / max(total, 1),
        "train/accuracy": correct / max(total, 1),
    }

    if wandb_active:
        import wandb

        metrics["train/lr"] = optimizer.param_groups[0]["lr"]
        wandb.log(metrics, step=epoch)

    return metrics


@torch.no_grad()
def _validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    wandb_active: bool,
    cfg: DictConfig,
) -> dict:
    """
    Run validation. Returns metrics dict.

    Also logs confusion matrix and misclassified samples to W&B
    at configured intervals.
    """
    model.eval()
    flip_tta = bool(getattr(getattr(cfg, "eval", None), "flip_tta", False))
    use_amp = bool(cfg.training.mixed_precision)
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    all_weapons = []

    pbar = tqdm(dataloader, desc=f"Val   E{epoch}", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long).to(device)
        weapon_ids = torch.as_tensor(batch["weapon_id"], dtype=torch.long).to(device) if "weapon_id" in batch else None
        kinematics = batch["kinematics"].to(device) if "kinematics" in batch else None

        with autocast_context(device, use_amp):
            if flip_tta:
                logits = predict_with_flip_tta(model, pixel_values, weapon_ids, kinematics)
            else:
                logits = model(pixel_values, weapon_id=weapon_ids, kinematics=kinematics)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_paths.extend(batch["path"])
        if weapon_ids is not None:
            all_weapons.extend(weapon_ids.cpu().tolist())
        else:
            all_weapons.extend([0] * labels.size(0))

    val_acc = correct / max(total, 1)
    metrics = {
        "val/loss": total_loss / max(total, 1),
        "val/accuracy": val_acc,
        "val/macro_f1": macro_f1(all_labels, all_preds),
    }
    meta = _clip_metadata(dataloader)
    if meta:
        hit_types = [meta.get(p, ("", ""))[0] for p in all_paths]
        weapon_names_by_clip = [meta.get(p, ("", ""))[1] for p in all_paths]
        on_on_f1 = ambiguous_macro_f1(
            all_labels, all_preds, hit_types, weapon_names_by_clip
        )
        if on_on_f1 == on_on_f1:  # omitted (not NaN) when no ambiguous clips
            metrics["val/on_on_macro_f1"] = on_on_f1
    class_names = list(cfg.model.class_names)
    for name, rec in zip(class_names, per_class_recall(all_labels, all_preds)):
        if not np.isnan(rec):
            metrics[f"val/recall_{name}"] = float(rec)

    # Compute per-weapon metrics
    weapon_names = {0: "foil", 1: "sabre", 2: "epee"}
    weapon_correct = {0: 0, 1: 0, 2: 0}
    weapon_total = {0: 0, 1: 0, 2: 0}

    for y_true, y_pred, w_id in zip(all_labels, all_preds, all_weapons):
        weapon_total[w_id] += 1
        if y_true == y_pred:
            weapon_correct[w_id] += 1

    for w_id, w_name in weapon_names.items():
        if weapon_total[w_id] > 0:
            w_acc = weapon_correct[w_id] / weapon_total[w_id]
            metrics[f"val/{w_name}_accuracy"] = w_acc
            log.info(f"  Val {w_name.capitalize()} Acc: {w_acc:.4f} ({weapon_correct[w_id]}/{weapon_total[w_id]})")

    if wandb_active:
        import wandb

        wandb.log(metrics, step=epoch)

        # Log confusion matrix and misclassified samples periodically
        log_media = (
            epoch % cfg.wandb.log_media_every_n_epochs == 0
            or epoch == cfg.training.epochs - 1
        )
        if log_media:
            _log_confusion_matrix(all_labels, all_preds, cfg, epoch)
            _log_misclassified(
                all_labels,
                all_preds,
                all_probs,
                all_paths,
                all_weapons,
                cfg,
                epoch,
            )

    return metrics


def _log_confusion_matrix(
    labels: list, preds: list, cfg: DictConfig, epoch: int
):
    """Log confusion matrix to W&B."""
    import wandb

    class_names = list(cfg.model.class_names)
    wandb.log(
        {
            "val/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=labels,
                preds=preds,
                class_names=class_names,
                title=f"Confusion Matrix (Epoch {epoch})",
            )
        },
        step=epoch,
    )


def _log_misclassified(
    labels: list,
    preds: list,
    probs: list,
    paths: list,
    weapons: list,
    cfg: DictConfig,
    epoch: int,
):
    """Log misclassified video clips to W&B as a Table with thumbnails and weapon type."""
    import wandb
    from PIL import Image

    class_names = list(cfg.model.class_names)
    weapon_names = {0: "Foil", 1: "Sabre", 2: "Epee"}
    max_samples = cfg.wandb.max_misclassified_samples

    # Find misclassified indices, sorted by confidence (most confident errors first)
    misclassified = []
    for i, (true, pred) in enumerate(zip(labels, preds)):
        if true != pred:
            confidence = probs[i][pred]
            misclassified.append((i, confidence))

    # Sort by confidence descending (most confident mistakes first)
    misclassified.sort(key=lambda x: x[1], reverse=True)
    misclassified = misclassified[:max_samples]

    if not misclassified:
        return

    # Build W&B Table
    table = wandb.Table(
        columns=[
            "Thumbnail",
            "Weapon",
            "True Label",
            "Predicted",
            "Confidence",
            "Filename",
        ]
    )

    for idx, confidence in misclassified:
        path = paths[idx]
        true_label = class_names[labels[idx]]
        pred_label = class_names[preds[idx]]
        w_name = weapon_names.get(weapons[idx], "Unknown") if idx < len(weapons) else "Unknown"

        # Extract a middle-frame thumbnail
        try:
            thumbnail = _extract_thumbnail(path, cfg.data.num_frames)
            table.add_data(
                wandb.Image(thumbnail),
                w_name,
                true_label,
                pred_label,
                f"{confidence:.3f}",
                os.path.basename(path),
            )
        except Exception as e:
            log.warning(f"Could not extract thumbnail from {path}: {e}")
            table.add_data(
                None,
                w_name,
                true_label,
                pred_label,
                f"{confidence:.3f}",
                os.path.basename(path),
            )

    wandb.log({f"val/misclassified_epoch_{epoch}": table}, step=epoch)
    log.info(f"Logged {len(misclassified)} misclassified samples to W&B")


def _extract_thumbnail(path: str, num_frames: int) -> "Image.Image":
    """Extract the middle frame from a video as a PIL Image."""
    from src.utils import extract_thumbnail

    return extract_thumbnail(path)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main training entry point."""
    log.info("=" * 60)
    log.info("Fencing AI — Modern Training Pipeline")
    log.info("=" * 60)
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")

    # --- Device ---
    from src.model import get_device, build_model

    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")

    # --- W&B ---
    wandb_active = _setup_wandb(cfg)

    # --- Seed ---
    seed = int(cfg.training.seed)
    seed_everything(seed)

    # --- Data ---
    from src.dataset import discover_clips, FencingVideoDataset, kinematics_dataset_kwargs
    from src.splits import (
        build_splits,
        resolve_checkpoint_dir,
        save_split_manifest,
        select_split,
    )
    from src.transforms import build_train_transforms, build_eval_transforms

    samples = discover_clips(
        clips_dir=getattr(cfg.data, "clips_dir", None),
        weapon=getattr(cfg.data, "weapon", "foil"),
        include_flipped=getattr(cfg.data, "include_flipped", False),
        flipped_dir=getattr(cfg.data, "flipped_dir", None),
        max_clips=getattr(cfg.data, "max_clips", None),
        seed=cfg.data.split_seed,
        exclude_weapons=list(getattr(cfg.data, "exclude_weapons", ["epee"])),
    )

    if len(samples) == 0:
        log.error(
            "No clips found! Please ensure clips (manifest.csv or L/R/T-prefixed "
            f".mp4 files) exist in {cfg.data.clips_dir}"
        )
        sys.exit(1)

    cv_folds = int(getattr(cfg.data, "cv_folds", 0) or 0)
    cv_fold = int(getattr(cfg.data, "cv_fold", 0) or 0)
    split_records = build_splits(
        samples,
        group_by=cfg.data.group_by,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.data.split_seed,
        cv_folds=cv_folds,
        cv_fold=cv_fold,
        use_manifest_split=getattr(cfg.data, "use_manifest_split", False),
    )
    train_samples = select_split(split_records, "train")
    val_samples = select_split(split_records, "val")

    # Persist the split next to the checkpoint; evaluate.py reads it back.
    checkpoint_dir = resolve_checkpoint_dir(cfg)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_split_manifest(split_records, checkpoint_dir)

    train_transform = build_train_transforms(cfg)
    eval_transform = build_eval_transforms(cfg)

    sampling_mode = getattr(cfg.data, "sampling_mode", "onset_window")
    log.info(f"Using temporal frame sampling mode: {sampling_mode}")
    window_kwargs = dict(
        window_pre_s=cfg.data.window_pre_s, window_post_s=cfg.data.window_post_s
    )

    kin_kwargs = kinematics_dataset_kwargs(cfg)

    train_dataset = FencingVideoDataset(
        samples=train_samples,
        num_frames=cfg.data.num_frames,
        transform=train_transform,
        sampling_mode=sampling_mode,
        temporal_jitter=cfg.augmentation.train.temporal_jitter.enabled,
        max_shift=cfg.augmentation.train.temporal_jitter.max_shift,
        **kin_kwargs,
        **window_kwargs,
    )
    val_dataset = FencingVideoDataset(
        samples=val_samples,
        num_frames=cfg.data.num_frames,
        transform=eval_transform,
        sampling_mode=sampling_mode,
        **kin_kwargs,
        **window_kwargs,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
        generator=make_generator(seed),
        worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        worker_init_fn=seed_worker,
    )

    # --- Model ---
    model = build_model(cfg).to(device)

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    scheduler = _build_scheduler(optimizer, cfg, steps_per_epoch)

    # --- Training Loop ---
    _selection_key(cfg.training.selection_metric)  # validate the name early
    selection_warned: set = set()
    selection_name = cfg.training.selection_metric
    best_score = -1.0
    best_val_acc = 0.0
    patience_counter = 0

    log.info(f"Starting training for {cfg.training.epochs} epochs")
    log.info(f"  Batch size: {cfg.training.batch_size}")
    log.info(f"  Train batches/epoch: {len(train_loader)}")
    log.info(f"  Val batches/epoch: {len(val_loader)}")

    for epoch in range(cfg.training.epochs):
        log.info(f"\n--- Epoch {epoch + 1}/{cfg.training.epochs} ---")

        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=cfg.training.mixed_precision,
            grad_clip_norm=cfg.training.grad_clip_norm,
            epoch=epoch,
            wandb_active=wandb_active,
        )

        val_metrics = _validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            wandb_active=wandb_active,
            cfg=cfg,
        )

        log.info(
            f"  Train Loss: {train_metrics['train/loss']:.4f}  "
            f"Acc: {train_metrics['train/accuracy']:.4f}"
        )
        log.info(
            f"  Val   Loss: {val_metrics['val/loss']:.4f}  "
            f"Acc: {val_metrics['val/accuracy']:.4f}  "
            f"MacroF1: {val_metrics['val/macro_f1']:.4f}"
            + (
                f"  OnOnMacroF1: {val_metrics['val/on_on_macro_f1']:.4f}"
                if "val/on_on_macro_f1" in val_metrics
                else ""
            )
        )

        # --- Checkpointing ---
        val_acc = val_metrics["val/accuracy"]
        score, selection_name = _selection_score(
            val_metrics, cfg.training.selection_metric, selection_warned
        )
        if score > best_score:
            best_score = score
            best_val_acc = val_acc
            patience_counter = 0
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_metrics["val/macro_f1"],
                    "selection_metric": selection_name,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            log.info(
                f"  ✓ New best model saved ({selection_name}={score:.4f})"
            )

            # Save as W&B artifact
            if wandb_active:
                import wandb

                artifact = wandb.Artifact(
                    name="best-model",
                    type="model",
                    metadata={
                        "val_accuracy": val_acc,
                        "val_macro_f1": val_metrics["val/macro_f1"],
                        "epoch": epoch,
                    },
                )
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)
        else:
            patience_counter += 1
            log.info(
                f"  No improvement (patience: {patience_counter}/"
                f"{cfg.training.early_stopping_patience})"
            )

        # --- Early Stopping ---
        if (
            cfg.training.early_stopping_patience > 0
            and patience_counter >= cfg.training.early_stopping_patience
        ):
            log.info(
                f"Early stopping triggered after {epoch + 1} epochs "
                f"(best {selection_name}={best_score:.4f})"
            )
            break

    log.info(
        f"\nTraining complete! Best {cfg.training.selection_metric}: "
        f"{best_score:.4f} (val accuracy {best_val_acc:.4f})"
    )

    # --- Final W&B summary ---
    if wandb_active:
        import wandb

        wandb.summary["best_val_accuracy"] = best_val_acc
        wandb.summary[f"best_{selection_name}"] = best_score
        wandb.finish()

    return best_val_acc


if __name__ == "__main__":
    main()
