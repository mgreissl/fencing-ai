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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> dict:
    """
    Run full evaluation on a dataset.

    Returns:
        Dict with all evaluation metrics and per-sample results.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []
    all_weapons = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(dataloader, desc="Evaluating", leave=True)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = torch.as_tensor(batch["label"], dtype=torch.long).to(device)
        weapon_ids = torch.as_tensor(batch["weapon_id"], dtype=torch.long).to(device) if "weapon_id" in batch else None

        amp_device = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=amp_device, enabled=device.type in ["cuda", "mps"]):
            logits = model(pixel_values, weapon_id=weapon_ids)
            loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        all_paths.extend(batch["path"])
        if weapon_ids is not None:
            all_weapons.extend(weapon_ids.cpu().tolist())
        else:
            all_weapons.extend([0] * labels.size(0))

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
    )
    cm = sk_confusion_matrix(all_labels, all_preds)

    results = {
        "accuracy": accuracy,
        "loss": total_loss / max(len(all_labels), 1),
        "report": report,
        "report_str": report_str,
        "confusion_matrix": cm,
        "all_preds": all_preds,
        "all_labels": all_labels,
        "all_probs": all_probs,
        "all_paths": all_paths,
        "all_weapons": all_weapons,
    }
    return results


def log_results_to_wandb(
    results: dict,
    class_names: list[str],
    cfg: DictConfig,
):
    """Log comprehensive evaluation results to W&B."""
    import wandb
    from PIL import Image

    # --- Summary Metrics ---
    wandb.summary["test/accuracy"] = results["accuracy"]
    wandb.summary["test/loss"] = results["loss"]

    # Per-class metrics
    for cls_name in class_names:
        if cls_name in results["report"]:
            for metric in ["precision", "recall", "f1-score"]:
                wandb.summary[f"test/{cls_name}/{metric}"] = results["report"][
                    cls_name
                ][metric]

    # --- Confusion Matrix ---
    wandb.log(
        {
            "test/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=results["all_labels"],
                preds=results["all_preds"],
                class_names=class_names,
                title="Test Set Confusion Matrix",
            )
        }
    )

    # --- Misclassified Samples Table ---
    misclassified = []
    for i, (true, pred) in enumerate(
        zip(results["all_labels"], results["all_preds"])
    ):
        if true != pred:
            confidence = results["all_probs"][i][pred]
            misclassified.append((i, confidence))

    # Sort by confidence descending (most confident errors are most interesting)
    misclassified.sort(key=lambda x: x[1], reverse=True)

    max_samples = cfg.wandb.max_misclassified_samples
    misclassified = misclassified[:max_samples]

    if misclassified:
        table = wandb.Table(
            columns=[
                "Thumbnail",
                "True Label",
                "Predicted",
                "Confidence",
                "P(Left)",
                "P(Tie)",
                "P(Right)",
                "Filename",
            ]
        )

        for idx, confidence in misclassified:
            path = results["all_paths"][idx]
            true_label = class_names[results["all_labels"][idx]]
            pred_label = class_names[results["all_preds"][idx]]
            probs = results["all_probs"][idx]

            try:
                thumbnail = _extract_thumbnail(path)
                table.add_data(
                    wandb.Image(thumbnail),
                    true_label,
                    pred_label,
                    f"{confidence:.3f}",
                    f"{probs[0]:.3f}",
                    f"{probs[1]:.3f}",
                    f"{probs[2]:.3f}",
                    os.path.basename(path),
                )
            except Exception as e:
                log.warning(f"Could not extract thumbnail from {path}: {e}")

        wandb.log({"test/misclassified": table})
        log.info(f"Logged {len(misclassified)} misclassified samples to W&B")

    # --- Per-Class Distribution Bar Chart ---
    label_counts = {}
    for label in results["all_labels"]:
        name = class_names[label]
        label_counts[name] = label_counts.get(name, 0) + 1

    wandb.log(
        {
            "test/class_distribution": wandb.plot.bar(
                wandb.Table(
                    data=[[k, v] for k, v in label_counts.items()],
                    columns=["Class", "Count"],
                ),
                "Class",
                "Count",
                title="Test Set Class Distribution",
            )
        }
    )


def _extract_thumbnail(path: str) -> "Image.Image":
    """Extract the middle frame from a video as a PIL Image."""
    from src.utils import extract_thumbnail

    return extract_thumbnail(path)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    log.info("=" * 60)
    log.info("Fencing AI — Evaluation")
    log.info("=" * 60)

    from src.model import get_device, build_model
    from src.dataset import discover_clips, make_splits, FencingVideoDataset
    from src.transforms import build_eval_transforms

    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")

    # --- Load Checkpoint ---
    ckpt_path = Path(cfg.training.checkpoint_dir) / "best_model.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    log.info(
        f"Loaded model from epoch {checkpoint.get('epoch', '?')} "
        f"(val_acc={checkpoint.get('val_accuracy', '?')})"
    )

    # --- Data ---
    samples = discover_clips(
        clips_dir=getattr(cfg.data, "clips_dir", None),
        weapon=getattr(cfg.data, "weapon", "foil"),
        include_flipped=getattr(cfg.data, "include_flipped", False),
        flipped_dir=getattr(cfg.data, "flipped_dir", None),
        max_clips=getattr(cfg.data, "max_clips", None),
    )

    _, _, test_samples = make_splits(
        samples,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        seed=cfg.data.split_seed,
    )

    eval_transform = build_eval_transforms(cfg)
    sampling_mode = getattr(cfg.data, "sampling_mode", "end_weighted")
    test_dataset = FencingVideoDataset(
        samples=test_samples,
        num_frames=cfg.data.num_frames,
        transform=eval_transform,
        sampling_mode=sampling_mode,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    class_names = list(cfg.model.class_names)

    # --- Evaluate ---
    log.info(f"Evaluating on {len(test_dataset)} test clips...")
    results = evaluate(model, test_loader, device, class_names)

    # --- Print Results ---
    log.info(f"\n{'=' * 40}")
    log.info(f"TEST ACCURACY: {results['accuracy']:.4f} ({results['accuracy'] * 100:.1f}%)")
    log.info(f"TEST LOSS:     {results['loss']:.4f}")
    log.info(f"{'=' * 40}")
    log.info(f"\nClassification Report:\n{results['report_str']}")
    log.info(f"\nConfusion Matrix:\n{results['confusion_matrix']}")

    weapon_names = {0: "Foil", 1: "Sabre", 2: "Epee"}
    weapon_correct = {0: 0, 1: 0, 2: 0}
    weapon_total = {0: 0, 1: 0, 2: 0}
    for t, p, w in zip(results["all_labels"], results["all_preds"], results["all_weapons"]):
        weapon_total[w] += 1
        if t == p:
            weapon_correct[w] += 1

    log.info("\nPer-Weapon Test Accuracy:")
    for w_id, w_name in weapon_names.items():
        if weapon_total[w_id] > 0:
            w_acc = weapon_correct[w_id] / weapon_total[w_id]
            log.info(f"  {w_name:<6}: {w_acc:.4f} ({w_acc * 100:.1f}%) [{weapon_correct[w_id]}/{weapon_total[w_id]}]")

    # Count misclassified
    n_misclassified = sum(
        1
        for t, p in zip(results["all_labels"], results["all_preds"])
        if t != p
    )
    log.info(
        f"\nMisclassified: {n_misclassified}/{len(results['all_labels'])} "
        f"({100 * n_misclassified / max(len(results['all_labels']), 1):.1f}%)"
    )

    # --- Log to W&B ---
    if cfg.wandb.enabled:
        import wandb

        try:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=f"{cfg.wandb.run_name or 'eval'}-test",
                tags=list(cfg.wandb.tags) + ["evaluation"],
                config=OmegaConf.to_container(cfg, resolve=True),
                job_type="evaluation",
            )
        except Exception as e:
            log.warning(f"W&B online init failed ({e}); falling back to offline mode")
            wandb.init(
                project=cfg.wandb.project,
                name=f"{cfg.wandb.run_name or 'eval'}-test",
                tags=list(cfg.wandb.tags) + ["evaluation"],
                config=OmegaConf.to_container(cfg, resolve=True),
                job_type="evaluation",
                mode="offline",
            )
        log_results_to_wandb(results, class_names, cfg)
        wandb.finish()
        log.info("Results logged to W&B")

    return results["accuracy"]


if __name__ == "__main__":
    main()
