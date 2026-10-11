import src.compat  # noqa: F401 — must be first to patch Hydra for Python 3.14+

import json
import logging
import os
import sys
from pathlib import Path
from typing import Sequence

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

from src.inference import predict_with_flip_tta
from src.metrics import (
    LABEL_LETTERS,
    evaluation_report,
    hybrid_predict,
    write_predictions,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: list[str],
    flip_tta: bool = False,
) -> dict:
    """
    Run full evaluation on a dataset.

    With ``flip_tta`` the logits are averaged with those of the mirrored clip
    (see ``predict_with_flip_tta``).

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
        kinematics = batch["kinematics"].to(device) if "kinematics" in batch else None

        amp_device = "cuda" if device.type == "cuda" else ("mps" if device.type == "mps" else "cpu")
        with torch.amp.autocast(device_type=amp_device, enabled=device.type in ["cuda", "mps"]):
            if flip_tta:
                logits = predict_with_flip_tta(model, pixel_values, weapon_ids, kinematics)
            else:
                logits = model(pixel_values, weapon_id=weapon_ids, kinematics=kinematics)
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
    class_indices = list(range(len(class_names)))
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        labels=class_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        all_labels,
        all_preds,
        labels=class_indices,
        target_names=class_names,
        zero_division=0,
    )
    cm = sk_confusion_matrix(all_labels, all_preds, labels=class_indices)

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
    extra = results.get("report_extra")
    if extra is not None:
        wandb.summary["test/macro_f1"] = extra["overall"]["macro_f1"]
        wandb.summary["test/ece"] = extra["overall"]["ece"]
        wandb.summary["test/ambiguous_accuracy"] = extra["ambiguous"]["accuracy"]
        wandb.summary["test/ambiguous_macro_f1"] = extra["ambiguous"]["macro_f1"]
        if "model_only" in extra:
            wandb.summary["test/model_only_accuracy"] = extra["model_only"]["overall"]["accuracy"]
        wandb.summary["test/baseline_majority_accuracy"] = extra["baselines"]["majority"]["accuracy"]
        wandb.summary["test/baseline_light_only_accuracy"] = extra["baselines"]["light_only"]["accuracy"]
        for hit, m in extra["per_hit_type"].items():
            wandb.summary[f"test/hit_type/{hit}/accuracy"] = m["accuracy"]
        for weapon, m in extra["per_weapon"].items():
            wandb.summary[f"test/weapon/{weapon}/accuracy"] = m["accuracy"]

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


def merge_checkpoint_config(
    base_cfg: DictConfig, checkpoint_cfg: dict, overrides: Sequence[str] = ()
) -> DictConfig:
    """Config used for evaluation: checkpoint settings win over defaults.

    Precedence (low to high): current defaults (fills keys that older
    checkpoints lack) < config stored in the checkpoint < explicit CLI
    overrides (``key=value`` strings; ``~``-deletions and ``hydra.*`` skipped).
    """
    explicit = []
    for item in overrides:
        item = item.strip()
        if not item or item.startswith("~") or "=" not in item:
            continue
        item = item.lstrip("+")
        if item.startswith("hydra."):
            continue
        explicit.append(item)
    merged = OmegaConf.merge(
        base_cfg, OmegaConf.create(checkpoint_cfg), OmegaConf.from_dotlist(explicit)
    )
    OmegaConf.resolve(merged)
    return merged


def _cli_overrides() -> list[str]:
    """Explicit command-line overrides of this Hydra run."""
    from hydra.core.hydra_config import HydraConfig

    return list(HydraConfig.get().overrides.task)


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if value != value else value  # NaN -> null
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def prediction_rows(records: Sequence, results: dict) -> list[dict]:
    """Per-clip rows for predictions.csv (records are in dataloader order).

    ``pred`` is the system prediction (``system_preds`` if present, i.e. with
    hybrid rules applied), ``model_pred`` the raw model output.
    """
    rows = []
    model_preds = results["all_preds"]
    system_preds = results.get("system_preds", model_preds)
    from_rule = results.get("from_rule", [False] * len(model_preds))
    for rec, label, model_pred, pred, rule, probs in zip(
        records,
        results["all_labels"],
        model_preds,
        system_preds,
        from_rule,
        results["all_probs"],
    ):
        rows.append(
            {
                "clip_path": rec.clip_path,
                "match_id": rec.match_id,
                "hit_type": rec.hit_type,
                "weapon": rec.weapon,
                "label": LABEL_LETTERS[label],
                "pred": LABEL_LETTERS[pred],
                "p_L": f"{probs[0]:.6f}",
                "p_T": f"{probs[1]:.6f}",
                "p_R": f"{probs[2]:.6f}",
                "model_pred": LABEL_LETTERS[model_pred],
                "source": "rule" if rule else "model",
            }
        )
    return rows


def _log_report(report: dict) -> None:
    overall = report["overall"]
    log.info(
        f"Overall (system): acc={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f} "
        f"ECE={overall['ece']:.4f} recall(L/T/R)="
        + "/".join("n/a" if r != r else f"{r:.3f}" for r in overall["recall"])
    )
    amb = report["ambiguous"]
    log.info(
        f"Ambiguous clips (On-On, foil On-Off/Off-On; model decides): n={amb['n']} "
        f"acc={amb['accuracy']:.4f} macro_f1={amb['macro_f1']:.4f}"
    )
    model_only = report.get("model_only")
    if model_only is not None:
        mo = model_only["overall"]
        log.info(
            f"Model only (no rules): acc={mo['accuracy']:.4f} "
            f"macro_f1={mo['macro_f1']:.4f}"
        )
    log.info("Per hit type:")
    for name, m in report["per_hit_type"].items():
        log.info(f"  {name:<8} n={m['n']:<4} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    if report["on_on"] is not None:
        log.info(f"On-On accuracy: {report['on_on']['accuracy']:.4f} (n={report['on_on']['n']})")
    log.info("Per weapon:")
    for name, m in report["per_weapon"].items():
        log.info(f"  {name:<6} n={m['n']:<4} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")
    base = report["baselines"]
    log.info(
        f"Baseline majority ({base['majority']['predicts']}): "
        f"acc={base['majority']['accuracy']:.4f} macro_f1={base['majority']['macro_f1']:.4f}"
    )
    light = base["light_only"]
    on_on = light["on_on_accuracy"]
    log.info(
        f"Baseline light-only: acc={light['accuracy']:.4f} macro_f1={light['macro_f1']:.4f}"
        + ("" if on_on is None else f" on_on_acc={on_on:.4f}")
    )


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    """Main evaluation entry point."""
    log.info("=" * 60)
    log.info("Fencing AI — Evaluation")
    log.info("=" * 60)

    from src.model import get_device, build_model
    from src.dataset import (
        FencingVideoDataset,
        filter_readable_clips,
        kinematics_dataset_kwargs,
    )
    from src.splits import load_split_manifest, resolve_checkpoint_dir, select_split
    from src.transforms import build_eval_transforms

    # --- Load Checkpoint ---
    checkpoint_dir = resolve_checkpoint_dir(cfg)
    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        log.error(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    # Split assignment saved by training; never recomputed here.
    split_records = load_split_manifest(checkpoint_dir)

    # Model and data settings come from the checkpoint; CLI overrides only
    # where given explicitly.
    log.info(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" not in checkpoint:
        log.error("Checkpoint has no stored config; cannot rebuild the model")
        sys.exit(1)
    cfg = merge_checkpoint_config(cfg, checkpoint["config"], _cli_overrides())

    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")

    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    log.info(
        f"Loaded model from epoch {checkpoint.get('epoch', '?')} "
        f"(val_acc={checkpoint.get('val_accuracy', '?')})"
    )

    # --- Data ---
    eval_split = cfg.eval.split
    test_samples = filter_readable_clips(select_split(split_records, eval_split))
    train_samples = select_split(split_records, "train")
    if not test_samples:
        log.error(f"No readable clips in split '{eval_split}'")
        sys.exit(1)

    eval_transform = build_eval_transforms(cfg)
    test_dataset = FencingVideoDataset(
        samples=test_samples,
        num_frames=cfg.data.num_frames,
        transform=eval_transform,
        sampling_mode=cfg.data.sampling_mode,
        **kinematics_dataset_kwargs(cfg),
        window_pre_s=cfg.data.window_pre_s,
        window_post_s=cfg.data.window_post_s,
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
    log.info(f"Evaluating on {len(test_dataset)} '{eval_split}' clips...")
    results = evaluate(
        model,
        test_loader,
        device,
        class_names,
        flip_tta=bool(getattr(cfg.eval, "flip_tta", False)),
    )

    # --- Print Results ---
    log.info(f"\n{'=' * 40}")
    log.info(f"{eval_split.upper()} ACCURACY (model only): {results['accuracy']:.4f} ({results['accuracy'] * 100:.1f}%)")
    log.info(f"{eval_split.upper()} LOSS:     {results['loss']:.4f}")
    log.info(f"{'=' * 40}")
    log.info(f"\nClassification Report:\n{results['report_str']}")
    log.info(f"\nConfusion Matrix:\n{results['confusion_matrix']}")

    hit_types = [r.hit_type for r in test_samples]
    weapons = [r.weapon for r in test_samples]
    system_preds = results["all_preds"]
    if bool(getattr(cfg.eval, "hybrid_rules", False)):
        system_preds, from_rule = hybrid_predict(results["all_preds"], hit_types, weapons)
        results["from_rule"] = from_rule.tolist()
        log.info(
            f"Hybrid rules: {int(from_rule.sum())}/{len(from_rule)} clips decided "
            "by the lights, the rest by the model"
        )
    results["system_preds"] = list(map(int, system_preds))

    report = evaluation_report(
        y_true=results["all_labels"],
        y_pred=system_preds,
        model_pred=results["all_preds"],
        probs=np.asarray(results["all_probs"]),
        hit_types=hit_types,
        weapons=weapons,
        train_labels=[r.label_id for r in train_samples],
        train_hit_types=[r.hit_type for r in train_samples],
        ece_bins=cfg.eval.ece_bins,
        train_weapons=[r.weapon for r in train_samples],
    )
    results["report_extra"] = report
    _log_report(report)

    # --- Save per-clip predictions + metrics ---
    out_dir = Path(cfg.eval.output_dir) if cfg.eval.output_dir else checkpoint_dir
    write_predictions(
        prediction_rows(test_samples, results), out_dir / f"predictions_{eval_split}.csv"
    )
    metrics_path = out_dir / f"metrics_{eval_split}.json"
    metrics_path.write_text(
        json.dumps(_to_jsonable({"loss": results["loss"], **report}), indent=2),
        encoding="utf-8",
    )
    log.info(f"Saved predictions and metrics to {out_dir}")

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
