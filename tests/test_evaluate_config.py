import pytest
from omegaconf import OmegaConf

from src.evaluate import merge_checkpoint_config, prediction_rows
from src.manifest import ClipRecord
from src.train import _selection_key

BASE = OmegaConf.create(
    {
        "data": {
            "num_frames": 16,
            "sampling_mode": "onset_window",
            "window_pre_s": 1.0,
            "window_post_s": 0.25,
        },
        "model": {"use_kinematics": True, "head": {"hidden_dim": 256}},
        "training": {"device": "auto", "batch_size": 4, "checkpoint_dir": "checkpoints"},
    }
)
CKPT = {
    "data": {"num_frames": 8, "sampling_mode": "end_weighted"},
    "model": {"use_kinematics": False, "head": {"hidden_dim": 128}},
    "training": {"device": "mps", "batch_size": 2},
}


def test_checkpoint_settings_beat_defaults_and_fill_missing():
    cfg = merge_checkpoint_config(BASE, CKPT, [])
    assert cfg.data.num_frames == 8
    assert cfg.data.sampling_mode == "end_weighted"
    assert cfg.model.head.hidden_dim == 128
    assert cfg.model.use_kinematics is False
    assert cfg.data.window_pre_s == 1.0  # not in checkpoint -> default
    assert cfg.training.checkpoint_dir == "checkpoints"


def test_only_explicit_overrides_win():
    cfg = merge_checkpoint_config(
        BASE, CKPT, ["training.device=cpu", "+eval.split=val", "~foo", "hydra.run.dir=x"]
    )
    assert cfg.training.device == "cpu"
    assert cfg.training.batch_size == 2  # checkpoint, not CLI default
    assert cfg.eval.split == "val"
    assert "hydra" not in cfg


def test_override_value_types_parsed():
    cfg = merge_checkpoint_config(BASE, CKPT, ["data.num_frames=12", "data.x=null"])
    assert cfg.data.num_frames == 12 and cfg.data.x is None


def test_selection_key():
    assert _selection_key("macro_f1") == "val/macro_f1"
    assert _selection_key("accuracy") == "val/accuracy"
    with pytest.raises(ValueError):
        _selection_key("loss")


def test_prediction_rows_order_and_letters():
    recs = [
        ClipRecord("a.mp4", "L", "foil", match_id="m1", hit_type="On-No"),
        ClipRecord("b.mp4", "R", "epee", match_id="m2"),
    ]
    rows = prediction_rows(
        recs,
        {
            "all_labels": [0, 2],
            "all_preds": [1, 2],
            "all_probs": [[0.1, 0.8, 0.1], [0.0, 0.1, 0.9]],
        },
    )
    assert [r["clip_path"] for r in rows] == ["a.mp4", "b.mp4"]
    assert rows[0]["label"] == "L" and rows[0]["pred"] == "T"
    assert rows[1]["match_id"] == "m2" and rows[1]["p_R"] == "0.900000"


def test_prediction_rows_carry_system_and_model_pred():
    recs = [ClipRecord("a.mp4", "L", "foil", match_id="m1", hit_type="On-No")]
    rows = prediction_rows(
        recs,
        {
            "all_labels": [0],
            "all_preds": [2],  # raw model prediction
            "all_probs": [[0.1, 0.1, 0.8]],
            "system_preds": [0],
            "from_rule": [True],
        },
    )
    assert rows[0]["pred"] == "L" and rows[0]["model_pred"] == "R"
    assert rows[0]["source"] == "rule"


EVAL_BASE = OmegaConf.create(
    {
        **OmegaConf.to_container(BASE),
        "eval": {"split": "test", "flip_tta": True, "hybrid_rules": True},
    }
)
EVAL_CKPT = {
    **CKPT,
    "eval": {"split": "val", "flip_tta": False, "legacy_key": 1},
}


def test_checkpoint_eval_settings_do_not_override_current_eval_config():
    cfg = merge_checkpoint_config(EVAL_BASE, EVAL_CKPT, [])
    assert cfg.eval.split == "test"
    assert cfg.eval.flip_tta is True
    assert cfg.eval.hybrid_rules is True
    assert "legacy_key" not in cfg.eval
    assert cfg.data.num_frames == 8  # non-eval settings still from checkpoint


def test_explicit_eval_override_still_wins():
    cfg = merge_checkpoint_config(EVAL_BASE, EVAL_CKPT, ["eval.split=val"])
    assert cfg.eval.split == "val"
    assert cfg.eval.flip_tta is True


def test_checkpoint_dict_is_not_mutated():
    ckpt = {**EVAL_CKPT}
    merge_checkpoint_config(EVAL_BASE, ckpt, [])
    assert "eval" in ckpt


def test_effective_eval_summary_lists_all_eval_keys():
    from src.evaluate import effective_eval_summary

    cfg = merge_checkpoint_config(EVAL_BASE, EVAL_CKPT, [])
    text = effective_eval_summary(cfg)
    assert "split=test" in text and "flip_tta=True" in text
    assert "hybrid_rules=True" in text
