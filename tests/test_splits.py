import random

import pytest

from src.manifest import ClipRecord
from src.splits import (
    assert_no_group_leak,
    groups_by_split,
    has_complete_split,
    make_group_kfold_splits,
    make_group_splits,
    select_split,
)


def _records(n_matches=40, seed=0, with_tournament=True):
    rng = random.Random(seed)
    recs = []
    for m in range(n_matches):
        for i in range(rng.randint(1, 6)):
            label = rng.choice("LTR")
            recs.append(
                ClipRecord(
                    clip_path=f"data/clips/foil/{label}_foil_m{m}_{i}.mp4",
                    label=label,
                    weapon="foil",
                    match_id=f"m{m}",
                    tournament=f"t{m % 8}" if with_tournament else "",
                )
            )
    return recs


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_no_match_in_two_splits(seed):
    out = make_group_splits(_records(), seed=seed)
    assert len(out) == len(_records())
    assert_no_group_leak(out, "match")
    parts = groups_by_split(out, "match")
    assert parts["train"].isdisjoint(parts["val"])
    assert parts["train"].isdisjoint(parts["test"])
    assert parts["val"].isdisjoint(parts["test"])
    assert all(len(select_split(out, s)) > 0 for s in ("train", "val", "test"))


def test_no_tournament_in_two_splits():
    out = make_group_splits(_records(), group_by="tournament")
    assert_no_group_leak(out, "tournament")
    # tournament grouping implies match grouping
    assert_no_group_leak(out, "match")


def test_split_is_deterministic():
    a = make_group_splits(_records(), seed=7)
    b = make_group_splits(_records(), seed=7)
    assert [r.split for r in a] == [r.split for r in b]


def test_ratios_roughly_respected():
    out = make_group_splits(_records(n_matches=100))
    n = len(out)
    assert 0.6 < len(select_split(out, "train")) / n < 0.8


def test_missing_match_id_each_clip_own_group():
    recs = [
        ClipRecord(f"c{i}.mp4", "LTR"[i % 3], "foil", match_id="") for i in range(30)
    ]
    out = make_group_splits(recs)
    assert_no_group_leak(out)
    assert len(select_split(out, "test")) > 0


def test_leak_detected():
    recs = [
        ClipRecord("a.mp4", "L", "foil", match_id="m", split="train"),
        ClipRecord("b.mp4", "L", "foil", match_id="m", split="test"),
    ]
    with pytest.raises(AssertionError):
        assert_no_group_leak(recs)


def test_invalid_ratios():
    with pytest.raises(ValueError):
        make_group_splits(_records(), 0.5, 0.2, 0.2)


def test_kfold_disjoint_and_covering():
    recs = _records()
    test_groups = []
    for fold in range(5):
        out = make_group_kfold_splits(recs, n_folds=5, fold=fold)
        assert_no_group_leak(out)
        assert has_complete_split(out)
        test_groups.append(groups_by_split(out)["test"])
    # every match is test exactly once
    union = set().union(*test_groups)
    assert sum(len(g) for g in test_groups) == len(union)
    assert len(union) == len({r.match_id for r in recs})


def test_kfold_validates_args():
    with pytest.raises(ValueError):
        make_group_kfold_splits(_records(), n_folds=2, fold=0)
    with pytest.raises(ValueError):
        make_group_kfold_splits(_records(), n_folds=5, fold=5)


def test_stratified_subset_size_and_balance():
    from src.splits import stratified_subset

    recs = _records(n_matches=60)
    sub = stratified_subset(recs, 50, seed=3)
    assert len(sub) == 50
    assert {r.label for r in sub} == {r.label for r in recs}
    # keeps original order, deterministic
    assert [r.clip_path for r in sub] == [r.clip_path for r in recs if r in set(sub)]
    assert sub == stratified_subset(recs, 50, seed=3)
    assert sub != stratified_subset(recs, 50, seed=4)


def test_stratified_subset_noop_and_tiny():
    from src.splits import stratified_subset

    recs = _records(n_matches=10)
    assert stratified_subset(recs, 10_000) == recs
    assert len(stratified_subset(recs, 3)) == 3
    assert len({r.label for r in stratified_subset(recs, 3)}) == 3


def test_build_splits_modes(tmp_path):
    from src.splits import (
        SPLIT_FILENAME,
        build_splits,
        load_split_manifest,
        save_split_manifest,
    )

    recs = _records()
    ratio = build_splits(recs, seed=1)
    assert has_complete_split(ratio)
    cv = build_splits(recs, seed=1, cv_folds=4, cv_fold=2)
    assert has_complete_split(cv)
    # manifest split honoured only when requested and complete
    pre = [r if i else r for i, r in enumerate(ratio)]
    assert build_splits(pre, use_manifest_split=True) == pre
    assert build_splits(recs, use_manifest_split=True) != recs  # incomplete -> computed

    save_split_manifest(ratio, tmp_path)
    assert (tmp_path / SPLIT_FILENAME).is_file()
    assert load_split_manifest(tmp_path) == ratio


def test_load_split_manifest_missing_fails(tmp_path):
    from src.splits import load_split_manifest

    with pytest.raises(FileNotFoundError, match="never recomputed"):
        load_split_manifest(tmp_path)


def test_load_split_manifest_rejects_incomplete(tmp_path):
    from src.manifest import write_manifest
    from src.splits import SPLIT_FILENAME, load_split_manifest

    write_manifest(_records(3), tmp_path / SPLIT_FILENAME)
    with pytest.raises(ValueError, match="split"):
        load_split_manifest(tmp_path)


@pytest.mark.parametrize("cv_folds, expected", [(0, ""), (4, "fold2")])
def test_resolve_checkpoint_dir_and_split_roundtrip(tmp_path, cv_folds, expected):
    from omegaconf import OmegaConf

    from src.splits import (
        load_split_manifest,
        resolve_checkpoint_dir,
        save_split_manifest,
    )

    cfg = OmegaConf.create(
        {
            "training": {"checkpoint_dir": str(tmp_path)},
            "data": {"cv_folds": cv_folds, "cv_fold": 2},
        }
    )
    directory = resolve_checkpoint_dir(cfg)
    assert directory == (tmp_path / expected if expected else tmp_path)
    directory.mkdir(parents=True, exist_ok=True)
    recs = build_splits_for_test()
    save_split_manifest(recs, directory)  # what training does
    assert load_split_manifest(resolve_checkpoint_dir(cfg)) == recs  # evaluate


def build_splits_for_test():
    from src.splits import build_splits

    return build_splits(_records(), seed=1)
