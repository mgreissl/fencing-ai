from src.dataset import discover_clips
from src.manifest import MANIFEST_COLUMNS, ClipRecord


def test_discover_clips_uses_manifest(tmp_path):
    row = dict.fromkeys(MANIFEST_COLUMNS, "")
    row.update(
        clip_path="data/clips/foil/R_foil_mx_3.mp4",
        label="R",
        weapon="foil",
        match_id="mx",
        hit_type="No-On",
    )
    (tmp_path / "manifest.csv").write_text(
        ",".join(MANIFEST_COLUMNS) + "\n" + ",".join(row[c] for c in MANIFEST_COLUMNS) + "\n"
    )
    (tmp_path / "L_foil_other_1.mp4").write_bytes(b"")  # ignored: manifest wins
    records = discover_clips(clips_dir=str(tmp_path), weapon="foil", validate=False)
    assert len(records) == 1
    assert records[0].hit_type == "No-On"
    assert records[0].label_id == 2


def test_discover_clips_legacy_scan(tmp_path):
    (tmp_path / "L_foil_a_b_1.mp4").write_bytes(b"")
    (tmp_path / "T_foil_a_b_2.mp4").write_bytes(b"")
    records = discover_clips(clips_dir=str(tmp_path), weapon="foil", validate=False)
    assert {r.match_id for r in records} == {"a_b"}
    assert sorted(r.label for r in records) == ["L", "T"]


import os

import pytest

import src.dataset as dataset_module
from src.dataset import resolve_weapons


def test_resolve_weapons_single_ignores_exclusion():
    assert resolve_weapons("epee", ["epee"]) == ["epee"]
    assert resolve_weapons("Foil", ["epee"]) == ["foil"]


@pytest.mark.parametrize("mode", ["multi", "all"])
def test_resolve_weapons_multi_drops_excluded(mode):
    assert resolve_weapons(mode, ["epee"]) == ["foil", "sabre"]
    assert resolve_weapons(mode, []) == ["foil", "sabre", "epee"]
    assert resolve_weapons(mode, ["epee", "sabre"]) == ["foil"]


def test_resolve_weapons_rejects_everything_excluded():
    with pytest.raises(ValueError):
        resolve_weapons("multi", ["foil", "sabre", "epee"])


def test_discover_clips_multi_excludes_epee_by_default(monkeypatch):
    loaded = []

    def fake_load(directory, weapon):
        loaded.append(weapon)
        return [ClipRecord(f"data/clips/{weapon}/L_{weapon}_a_1.mp4", "L", weapon)]

    monkeypatch.setattr(dataset_module, "load_weapon_records", fake_load)
    monkeypatch.setattr(os.path, "isdir", lambda p: True)
    records = discover_clips(weapon="multi", validate=False)
    assert loaded == ["foil", "sabre"]
    assert {r.weapon for r in records} == {"foil", "sabre"}
    loaded.clear()
    discover_clips(weapon="multi", exclude_weapons=[], validate=False)
    assert loaded == ["foil", "sabre", "epee"]
