from src.dataset import discover_clips
from src.manifest import MANIFEST_COLUMNS


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
