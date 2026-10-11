import pytest

from src.manifest import (
    MANIFEST_COLUMNS,
    ClipRecord,
    ManifestError,
    load_manifest,
    load_weapon_records,
    merge_records,
    parse_legacy_filename,
    write_manifest,
)

HEADER = ",".join(MANIFEST_COLUMNS)


def _row(**kw):
    base = dict.fromkeys(MANIFEST_COLUMNS, "")
    base.update(
        clip_path="data/clips/foil/L_foil_m1_0.mp4",
        label="L",
        weapon="foil",
        match_id="m1",
    )
    base.update(kw)
    return ",".join(str(base[c]) for c in MANIFEST_COLUMNS)


def _write(tmp_path, rows, header=HEADER):
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_load_manifest_parses_types(tmp_path):
    path = _write(
        tmp_path,
        [_row(hit_type="On-No", n_frames=60, light_frame_in_clip=45.5, clip_fps=30)],
    )
    (rec,) = load_manifest(path)
    assert rec.label_id == 0
    assert rec.hit_type == "On-No"
    assert rec.n_frames == 60 and isinstance(rec.n_frames, int)
    assert rec.light_frame_in_clip == 45.5
    assert rec.light_time_s is None
    assert rec.split == ""


def test_load_manifest_rejects_bad_label(tmp_path):
    with pytest.raises(ManifestError, match="label"):
        load_manifest(_write(tmp_path, [_row(label="X")]))


def test_load_manifest_rejects_bad_hit_type(tmp_path):
    with pytest.raises(ManifestError, match="hit_type"):
        load_manifest(_write(tmp_path, [_row(hit_type="Maybe")]))


def test_load_manifest_missing_column(tmp_path):
    with pytest.raises(ManifestError, match="missing"):
        load_manifest(_write(tmp_path, ["a,b"], header="clip_path,label"))


def test_load_manifest_duplicate_path(tmp_path):
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(_write(tmp_path, [_row(), _row()]))


def test_write_roundtrip(tmp_path):
    rec = ClipRecord(
        clip_path="data/clips/epee/T_epee_a_b_3.mp4",
        label="T",
        weapon="epee",
        match_id="a_b",
        hit_type="On-On",
        n_frames=48,
        light_frame_in_clip=30.0,
        split="test",
    )
    out = tmp_path / "sub" / "m.csv"
    write_manifest([rec], out)
    assert load_manifest(out) == [rec]


def test_legacy_new_style_match_id():
    info = parse_legacy_filename("T_foil_Paris_2023_final_12.mp4")
    assert info["label"] == "T"
    assert info["weapon"] == "foil"
    assert info["match_id"] == "Paris_2023_final"
    assert info["grouped"]


def test_legacy_flipped_keeps_match():
    info = parse_legacy_filename("L_sabre_x_y_7-flipped.mp4")
    assert info["match_id"] == "x_y"


def test_legacy_old_style_ungrouped():
    info = parse_legacy_filename("L1234-5.mp4")
    assert info["label"] == "L" and not info["grouped"]
    flipped = parse_legacy_filename("L1234-5-flipped.mp4")
    assert flipped["match_id"] == info["match_id"]
    other = parse_legacy_filename("L1234-6.mp4")
    assert other["match_id"] != info["match_id"]


def test_legacy_no_label():
    assert parse_legacy_filename("x_foil_a_1.mp4") is None


def test_load_weapon_records_fallback_scan(tmp_path, caplog):
    (tmp_path / "L_foil_m1_0.mp4").write_bytes(b"")
    (tmp_path / "R_foil_m2_1.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    records = load_weapon_records(tmp_path, "foil", root=tmp_path)
    assert [r.match_id for r in records] == ["m1", "m2"]
    assert records[0].clip_path == "L_foil_m1_0.mp4"
    assert records[0].hit_type == "" and records[0].n_frames is None


def test_load_weapon_records_prefers_manifest(tmp_path):
    _write(tmp_path, [_row()])
    (tmp_path / "R_foil_zzz_1.mp4").write_bytes(b"")
    records = load_weapon_records(tmp_path, "foil")
    assert len(records) == 1 and records[0].match_id == "m1"


def test_merge_rejects_duplicates():
    rec = ClipRecord("a.mp4", "L", "foil")
    with pytest.raises(ManifestError):
        merge_records([[rec], [rec]])
