"""Clip manifest handling.

A manifest is ``data/clips/{weapon}/manifest.csv`` with one row per clip.
When no manifest exists, clips are discovered by scanning ``*.mp4`` and
parsing the filename (legacy fallback).
"""

from __future__ import annotations

import csv
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

LABEL_MAP = {"L": 0, "T": 1, "R": 2}
WEAPON_MAP = {"foil": 0, "sabre": 1, "epee": 2}
HIT_TYPES = ("On-On", "On-No", "No-On", "On-Off", "Off-On")
VALID_SPLITS = ("train", "val", "test")

MANIFEST_COLUMNS = (
    "clip_path",
    "label",
    "weapon",
    "match_id",
    "source_url",
    "tournament",
    "hit_type",
    "score_left_before",
    "score_right_before",
    "score_left_after",
    "score_right_after",
    "light_time_s",
    "source_fps",
    "clip_fps",
    "n_frames",
    "light_frame_in_clip",
    "split",
)
REQUIRED_COLUMNS = ("clip_path", "label", "weapon", "match_id")

_INT_FIELDS = (
    "score_left_before",
    "score_right_before",
    "score_left_after",
    "score_right_after",
    "n_frames",
)
_FLOAT_FIELDS = ("light_time_s", "source_fps", "clip_fps", "light_frame_in_clip")

_LEGACY_NEW = re.compile(
    r"^(?P<label>[LTR])_(?P<weapon>foil|sabre|epee)_(?P<match>.+)_(?P<n>\d+)"
    r"(?:-flipped)?$"
)
_LEGACY_OLD = re.compile(r"^(?P<label>[LTR])(?P<rest>.*?)(?:-flipped)?$")


class ManifestError(ValueError):
    """Raised when a manifest is malformed."""


@dataclass(frozen=True)
class ClipRecord:
    """One labelled clip plus its metadata. Missing values are None / ''."""

    clip_path: str
    label: str
    weapon: str
    match_id: str = ""
    source_url: str = ""
    tournament: str = ""
    hit_type: str = ""
    score_left_before: Optional[int] = None
    score_right_before: Optional[int] = None
    score_left_after: Optional[int] = None
    score_right_after: Optional[int] = None
    light_time_s: Optional[float] = None
    source_fps: Optional[float] = None
    clip_fps: Optional[float] = None
    n_frames: Optional[int] = None
    light_frame_in_clip: Optional[float] = None
    split: str = ""

    @property
    def label_id(self) -> int:
        return LABEL_MAP[self.label]

    @property
    def weapon_id(self) -> int:
        return WEAPON_MAP.get(self.weapon, 0)

    def abs_path(self, root: Path = REPO_ROOT) -> str:
        """Absolute filesystem path of the clip."""
        path = Path(self.clip_path)
        if not path.is_absolute():
            path = Path(root) / path
        return str(path)


def _to_posix_rel(path: Path, root: Path = REPO_ROOT) -> str:
    """Path relative to ``root`` (posix) if inside it, else absolute posix."""
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_number(value: Optional[str], cast, column: str, row_no: int):
    if value is None:
        return None
    value = value.strip()
    if value == "" or value.lower() in ("nan", "none", "null"):
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ManifestError(
            f"row {row_no}: column '{column}' is not numeric: {value!r}"
        ) from exc
    if math.isnan(number):
        return None
    return cast(number)


def parse_row(row: dict, row_no: int = 0) -> ClipRecord:
    """Validate and convert one manifest row (dict of strings)."""
    label = (row.get("label") or "").strip().upper()
    if label not in LABEL_MAP:
        raise ManifestError(f"row {row_no}: invalid label {label!r} (want L/T/R)")
    clip_path = (row.get("clip_path") or "").strip().replace("\\", "/")
    if not clip_path:
        raise ManifestError(f"row {row_no}: empty clip_path")
    weapon = (row.get("weapon") or "").strip().lower()
    if weapon not in WEAPON_MAP:
        raise ManifestError(f"row {row_no}: invalid weapon {weapon!r}")
    hit_type = (row.get("hit_type") or "").strip()
    if hit_type and hit_type not in HIT_TYPES:
        raise ManifestError(f"row {row_no}: invalid hit_type {hit_type!r}")
    split = (row.get("split") or "").strip().lower()
    if split and split not in VALID_SPLITS:
        raise ManifestError(f"row {row_no}: invalid split {split!r}")

    values: dict = {
        "clip_path": clip_path,
        "label": label,
        "weapon": weapon,
        "match_id": (row.get("match_id") or "").strip(),
        "source_url": (row.get("source_url") or "").strip(),
        "tournament": (row.get("tournament") or "").strip(),
        "hit_type": hit_type,
        "split": split,
    }
    for name in _INT_FIELDS:
        values[name] = _parse_number(row.get(name), int, name, row_no)
    for name in _FLOAT_FIELDS:
        values[name] = _parse_number(row.get(name), float, name, row_no)
    return ClipRecord(**values)


def load_manifest(path: Path | str) -> list[ClipRecord]:
    """Load and validate a manifest CSV."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ManifestError(f"{path}: missing required columns {missing}")
        records = [parse_row(row, i) for i, row in enumerate(reader, start=2)]
    seen: set[str] = set()
    for rec in records:
        if rec.clip_path in seen:
            raise ManifestError(f"{path}: duplicate clip_path {rec.clip_path}")
        seen.add(rec.clip_path)
    return records


def _format(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return repr(value)
    return str(value)


def write_manifest(records: Iterable[ClipRecord], path: Path | str) -> None:
    """Write records in the manifest column order (including ``split``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for rec in records:
            row = asdict(rec)
            writer.writerow({k: _format(row[k]) for k in MANIFEST_COLUMNS})


def parse_legacy_filename(
    filename: str, default_weapon: str = "foil"
) -> Optional[dict]:
    """Parse a legacy clip filename.

    ``L_foil_<match>_<n>.mp4`` gives match_id ``<match>`` (everything between
    ``{weapon}_`` and the last ``_{n}``). Older ``L1234-5.mp4`` names carry no
    match information: ``match_id`` is empty and ``grouped`` is False.

    Returns None when the name has no L/T/R label prefix.
    """
    stem = Path(filename).name
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    new = _LEGACY_NEW.match(stem)
    if new:
        return {
            "label": new["label"],
            "weapon": new["weapon"],
            "match_id": new["match"],
            "grouped": True,
        }
    old = _LEGACY_OLD.match(stem)
    if old:
        return {
            "label": old["label"],
            "weapon": default_weapon,
            # No match info: use the base clip id so flipped copies still
            # stay together, but every original clip is its own group.
            "match_id": f"clip:{old['rest']}",
            "grouped": False,
        }
    return None


def scan_legacy_dir(
    directory: Path | str, weapon: str, root: Path = REPO_ROOT
) -> list[ClipRecord]:
    """Fallback discovery: scan ``*.mp4`` files and parse their names."""
    directory = Path(directory)
    records: list[ClipRecord] = []
    ungrouped = 0
    for file in sorted(directory.iterdir()):
        if file.suffix.lower() != ".mp4":
            continue
        info = parse_legacy_filename(file.name, default_weapon=weapon)
        if info is None:
            continue
        if not info["grouped"]:
            ungrouped += 1
        records.append(
            ClipRecord(
                clip_path=_to_posix_rel(file, root),
                label=info["label"],
                weapon=info["weapon"] if info["grouped"] else weapon,
                match_id=info["match_id"],
            )
        )
    if ungrouped:
        log.warning(
            "%d clips in %s use the old naming scheme without match ids; "
            "grouping by match is impossible (each clip is its own group). "
            "Generate a manifest to get leakage-free splits.",
            ungrouped,
            directory,
        )
    return records


def load_weapon_records(
    directory: Path | str, weapon: str, root: Path = REPO_ROOT
) -> list[ClipRecord]:
    """Load ``manifest.csv`` from ``directory`` if present, else scan it."""
    directory = Path(directory)
    manifest = directory / "manifest.csv"
    if manifest.is_file():
        records = load_manifest(manifest)
        log.info("Loaded manifest %s (%d clips)", manifest, len(records))
        return records
    log.warning("No manifest in %s; falling back to filename parsing", directory)
    return scan_legacy_dir(directory, weapon, root)


def merge_records(groups: Sequence[Sequence[ClipRecord]]) -> list[ClipRecord]:
    """Concatenate per-weapon record lists, rejecting duplicate clip paths."""
    merged: list[ClipRecord] = []
    seen: set[str] = set()
    for group in groups:
        for rec in group:
            if rec.clip_path in seen:
                raise ManifestError(f"duplicate clip_path across weapons: {rec.clip_path}")
            seen.add(rec.clip_path)
            merged.append(rec)
    return merged
