"""Tests for audit sampling and reporting (#11)."""

import csv
import json
import pytest
from pathlib import Path

from src.manifest import ClipRecord, write_manifest
from scripts.audit_sample import stratified_sample, write_audit_sheet, write_audit_index_html
from scripts.audit_report import load_audit_sheet, compute_noise_rates, wilson_ci


class TestStratifiedSample:
    """Test stratified random sampling of clips."""

    def test_stratified_sample_basic(self):
        """Sample respects stratification and random seed."""
        records = [
            ClipRecord(f"clip_{i}.mp4", label=['L', 'T', 'R'][i % 3], weapon=['foil', 'sabre'][i % 2])
            for i in range(30)
        ]
        sample1 = stratified_sample(records, n=10, seed=42)
        sample2 = stratified_sample(records, n=10, seed=42)
        assert sample1 == sample2, "Same seed should produce identical samples"
        assert len(sample1) == 10

    def test_stratified_sample_exact_n(self):
        """Sample size never exceeds requested n."""
        records = [
            ClipRecord(f"clip_{i}.mp4", label='L', weapon='foil')
            for i in range(5)
        ]
        sample = stratified_sample(records, n=100, seed=42)
        assert len(sample) == 5, "Sample size should not exceed total records"

    def test_stratified_sample_respects_strata(self):
        """Larger strata get more samples."""
        records = (
            [ClipRecord(f"L_{i}.mp4", label='L', weapon='foil') for i in range(20)] +
            [ClipRecord(f"T_{i}.mp4", label='T', weapon='foil') for i in range(5)]
        )
        sample = stratified_sample(records, n=10, seed=42)
        l_count = sum(1 for r in sample if r.label == 'L')
        t_count = sum(1 for r in sample if r.label == 'T')
        assert l_count > t_count, "Larger stratum (L) should have more samples"

    def test_stratified_sample_by_hittype(self):
        """Stratification can include hit_type."""
        records = [
            ClipRecord(f"clip_{i}.mp4", label='L', weapon='foil', hit_type='On-On')
            for i in range(10)
        ] + [
            ClipRecord(f"clip_{i}.mp4", label='L', weapon='foil', hit_type='On-No')
            for i in range(20)
        ]
        sample = stratified_sample(records, n=10, seed=42, stratify_by_hittype=True)
        assert len(sample) == 10
        on_on = sum(1 for r in sample if r.hit_type == 'On-On')
        on_no = sum(1 for r in sample if r.hit_type == 'On-No')
        assert on_on > 0 and on_no > 0, "Both strata should be represented"


class TestAuditSheet:
    """Test audit sheet generation."""

    def test_write_audit_sheet(self, tmp_path):
        """Audit sheet CSV has correct columns and data."""
        records = [
            ClipRecord("clip_1.mp4", label='L', weapon='foil', hit_type='On-On'),
            ClipRecord("clip_2.mp4", label='T', weapon='sabre', hit_type=''),
        ]
        path = tmp_path / "audit.csv"
        write_audit_sheet(records, path)

        assert path.exists()
        with path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]['clip_path'] == 'clip_1.mp4'
        assert rows[0]['auto_label'] == 'L'
        assert rows[0]['human_label'] == ''
        assert rows[0]['notes'] == ''
        assert rows[1]['hit_type'] == ''

    def test_write_audit_index_html(self, tmp_path):
        """Audit index HTML has video tags and metadata."""
        records = [
            ClipRecord("clip_1.mp4", label='L', weapon='foil', match_id='m1', hit_type='On-On'),
        ]
        # Create a dummy video file so abs_path works
        (tmp_path / "clip_1.mp4").write_bytes(b"")

        path = tmp_path / "index.html"
        write_audit_index_html(records, path, repo_root=tmp_path)

        assert path.exists()
        html = path.read_text()
        assert "<video controls" in html
        assert "clip_1.mp4" in html
        assert "foil" in html
        assert "On-On" in html
        assert "m1" in html


class TestWilsonCI:
    """Test Wilson score confidence intervals."""

    def test_wilson_ci_zero_errors(self):
        """0/N should give [0, upper]."""
        lower, upper = wilson_ci(0, 100)
        assert lower == 0.0
        assert 0.0 < upper < 0.05

    def test_wilson_ci_all_errors(self):
        """N/N should give [lower, 1]."""
        lower, upper = wilson_ci(100, 100)
        assert upper >= 0.999
        assert 0.95 < lower < 1.0

    def test_wilson_ci_half(self):
        """50/100 should give [lower, upper] around 0.5."""
        lower, upper = wilson_ci(50, 100)
        assert lower < 0.5 < upper
        assert 0.4 < lower < 0.5
        assert 0.5 < upper < 0.6

    def test_wilson_ci_symmetry(self):
        """CI should be tighter for larger n."""
        lower_small, upper_small = wilson_ci(1, 10)
        lower_big, upper_big = wilson_ci(10, 100)
        assert upper_small - lower_small > upper_big - lower_big


class TestLoadAuditSheet:
    """Test loading filled audit sheets."""

    def test_load_audit_sheet_skips_empty_human(self, tmp_path):
        """Rows with empty human_label are skipped."""
        path = tmp_path / "audit.csv"
        path.write_text(
            "clip_path,weapon,hit_type,auto_label,human_label,notes\n"
            "clip_1.mp4,foil,On-On,L,L,\n"
            "clip_2.mp4,foil,On-No,T,,\n"  # empty human_label
            "clip_3.mp4,sabre,On-On,R,R,\n",
            encoding="utf-8"
        )
        rows = load_audit_sheet(path)
        assert len(rows) == 2
        assert rows[0]['clip_path'] == 'clip_1.mp4'
        assert rows[1]['clip_path'] == 'clip_3.mp4'

    def test_load_audit_sheet_unidentifiable(self, tmp_path):
        """Rows with human_label='X' are loaded but tracked separately."""
        path = tmp_path / "audit.csv"
        path.write_text(
            "clip_path,weapon,hit_type,auto_label,human_label,notes\n"
            "clip_1.mp4,foil,On-On,L,L,\n"
            "clip_2.mp4,foil,On-No,T,X,unclear\n",
            encoding="utf-8"
        )
        rows = load_audit_sheet(path)
        assert len(rows) == 2  # X is still loaded
        assert rows[1]['human_label'] == 'X'


class TestComputeNoiseRates:
    """Test noise rate computation and CI calculations."""

    def test_compute_noise_rates_perfect_labels(self, tmp_path):
        """Perfect labeling should have 0% error rate."""
        rows = [
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},
            {'auto_label': 'T', 'human_label': 'T', 'weapon': 'foil'},
            {'auto_label': 'R', 'human_label': 'R', 'weapon': 'sabre'},
        ]
        results = compute_noise_rates(rows)

        assert results['total']['n_clips'] == 3
        assert results['total']['n_errors'] == 0
        assert results['total']['error_rate'] == 0.0
        assert results['total']['ci_lower'] == 0.0

    def test_compute_noise_rates_all_wrong(self):
        """All mismatches should give 100% error rate."""
        rows = [
            {'auto_label': 'L', 'human_label': 'T', 'weapon': 'foil'},
            {'auto_label': 'T', 'human_label': 'R', 'weapon': 'foil'},
            {'auto_label': 'R', 'human_label': 'L', 'weapon': 'sabre'},
        ]
        results = compute_noise_rates(rows)

        assert results['total']['n_clips'] == 3
        assert results['total']['n_errors'] == 3
        assert results['total']['error_rate'] == 1.0
        assert results['total']['ci_upper'] == 1.0

    def test_compute_noise_rates_per_class(self):
        """Per-class error rates are computed correctly (by human_label)."""
        rows = [
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},  # correct
            {'auto_label': 'T', 'human_label': 'L', 'weapon': 'foil'},  # error
            {'auto_label': 'T', 'human_label': 'T', 'weapon': 'foil'},
        ]
        results = compute_noise_rates(rows)

        assert results['by_class']['L']['n_clips'] == 3
        assert results['by_class']['L']['n_errors'] == 1
        assert results['by_class']['L']['error_rate'] == pytest.approx(1/3)

    def test_compute_noise_rates_per_weapon(self):
        """Per-weapon error rates are computed correctly."""
        rows = [
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},
            {'auto_label': 'T', 'human_label': 'R', 'weapon': 'foil'},  # error
            {'auto_label': 'R', 'human_label': 'R', 'weapon': 'sabre'},
        ]
        results = compute_noise_rates(rows)

        assert results['by_weapon']['foil']['n_clips'] == 2
        assert results['by_weapon']['foil']['n_errors'] == 1
        assert results['by_weapon']['sabre']['n_clips'] == 1
        assert results['by_weapon']['sabre']['n_errors'] == 0

    def test_compute_noise_rates_confusion_matrix(self):
        """Confusion matrix is built correctly."""
        rows = [
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},
            {'auto_label': 'L', 'human_label': 'T', 'weapon': 'foil'},
            {'auto_label': 'T', 'human_label': 'L', 'weapon': 'foil'},
        ]
        results = compute_noise_rates(rows)

        cm = results['confusion_matrix']
        assert cm['L']['L'] == 1
        assert cm['T']['L'] == 1
        assert cm['L']['T'] == 1

    def test_compute_noise_rates_unidentifiable(self):
        """Human label 'X' is tracked separately."""
        rows = [
            {'auto_label': 'L', 'human_label': 'L', 'weapon': 'foil'},
            {'auto_label': 'T', 'human_label': 'X', 'weapon': 'foil'},
            {'auto_label': 'R', 'human_label': 'X', 'weapon': 'sabre'},
        ]
        results = compute_noise_rates(rows)

        assert results['total']['n_clips'] == 1  # Only 'L' counted
        assert results['unidentifiable'] == 2
        assert 'X' not in results['by_class']


class TestAuditIntegration:
    """Integration tests for audit sample + report workflow."""

    def test_full_workflow(self, tmp_path):
        """End-to-end: sample -> fill -> report."""
        # Create a fake manifest
        records = [
            ClipRecord(f"L_foil_{i}.mp4", label='L', weapon='foil', match_id=f'm{i}')
            for i in range(10)
        ] + [
            ClipRecord(f"T_sabre_{i}.mp4", label='T', weapon='sabre', match_id=f'm{i}')
            for i in range(10)
        ]

        # Sample
        sample = stratified_sample(records, n=10, seed=42)
        assert len(sample) == 10

        # Write audit sheet
        sheet_path = tmp_path / "audit.csv"
        write_audit_sheet(sample, sheet_path)

        # Simulate human filling it
        with sheet_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        with sheet_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for i, row in enumerate(rows):
                # Some correct, some wrong
                row['human_label'] = row['auto_label'] if i % 2 == 0 else 'X'
                writer.writerow(row)

        # Compute report
        loaded = load_audit_sheet(sheet_path)
        results = compute_noise_rates(loaded)

        assert results['total']['n_clips'] >= 0
        assert 'unidentifiable' in results
        assert 'confusion_matrix' in results
