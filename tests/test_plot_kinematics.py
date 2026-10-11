"""Tests for kinematic feature plotting (#32)."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.kinematics import (
    KINEMATIC_FEATURE_DIM,
    KINEMATIC_FEATURE_NAMES,
)
from scripts.plot_kinematics import find_feature_index, plot_kinematics


class TestFindFeatureIndex:
    """Test feature index lookup."""

    def test_find_valid_feature(self):
        """Valid feature names map to correct indices."""
        idx = find_feature_index("left_wrist_vx")
        assert isinstance(idx, int)
        assert 0 <= idx < KINEMATIC_FEATURE_DIM
        assert KINEMATIC_FEATURE_NAMES[idx] == "left_wrist_vx"

    def test_find_all_features(self):
        """All feature names are findable."""
        for name in KINEMATIC_FEATURE_NAMES:
            idx = find_feature_index(name)
            assert KINEMATIC_FEATURE_NAMES[idx] == name

    def test_find_invalid_feature(self):
        """Invalid feature name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kinematic feature"):
            find_feature_index("nonexistent_feature")

    def test_find_case_sensitive(self):
        """Feature lookup is case-sensitive."""
        with pytest.raises(ValueError):
            find_feature_index("LEFT_WRIST_VX")  # uppercase


class TestPlotKinematics:
    """Test kinematic feature plotting."""

    def _make_features(self, n_frames: int = 100) -> np.ndarray:
        """Synthetic kinematic features for testing."""
        features = np.zeros((n_frames, KINEMATIC_FEATURE_DIM), dtype=np.float32)

        # Add some synthetic signals
        t = np.arange(n_frames)
        left_wrist_vx = find_feature_index("left_wrist_vx")
        left_valid = find_feature_index("left_valid")

        # Sine wave velocity
        features[:, left_wrist_vx] = np.sin(2 * np.pi * t / n_frames)
        # Valid for first 80% of frames
        features[:80, left_valid] = 1.0

        return features

    def test_plot_kinematics_basic(self, tmp_path):
        """Plot generates without error."""
        features = self._make_features(100)
        output_path = tmp_path / "plot.png"

        # Should not raise
        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_with_light_frame(self, tmp_path):
        """Plot includes light frame marker."""
        features = self._make_features(100)
        output_path = tmp_path / "plot_with_light.png"

        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=50.0,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_no_output_path(self, tmp_path):
        """Plot can be generated without saving (for stdout/display)."""
        features = self._make_features(100)

        # Should not raise
        with patch('matplotlib.pyplot.show'):
            plot_kinematics(
                "test_clip.mp4",
                features,
                light_frame=None,
                fps=25.0,
                output_path=None,
            )

    def test_plot_kinematics_empty_features(self, tmp_path):
        """Empty feature array is handled gracefully."""
        features = np.zeros((0, KINEMATIC_FEATURE_DIM), dtype=np.float32)
        output_path = tmp_path / "empty.png"

        # Should not crash
        plot_kinematics(
            "empty_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

    def test_plot_kinematics_single_frame(self, tmp_path):
        """Single frame is plotted without error."""
        features = self._make_features(1)
        output_path = tmp_path / "single_frame.png"

        plot_kinematics(
            "single_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_with_nan_values(self, tmp_path):
        """NaN values are handled (typically from invalid frames)."""
        features = self._make_features(100)
        # Inject some NaNs
        features[10:20, find_feature_index("left_wrist_vx")] = np.nan

        output_path = tmp_path / "with_nan.png"
        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_fps_none(self, tmp_path):
        """Plot works without FPS (uses frame indices as X-axis)."""
        features = self._make_features(100)
        output_path = tmp_path / "no_fps.png"

        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=None,
            fps=None,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_light_frame_without_fps(self, tmp_path):
        """Light frame marker works even without FPS."""
        features = self._make_features(100)
        output_path = tmp_path / "light_no_fps.png"

        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=50.0,
            fps=None,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_large_array(self, tmp_path):
        """Large feature arrays are plotted efficiently."""
        features = self._make_features(10000)  # 10k frames
        output_path = tmp_path / "large.png"

        plot_kinematics(
            "large_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()

    def test_plot_kinematics_creates_parent_dirs(self, tmp_path):
        """Plot creates parent directories if needed."""
        features = self._make_features(100)
        output_path = tmp_path / "subdir" / "nested" / "plot.png"

        plot_kinematics(
            "test_clip.mp4",
            features,
            light_frame=None,
            fps=25.0,
            output_path=output_path,
        )

        assert output_path.exists()
        assert output_path.parent.exists()
