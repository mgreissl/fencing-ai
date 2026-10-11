"""Tests for the version-independent digit classifier in src/overlay.py."""

import warnings

import numpy as np
import pytest

from src import overlay


def make_classifier(n_classes=4, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    return overlay.DigitClassifier(
        coef=rng.normal(size=(n_classes, n_features)),
        intercept=rng.normal(size=n_classes),
        classes=np.arange(n_classes),
    )


def test_predict_proba_rows_sum_to_one():
    clf = make_classifier()
    x = np.random.default_rng(1).normal(size=(5, 6))
    proba = clf.predict_proba(x)
    assert proba.shape == (5, 4)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_predict_is_argmax_of_proba():
    clf = make_classifier()
    x = np.random.default_rng(2).normal(size=(8, 6))
    assert (clf.predict(x) == clf.predict_proba(x).argmax(axis=1)).all()


def test_softmax_is_stable_for_large_logits():
    clf = overlay.DigitClassifier(
        coef=np.array([[1000.0], [-1000.0]]),
        intercept=np.zeros(2),
        classes=np.arange(2),
    )
    proba = clf.predict_proba(np.array([[255.0]]))
    assert np.isfinite(proba).all()
    assert proba[0, 0] == pytest.approx(1.0)


def test_npz_roundtrip(tmp_path):
    clf = make_classifier()
    path = tmp_path / "clf.npz"
    clf.save_npz(path)
    loaded = overlay.DigitClassifier.load_npz(path)
    x = np.random.default_rng(3).normal(size=(4, 6))
    assert np.allclose(loaded.predict_proba(x), clf.predict_proba(x))
    assert (loaded.classes == clf.classes).all()


def test_shipped_npz_matches_shipped_pickle():
    cfg = overlay.load_overlay("fie_default")
    npz_clf = overlay.DigitClassifier.load_npz(cfg.classifier_npz)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pkl_clf = overlay.load_pickled_classifier(cfg.classifier_pickle)
    x = np.random.default_rng(4).integers(0, 256, size=(20, 960)).astype(np.uint8)
    assert np.allclose(npz_clf.predict_proba(x), pkl_clf.predict_proba(x))


def test_shipped_npz_matches_sklearn_predict_proba():
    sklearn_linear = pytest.importorskip("sklearn.linear_model")
    assert sklearn_linear is not None
    import pickle

    cfg = overlay.load_overlay("fie_default")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(cfg.classifier_pickle, "rb") as fh:
            model = pickle.load(fh)
        model.classes_ = np.arange(16)
        x = np.random.default_rng(5).integers(0, 256, size=(10, 960)).astype(np.uint8)
        expected = model.predict_proba(x)
    npz_clf = overlay.DigitClassifier.load_npz(cfg.classifier_npz)
    assert np.allclose(npz_clf.predict_proba(x), expected, atol=1e-9)


def test_load_falls_back_to_pickle_when_npz_missing(tmp_path):
    cfg = overlay.load_overlay("fie_default")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = overlay.load_digit_classifier(
            npz_path=tmp_path / "missing.npz", pickle_path=cfg.classifier_pickle
        )
    assert clf.coef.shape == (16, 960)


def test_load_prefers_npz(tmp_path):
    clf = make_classifier()
    path = tmp_path / "clf.npz"
    clf.save_npz(path)
    loaded = overlay.load_digit_classifier(
        npz_path=path, pickle_path=tmp_path / "does_not_exist.pkl"
    )
    assert loaded.coef.shape == clf.coef.shape


def test_load_without_any_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        overlay.load_digit_classifier(
            npz_path=tmp_path / "a.npz", pickle_path=tmp_path / "b.pkl"
        )


def test_export_npz_from_pickle(tmp_path):
    cfg = overlay.load_overlay("fie_default")
    out = tmp_path / "exported.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        overlay.export_classifier_npz(cfg.classifier_pickle, out)
    assert overlay.DigitClassifier.load_npz(out).coef.shape == (16, 960)


class FixedClassifier:
    """Returns a fixed probability row per call, for read_score tests."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.calls = 0

    def predict_proba(self, x):
        out = np.array(self.rows[self.calls : self.calls + len(x)])
        self.calls += len(x)
        return out


def one_hot(digit, confidence):
    row = np.full(16, (1 - confidence) / 15)
    row[digit] = confidence
    return row


def test_read_score_confident():
    cfg = overlay.load_overlay("fie_default")
    clf = FixedClassifier([one_hot(7, 0.99), one_hot(12, 0.95)])
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    read = overlay.read_score(frame, cfg, clf)
    assert (read.left, read.right) == (7, 12)
    assert read.confidence == pytest.approx(0.95)


def test_read_score_low_confidence_is_rejected():
    cfg = overlay.load_overlay("fie_default")
    clf = FixedClassifier([one_hot(7, 0.99), one_hot(12, 0.5)])
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    assert overlay.read_score(frame, cfg, clf) is None


def test_read_score_threshold_override():
    cfg = overlay.load_overlay("fie_default")
    clf = FixedClassifier([one_hot(7, 0.99), one_hot(12, 0.5)])
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    read = overlay.read_score(frame, cfg, clf, min_confidence=0.4)
    assert (read.left, read.right) == (7, 12)
