import numpy as np

from src.evaluation.metrics import (
    binary_metrics,
    find_best_binary_threshold,
    find_best_multilabel_thresholds,
    localization_metrics,
)


def test_binary_metrics_perfect():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    m = binary_metrics(y, p, 0.5)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["specificity"] == 1.0
    assert m["auroc"] == 1.0


def test_threshold_tuning_returns_valid_threshold():
    y = [0, 0, 1, 1]
    p = [0.05, 0.4, 0.45, 0.9]
    t, m = find_best_binary_threshold(y, p)
    assert 0.05 <= t <= 0.95
    assert m["f1"] >= binary_metrics(y, p, 0.5)["f1"]


def test_multilabel_thresholds_shape():
    y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    p = np.array([[0.8, 0.2], [0.7, 0.8], [0.2, 0.7], [0.1, 0.2]])
    thresholds, metrics = find_best_multilabel_thresholds(y, p)
    assert thresholds.shape == (2,)
    assert metrics["macro_f1"] == 1.0


def test_localization_metrics_perfect():
    target = np.zeros((1, 1, 2, 2), dtype=float)
    target[0, 0, 0, 0] = 1
    prob = target.copy()
    m = localization_metrics(target, prob, 0.5)
    assert m["mean_iou"] == 1.0
    assert m["mean_pixel_f1"] == 1.0
    assert m["pointing_accuracy"] == 1.0
