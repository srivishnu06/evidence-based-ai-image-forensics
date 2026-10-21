from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    hamming_loss,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_div(num, den):
    return float(num / den) if den else 0.0


def binary_metrics(y_true, y_prob, threshold: float = 0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    result = {
        "threshold": float(threshold),
        "support": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _safe_div(tn, tn + fp),
        "npv": _safe_div(tn, tn + fn),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(y_true) else 0.0,
        "false_positive_rate": _safe_div(fp, fp + tn),
        "false_negative_rate": _safe_div(fn, fn + tp),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.c_[1.0 - y_prob, y_prob], labels=[0, 1])),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": cm.tolist(),
    }

    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
        result["average_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        result["auroc"] = float("nan")
        result["average_precision"] = float("nan")

    return result


def find_best_binary_threshold(
    y_true,
    y_prob,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        return 0.5, binary_metrics(y_true, y_prob, 0.5)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5, binary_metrics(y_true, y_prob, 0.5)

    denom = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )

    valid = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    if not valid.any():
        threshold = 0.5
    else:
        valid_idx = np.where(valid)[0]
        best_f1 = np.max(f1[valid_idx])
        tied = valid_idx[np.isclose(f1[valid_idx], best_f1)]
        threshold = float(thresholds[tied[np.argmin(np.abs(thresholds[tied] - 0.5))]])

    return threshold, binary_metrics(y_true, y_prob, threshold)


def _threshold_array(threshold, n_classes):
    if np.isscalar(threshold):
        return np.full(n_classes, float(threshold), dtype=float)
    arr = np.asarray(threshold, dtype=float)
    if arr.shape != (n_classes,):
        raise ValueError(f"Expected {n_classes} thresholds, got shape {arr.shape}")
    return arr


def multilabel_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = _threshold_array(threshold, y_true.shape[1])
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    per_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_support = y_true.sum(axis=0).astype(int)

    result = {
        "thresholds": thresholds.tolist(),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "per_class_precision": per_precision.tolist(),
        "per_class_recall": per_recall.tolist(),
        "per_class_f1": per_f1.tolist(),
        "per_class_support": per_support.tolist(),
    }

    for average, key in [("macro", "map_macro"), ("micro", "map_micro"), ("weighted", "map_weighted")]:
        try:
            result[key] = float(average_precision_score(y_true, y_prob, average=average))
        except ValueError:
            result[key] = float("nan")

    return result


def find_best_multilabel_thresholds(y_true, y_prob, min_threshold=0.05, max_threshold=0.95):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    thresholds = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            thresholds.append(0.5)
            continue
        t, _ = find_best_binary_threshold(
            y_true[:, c], y_prob[:, c], min_threshold=min_threshold, max_threshold=max_threshold
        )
        thresholds.append(float(t))
    thresholds = np.asarray(thresholds, dtype=float)
    return thresholds, multilabel_metrics(y_true, y_prob, thresholds)


def localization_metrics(target, probability, threshold: float = 0.5):
    target = np.asarray(target)
    probability = np.asarray(probability)

    pred = probability >= threshold
    gt = target > 0.0

    ious = []
    pixel_f1s = []
    pixel_precisions = []
    pixel_recalls = []
    pointing_hits = []
    per_class = []

    n, c, h, w = gt.shape

    for j in range(c):
        class_ious = []
        class_f1s = []
        class_points = []
        for i in range(n):
            g = gt[i, j]
            if not g.any():
                continue

            p = pred[i, j]
            intersection = np.logical_and(g, p).sum()
            union = np.logical_or(g, p).sum()
            tp = intersection
            fp = np.logical_and(~g, p).sum()
            fn = np.logical_and(g, ~p).sum()

            iou = intersection / union if union else 0.0
            precision = _safe_div(tp, tp + fp)
            recall = _safe_div(tp, tp + fn)
            denom = 2 * tp + fp + fn
            f1 = (2 * tp) / denom if denom else 0.0

            flat_idx = np.argmax(probability[i, j])
            yy, xx = np.unravel_index(flat_idx, (h, w))
            hit = float(g[yy, xx])

            ious.append(iou)
            pixel_precisions.append(precision)
            pixel_recalls.append(recall)
            pixel_f1s.append(f1)
            pointing_hits.append(hit)
            class_ious.append(iou)
            class_f1s.append(f1)
            class_points.append(hit)

        per_class.append({
            "evaluated_instances": len(class_ious),
            "mean_iou": float(np.mean(class_ious)) if class_ious else 0.0,
            "mean_pixel_f1": float(np.mean(class_f1s)) if class_f1s else 0.0,
            "pointing_accuracy": float(np.mean(class_points)) if class_points else 0.0,
        })

    return {
        "threshold": float(threshold),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_pixel_precision": float(np.mean(pixel_precisions)) if pixel_precisions else 0.0,
        "mean_pixel_recall": float(np.mean(pixel_recalls)) if pixel_recalls else 0.0,
        "mean_pixel_f1": float(np.mean(pixel_f1s)) if pixel_f1s else 0.0,
        "pointing_accuracy": float(np.mean(pointing_hits)) if pointing_hits else 0.0,
        "evaluated_artifact_instances": len(ious),
        "per_class": per_class,
    }


def find_best_localization_threshold(target, probability, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)
    scored = []
    for threshold in thresholds:
        metrics = localization_metrics(target, probability, float(threshold))
        scored.append((metrics["mean_pixel_f1"], metrics["mean_iou"], float(threshold), metrics))
    scored.sort(key=lambda x: (x[0], x[1], -abs(x[2] - 0.5)), reverse=True)
    _, _, threshold, metrics = scored[0]
    return threshold, metrics
