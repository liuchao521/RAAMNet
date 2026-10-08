"""Binary HSI-CD metrics: OA, Kappa, F1, Precision, and Recall."""

from __future__ import annotations

import numpy as np


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.int64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same flattened shape")
    if not set(np.unique(target)).issubset({0, 1}) or not set(np.unique(prediction)).issubset({0, 1}):
        raise ValueError("binary_metrics expects labels in {0, 1}")
    true_positive = int(np.logical_and(prediction == 1, target == 1).sum())
    true_negative = int(np.logical_and(prediction == 0, target == 0).sum())
    false_positive = int(np.logical_and(prediction == 1, target == 0).sum())
    false_negative = int(np.logical_and(prediction == 0, target == 1).sum())
    total = max(true_positive + true_negative + false_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    oa = (true_positive + true_negative) / total
    predicted_positive = (true_positive + false_positive) / total
    actual_positive = (true_positive + false_negative) / total
    expected = predicted_positive * actual_positive + (1.0 - predicted_positive) * (1.0 - actual_positive)
    kappa = (oa - expected) / max(1.0 - expected, 1e-12)
    return {"oa": oa, "kappa": kappa, "f1": f1, "precision": precision, "recall": recall}
