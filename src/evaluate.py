from __future__ import annotations

from typing import Iterable, Sequence


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def accuracy_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if not y_true:
        return 0.0
    correct = sum(int(true_label == pred_label) for true_label, pred_label in zip(y_true, y_pred))
    return correct / len(y_true)


def precision_recall_f1_for_label(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_id: int,
) -> dict[str, float]:
    tp = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == label_id and pred_label == label_id)
    fp = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label != label_id and pred_label == label_id)
    fn = sum(1 for true_label, pred_label in zip(y_true, y_pred) if true_label == label_id and pred_label != label_id)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Iterable[int] | None = None,
) -> dict[str, float]:
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    label_list = list(labels)

    per_label_scores = [
        precision_recall_f1_for_label(y_true=y_true, y_pred=y_pred, label_id=label_id)
        for label_id in label_list
    ]
    macro_precision = sum(item["precision"] for item in per_label_scores) / len(per_label_scores) if per_label_scores else 0.0
    macro_recall = sum(item["recall"] for item in per_label_scores) / len(per_label_scores) if per_label_scores else 0.0
    macro_f1 = sum(item["f1"] for item in per_label_scores) / len(per_label_scores) if per_label_scores else 0.0

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
    }


def select_metrics(
    computed_metrics: dict[str, float],
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    if not metric_names:
        return dict(computed_metrics)
    return {
        metric_name: computed_metrics[metric_name]
        for metric_name in metric_names
        if metric_name in computed_metrics
    }
