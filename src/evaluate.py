from __future__ import annotations

from typing import Iterable, Mapping, Sequence


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
    support = sum(1 for true_label in y_true if true_label == label_id)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": float(support),
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
    computed_metrics: Mapping[str, float],
    metric_names: Sequence[str] | None = None,
) -> dict[str, float]:
    if not metric_names:
        return dict(computed_metrics)
    return {
        metric_name: float(computed_metrics[metric_name])
        for metric_name in metric_names
        if metric_name in computed_metrics
    }


def compute_detailed_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Sequence[int],
    id2label: Mapping[int, str],
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")

    label_list = list(labels)
    summary = compute_classification_metrics(y_true=y_true, y_pred=y_pred, labels=label_list)

    confusion = []
    for true_label in label_list:
        row = []
        for pred_label in label_list:
            row.append(
                sum(
                    1
                    for y_t, y_p in zip(y_true, y_pred)
                    if y_t == true_label and y_p == pred_label
                )
            )
        confusion.append(row)

    per_class = {}
    for label_id in label_list:
        label_name = id2label[label_id]
        scores = precision_recall_f1_for_label(y_true=y_true, y_pred=y_pred, label_id=label_id)
        per_class[label_name] = {
            "label_id": label_id,
            "precision": round(float(scores["precision"]), 6),
            "recall": round(float(scores["recall"]), 6),
            "f1": round(float(scores["f1"]), 6),
            "support": int(scores["support"]),
        }

    classification_report = {
        label_name: {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1-score": metrics["f1"],
            "support": metrics["support"],
        }
        for label_name, metrics in per_class.items()
    }
    classification_report["accuracy"] = round(float(summary["accuracy"]), 6)
    classification_report["macro avg"] = {
        "precision": round(float(summary["macro_precision"]), 6),
        "recall": round(float(summary["macro_recall"]), 6),
        "f1-score": round(float(summary["macro_f1"]), 6),
        "support": int(len(y_true)),
    }

    return {
        "accuracy": round(float(summary["accuracy"]), 6),
        "macro_precision": round(float(summary["macro_precision"]), 6),
        "macro_recall": round(float(summary["macro_recall"]), 6),
        "macro_f1": round(float(summary["macro_f1"]), 6),
        "per_class_metrics": per_class,
        "classification_report": classification_report,
        "confusion_matrix": confusion,
        "labels": [id2label[label_id] for label_id in label_list],
    }
