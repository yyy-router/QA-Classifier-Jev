from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassificationMetrics:
    labels: list[str]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_label: dict[str, dict[str, float | int]]
    confusion_matrix: dict[str, dict[str, int]]
    total: int

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "total": self.total,
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_label": self.per_label,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> ClassificationMetrics:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("Cannot compute metrics for an empty prediction set")

    label_order = labels or sorted(set(y_true) | set(y_pred))
    confusion = {actual: {predicted: 0 for predicted in label_order} for actual in label_order}

    for actual, predicted in zip(y_true, y_pred):
        if actual not in confusion:
            confusion[actual] = {label: 0 for label in label_order}
        if predicted not in confusion[actual]:
            for row in confusion.values():
                row.setdefault(predicted, 0)
            label_order.append(predicted)
        confusion[actual][predicted] += 1

    correct = sum(1 for actual, predicted in zip(y_true, y_pred) if actual == predicted)
    per_label: dict[str, dict[str, float | int]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in label_order:
        tp = confusion.get(label, {}).get(label, 0)
        fp = sum(confusion.get(other, {}).get(label, 0) for other in label_order if other != label)
        fn = sum(count for pred, count in confusion.get(label, {}).items() if pred != label)
        support = sum(confusion.get(label, {}).values())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    total = len(y_true)
    return ClassificationMetrics(
        labels=label_order,
        total=total,
        accuracy=correct / total,
        macro_precision=sum(precisions) / len(precisions),
        macro_recall=sum(recalls) / len(recalls),
        macro_f1=sum(f1s) / len(f1s),
        per_label=per_label,
        confusion_matrix=confusion,
    )
