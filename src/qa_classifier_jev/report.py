from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from qa_classifier_jev.metrics import compute_classification_metrics
from qa_classifier_jev.paths import (
    DEFAULT_FAILURES_PATH,
    DEFAULT_LOW_CONFIDENCE_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MISCLASSIFIED_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_REPORT_PATH,
)
from qa_classifier_jev.schema import load_label_schema


PredictionRow = dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build human-readable reports from Jev outputs.")
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS_PATH),
        help="Prediction JSONL path.",
    )
    parser.add_argument(
        "--failures",
        default=str(DEFAULT_FAILURES_PATH),
        help="Failure JSONL path.",
    )
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Metrics JSON path.")
    parser.add_argument("--schema", default=None, help="Path to label schema JSON.")
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--misclassified-csv",
        default=str(DEFAULT_MISCLASSIFIED_PATH),
        help="CSV path for misclassified examples.",
    )
    parser.add_argument(
        "--low-confidence-csv",
        default=str(DEFAULT_LOW_CONFIDENCE_PATH),
        help="CSV path for low-confidence examples.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.6,
        help="Confidence threshold for low-confidence export.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of examples to show in the report.",
    )
    return parser


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {jsonl_path}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def load_metrics(path: str | Path) -> dict[str, Any] | None:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None
    with metrics_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Metrics file must contain an object: {metrics_path}")
    return raw


def build_metrics_from_predictions(
    predictions: list[PredictionRow],
    label_names: list[str],
) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    for row in predictions:
        if row.get("gold_label") and row.get("predicted_label"):
            y_true.append(str(row["gold_label"]))
            y_pred.append(str(row["predicted_label"]))
    if not y_true:
        return {
            "labels": label_names,
            "total": 0,
            "completed": 0,
            "message": "No successful labeled predictions are available.",
        }
    metrics = compute_classification_metrics(y_true, y_pred, labels=label_names).to_dict()
    metrics["completed"] = len(y_true)
    metrics["requested"] = len(y_true)
    metrics["usage"] = collect_usage(predictions)
    return metrics


def collect_usage(predictions: list[PredictionRow]) -> dict[str, int]:
    usage_total: dict[str, int] = {}
    for row in predictions:
        usage = row.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int):
                usage_total[key] = usage_total.get(key, 0) + value
    return usage_total


def get_misclassified(predictions: list[PredictionRow]) -> list[PredictionRow]:
    return [
        row
        for row in predictions
        if row.get("gold_label")
        and row.get("predicted_label")
        and row.get("gold_label") != row.get("predicted_label")
    ]


def get_low_confidence(predictions: list[PredictionRow], threshold: float) -> list[PredictionRow]:
    rows = [
        row
        for row in predictions
        if isinstance(row.get("confidence"), (int, float)) and float(row["confidence"]) < threshold
    ]
    return sorted(rows, key=lambda row: float(row["confidence"]))


def write_rows_csv(path: str | Path, rows: list[PredictionRow]) -> None:
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "gold_label",
        "predicted_label",
        "confidence",
        "text",
        "probabilities",
        "model",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "index": row.get("index", ""),
                    "gold_label": row.get("gold_label", ""),
                    "predicted_label": row.get("predicted_label", ""),
                    "confidence": row.get("confidence", ""),
                    "text": row.get("text", ""),
                    "probabilities": json.dumps(row.get("probabilities") or {}, ensure_ascii=False),
                    "model": row.get("model", ""),
                }
            )


def format_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.2%}"
    return ""


def format_number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int):
        return str(value)
    return ""


def truncate_text(text: Any, max_length: int = 120) -> str:
    clean = str(text or "").replace("\n", " ").replace("|", "\\|").strip()
    if len(clean) <= max_length:
        return clean
    return clean[: max_length - 3] + "..."


def build_markdown_report(
    metrics: dict[str, Any],
    predictions: list[PredictionRow],
    failures: list[dict[str, Any]],
    misclassified: list[PredictionRow],
    low_confidence: list[PredictionRow],
    confidence_threshold: float,
    top_n: int,
) -> str:
    lines = [
        "# Jev Evaluation Report",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Requested | {metrics.get('requested', metrics.get('total', len(predictions)))} |",
        f"| Completed | {metrics.get('completed', metrics.get('total', len(predictions)))} |",
        f"| Failures | {len(failures)} |",
        f"| Accuracy | {format_percent(metrics.get('accuracy'))} |",
        f"| Macro precision | {format_percent(metrics.get('macro_precision'))} |",
        f"| Macro recall | {format_percent(metrics.get('macro_recall'))} |",
        f"| Macro F1 | {format_percent(metrics.get('macro_f1'))} |",
        "",
    ]

    usage = metrics.get("usage")
    if isinstance(usage, dict) and usage:
        lines.extend(["## Usage", "", "| Field | Value |", "|---|---:|"])
        for key, value in sorted(usage.items()):
            lines.append(f"| {key} | {value} |")
        lines.append("")

    per_label = metrics.get("per_label")
    if isinstance(per_label, dict) and per_label:
        lines.extend(
            [
                "## Per-Label Metrics",
                "",
                "| Label | Support | Precision | Recall | F1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label, values in per_label.items():
            if not isinstance(values, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(label),
                        format_number(values.get("support")),
                        format_percent(values.get("precision")),
                        format_percent(values.get("recall")),
                        format_percent(values.get("f1")),
                    ]
                )
                + " |"
            )
        lines.append("")

    confusion = metrics.get("confusion_matrix")
    labels = metrics.get("labels")
    if isinstance(confusion, dict) and isinstance(labels, list) and labels:
        header = "| Actual \\ Predicted | " + " | ".join(str(label) for label in labels) + " |"
        separator = "|---" + "|---:" * len(labels) + "|"
        lines.extend(["## Confusion Matrix", "", header, separator])
        for actual in labels:
            row = confusion.get(actual, {})
            if not isinstance(row, dict):
                row = {}
            counts = [str(row.get(predicted, 0)) for predicted in labels]
            lines.append(f"| {actual} | " + " | ".join(counts) + " |")
        lines.append("")

    lines.extend(
        [
            "## Misclassified Examples",
            "",
            f"Total: {len(misclassified)}",
            "",
            "| # | Gold | Predicted | Confidence | Text |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in misclassified[:top_n]:
        lines.append(
            f"| {row.get('index', '')} | {row.get('gold_label', '')} | "
            f"{row.get('predicted_label', '')} | {format_number(row.get('confidence'))} | "
            f"{truncate_text(row.get('text'))} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Low-Confidence Examples",
            "",
            f"Threshold: `< {confidence_threshold}`",
            "",
            "| # | Gold | Predicted | Confidence | Text |",
            "|---:|---|---|---:|---|",
        ]
    )
    for row in low_confidence[:top_n]:
        lines.append(
            f"| {row.get('index', '')} | {row.get('gold_label', '')} | "
            f"{row.get('predicted_label', '')} | {format_number(row.get('confidence'))} | "
            f"{truncate_text(row.get('text'))} |"
        )
    lines.append("")

    if failures:
        lines.extend(["## Failures", "", "| # | Error type | Error |", "|---:|---|---|"])
        for row in failures[:top_n]:
            lines.append(
                f"| {row.get('index', '')} | {row.get('error_type', '')} | "
                f"{truncate_text(row.get('error'))} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    schema = load_label_schema(args.schema) if args.schema else load_label_schema()
    predictions = load_jsonl(args.predictions)
    failures = load_jsonl(args.failures)
    if not predictions:
        raise SystemExit(f"No predictions found: {args.predictions}")

    metrics = load_metrics(args.metrics) or build_metrics_from_predictions(
        predictions=predictions,
        label_names=schema.label_names,
    )
    misclassified = get_misclassified(predictions)
    low_confidence = get_low_confidence(predictions, args.confidence_threshold)

    write_rows_csv(args.misclassified_csv, misclassified)
    write_rows_csv(args.low_confidence_csv, low_confidence)

    report = build_markdown_report(
        metrics=metrics,
        predictions=predictions,
        failures=failures,
        misclassified=misclassified,
        low_confidence=low_confidence,
        confidence_threshold=args.confidence_threshold,
        top_n=args.top_n,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print(f"Wrote misclassified examples: {args.misclassified_csv}")
    print(f"Wrote low-confidence examples: {args.low_confidence_csv}")
