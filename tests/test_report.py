import csv
import json

from qa_classifier_jev.report import (
    build_markdown_report,
    build_metrics_from_predictions,
    get_low_confidence,
    get_misclassified,
    load_jsonl,
    write_rows_csv,
)


def test_report_helpers_build_metrics_and_slices() -> None:
    predictions = [
        {
            "index": 1,
            "text": "What is DNA?",
            "gold_label": "Definition",
            "predicted_label": "Definition",
            "confidence": 0.9,
            "usage": {"input_tokens": 12},
        },
        {
            "index": 2,
            "text": "Who was born first?",
            "gold_label": "Reason",
            "predicted_label": "Fact",
            "confidence": 0.42,
            "usage": {"input_tokens": 14},
        },
    ]

    metrics = build_metrics_from_predictions(predictions, ["Fact", "Definition", "Reason"])

    assert metrics["completed"] == 2
    assert metrics["usage"]["input_tokens"] == 26
    assert len(get_misclassified(predictions)) == 1
    assert get_low_confidence(predictions, 0.6)[0]["index"] == 2


def test_load_jsonl_and_write_rows_csv(tmp_path) -> None:
    jsonl_path = tmp_path / "predictions.jsonl"
    jsonl_path.write_text(
        json.dumps({"index": 1, "predicted_label": "Fact"}) + "\n",
        encoding="utf-8",
    )

    rows = load_jsonl(jsonl_path)
    csv_path = tmp_path / "rows.csv"
    write_rows_csv(csv_path, rows)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))

    assert rows[0]["predicted_label"] == "Fact"
    assert csv_rows[0]["predicted_label"] == "Fact"


def test_build_markdown_report_contains_sections() -> None:
    report = build_markdown_report(
        metrics={
            "labels": ["Fact"],
            "requested": 1,
            "completed": 1,
            "accuracy": 1.0,
            "macro_precision": 1.0,
            "macro_recall": 1.0,
            "macro_f1": 1.0,
            "per_label": {"Fact": {"support": 1, "precision": 1.0, "recall": 1.0, "f1": 1.0}},
            "confusion_matrix": {"Fact": {"Fact": 1}},
            "usage": {"input_tokens": 10},
        },
        predictions=[],
        failures=[],
        misclassified=[],
        low_confidence=[],
        confidence_threshold=0.6,
        top_n=10,
    )

    assert "# Jev Evaluation Report" in report
    assert "## Confusion Matrix" in report
