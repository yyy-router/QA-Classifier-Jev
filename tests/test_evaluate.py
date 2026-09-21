import json

from qa_classifier_jev.data import QuestionRecord
from qa_classifier_jev.evaluate import build_metrics, is_retryable_error, load_existing_predictions


def test_load_existing_predictions_reads_successful_rows(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        "\n".join(
            [
                json.dumps({"index": 1, "predicted_label": "Fact"}),
                json.dumps({"index": 2, "predicted_label": ""}),
                json.dumps({"index": "3", "predicted_label": "Reason"}),
            ]
        ),
        encoding="utf-8",
    )

    predictions = load_existing_predictions(predictions_path)

    assert list(predictions) == [1]
    assert predictions[1]["predicted_label"] == "Fact"


def test_build_metrics_uses_completed_predictions_only() -> None:
    records = [
        QuestionRecord(text="q1", label="Fact"),
        QuestionRecord(text="q2", label="Reason"),
    ]
    predictions = {1: {"predicted_label": "Fact", "usage": {"input_tokens": 12}}}

    metrics = build_metrics(records, predictions, ["Fact", "Definition", "Reason"])

    assert metrics["completed"] == 1
    assert metrics["requested"] == 2
    assert metrics["usage"]["input_tokens"] == 12
    assert metrics["accuracy"] == 1.0


def test_timeout_is_retryable() -> None:
    import requests

    assert is_retryable_error(requests.Timeout("slow"))
