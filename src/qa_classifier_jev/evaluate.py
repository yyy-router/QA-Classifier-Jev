from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import requests

from qa_classifier_jev.data import QuestionRecord, load_json_dataset
from qa_classifier_jev.env import load_project_dotenv
from qa_classifier_jev.metrics import compute_classification_metrics
from qa_classifier_jev.paths import (
    DEFAULT_FAILURES_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_TEST_DATASET,
)
from qa_classifier_jev.schema import LabelSchema, load_label_schema
from qa_classifier_jev.typesafe_client import JevPrediction, TypeSafeJevClient


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev on a labeled question dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_TEST_DATASET), help="Labeled JSON dataset.")
    parser.add_argument("--schema", default=None, help="Path to label schema JSON.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH), help="JSONL output path.")
    parser.add_argument("--failures", default=str(DEFAULT_FAILURES_PATH), help="Failure JSONL output path.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Metrics JSON output path.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries for transient API failures.")
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Base retry wait in seconds.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing prediction JSONL.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first request payload only.")
    return parser


def load_existing_predictions(path: str | Path) -> dict[int, dict[str, Any]]:
    predictions_path = Path(path)
    if not predictions_path.exists():
        return {}

    rows: dict[int, dict[str, Any]] = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {predictions_path}") from exc
            index = row.get("index")
            predicted_label = row.get("predicted_label")
            if isinstance(index, int) and isinstance(predicted_label, str) and predicted_label:
                rows[index] = row
    return rows


def prediction_to_row(index: int, record: QuestionRecord, prediction: JevPrediction) -> dict[str, Any]:
    return {
        "index": index,
        "text": record.text,
        "gold_label": record.label,
        "predicted_label": prediction.label,
        "confidence": prediction.confidence,
        "probabilities": prediction.probabilities,
        "model": prediction.model,
        "usage": prediction.usage,
    }


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_retryable_error(error: Exception) -> bool:
    if isinstance(error, (requests.Timeout, requests.ConnectionError)):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code in RETRYABLE_STATUS_CODES
    return False


def classify_with_retries(
    client: TypeSafeJevClient,
    record: QuestionRecord,
    schema: LabelSchema,
    max_retries: int,
    retry_wait: float,
) -> JevPrediction:
    attempts = max(1, max_retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return client.classify(record.text, schema)
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not is_retryable_error(exc):
                break
            time.sleep(retry_wait * attempt)
    assert last_error is not None
    raise last_error


def build_metrics(
    records: list[QuestionRecord],
    predictions: dict[int, dict[str, Any]],
    label_names: list[str],
) -> dict[str, Any]:
    y_true: list[str] = []
    y_pred: list[str] = []
    total_usage: dict[str, int] = {}

    for index, record in enumerate(records, start=1):
        row = predictions.get(index)
        if row is None:
            continue
        predicted_label = row.get("predicted_label")
        if not isinstance(predicted_label, str):
            continue

        y_true.append(record.label or "")
        y_pred.append(predicted_label)

        usage = row.get("usage")
        if isinstance(usage, dict):
            for key, value in usage.items():
                if isinstance(value, int):
                    total_usage[key] = total_usage.get(key, 0) + value

    if not y_true:
        return {
            "labels": label_names,
            "total": 0,
            "completed": 0,
            "message": "No successful predictions are available yet.",
        }

    metrics = compute_classification_metrics(y_true, y_pred, labels=label_names).to_dict()
    metrics["completed"] = len(y_true)
    metrics["requested"] = len(records)
    metrics["usage"] = total_usage
    return metrics


def main() -> None:
    load_project_dotenv()
    args = build_parser().parse_args()
    schema = load_label_schema(args.schema) if args.schema else load_label_schema()
    records = load_json_dataset(args.dataset)
    if args.limit is not None:
        records = records[: args.limit]

    labeled_records = [record for record in records if record.label is not None]
    if len(labeled_records) != len(records):
        raise SystemExit("All evaluation records must include a label")
    if not records:
        raise SystemExit("Dataset is empty")

    client = TypeSafeJevClient(timeout=args.timeout)
    if args.dry_run:
        print(json.dumps(client.build_payload(records[0].text, schema), ensure_ascii=False, indent=2))
        return

    predictions_path = Path(args.predictions)
    failures_path = Path(args.failures)
    metrics_path = Path(args.metrics)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    predictions = {} if args.no_resume else load_existing_predictions(predictions_path)
    if predictions:
        print(f"Loaded {len(predictions)} cached predictions from {predictions_path}")

    failures = 0
    for index, record in enumerate(records, start=1):
        if index in predictions:
            print(f"[{index}/{len(records)}] cached {record.label} -> {predictions[index]['predicted_label']}")
            continue

        try:
            prediction = classify_with_retries(
                client=client,
                record=record,
                schema=schema,
                max_retries=args.max_retries,
                retry_wait=args.retry_wait,
            )
        except Exception as exc:
            failures += 1
            failure_row = {
                "index": index,
                "text": record.text,
                "gold_label": record.label,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            write_jsonl_row(failures_path, failure_row)
            print(f"[{index}/{len(records)}] failed {type(exc).__name__}: {exc}")
            continue

        row = prediction_to_row(index, record, prediction)
        predictions[index] = row
        write_jsonl_row(predictions_path, row)
        print(f"[{index}/{len(records)}] {record.label} -> {prediction.label}")

    metrics = build_metrics(records, predictions, schema.label_names)
    metrics["failures"] = failures
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
