from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_classifier_jev.data import load_json_dataset
from qa_classifier_jev.env import load_project_dotenv
from qa_classifier_jev.metrics import compute_classification_metrics
from qa_classifier_jev.paths import (
    DEFAULT_METRICS_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_TEST_DATASET,
)
from qa_classifier_jev.schema import load_label_schema
from qa_classifier_jev.typesafe_client import TypeSafeJevClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Jev on a labeled question dataset.")
    parser.add_argument("--dataset", default=str(DEFAULT_TEST_DATASET), help="Labeled JSON dataset.")
    parser.add_argument("--schema", default=None, help="Path to label schema JSON.")
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS_PATH), help="JSONL output path.")
    parser.add_argument("--metrics", default=str(DEFAULT_METRICS_PATH), help="Metrics JSON output path.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N records.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print the first request payload only.")
    return parser


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
    metrics_path = Path(args.metrics)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    y_true: list[str] = []
    y_pred: list[str] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    with predictions_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records, start=1):
            prediction = client.classify(record.text, schema)
            y_true.append(record.label or "")
            y_pred.append(prediction.label)

            for key in total_usage:
                value = prediction.usage.get(key)
                if isinstance(value, int):
                    total_usage[key] += value

            row = {
                "index": index,
                "text": record.text,
                "gold_label": record.label,
                "predicted_label": prediction.label,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "model": prediction.model,
                "usage": prediction.usage,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{index}/{len(records)}] {record.label} -> {prediction.label}")

    metrics = compute_classification_metrics(y_true, y_pred, labels=schema.label_names).to_dict()
    metrics["usage"] = total_usage
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
