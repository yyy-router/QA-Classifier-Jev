from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "config" / "label_schema_v1.json"
DEFAULT_TEST_DATASET = PROJECT_ROOT / "data" / "eval" / "test.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_PREDICTIONS_PATH = DEFAULT_OUTPUT_DIR / "jev_predictions.jsonl"
DEFAULT_FAILURES_PATH = DEFAULT_OUTPUT_DIR / "jev_failures.jsonl"
DEFAULT_METRICS_PATH = DEFAULT_OUTPUT_DIR / "metrics.json"
