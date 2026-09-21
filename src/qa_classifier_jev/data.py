from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QuestionRecord:
    text: str
    label: str | None = None


def load_json_dataset(path: str | Path) -> list[QuestionRecord]:
    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a JSON list: {dataset_path}")

    records: list[QuestionRecord] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Dataset item #{index} must be an object")
        text = item.get("text") or item.get("question")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Dataset item #{index} is missing non-empty text/question")
        label = item.get("label")
        records.append(QuestionRecord(text=text.strip(), label=str(label) if label else None))
    return records


def load_lines(path: str | Path) -> list[QuestionRecord]:
    text_path = Path(path)
    with text_path.open("r", encoding="utf-8") as f:
        return [QuestionRecord(line.strip()) for line in f if line.strip()]
