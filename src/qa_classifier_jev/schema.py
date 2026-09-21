from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from qa_classifier_jev.paths import DEFAULT_SCHEMA_PATH


@dataclass(frozen=True)
class LabelSchema:
    name: str
    instructions: str
    labels: dict[str, str]

    @property
    def label_names(self) -> list[str]:
        return list(self.labels.keys())


def load_label_schema(path: str | Path = DEFAULT_SCHEMA_PATH) -> LabelSchema:
    schema_path = Path(path)
    with schema_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    labels = raw.get("labels")
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Schema must contain a non-empty labels object: {schema_path}")

    return LabelSchema(
        name=str(raw.get("name", schema_path.stem)),
        instructions=str(raw["instructions"]),
        labels={str(label): str(criteria) for label, criteria in labels.items()},
    )
