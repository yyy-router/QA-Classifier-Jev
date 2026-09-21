from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests

from qa_classifier_jev.schema import LabelSchema


QUESTION_ID = "question_type"


@dataclass(frozen=True)
class JevPrediction:
    label: str
    confidence: float | None
    probabilities: dict[str, float]
    model: str | None
    usage: dict[str, Any]
    raw_response: dict[str, Any]


class TypeSafeJevClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.base_url = (base_url or os.getenv("TYPESAFE_BASE_URL") or "https://api.typesafe.ai").rstrip(
            "/"
        )
        self.model = model or os.getenv("TYPESAFE_DEFAULT_MODEL") or "jev-latest"
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/systemone"

    def build_payload(self, question_text: str, schema: LabelSchema) -> dict[str, Any]:
        return {
            "state": question_text,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": schema.instructions,
                    "criteria": schema.labels,
                }
            },
        }

    def classify(self, question_text: str, schema: LabelSchema) -> JevPrediction:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for live Jev classification")

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=self.build_payload(question_text, schema),
            timeout=self.timeout,
        )
        response.raise_for_status()
        return parse_choice_response(response.json())


def parse_choice_response(raw: dict[str, Any], question_id: str = QUESTION_ID) -> JevPrediction:
    try:
        answer = raw["answers"][question_id]
    except KeyError as exc:
        raise ValueError(f"Response is missing answers.{question_id}") from exc

    if answer.get("type") != "choice":
        raise ValueError(f"Expected a choice answer, got: {answer.get('type')!r}")

    label = answer.get("choice")
    if not isinstance(label, str) or not label:
        raise ValueError("Choice response is missing a non-empty choice")

    probabilities = answer.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        raise ValueError("Choice response probabilities must be an object")

    confidence = answer.get("confidence")
    return JevPrediction(
        label=label,
        confidence=float(confidence) if confidence is not None else None,
        probabilities={str(key): float(value) for key, value in probabilities.items()},
        model=raw.get("model"),
        usage=dict(raw.get("usage") or {}),
        raw_response=raw,
    )
