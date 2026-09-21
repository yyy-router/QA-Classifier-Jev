from __future__ import annotations

import argparse
import json
from pathlib import Path

from qa_classifier_jev.data import load_lines
from qa_classifier_jev.env import load_project_dotenv
from qa_classifier_jev.schema import load_label_schema
from qa_classifier_jev.typesafe_client import TypeSafeJevClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify questions with TypeSafe Jev.")
    parser.add_argument("-q", "--question", help="Single question to classify.")
    parser.add_argument("-f", "--file", help="Text file with one question per line.")
    parser.add_argument("--schema", default=None, help="Path to label schema JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Jev request payload only.")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout in seconds.")
    return parser


def main() -> None:
    load_project_dotenv()
    args = build_parser().parse_args()
    schema = load_label_schema(args.schema) if args.schema else load_label_schema()
    client = TypeSafeJevClient(timeout=args.timeout)

    if args.question:
        questions = [args.question]
    elif args.file:
        questions = [record.text for record in load_lines(args.file)]
    else:
        raise SystemExit("Pass --question or --file")

    if args.dry_run:
        print(json.dumps(client.build_payload(questions[0], schema), ensure_ascii=False, indent=2))
        return

    for question in questions:
        prediction = client.classify(question, schema)
        print(
            json.dumps(
                {
                    "text": question,
                    "label": prediction.label,
                    "confidence": prediction.confidence,
                    "probabilities": prediction.probabilities,
                },
                ensure_ascii=False,
            )
        )
