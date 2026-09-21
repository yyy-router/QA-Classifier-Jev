# QA-Classifier-Jev

A lightweight question classification experiment powered by
[TypeSafe AI Jev](https://typesafe.ai/). The project explores whether Jev's
System One `choice` decisions can replace or complement a small local question
classifier for QA/RAG routing.

The initial task is a three-way classifier:

- `Fact`
- `Definition`
- `Reason`

## Features

- Jev HTTP client for `POST /v1/systemone`
- Reusable label schema in JSON
- Single-question and batch classification CLI
- Evaluation CLI for labeled JSON datasets
- Markdown and CSV report generation for completed evaluations
- Metrics without heavyweight ML dependencies
- Conda-first development environment
- Local-only handling for API keys and experiment outputs

## Installation

Create a project-local Conda environment:

```bat
conda env create -f environment.yml --prefix .\.conda
conda activate .\.conda
pip install -e .
```

The environment is stored inside the repository working tree at `.conda/`, which
is ignored by Git. On Windows, `cmd` tends to handle path-based Conda activation
more reliably than PowerShell unless PowerShell has been initialized with
`conda init powershell`.

Create a local environment file:

```bat
copy .env.example .env
```

Set your TypeSafe API key in `.env`:

```text
TYPESAFE_API_KEY=your_api_key_here
TYPESAFE_BASE_URL=https://api.typesafe.ai
TYPESAFE_DEFAULT_MODEL=jev-latest
```

## Quick Start

Preview the Jev request payload without calling the API:

```powershell
qa-jev-classify --question "Why is the sky blue?" --dry-run
```

Classify one question:

```powershell
qa-jev-classify --question "Why is the sky blue?"
```

Evaluate the bundled test split:

```powershell
qa-jev-evaluate --dataset data/eval/test.json --limit 20
```

Evaluation is resumable by default. Existing rows in `outputs/jev_predictions.jsonl`
are reused, and new successful predictions are appended. Transient API failures
are retried before being written to `outputs/jev_failures.jsonl`.

Full evaluation writes:

```text
outputs/jev_predictions.jsonl
outputs/jev_failures.jsonl
outputs/metrics.json
```

Generate a readable report after evaluation:

```powershell
qa-jev-report
```

The report command writes:

```text
outputs/report.md
outputs/misclassified.csv
outputs/low_confidence.csv
```

## Dataset Format

Evaluation data is expected to be a JSON list:

```json
[
  {
    "text": "What is photosynthesis?",
    "label": "Definition"
  }
]
```

The repository includes lightweight evaluation splits under `data/eval/`. Larger
historical training artifacts can be placed under `history/`, which is ignored by
Git so the repository can stay small and free of model files.

## Label Schema

The default label schema lives at:

```text
config/label_schema_v1.json
```

Jev receives the schema as `choice.criteria`, then returns a predicted label,
probabilities, confidence, model name, and usage metadata.

## Project Layout

```text
.
|-- config/
|   `-- label_schema_v1.json
|-- src/
|   `-- qa_classifier_jev/
|       |-- classify.py
|       |-- evaluate.py
|       |-- metrics.py
|       |-- report.py
|       |-- schema.py
|       `-- typesafe_client.py
|-- tests/
|-- environment.yml
|-- pyproject.toml
`-- README.md
```

## Development

Run the test suite:

```powershell
pytest
```

Compile-check the package:

```powershell
python -m compileall src tests
```

Lint with Ruff:

```powershell
ruff check .
```

## Roadmap

- Compare Jev results against the historical SetFit classifier.
- Add confidence-threshold analysis.
- Add optional visualization for confidence and class-level errors.

## References

- TypeSafe quick start: https://docs.typesafe.ai/introduction/quickstart
- TypeSafe HTTP API reference: https://docs.typesafe.ai/api
