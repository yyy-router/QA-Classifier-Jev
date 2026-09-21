from qa_classifier_jev.typesafe_client import parse_choice_response


def test_parse_choice_response() -> None:
    prediction = parse_choice_response(
        {
            "model": "jev-1.13.0",
            "answers": {
                "question_type": {
                    "type": "choice",
                    "choice": "Fact",
                    "confidence": 0.8,
                    "probabilities": {"Fact": 0.9, "Definition": 0.05, "Reason": 0.05},
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 3},
        }
    )

    assert prediction.label == "Fact"
    assert prediction.confidence == 0.8
    assert prediction.probabilities["Fact"] == 0.9
