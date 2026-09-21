from qa_classifier_jev.metrics import compute_classification_metrics


def test_compute_metrics_for_three_class_problem() -> None:
    metrics = compute_classification_metrics(
        ["Fact", "Fact", "Definition", "Reason"],
        ["Fact", "Reason", "Definition", "Reason"],
        labels=["Fact", "Definition", "Reason"],
    )

    assert metrics.total == 4
    assert metrics.accuracy == 0.75
    assert metrics.per_label["Fact"]["support"] == 2
    assert metrics.confusion_matrix["Fact"]["Reason"] == 1
