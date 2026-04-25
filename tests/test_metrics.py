from skeval.metrics import compute_metrics


def test_compute_metrics():
    y_true = ["fact", "emotion", "fact", "opinion"]
    y_pred = ["fact", "fact", "fact", "opinion"]

    results = compute_metrics(y_true, y_pred)

    # 3 correct out of 4
    assert results["accuracy"] == 0.75

    # Check per-class metrics
    assert "fact" in results["per_class"]
    assert "emotion" in results["per_class"]
    assert "opinion" in results["per_class"]

    # opinion was perfectly predicted
    assert results["per_class"]["opinion"]["precision"] == 1.0
    assert results["per_class"]["opinion"]["recall"] == 1.0

    # Confusion matrix structure
    assert "confusion_matrix" in results
    assert len(results["confusion_matrix"]) == 3  # 3 unique classes


def test_compute_metrics_zero_division():
    # Test when a class is in true labels but never predicted
    y_true = ["fact", "emotion"]
    y_pred = ["fact", "fact"]

    results = compute_metrics(y_true, y_pred)
    assert results["accuracy"] == 0.5
    assert results["per_class"]["emotion"]["precision"] == 0.0
    assert results["per_class"]["emotion"]["recall"] == 0.0
