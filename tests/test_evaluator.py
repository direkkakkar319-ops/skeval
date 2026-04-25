import pytest

from sentinel.evaluator import Evaluator


def test_evaluator():
    evaluator = Evaluator()

    predictions = ["fact", "emotion"]
    ground_truth = ["fact", "opinion"]

    results = evaluator.evaluate(predictions, ground_truth)

    assert results["accuracy"] == 0.5
    assert "confusion_matrix" in results
    assert "per_class" in results


def test_evaluator_mismatch_length():
    evaluator = Evaluator()

    with pytest.raises(ValueError, match="Mismatch in lengths"):
        evaluator.evaluate(["fact"], ["fact", "emotion"])
