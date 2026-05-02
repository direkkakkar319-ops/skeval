from typing import Any, Dict, List

from skeval.metrics.metrics import compute_metrics


class Evaluator:
    """Evaluates classifier predictions against ground-truth labels."""

    def __init__(self):
        pass

    def evaluate(
        self, predictions: List[str], ground_truth: List[str]
    ) -> Dict[str, Any]:
        """Compute a suite of classification metrics for a set of predictions.

        Args:
            predictions: Predicted labels produced by ``SentenceClassifier.predict``.
            ground_truth: True labels aligned with ``predictions``.

        Returns:
            Dictionary with the following keys:

            - ``accuracy`` (float): Overall accuracy.
            - ``per_class`` (dict): Per-label precision, recall, F1, and support.
            - ``macro_avg`` (dict): Macro-averaged precision, recall, and F1.
            - ``weighted_avg`` (dict): Weighted-averaged precision, recall, and F1.
            - ``confusion_matrix`` (list[list[int]]): Row = true, column = predicted.
            - ``labels`` (list[str]): Sorted list of unique labels.

        Raises:
            ValueError: If either list is empty or the lengths do not match.
        """
        if not isinstance(predictions, (list, tuple)) or len(predictions) == 0:
            raise ValueError("predictions must be a non-empty list of strings.")
        if not isinstance(ground_truth, (list, tuple)) or len(ground_truth) == 0:
            raise ValueError("ground_truth must be a non-empty list of strings.")
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Mismatch in lengths: "
                f"{len(predictions)} predictions vs "
                f"{len(ground_truth)} truth labels."
            )

        results = compute_metrics(ground_truth, predictions)
        return results
