from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Compute classification metrics given true and predicted label sequences.

    Args:
        y_true: Ground-truth label strings.
        y_pred: Predicted label strings in the same order as ``y_true``.

    Returns:
        Dictionary with the following keys:

        - ``accuracy`` (float): Fraction of correctly classified samples.
        - ``per_class`` (dict): Per-label dict of ``precision``, ``recall``,
          ``f1-score``, and ``support``.
        - ``macro_avg`` (dict): Macro-averaged precision, recall, and F1.
        - ``weighted_avg`` (dict): Support-weighted precision, recall, and F1.
        - ``confusion_matrix`` (list[list[int]]): Confusion matrix where rows
          correspond to true labels and columns to predicted labels, ordered by
          ``labels``.
        - ``labels`` (list[str]): Sorted union of all labels seen in either
          ``y_true`` or ``y_pred``.
    """
    acc = accuracy_score(y_true, y_pred)

    # zero_division=0 prevents warnings if a class is never predicted
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Extract unique labels in sorted order to map the confusion matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": acc,
        "per_class": {
            label: {
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1-score": report[label]["f1-score"],
                "support": report[label]["support"],
            }
            for label in labels
            if label in report
        },
        "macro_avg": report.get("macro avg", {}),
        "weighted_avg": report.get("weighted avg", {}),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }
