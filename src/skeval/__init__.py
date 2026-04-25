"""skeval — Semantic Evaluation Layer for LLMs.

This package provides tools to classify and evaluate LLM outputs based on
semantic sentence types: facts, emotions, opinions, and instructions.

Example
-------
    >>> from skeval.classifier import SentenceClassifier
    >>> from skeval.evaluator import Evaluator
    >>> classifier = SentenceClassifier()
    >>> evaluator = Evaluator()
    >>> predictions = [classifier.predict(s) for s in ["Water boils at 100°C"]]
    >>> results = evaluator.evaluate(predictions, ["fact"])
    >>> print(results)
"""

from skeval.classifier import SentenceClassifier
from skeval.evaluator import Evaluator
from skeval.metrics import compute_metrics

__version__ = "0.1.1"
__author__ = "skeval Contributors"
__license__ = "MIT"

__all__ = [
    "SentenceClassifier",
    "Evaluator",
    "compute_metrics",
    "__version__",
]
