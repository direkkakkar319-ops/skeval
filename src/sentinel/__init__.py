"""Sentinel AI — Semantic Evaluation Layer for LLMs.

This package provides tools to classify and evaluate LLM outputs based on
semantic sentence types: facts, emotions, opinions, and instructions.

Example
-------
    >>> from sentinel.classifier import SentenceClassifier
    >>> from sentinel.evaluator import Evaluator
    >>> classifier = SentenceClassifier()
    >>> evaluator = Evaluator()
    >>> predictions = [classifier.predict(s) for s in ["Water boils at 100°C"]]
    >>> results = evaluator.evaluate(predictions, ["fact"])
    >>> print(results)
"""

from sentinel.classifier import SentenceClassifier
from sentinel.evaluator import Evaluator
from sentinel.metrics import compute_metrics

__version__ = "0.1.0"
__author__ = "Sentinel AI Contributors"
__license__ = "MIT"

__all__ = [
    "SentenceClassifier",
    "Evaluator",
    "compute_metrics",
    "__version__",
]
