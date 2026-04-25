Changelog
=========

All notable changes to Sentinel AI are documented here.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

----

0.1.1 — 2026-04-25
-------------------

**Fixed**

* CI workflow: updated ``actions/checkout`` to ``v4`` and ``actions/setup-python`` to ``v5`` (``v6`` does not exist and caused CI failures)
* README: corrected ``predict()`` usage example — method takes a list of strings, not a single string
* README: corrected example output keys (``per_class_f1`` → ``per_class``)
* README: fixed install URL placeholder (``your-username`` → correct repo path)

----

0.1.0 — 2026-04-25
-------------------

First public release.

**Added**

* :class:`~sentinel.classifier.SentenceClassifier` — train, predict, save, and load a PyTorch sentence classifier
* :class:`~sentinel.classifier.BasicTextClassifier` — ``EmbeddingBag + Linear`` neural network architecture
* :class:`~sentinel.evaluator.Evaluator` — evaluate predicted labels against ground truth
* :func:`~sentinel.metrics.compute_metrics` — accuracy, per-class precision / recall / F1, confusion matrix via scikit-learn
* :class:`~sentinel.dataset.loader.DatasetLoader` — load training data from CSV or JSON Lines files
* :class:`~sentinel.dataset.loader.SentenceDataset` — PyTorch ``Dataset`` wrapper with variable-length collation
* :class:`~sentinel.utils.helpers.VocabBuilder` — bag-of-words tokenizer with ``<PAD>`` / ``<UNK>`` support
* :class:`~sentinel.utils.helpers.LabelEncoder` — string label ↔ integer index mapping
* ``scripts/train_model.py`` — CLI script for training from file
* ``scripts/evaluate_llm.py`` — CLI script for evaluation from file
* Sphinx documentation
* Full pytest test suite
