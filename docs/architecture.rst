Architecture
============

skeval is organized as a set of composable modules. Each module has a single responsibility, and they are designed to be used together or independently.

----

Module Overview
---------------

.. code-block:: text

   skeval/
   ├── classifier/          # Training and inference
   │   └── sentence_classifier.py
   ├── dataset/             # Data loading and batching
   │   └── loader.py
   ├── evaluator/           # Prediction evaluation
   │   └── evaluator.py
   ├── metrics/             # Metric computation
   │   └── metrics.py
   └── utils/               # Shared building blocks
       └── helpers.py

----

Data Flow
---------

A typical training and evaluation run follows this path:

.. code-block:: text

   Raw text + labels
         │
         ▼
   VocabBuilder.build()        ← builds word → index mapping
   LabelEncoder.build()        ← builds label → index mapping
         │
         ▼
   SentenceDataset             ← wraps text/label pairs as a PyTorch Dataset
         │
         ▼
   DataLoader (collate_fn)     ← pads variable-length sequences into batches
         │
         ▼
   BasicTextClassifier         ← EmbeddingBag + Linear
         │
         ▼
   CrossEntropyLoss + Adam     ← standard training loop
         │
         ▼
   SentenceClassifier.predict()
         │
         ▼
   Evaluator.evaluate()
         │
         ▼
   compute_metrics()           ← accuracy, per-class F1, confusion matrix

----

Key Components
--------------

VocabBuilder
^^^^^^^^^^^^

Located in :mod:`skeval.utils.helpers`.

Builds a bag-of-words vocabulary from a list of sentences. Text is normalized (lowercased, punctuation stripped) before tokenization. Two special tokens are always present:

* ``<PAD>`` at index ``0`` — used as a placeholder for empty inputs
* ``<UNK>`` at index ``1`` — maps words not seen during training

The ``min_freq`` parameter (default ``1``) filters out rare tokens.

LabelEncoder
^^^^^^^^^^^^

Located in :mod:`skeval.utils.helpers`.

Maps string labels to integer indices and back. Labels are sorted alphabetically before indexing so the mapping is deterministic across runs.

BasicTextClassifier
^^^^^^^^^^^^^^^^^^^

Located in :mod:`skeval.classifier.sentence_classifier`.

A two-layer PyTorch model:

1. :class:`torch.nn.EmbeddingBag` — looks up word embeddings and averages them across the sentence. This produces a fixed-size sentence vector regardless of sentence length.
2. :class:`torch.nn.Linear` — maps the sentence vector to class logits.

The model is intentionally minimal: it is fast to train on small datasets, interpretable, and easy to extend. The ``embed_dim`` hyperparameter controls the size of the internal representation (default ``64``).

SentenceDataset and collate_fn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Located in :mod:`skeval.dataset.loader`.

:class:`~skeval.dataset.loader.SentenceDataset` wraps encoded sentences and labels as a standard :class:`torch.utils.data.Dataset`. Because sentences have different lengths, the custom ``collate_fn`` packs them into a single 1-D tensor and computes ``offsets`` — the starting index of each sentence — so that :class:`~torch.nn.EmbeddingBag` can process the whole batch in one call.

Evaluator and compute_metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Located in :mod:`skeval.evaluator.evaluator` and :mod:`skeval.metrics.metrics`.

:class:`~skeval.evaluator.Evaluator` is a thin wrapper that validates input lengths and delegates to :func:`~skeval.metrics.compute_metrics`. The metrics function uses scikit-learn internally for accuracy, per-class precision/recall/F1, and the confusion matrix, then reshapes the results into a consistent dictionary.

----

Extending the Label Taxonomy
-----------------------------

The four default categories (``fact``, ``emotion``, ``opinion``, ``instruction``) are not hard-coded anywhere. Both :class:`~skeval.utils.helpers.LabelEncoder` and :class:`~skeval.classifier.SentenceClassifier` infer the label set from your training data at call time. To add a new category, include sentences with that label in your training set — no code changes required.

----

Saving and Loading
------------------

:meth:`~skeval.classifier.SentenceClassifier.save` writes two files:

* ``model.pt`` — PyTorch ``state_dict`` of the trained weights
* ``metadata.json`` — ``embed_dim``, the full ``word2idx`` / ``idx2word`` vocabulary, and the full ``label2idx`` / ``idx2label`` mapping

:meth:`~skeval.classifier.SentenceClassifier.load` reconstructs the ``VocabBuilder`` and ``LabelEncoder`` from ``metadata.json``, then rebuilds the ``BasicTextClassifier`` with the correct architecture before loading the weights. This means a saved model is fully portable across machines.
