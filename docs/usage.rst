Usage Guide
===========

Python API
----------

Training a Classifier
^^^^^^^^^^^^^^^^^^^^^

Create a :class:`~sentinel.classifier.SentenceClassifier`, pass your sentences and labels, and call :meth:`~sentinel.classifier.SentenceClassifier.train`:

.. code-block:: python

   from sentinel.classifier import SentenceClassifier

   classifier = SentenceClassifier(embed_dim=64)

   sentences = [
       "Water boils at 100 degrees Celsius",
       "Paris is the capital of France",
       "I am feeling very sad today",
       "This is the worst day of my life",
       "I think this movie is amazing",
       "In my opinion, pizza is the best food",
       "Please close the door",
       "Open the window right now",
   ]
   labels = [
       "fact", "fact",
       "emotion", "emotion",
       "opinion", "opinion",
       "instruction", "instruction",
   ]

   classifier.train(sentences, labels, epochs=20, lr=0.01)

The label vocabulary is inferred automatically from the labels you provide — you are not limited to the four default categories.

Making Predictions
^^^^^^^^^^^^^^^^^^

:meth:`~sentinel.classifier.SentenceClassifier.predict` takes a list of strings and returns a list of predicted label strings:

.. code-block:: python

   predictions = classifier.predict([
       "The sky is blue",
       "I am so happy",
       "I believe dogs are better than cats",
       "Turn off the lights",
   ])
   print(predictions)
   # ['fact', 'emotion', 'opinion', 'instruction']

Saving and Loading
^^^^^^^^^^^^^^^^^^

Save the trained model and vocabulary to a directory:

.. code-block:: python

   classifier.save("saved_model/")

Load it back in a new session:

.. code-block:: python

   from sentinel.classifier import SentenceClassifier

   classifier = SentenceClassifier()
   classifier.load("saved_model/")

   predictions = classifier.predict(["Water is wet"])

Two files are written: ``model.pt`` (PyTorch weights) and ``metadata.json`` (vocabulary and label mappings).

Evaluating Predictions
^^^^^^^^^^^^^^^^^^^^^^

Pass predictions and ground-truth labels to :class:`~sentinel.evaluator.Evaluator`:

.. code-block:: python

   from sentinel.evaluator import Evaluator

   evaluator = Evaluator()
   results = evaluator.evaluate(predictions, ground_truth)

   print(results["accuracy"])
   print(results["per_class"])
   print(results["confusion_matrix"])

The returned dictionary contains:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Description
   * - ``accuracy``
     - Overall fraction of correct predictions
   * - ``per_class``
     - Dict of ``{label: {precision, recall, f1-score, support}}``
   * - ``macro_avg``
     - Unweighted average of per-class metrics
   * - ``weighted_avg``
     - Support-weighted average of per-class metrics
   * - ``confusion_matrix``
     - 2-D list; rows = true labels, columns = predicted labels
   * - ``labels``
     - Sorted list of all class names used to index the matrix

Loading Data from Files
^^^^^^^^^^^^^^^^^^^^^^^

Use :class:`~sentinel.dataset.loader.DatasetLoader` to read CSV or JSON Lines files:

.. code-block:: python

   from sentinel.dataset.loader import DatasetLoader

   # CSV
   sentences, labels = DatasetLoader.load_csv(
       "data/train.csv", text_col="text", label_col="label"
   )

   # JSON Lines
   sentences, labels = DatasetLoader.load_json(
       "data/train.jsonl", text_key="text", label_key="label"
   )

   classifier.train(sentences, labels)

----

Command-Line Interface
----------------------

After installing the package, a ``sentinel`` command is available:

.. code-block:: bash

   sentinel --help
   sentinel train --help
   sentinel evaluate --help

Check the installed version:

.. code-block:: bash

   sentinel --version

Training via CLI
^^^^^^^^^^^^^^^^

.. code-block:: bash

   sentinel train \
       --data data/train.csv \
       --text-col text \
       --label-col label \
       --save-dir saved_model/ \
       --epochs 20 \
       --batch-size 32 \
       --embed-dim 64 \
       --lr 0.005

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--data``
     - Path to ``.csv`` or ``.jsonl`` training file (required)
   * - ``--text-col``
     - Column / key name that holds the sentence text (required)
   * - ``--label-col``
     - Column / key name that holds the label (required)
   * - ``--save-dir``
     - Directory to write ``model.pt`` and ``metadata.json`` (required)
   * - ``--embed-dim``
     - Embedding dimension (default: ``64``)
   * - ``--epochs``
     - Number of training epochs (default: ``10``)
   * - ``--batch-size``
     - Mini-batch size (default: ``32``)
   * - ``--lr``
     - Learning rate (default: ``0.005``)

Evaluation via CLI
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   sentinel evaluate \
       --model-dir saved_model/ \
       --data data/test.csv \
       --text-col text \
       --label-col label \
       --output report.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--model-dir``
     - Directory containing ``model.pt`` and ``metadata.json`` (required)
   * - ``--data``
     - Path to test ``.csv`` or ``.jsonl`` file (required)
   * - ``--text-col``
     - Column / key name for the sentence text (required)
   * - ``--label-col``
     - Column / key name for the ground-truth label (required)
   * - ``--output``
     - Optional path to save the JSON results file

----

Legacy Scripts
--------------

Sentinel AI also ships standalone scripts in ``scripts/`` that work without installation.

Training
^^^^^^^^

Train a model from a CSV or JSONL file and save it to disk:

.. code-block:: bash

   python scripts/train_model.py \
       --data data/train.csv \
       --text-col text \
       --label-col label \
       --save-dir saved_model/ \
       --epochs 20 \
       --batch-size 32 \
       --embed-dim 64 \
       --lr 0.005

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--data``
     - Path to ``.csv`` or ``.jsonl`` training file (required)
   * - ``--text-col``
     - Column / key name that holds the sentence text (required)
   * - ``--label-col``
     - Column / key name that holds the label (required)
   * - ``--save-dir``
     - Directory to write ``model.pt`` and ``metadata.json`` (required)
   * - ``--embed-dim``
     - Embedding dimension (default: ``64``)
   * - ``--epochs``
     - Number of training epochs (default: ``10``)
   * - ``--batch-size``
     - Mini-batch size (default: ``32``)
   * - ``--lr``
     - Learning rate (default: ``0.005``)

Evaluation
^^^^^^^^^^

Load a trained model and evaluate it on a held-out test set:

.. code-block:: bash

   python scripts/evaluate_llm.py \
       --model-dir saved_model/ \
       --data data/test.csv \
       --text-col text \
       --label-col label \
       --output report.json

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Description
   * - ``--model-dir``
     - Directory containing ``model.pt`` and ``metadata.json`` (required)
   * - ``--data``
     - Path to test ``.csv`` or ``.jsonl`` file (required)
   * - ``--text-col``
     - Column / key name for the sentence text (required)
   * - ``--label-col``
     - Column / key name for the ground-truth label (required)
   * - ``--output``
     - Optional path to save the JSON results file

The script prints accuracy, per-class F1, and the confusion matrix, and optionally writes them to the path given by ``--output``.
