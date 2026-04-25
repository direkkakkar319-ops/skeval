Usage Guide
===========

Command Line Interface
----------------------

Sentinel AI comes with powerful command line tools to train and evaluate models without writing any Python code.

Training a Model
^^^^^^^^^^^^^^^^

Use the ``train_model.py`` script to train a custom PyTorch model on your CSV or JSONL dataset.

.. code-block:: bash

   python scripts/train_model.py \
       --data data.jsonl \
       --text-col text \
       --label-col label \
       --save-dir saved_model/ \
       --epochs 20 \
       --batch-size 32

Evaluating a Model
^^^^^^^^^^^^^^^^^^

Once you have a trained model, you can evaluate it on a held-out test set using ``evaluate_llm.py``.

.. code-block:: bash

   python scripts/evaluate_llm.py \
       --model-dir saved_model/ \
       --data test_data.csv \
       --text-col sentence \
       --label-col actual_label \
       --output evaluation_report.json

This will print a comprehensive evaluation report including accuracy, per-class F1-scores, and a confusion matrix, and save it to the specified JSON file.
