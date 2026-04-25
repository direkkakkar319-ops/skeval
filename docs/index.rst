Welcome to Sentinel AI!
=======================

Sentinel AI is a robust PyTorch-powered library for classifying and evaluating semantic concepts in text. It allows you to build custom deep learning models from the ground up to detect facts, emotions, opinions, instructions, and more.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   usage
   api

Installation
------------

You can install Sentinel AI directly using pip:

.. code-block:: bash

   git clone https://github.com/direkkakkar319-ops/Sentinel.AI.git
   cd Sentinel.AI
   pip install -e .

Quickstart
----------

You can quickly train a model right from the python API:

.. code-block:: python

   from sentinel.classifier import SentenceClassifier

   sentences = ["Water boils at 100C", "I feel happy"]
   labels = ["fact", "emotion"]

   classifier = SentenceClassifier()
   classifier.train(sentences, labels, epochs=10)

   predictions = classifier.predict(["The sky is blue"])
   print(predictions)

See the :doc:`usage` section for information on the command line interface.
