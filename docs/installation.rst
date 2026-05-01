Installation
============

Requirements
------------

* Python 3.9 or later
* PyTorch 2.0 or later
* scikit-learn 1.3 or later
* pandas 2.0 or later

Install from PyPI
-----------------

.. code-block:: bash

   pip install skeval

Install from Source
-------------------

.. code-block:: bash

   git clone https://github.com/direkkakkar319-ops/skeval.git
   cd skeval
   pip install -e .

Install with Transformer Support
---------------------------------

Required for the future ``TransformerClassifier`` (v0.3.0+):

.. code-block:: bash

   pip install "skeval[transformers]"

Install with Dev Extras
-----------------------

Includes pytest, black, mypy, flake8, and other dev tooling:

.. code-block:: bash

   pip install -e ".[dev]"

Install with Docs Extras
------------------------

Includes Sphinx and the Read the Docs theme:

.. code-block:: bash

   pip install -e ".[docs]"

Verify Installation
-------------------

.. code-block:: python

   import skeval
   print(skeval.__version__)  # 0.1.2
