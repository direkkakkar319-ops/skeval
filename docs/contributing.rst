Contributing
============

Contributions are welcome and appreciated. This page covers how to get set up and what to keep in mind before opening a pull request.

----

Getting Started
---------------

1. Fork the repository on GitHub.
2. Clone your fork and create a new branch:

   .. code-block:: bash

      git clone https://github.com/<your-username>/Sentinel.AI.git
      cd Sentinel.AI
      git checkout -b my-feature

3. Install the package in editable mode with dev dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. (Optional) install pre-commit hooks:

   .. code-block:: bash

      pre-commit install

----

Running Tests
-------------

.. code-block:: bash

   pytest

For a coverage report:

.. code-block:: bash

   pytest --cov=sentinel --cov-report=term-missing

All tests must pass before a pull request can be merged.

----

Code Style
----------

This project uses:

* **black** for formatting (line length 88)
* **isort** for import ordering
* **flake8** for linting
* **mypy** for type checking

Run them all at once:

.. code-block:: bash

   black src/ tests/
   isort src/ tests/
   flake8 src/ tests/
   mypy src/

----

Building the Docs Locally
--------------------------

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make html

The built site is written to ``docs/_build/html/``. Open ``index.html`` in a browser to preview.

----

Submitting a Pull Request
--------------------------

* Keep pull requests focused — one feature or fix per PR.
* Add tests for any new behaviour.
* Update ``docs/changelog.rst`` with a brief entry under a new version heading if applicable.
* Make sure ``pytest``, ``black --check``, and ``mypy`` all pass locally before pushing.

If you are unsure whether your change fits the project direction, open an issue first to discuss it.

----

Code of Conduct
---------------

This project follows the `Contributor Covenant <https://www.contributor-covenant.org/>`_ Code of Conduct. By participating you agree to uphold a respectful and inclusive environment for everyone.
