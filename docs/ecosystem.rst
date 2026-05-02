Ecosystem Compatibility
=======================

``skeval`` implements the full scikit-learn estimator interface (``fit``,
``predict``, ``score``, ``get_params``, ``set_params``), which means it works
out-of-the-box with every library that consumes sklearn-compatible estimators.

.. contents:: Libraries covered
   :local:
   :depth: 1

----

scikit-learn
------------

``SentenceClassifier`` passes ``sklearn.utils.estimator_checks.check_estimator``
and works directly with all sklearn model-selection utilities.

**GridSearchCV**

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from skeval.classifier import SentenceClassifier

   param_grid = {"embed_dim": [32, 64], "epochs": [20, 40], "lr": [0.005, 0.01]}
   search = GridSearchCV(SentenceClassifier(random_state=42), param_grid, cv=2)
   search.fit(sentences, labels)
   print(search.best_params_)

**cross_val_score**

.. code-block:: python

   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   scores = cross_val_score(
       SentenceClassifier(embed_dim=64, epochs=30, random_state=42),
       sentences, labels, cv=3, scoring="accuracy",
   )
   print(scores.mean())

See also :doc:`usage` for a full training example.

----

skore
-----

`skore <https://skore.probabl.ai>`_ is an open-source ML experiment tracker
that integrates with any sklearn-compatible estimator.

.. code-block:: bash

   pip install skore

.. code-block:: python

   import skore
   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   project = skore.open("my_project", overwrite=True)

   clf = SentenceClassifier(embed_dim=64, epochs=40, random_state=42)
   scores = cross_val_score(clf, sentences, labels, cv=2)

   project.put("cv_accuracy", scores.mean())
   project.put("params", clf.get_params())

Launch the interactive UI to compare runs:

.. code-block:: bash

   skore launch my_project

A complete working script is in ``examples/07_skore.py``.

----

Optuna
------

`Optuna <https://optuna.org>`_ is a hyperparameter optimisation framework that
works with any Python callable.

.. code-block:: bash

   pip install optuna

.. code-block:: python

   import optuna
   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   def objective(trial):
       clf = SentenceClassifier(
           embed_dim=trial.suggest_categorical("embed_dim", [32, 64, 128]),
           epochs=trial.suggest_int("epochs", 10, 60, step=10),
           lr=trial.suggest_float("lr", 1e-3, 1e-1, log=True),
           random_state=42,
       )
       return cross_val_score(clf, sentences, labels, cv=2).mean()

   study = optuna.create_study(direction="maximize")
   study.optimize(objective, n_trials=20)
   print(study.best_params)

----

Ray Tune
--------

`Ray Tune <https://docs.ray.io/en/latest/tune/index.html>`_ runs distributed
hyperparameter searches and integrates with sklearn via its sklearn wrappers.

.. code-block:: bash

   pip install "ray[tune]"

.. code-block:: python

   from ray import tune
   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   def train_fn(config):
       clf = SentenceClassifier(**config, random_state=42)
       score = cross_val_score(clf, sentences, labels, cv=2).mean()
       tune.report({"accuracy": score})

   tuner = tune.Tuner(
       train_fn,
       param_space={
           "embed_dim": tune.choice([32, 64, 128]),
           "epochs": tune.choice([20, 40]),
           "lr": tune.loguniform(1e-3, 1e-1),
       },
   )
   results = tuner.fit()
   print(results.get_best_result().config)

----

MLflow
------

`MLflow <https://mlflow.org>`_ tracks experiments, parameters, and metrics.

.. code-block:: bash

   pip install mlflow

.. code-block:: python

   import mlflow
   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   with mlflow.start_run():
       params = {"embed_dim": 64, "epochs": 40, "lr": 0.01}
       mlflow.log_params(params)

       clf = SentenceClassifier(**params, random_state=42)
       scores = cross_val_score(clf, sentences, labels, cv=2)

       mlflow.log_metric("cv_accuracy_mean", scores.mean())
       mlflow.log_metric("cv_accuracy_std", scores.std())

       clf.fit(sentences, labels)
       mlflow.sklearn.log_model(clf, "skeval_model")

----

Weights & Biases
----------------

`W&B <https://wandb.ai>`_ provides experiment tracking, visualisation, and
model registry.

.. code-block:: bash

   pip install wandb

.. code-block:: python

   import wandb
   from sklearn.model_selection import cross_val_score
   from skeval.classifier import SentenceClassifier

   wandb.init(project="skeval-runs")

   config = wandb.config
   config.embed_dim = 64
   config.epochs = 40
   config.lr = 0.01

   clf = SentenceClassifier(
       embed_dim=config.embed_dim,
       epochs=config.epochs,
       lr=config.lr,
       random_state=42,
   )
   scores = cross_val_score(clf, sentences, labels, cv=2)
   wandb.log({"cv_accuracy": scores.mean()})

----

BentoML
-------

`BentoML <https://bentoml.com>`_ packages trained models into production-ready
APIs.

.. code-block:: bash

   pip install bentoml

.. code-block:: python

   import bentoml
   from skeval.classifier import SentenceClassifier

   clf = SentenceClassifier(embed_dim=64, epochs=40, random_state=42)
   clf.fit(sentences, labels)

   # Save the model to the BentoML model store
   bento_model = bentoml.picklable_model.save_model("skeval_classifier", clf)
   print(f"Saved: {bento_model}")

   # Load it back for serving
   loaded = bentoml.picklable_model.load_model("skeval_classifier:latest")
   print(loaded.predict(["Water boils at 100 degrees Celsius"]))

----

LIME
----

`LIME <https://github.com/marcotcr/lime>`_ explains individual predictions by
perturbing the input. It requires ``predict_proba()``, which ``SentenceClassifier``
provides as of v0.2.0.

.. code-block:: bash

   pip install lime

.. code-block:: python

   from lime.lime_text import LimeTextExplainer
   from skeval.classifier import SentenceClassifier

   clf = SentenceClassifier(embed_dim=64, epochs=40, random_state=42)
   clf.fit(sentences, labels)

   explainer = LimeTextExplainer(class_names=clf.label_encoder.idx2label.values())
   explanation = explainer.explain_instance(
       "Water boils at 100 degrees Celsius",
       clf.predict_proba,
       num_features=5,
   )
   explanation.show_in_notebook()

.. note::
   ``predict_proba()`` is available from v0.2.0 onwards (issue #41).

----

SHAP
----

`SHAP <https://shap.readthedocs.io>`_ computes Shapley values to explain model
output globally and locally.

.. code-block:: bash

   pip install shap

.. code-block:: python

   import shap
   import numpy as np
   from skeval.classifier import SentenceClassifier

   clf = SentenceClassifier(embed_dim=64, epochs=40, random_state=42)
   clf.fit(sentences, labels)

   # KernelExplainer works with any predict_proba function
   background = shap.sample(np.array(sentences), 10)
   explainer = shap.KernelExplainer(clf.predict_proba, background)
   shap_values = explainer.shap_values(np.array(test_sentences))
   shap.summary_plot(shap_values, np.array(test_sentences))

----

Summary table
-------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 20 20

   * - Library
     - Use case
     - Works today
     - Requires
   * - scikit-learn
     - GridSearchCV, cross_val_score, Pipeline
     - Yes
     - ``pip install skeval``
   * - skore
     - Experiment tracking & UI
     - Yes
     - ``pip install skore``
   * - Optuna
     - Bayesian hyperparameter search
     - Yes
     - ``pip install optuna``
   * - Ray Tune
     - Distributed hyperparameter search
     - Yes
     - ``pip install ray[tune]``
   * - MLflow
     - Experiment tracking & model registry
     - Yes
     - ``pip install mlflow``
   * - Weights & Biases
     - Experiment tracking & visualisation
     - Yes
     - ``pip install wandb``
   * - BentoML
     - Model serving & packaging
     - Yes
     - ``pip install bentoml``
   * - LIME
     - Local prediction explanations
     - Yes (v0.2.0+)
     - ``pip install lime``
   * - SHAP
     - Shapley value explanations
     - Yes (v0.2.0+)
     - ``pip install shap``
