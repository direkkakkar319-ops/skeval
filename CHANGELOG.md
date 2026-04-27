# Changelog

All notable changes to skeval are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

<!--
────────────────────────────────────────────────────────────────────────────
  DEVELOPER INSTRUCTIONS — HOW TO UPDATE THIS FILE
────────────────────────────────────────────────────────────────────────────

  When you make a change, add a bullet point under the correct heading in
  the [Unreleased] section below. Do NOT create a new version section —
  that happens only at release time.

  Pick the right heading for your change:

  ### Added      — new feature or capability that did not exist before
  ### Changed    — change to existing behaviour (not a bug fix)
  ### Deprecated — feature that still works but will be removed in a future version
  ### Removed    — feature that has been deleted
  ### Fixed      — bug fix
  ### Security   — security patch or vulnerability fix
  ### CI/Tooling — changes to workflows, linting, build, or developer tooling

  Format for each entry:
  - `ClassName.method()` or `module.function()` — one-line description of what changed and why

  At release time:
  1. Replace "## [Unreleased]" with "## [x.y.z] — YYYY-MM-DD"
  2. Add a fresh empty [Unreleased] section at the top (copy the template below)
  3. Update the version in pyproject.toml and src/skeval/__init__.py
────────────────────────────────────────────────────────────────────────────
-->

## [Unreleased]

### Added
<!-- New features go here -->

### Changed
<!-- Changes to existing behaviour go here -->

### Deprecated
<!-- Features that will be removed in a future version go here -->

### Removed
<!-- Deleted features go here -->

### Fixed
<!-- Bug fixes go here -->

### Security
<!-- Security patches go here -->

### CI / Tooling
<!-- Workflow, linting, build, or tooling changes go here -->

---

## [Unreleased] — v0.2.0

### Added
- `SentenceClassifier.fit(X, y)` — primary training method, sklearn-compatible, returns `self` for chaining
- `SentenceClassifier.score(X, y)` — returns accuracy directly without needing `Evaluator`
- `SentenceClassifier.get_params()` — returns all hyperparameters as a dict, required for `GridSearchCV`
- `SentenceClassifier.set_params(**params)` — sets hyperparameters in place, required for `GridSearchCV`
- `epochs`, `batch_size`, `lr` moved to `__init__()` so they are tunable via `GridSearchCV`
- `SentenceClassifier` now works inside `sklearn.pipeline.Pipeline` and `GridSearchCV`
- `random_state` parameter on `SentenceClassifier` for reproducible training runs
- Input validation on `fit()` and `predict()` — all bad inputs raise clear `ValueError` or `RuntimeError`
- Input validation on `Evaluator.evaluate()` — empty lists raise `ValueError`
- Full Google-style docstrings with Args, Returns, Raises, and Examples on all public methods
- `tqdm` progress bar wired into the training loop
- `pip install skeval[transformers]` optional extra for future `TransformerClassifier`

### Changed
- `train()` is now a deprecated alias for `fit()` — emits `DeprecationWarning`, will be removed in v0.3.0
- `transformers` and `datasets` moved from required to optional dependencies (`pip install skeval[transformers]`)
- `pyyaml` removed from dependencies (unused until config file support lands in v0.3.0)
- Dependency upper bounds added: `torch<3.0.0`, `scikit-learn<2.0.0`, `numpy<3.0.0`, `pandas<3.0.0`, `tqdm<5.0.0`

### Fixed
- `torch.load` now passes `weights_only=True` — prevents arbitrary code execution via pickle (CWE-502)

### CI / Tooling
- CI import test now actually imports `skeval` and prints version (was a no-op `import sys` placeholder)
- Coverage source corrected from `src/sentinel` to `src/skeval`
- `pytest` no longer suppressed with `|| true` — failures correctly fail the build
- `security.yml` action versions fixed (`@v6` → `@v4` / `@v5`)
- `bandit` format fixed (`-f custom` → `-r src/ -ll`)
- `pip-audit` upgraded pip before audit to avoid self-CVE; added `--skip-editable` for local package
- CVE-2026-3219 in pip 26.0.1 ignored with `--ignore-vuln` until a patched version is released
- Dead no-op workflows removed: `data.yml`, `docs.yml`, `danger.yml`, `precommit.yml`
- `code-quality.yml` rewritten — fixed broken action versions, removed suppressed checks, runs black + isort + flake8 only
- `codeql.yml` — removed PR trigger to save Actions minutes; runs on push to main and weekly schedule
- Manual CI gate added — collaborators trigger checks by commenting `/run-ci` on a PR
- `ENABLE_AUTO_CI` repository variable added — flip to `true` to enable automatic CI on every commit
- `.coverage` and `htmlcov/` added to `.gitignore`

---

## [0.1.2] — 2026-04-26

### Fixed
- `torch.load` uses `weights_only=True` to prevent CWE-502 deserialization vulnerability
- CI import test now imports `skeval` instead of running a no-op `import sys`
- Coverage source corrected from `src/sentinel` → `src/skeval` in both `tests.yml` and `pyproject.toml`
- `pytest` failure no longer suppressed with `|| true` in `tests.yml`
- `security.yml` action versions fixed: `actions/checkout@v6` → `@v4`, `actions/setup-python@v6` → `@v5`
- `bandit -f custom` (invalid format) → `bandit -r src/ -ll`
- `pip-audit` now installs the package and runs without `|| true`
- `pip-audit` adds `--skip-editable` to skip local package not on PyPI
- `pip-audit` upgrades pip before audit to resolve pip's own CVE
- CVE-2026-3219 (pip 26.0.1, no fix available) suppressed with `--ignore-vuln`
- `.coverage` and `htmlcov/` added to `.gitignore`

---

## [0.1.1] — 2026-04-25

### Fixed
- CI workflow: updated `actions/checkout` to `v4` and `actions/setup-python` to `v5`
- README: corrected `predict()` usage example
- README: corrected example output keys (`per_class_f1` → `per_class`)
- README: fixed install URL placeholder

---

## [0.1.0] — 2026-04-25

First public release.

### Added
- `SentenceClassifier` — train, predict, save, and load a PyTorch sentence classifier
- `BasicTextClassifier` — EmbeddingBag + Linear neural network architecture
- `Evaluator` — evaluate predicted labels against ground truth
- `compute_metrics` — accuracy, per-class precision / recall / F1, confusion matrix via scikit-learn
- `DatasetLoader` — load training data from CSV or JSON Lines files
- `SentenceDataset` — PyTorch Dataset wrapper with variable-length collation
- `VocabBuilder` — bag-of-words tokenizer with `<PAD>` / `<UNK>` support
- `LabelEncoder` — string label ↔ integer index mapping
- `skeval train` / `skeval evaluate` CLI commands
- Sphinx documentation
- Full pytest test suite
