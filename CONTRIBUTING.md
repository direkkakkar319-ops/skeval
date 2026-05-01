# Contributing to skeval

Thank you for your interest in contributing to skeval. This document explains how to get set up, what our standards are, and how to get your changes merged.

---

## Table of Contents

1. [Getting started](#getting-started)
2. [Branching strategy](#branching-strategy)
3. [Making changes](#making-changes)
4. [Code standards](#code-standards)
5. [Running tests](#running-tests)
6. [Submitting a pull request](#submitting-a-pull-request)
7. [Reporting bugs](#reporting-bugs)
8. [Suggesting features](#suggesting-features)

---

## Getting started

**1. Fork and clone the repository**
```bash
git clone https://github.com/your-username/skeval.git
cd skeval
```

**2. Create a virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

**3. Verify everything works**
```bash
pytest tests/
python -c "import skeval; print(skeval.__version__)"
```

---

## Branching strategy

- `main` — stable, always releasable
- Feature branches — named after the issue they fix, e.g. `fix/27-remove-unused-deps` or `feature/16-sklearn-api`

Always branch off `main` unless you are building on top of an open PR.

---

## Making changes

- One pull request per issue
- Keep changes focused — do not mix unrelated fixes in one PR
- Reference the issue number in your PR description: `Closes #16`
- Write or update tests for any code you change

---

## Code standards

We enforce the following automatically via CI:

| Tool | What it checks |
|---|---|
| `black` | Code formatting |
| `isort` | Import ordering |
| `flake8` | Style and common errors |
| `bandit` | Security issues |

Run them locally before pushing:
```bash
black src/ tests/
isort src/ tests/
flake8 src/ tests/
bandit -r src/ -ll
```

**Docstrings:** Every public class and function must have a Google-style docstring with `Args`, `Returns`, `Raises`, and at least one `Example`.

**Comments:** Only add a comment when the *why* is non-obvious. Do not explain what the code does — well-named variables do that already.

---

## Running tests

```bash
# Run the full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/skeval --cov-report=term-missing
```

All tests must pass before a PR can be merged. New features require new tests. Coverage must not drop below 85%.

---

## Submitting a pull request

1. Push your branch to your fork
2. Open a PR against `main` on the skeval repository
3. Fill in the PR template — include what you changed and why
4. A maintainer will comment `/run-ci` to trigger the CI checks
5. Address any review feedback
6. Once approved and CI passes, the PR will be merged

---

## Reporting bugs

Open a [GitHub issue](https://github.com/direkkakkar319-ops/skeval/issues/new?template=bug_report.md) using the bug report template. Include:

- What you did
- What you expected to happen
- What actually happened
- Your Python version, OS, and skeval version (`python -c "import skeval; print(skeval.__version__)"`)

---

## Suggesting features

Open a [GitHub issue](https://github.com/direkkakkar319-ops/skeval/issues/new?template=feature_request.md) using the feature request template. Describe:

- The problem you are trying to solve
- Your proposed solution
- Any alternatives you considered

---

By contributing you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
