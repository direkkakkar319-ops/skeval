.PHONY: help install install-dev format lint test clean build docs security pre-commit

# Default target executed when no arguments are given to make.
help:
	@echo "Available commands:"
	@echo "  install      Install project dependencies"
	@echo "  install-dev  Install development dependencies (formatting, linting, testing tools)"
	@echo "  format       Format code with black and isort"
	@echo "  lint         Run all linters (flake8, mypy, vulture, pydocstyle, codespell)"
	@echo "  test         Run unit tests with pytest and coverage"
	@echo "  security     Run security checks (bandit, pip-audit)"
	@echo "  build        Build the package (wheel and sdist)"
	@echo "  docs         Build Sphinx documentation"
	@echo "  pre-commit   Run pre-commit hooks manually on all files"
	@echo "  clean        Remove build artifacts, compiled files, and caches"

install:
	pip install --upgrade pip
	if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
	pip install -e . || true

install-dev: install
	pip install flake8 black isort mypy vulture pydocstyle codespell pytest pytest-cov bandit pip-audit pip-licenses sphinx sphinx-rtd-theme build twine pre-commit

format:
	black .
	isort .

lint:
	flake8 .
	mypy . || true
	vulture . --min-confidence 80 || true
	pydocstyle . || true
	codespell . || true

test:
	pytest --cov=./ --cov-report=term-missing

security:
	bandit -r . -f custom || true
	if [ -f requirements.txt ]; then pip-audit -r requirements.txt || true; fi

build: clean
	python -m build

docs:
	@echo "Building docs..."
	# Uncomment when docs are configured: cd docs && make html

pre-commit:
	pre-commit run --all-files

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .coverage
