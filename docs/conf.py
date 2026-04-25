# Configuration file for the Sphinx documentation builder.
import os
import sys

# Add src/ to path so sphinx can import sentinel
sys.path.insert(0, os.path.abspath("../src"))

project = "Sentinel AI"
copyright = "2026, Sentinel AI Team"
author = "Sentinel AI Team"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
