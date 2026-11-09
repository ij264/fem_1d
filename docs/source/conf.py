"""Sphinx configuration for the fem_1d API reference."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import fem_1d  # noqa: E402

project = "fem_1d"
release = version = fem_1d.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # NumPy-style docstrings
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

# matplotlib is an optional extra: let the plotting reference build without it
autodoc_mock_imports = ["matplotlib"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

html_theme = "furo"
html_title = f"fem_1d {release}"
