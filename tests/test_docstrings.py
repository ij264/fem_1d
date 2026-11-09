"""Run the Examples sections of the package docstrings as tests."""

import doctest
import importlib

import pytest

MODULES = ["assembly", "gll", "mesh", "plotting", "problem", "space", "stokes", "self_gravity"]


@pytest.mark.parametrize("name", MODULES)
def test_docstring_examples(name):
    module = importlib.import_module(f"fem_1d.{name}")
    result = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
