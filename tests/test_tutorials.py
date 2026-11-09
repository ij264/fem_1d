"""Run the code cells of every tutorial notebook, in order, in one namespace."""

import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TUTORIALS = sorted((Path(__file__).resolve().parent.parent / "tutorials").glob("*.ipynb"))


@pytest.mark.parametrize("path", TUTORIALS, ids=[p.name for p in TUTORIALS])
def test_tutorial_runs(path, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    nb = json.loads(path.read_text())
    namespace = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{path.name}[{i}]", "exec"), namespace)
    plt.close("all")
