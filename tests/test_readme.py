"""Run the Python code blocks of README.md, in order, in one namespace."""

import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"


def test_readme_examples_run(capsys):
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.S)
    assert len(blocks) >= 2
    namespace = {}
    for block in blocks:
        exec(compile(block, str(README), "exec"), namespace)
    out = capsys.readouterr().out
    assert "u(0.5) = 0.084963" in out  # the claimed value, ln(1.5)/ln(2) - 0.5
