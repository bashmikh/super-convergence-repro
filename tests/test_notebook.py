import json
from pathlib import Path

from convergence_repro.plots import END, START

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_valid():
    nb = json.loads((ROOT / "colab.ipynb").read_text())
    assert nb["nbformat"] == 4
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    for cmd in ("scripts/smoke.py", "scripts/run_all.py", "scripts/plots.py", "drive.mount"):
        assert cmd in src
    assert all(c.get("outputs", []) == [] for c in nb["cells"] if c["cell_type"] == "code")


def test_readme_markers():
    text = (ROOT / "README.md").read_text()
    assert START in text and END in text
