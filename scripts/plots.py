"""Figures, summary table and README results block."""
import argparse
from pathlib import Path

from convergence_repro.plots import make_all


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--readme", type=Path)
    a = p.parse_args()
    made = make_all(a.out, a.readme)
    print("figures:", ", ".join(made))


if __name__ == "__main__":
    main()
