"""Download CIFAR-10 once, keeping the archive in a cache directory (e.g. on Google Drive)."""
import argparse
from pathlib import Path

from convergence_repro.data import fetch_cifar10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"), help="local dataset directory")
    p.add_argument("--cache", type=Path, help="persistent archive cache")
    a = p.parse_args()
    fetch_cifar10(a.data, a.cache)
    print(f"CIFAR-10 ready in {a.data}" + (f"; archive cached in {a.cache}" if a.cache else ""))


if __name__ == "__main__":
    main()
