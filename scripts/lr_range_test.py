"""LR range test (E0) and its figure."""
import argparse
from pathlib import Path

from convergence_repro.plots import plot_range_test
from convergence_repro.runs import expand, load_config
from convergence_repro.train import run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "E0.yaml"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--data", default="data")
    p.add_argument("--device")
    a = p.parse_args()
    for cfg in expand(load_config(a.config)):
        cfg.update(data_root=a.data, device=a.device)
        log = run(cfg, a.out / cfg["run_id"])
        plot_range_test(log, a.out / "figures" / "e0_range_test")
        print(f"{cfg['run_id']}: {log['status']}")


if __name__ == "__main__":
    main()
