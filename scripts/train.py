"""Run a single configuration."""
import argparse
import json
from pathlib import Path

from convergence_repro.runs import load_config
from convergence_repro.train import run

OVERRIDES = ("wd", "max_lr", "lr0", "iters", "batch_size", "train_subset")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-id")
    p.add_argument("--data", default="data")
    p.add_argument("--device")
    for k in ("wd", "max_lr", "lr0"):
        p.add_argument(f"--{k.replace('_', '-')}", type=float)
    for k in ("iters", "batch_size", "train_subset"):
        p.add_argument(f"--{k.replace('_', '-')}", type=int)
    a = p.parse_args()

    exp = load_config(a.config)
    cfg = {k: v for k, v in exp.items() if k not in ("id", "seeds", "sweep")}
    cfg.update({k: getattr(a, k) for k in OVERRIDES if getattr(a, k) is not None})
    cfg.update(exp=exp["id"], seed=a.seed, data_root=a.data, device=a.device)
    cfg["run_id"] = a.run_id or f"{exp['id']}_manual_s{a.seed}"
    log = run(cfg, a.out / cfg["run_id"])
    print(json.dumps({"status": log["status"], **log["final"]}, indent=2))


if __name__ == "__main__":
    main()
