"""Run the experiment queue, skipping finished runs."""
import argparse
from pathlib import Path

from convergence_repro.runs import ORDER, apply_scale, expand, load_config
from convergence_repro.train import FINISHED, read_log, run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
FAKE = {"fake_data": True, "depth": 8, "batch_size": 100}


def queue(only=None, scale=1.0, configs=CONFIGS):
    ids = [e for e in ORDER if only is None or e in only]
    for exp_id in ids:
        path = configs / f"{exp_id}.yaml"
        if path.exists():
            for cfg in expand(load_config(path)):
                yield apply_scale(cfg, scale)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", type=lambda s: s.split(","))
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--data", default="data")
    p.add_argument("--device")
    p.add_argument("--fake", action="store_true", help="synthetic data, ResNet-8; for testing")
    a = p.parse_args()

    results = []
    for cfg in queue(a.only, a.scale):
        cfg.update(data_root=a.data, device=a.device, **(FAKE if a.fake else {}))
        rid, d = cfg["run_id"], a.out / cfg["run_id"]
        prev = read_log(d)
        if prev and prev["status"] in FINISHED:
            print(f"skip {rid}: {prev['status']}", flush=True)
            results.append((rid, prev["status"], prev["final"].get("test_acc")))
            continue
        print(f"run  {rid}: {cfg['schedule']} iters={cfg['iters']}", flush=True)
        try:
            log = run(cfg, d)
            results.append((rid, log["status"], log["final"].get("test_acc")))
        except Exception as e:
            print(f"fail {rid}: {type(e).__name__}: {e}", flush=True)
            results.append((rid, "failed", None))

    print("\nrun_id                      status     test_acc")
    for rid, status, acc in results:
        print(f"{rid:<28}{status:<11}{'' if acc is None else f'{acc:.4f}'}")


if __name__ == "__main__":
    main()
