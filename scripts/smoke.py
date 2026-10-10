"""Smoke test on synthetic CIFAR-shaped data: each schedule with interrupt/resume, timing, queue forecast."""
import argparse
import math
import shutil
import sys
from pathlib import Path

from convergence_repro.runs import CORE, ORDER, apply_scale, expand, load_config
from convergence_repro.train import run

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TINY = {"depth": 8, "batch_size": 100}


def check_schedule(name, base, out, iters):
    d = out / name
    shutil.rmtree(d, ignore_errors=True)
    cfg = {**base, "schedule": name, "iters": iters, "run_id": f"smoke_{name}",
           "max_lr": 4.0 if name == "linear" else 3.0}
    partial = run(cfg, d, stop_at=iters * 3 // 5)
    log = run(cfg, d)
    loss = log["steps"]["loss"]
    return {
        "interrupted": partial["status"] == "running",
        "resumed_to_end": log["status"] == "done" and log["steps"]["it"][-1] == iters,
        "finite_loss": all(math.isfinite(v) for v in loss),
        "loss_below_chance": min(loss) < math.log(10),
        "json_written": (d / "log.json").exists(),
    }


def sec_per_iter(base, out, iters):
    d = out / "timing"
    shutil.rmtree(d, ignore_errors=True)
    half = iters // 2
    cfg = {**base, "schedule": "onecycle", "iters": iters, "eval_every": half, "run_id": "smoke_timing"}
    ev = run(cfg, d)["evals"]
    return (ev[-1]["wall"] - ev[0]["wall"]) / (ev[-1]["it"] - ev[0]["it"])


def forecast(spi, scale):
    print(f"\nforecast at {spi:.3f} s/iter, scale={scale}:")
    total = 0.0
    for exp_id in ORDER:
        path = CONFIGS / f"{exp_id}.yaml"
        iters = sum(apply_scale(c, scale)["iters"] for c in expand(load_config(path)))
        total += iters * spi
        tag = "core" if exp_id in CORE else "opt "
        print(f"  {exp_id:<4} {tag} {iters:>6} it  {iters * spi / 3600:5.2f} h  (cumulative {total / 3600:5.2f} h)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--time-iters", type=int, default=120)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--device")
    p.add_argument("--tiny", action="store_true", help="ResNet-8, batch 100; for CPU tests")
    a = p.parse_args()

    base = {"eval_every": a.iters // 2, "ckpt_every": a.iters // 2, "log_every": 5,
            "fake_data": True, "device": a.device, **(TINY if a.tiny else {})}
    ok = True
    for name in ("onecycle", "step", "linear"):
        res = check_schedule(name, base, a.out, a.iters)
        ok &= all(res.values())
        print(f"{name:<9} " + "  ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in res.items()), flush=True)
    spi = sec_per_iter(base, a.out, a.time_iters)
    forecast(spi, a.scale)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
