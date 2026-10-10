"""Environment check: dependencies, GPU support, configs, output and data directories."""
import argparse
import importlib
import shutil
import sys
import warnings
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEPS = ("torch", "torchvision", "numpy", "matplotlib", "pandas", "yaml", "convergence_repro")
NUMERIC = ("max_lr", "lr0", "wd", "iters", "eval_every", "eval_subset", "train_subset")
MIN_GPU_GB = 8
CIFAR_GB = 0.4

results: list[tuple[str, str, str]] = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:<4}] {name:<18} {detail}", flush=True)


def check_python() -> None:
    ok = sys.version_info >= (3, 10)
    report("OK" if ok else "FAIL", "python", sys.version.split()[0])


def check_deps() -> bool:
    ok = True
    for name in DEPS:
        try:
            mod = importlib.import_module(name)
            report("OK", f"import {name}", getattr(mod, "__version__", ""))
        except Exception as e:
            report("FAIL", f"import {name}", f"{type(e).__name__}: {e}")
            ok = False
    return ok


def check_gpu(device: str) -> None:
    import torch

    from convergence_repro.models import resnet_cifar

    if device == "cpu":
        report("INFO", "gpu", "skipped (--device cpu)")
        return
    if not torch.cuda.is_available():
        report("FAIL", "gpu", "CUDA not available; select a GPU runtime")
        return
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    major, minor = torch.cuda.get_device_capability(idx)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()
    if arch not in supported:
        report("FAIL", "gpu arch", f"{name} ({arch}) not in torch build {supported}")
        return
    report("OK", "gpu arch", f"{name} ({arch}), CUDA {torch.version.cuda}")
    mem = torch.cuda.get_device_properties(idx).total_memory / 2**30
    report("OK" if mem >= MIN_GPU_GB else "WARN", "gpu memory",
           f"{mem:.1f} GB" + ("" if mem >= MIN_GPU_GB else f"; batch 1000 may not fit (<{MIN_GPU_GB} GB)"))
    try:
        model = resnet_cifar(8).cuda().to(memory_format=torch.channels_last)
        x = torch.randn(8, 3, 32, 32, device="cuda").contiguous(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(x).float().logsumexp(1).mean()
        loss.backward()
        torch.cuda.synchronize()
        report("OK", "gpu fp16 step", "forward/backward under autocast")
    except Exception as e:
        report("FAIL", "gpu fp16 step", f"{type(e).__name__}: {e}")


def check_configs() -> None:
    from convergence_repro.runs import ORDER, expand, load_config

    n, ok = 0, True
    for exp_id in ORDER:
        path = CONFIGS / f"{exp_id}.yaml"
        if not path.exists():
            report("FAIL", f"config {exp_id}", f"missing {path}")
            ok = False
            continue
        try:
            runs = expand(load_config(path))
            bad = [(r["run_id"], k) for r in runs for k in NUMERIC
                   if r.get(k) is not None and not isinstance(r[k], (int, float))]
            if bad:
                report("FAIL", f"config {exp_id}", f"non-numeric fields: {bad}")
                ok = False
                continue
            n += len(runs)
        except Exception as e:
            report("FAIL", f"config {exp_id}", f"{type(e).__name__}: {e}")
            ok = False
    if ok:
        report("OK", "configs", f"{len(ORDER)} experiments, {n} runs")


def check_out(out: Path) -> None:
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe, tmp = out / ".doctor", out / ".doctor.tmp"
        tmp.write_text("ok")
        tmp.replace(probe)
        probe.unlink()
        report("OK", "output dir", str(out))
    except Exception as e:
        report("FAIL", "output dir", f"{out}: {type(e).__name__}: {e}")


def check_data(root: Path) -> None:
    if (root / "cifar-10-batches-py").is_dir():
        report("OK", "cifar-10", str(root))
        return
    probe = root if root.exists() else next((p for p in root.parents if p.exists()), Path("."))
    free = shutil.disk_usage(probe).free / 2**30
    status = "INFO" if free > CIFAR_GB else "FAIL"
    report(status, "cifar-10", f"not found in {root}; will be downloaded ({free:.1f} GB free)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, help="results directory to check for write access")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    a = p.parse_args()
    warnings.filterwarnings("ignore", module="torch.cuda")

    check_python()
    if check_deps():
        check_gpu(a.device)
        check_configs()
    if a.out:
        check_out(a.out)
    check_data(a.data)

    failed = [r for r in results if r[0] == "FAIL"]
    print(f"\n{len(failed)} failed, {sum(r[0] == 'WARN' for r in results)} warnings")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
