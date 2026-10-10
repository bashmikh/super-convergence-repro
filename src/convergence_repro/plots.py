"""Figures and summary table built from run logs."""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .runs import ORDER
from .schedules import make_schedule
from .train import DEFAULTS

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PAPER = {  # ResNet-20, Fig. 4b; reference only, protocol differs
    "E1": "90.4 (CLR, ~20k it)",
    "E2": "88.6 (PC-LR)",
    "E3": "88.6 (PC-LR)",
}


def load_logs(out) -> list[dict]:
    logs = []
    for p in sorted(Path(out).glob("*/log.json")):
        log = json.loads(p.read_text())
        if log.get("evals"):
            logs.append(log)
    return logs


def _exp(log) -> str:
    return log["config"].get("exp") or log["run_id"].split("_")[0]


def _variant(log) -> str:
    return re.sub(r"_s\d+$", "", log["run_id"])


def _final(log) -> float:
    return log["evals"][-1]["test_acc"]


def _of(logs, exp, seed=None):
    return [l for l in logs if _exp(l) == exp and (seed is None or l["seed"] == seed)]


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_range_test(log, stem: Path) -> None:
    """Test accuracy and train/test loss versus learning rate (E0)."""
    ev, st = log["evals"], log["steps"]
    lr = [e["lr"] for e in ev]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(lr, [e["test_acc"] for e in ev], "o-", ms=3)
    a1.set(xlabel="learning rate", ylabel="test accuracy", title="LR range test")
    a2.plot(st["lr"], st["loss"], alpha=0.6, label="train (batch)")
    a2.plot(lr, [e["test_loss"] for e in ev], "o-", ms=3, label="test")
    a2.axvspan(0.3, 2.0, color="grey", alpha=0.15)
    a2.set(xlabel="learning rate", ylabel="loss", yscale="log")
    a2.legend()
    _save(fig, stem)


def plot_schedules(stem: Path) -> None:
    cases = [("1cycle 0.1-3, 5k", {"schedule": "onecycle", "iters": 5000}),
             ("step 0.35, 5k", {"schedule": "step", "iters": 5000}),
             ("step 0.35, 20k", {"schedule": "step", "iters": 20000})]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, c in cases:
        cfg = {**DEFAULTS, **c}
        f = make_schedule(cfg)
        its = np.arange(cfg["iters"])
        lrs = [f(i) for i in its]
        for ax in axes:
            ax.plot(its, lrs, label=label)
    axes[0].set(xlabel="iteration", ylabel="learning rate")
    axes[1].set(xlabel="iteration", yscale="log")
    for ax in axes:
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:g}k")
    axes[0].legend()
    _save(fig, stem)


def plot_main_curves(logs, stem: Path) -> bool:
    """Test accuracy vs iteration for E1/E2/E3: mean over seeds, min-max band."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for exp, label in (("E1", "1cycle, 5k"), ("E3", "step, 5k"), ("E2", "step, 20k")):
        runs = _of(logs, exp)
        if not runs:
            continue
        m = min(len(r["evals"]) for r in runs)
        its = np.array([e["it"] for e in runs[0]["evals"][:m]])
        acc = np.array([[e["test_acc"] for e in r["evals"][:m]] for r in runs])
        ax.plot(its, acc.mean(0), label=f"{exp}: {label} (n={len(runs)})")
        ax.fill_between(its, acc.min(0), acc.max(0), alpha=0.2)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set(xlabel="iteration", ylabel="test accuracy")
    ax.legend()
    _save(fig, stem)
    return True


def _points(logs, exp, key):
    return sorted((l["config"][key], _final(l)) for l in logs if _exp(l) == exp)


def plot_wd_ablation(logs, stem: Path) -> bool:
    """Final test accuracy vs weight decay; 1cycle (E4 + E1 seed 0) and step control (E4b)."""
    if not _of(logs, "E4"):
        return False
    pts = sorted(_points(logs, "E4", "wd") + [(l["config"]["wd"], _final(l)) for l in _of(logs, "E1", 0)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(*zip(*pts), "o-", label="1cycle, max LR 3")
    ctrl = _points(logs, "E4b", "wd")
    if ctrl:
        ax.plot(*zip(*ctrl), "s", ms=8, label="step 0.35 (E4b)")
    ax.set(xscale="log", xlabel="weight decay", ylabel="final test accuracy")
    ax.legend()
    _save(fig, stem)
    return True


def plot_small_data(logs, stem: Path) -> bool:
    """1cycle vs step on full data (E1/E3) and on 10k samples (E5)."""
    e5 = _of(logs, "E5")
    if not e5:
        return False

    def mean(runs):
        return float(np.mean([_final(r) for r in runs])) if runs else np.nan

    def by(schedule):
        return [r for r in e5 if r["config"]["schedule"] == schedule]

    one = [mean(_of(logs, "E1")), mean(by("onecycle"))]
    stp = [mean(_of(logs, "E3")), mean(by("step"))]
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.2, one, 0.4, label="1cycle")
    ax.bar(x + 0.2, stp, 0.4, label="step")
    for i in x:
        if not np.isnan(one[i]) and not np.isnan(stp[i]):
            ax.annotate(f"Δ={100 * (one[i] - stp[i]):+.1f} pp", (i, max(one[i], stp[i])),
                        ha="center", va="bottom")
    ax.set_xticks(x, ["50k samples", "10k samples"])
    ax.set(ylabel="final test accuracy", ylim=(0, 1.08))
    ax.legend(loc="lower right")
    _save(fig, stem)
    return True


def plot_maxlr(logs, stem: Path) -> bool:
    """Final test accuracy vs max LR (E6 + E1 seed 0)."""
    if not _of(logs, "E6"):
        return False
    pts = sorted(_points(logs, "E6", "max_lr") + [(l["config"]["max_lr"], _final(l)) for l in _of(logs, "E1", 0)])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(*zip(*pts), "o-")
    ax.set(xlabel="max learning rate", ylabel="final test accuracy")
    _save(fig, stem)
    return True


def _lr_label(c) -> str:
    s = c["schedule"]
    if s == "onecycle":
        return f"0.1-{c['max_lr']:g}"
    if s == "step":
        return f"{c['lr0']:g}"
    return f"0-{c['max_lr']:g}"


def summary_table(logs) -> pd.DataFrame:
    groups: dict[str, list] = {}
    for l in logs:
        groups.setdefault(_variant(l), []).append(l)
    rows = []
    for variant, runs in groups.items():
        c, exp = runs[0]["config"], _exp(runs[0])
        acc = np.array([_final(r) for r in runs]) * 100
        best = max(max(e["test_acc"] for e in r["evals"]) for r in runs) * 100
        times = [r.get("final", {}).get("total_time") or r["evals"][-1]["wall"] for r in runs]
        rows.append({
            "experiment": exp,
            "run": variant,
            "schedule": c["schedule"],
            "lr": _lr_label(c),
            "iters": c["iters"],
            "wd": f"{c['wd']:g}",
            "batch": c["batch_size"],
            "train_subset": c.get("train_subset") or "",
            "seeds": len(runs),
            "test_acc_%": f"{acc.mean():.2f} ± {acc.std(ddof=1):.2f}" if len(acc) > 1 else f"{acc[0]:.2f}",
            "best_acc_%": f"{best:.2f}",
            "time_min": f"{np.mean(times) / 60:.1f}",
            "status": ",".join(sorted({r["status"] for r in runs})),
            "paper_%": PAPER.get(exp, ""),
        })
    df = pd.DataFrame(rows)
    rank = {e: i for i, e in enumerate(ORDER)}
    df["_k"] = df["experiment"].map(lambda e: rank.get(e, len(ORDER)))
    return df.sort_values(["_k", "run"]).drop(columns="_k").reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def update_readme(readme, md: str) -> None:
    """Replace the block between RESULTS markers in `readme`."""
    readme = Path(readme)
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme}: results markers not found")
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    readme.write_text(f"{pre}{START}\n{md}\n{END}{post}")


def make_all(out, readme=None) -> list[str]:
    """Build every available figure and the summary table under `out`."""
    out = Path(out)
    figs = out / "figures"
    logs = load_logs(out)
    plot_schedules(figs / "schedules")
    made = ["schedules"]
    e0 = _of(logs, "E0")
    if e0:
        plot_range_test(e0[0], figs / "e0_range_test")
        made.append("e0_range_test")
    for name, fn in (("main_curves", plot_main_curves), ("wd_ablation", plot_wd_ablation),
                     ("small_data", plot_small_data), ("maxlr", plot_maxlr)):
        if fn(logs, figs / name):
            made.append(name)
    if logs:
        df = summary_table(logs)
        df.to_csv(out / "summary.csv", index=False)
        md = to_markdown(df)
        (out / "summary.md").write_text(md + "\n")
        if readme:
            update_readme(readme, md)
    return made
