import json

import pytest

from convergence_repro.plots import END, START, make_all, summary_table, to_markdown, update_readme


def make_log(run_id, exp, schedule, seed=0, wd=1e-4, max_lr=3.0, iters=1000, status="done",
             accs=(0.5, 0.8, 0.9), nan=False):
    its = [iters * (k + 1) // len(accs) for k in range(len(accs))]
    loss = [2.0, 1.0, float("nan") if nan else 0.5]
    return {
        "run_id": run_id, "seed": seed, "status": status,
        "config": {"exp": exp, "schedule": schedule, "wd": wd, "max_lr": max_lr, "lr0": 0.35,
                   "iters": iters, "batch_size": 1000},
        "steps": {"it": its, "lr": [0.1 * (k + 1) for k in range(3)], "mom": [0.9] * 3, "loss": loss},
        "evals": [{"it": i, "lr": 0.1 * (k + 1), "train_loss_ema": 1.0, "train_acc_ema": 0.6,
                   "test_loss": 1.0 - a / 2, "test_acc": a, "wall": 10.0 * (k + 1)}
                  for k, (i, a) in enumerate(zip(its, accs))],
        "final": {"test_acc": accs[-1], "best_test_acc": max(accs), "total_time": 30.0},
    }


def write(out, log):
    d = out / log["run_id"]
    d.mkdir(parents=True)
    (d / "log.json").write_text(json.dumps(log))


FULL = [
    make_log("E0_s0", "E0", "linear", max_lr=4.0, status="diverged", nan=True),
    make_log("E1_s0", "E1", "onecycle"), make_log("E1_s1", "E1", "onecycle", seed=1, accs=(0.5, 0.8, 0.92)),
    make_log("E2_s0", "E2", "step", iters=4000),
    make_log("E3_s0", "E3", "step"), make_log("E3_s1", "E3", "step", seed=1),
    make_log("E4_wd1e-3_s0", "E4", "onecycle", wd=1e-3, accs=(0.3, 0.5, 0.7)),
    make_log("E4_wd1e-5_s0", "E4", "onecycle", wd=1e-5),
    make_log("E4b_s0", "E4b", "step", wd=1e-3),
    make_log("E5_scheduleonecycle_s0", "E5", "onecycle"),
    make_log("E5_schedulestep_s0", "E5", "step", accs=(0.4, 0.6, 0.7)),
    make_log("E6_max_lr1_s0", "E6", "onecycle", max_lr=1.0),
]


def test_make_all_full(tmp_path):
    for log in FULL:
        write(tmp_path, log)
    readme = tmp_path / "README.md"
    readme.write_text(f"# T\n\n{START}\nold\n{END}\n\ntail\n")
    made = make_all(tmp_path, readme)
    assert set(made) == {"schedules", "e0_range_test", "main_curves", "wd_ablation", "small_data", "maxlr"}
    for name in made:
        assert (tmp_path / "figures" / f"{name}.png").exists()
        assert (tmp_path / "figures" / f"{name}.pdf").exists()
    assert (tmp_path / "summary.csv").exists()
    text = readme.read_text()
    assert "old" not in text and "E4b" in text and text.endswith("tail\n")


def test_make_all_partial(tmp_path):
    write(tmp_path, FULL[0])
    assert set(make_all(tmp_path)) == {"schedules", "e0_range_test"}


def test_make_all_empty(tmp_path):
    assert make_all(tmp_path) == ["schedules"]


def test_summary_seeds_and_paper():
    df = summary_table(FULL)
    e1 = df[df["experiment"] == "E1"].iloc[0]
    assert e1["seeds"] == 2 and "±" in e1["test_acc_%"] and e1["paper_%"].startswith("90.4")
    assert list(df["experiment"]).index("E0") < list(df["experiment"]).index("E2")
    md = to_markdown(df)
    assert md.splitlines()[1].startswith("|---")


def test_update_readme_requires_markers(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("no markers")
    with pytest.raises(ValueError):
        update_readme(p, "x")


def test_wd_ablation_labels_diverged(tmp_path, monkeypatch):
    from convergence_repro import plots

    log = make_log("E4_wd1e-3_s0", "E4", "onecycle", wd=1e-3, status="diverged", accs=(0.3, 0.5, 0.1))
    log["final"]["diverged_at"] = 2440
    texts = []
    monkeypatch.setattr(plots, "_save", lambda fig, stem: texts.extend(t.get_text() for t in fig.axes[0].texts))
    assert plots.plot_wd_ablation([log, FULL[1]], tmp_path / "wd")
    assert any("diverged" in t and "2440" in t for t in texts)
