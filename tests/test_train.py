import json
import math

import pytest
import torch.nn.functional as F

from convergence_repro import train
from convergence_repro.train import read_log, run

BASE = dict(depth=8, iters=20, batch_size=100, eval_every=10, ckpt_every=10, log_every=5,
            fake_data=True, device="cpu", schedule="onecycle", max_lr=1.0, run_id="t")


def test_run_completes(tmp_path):
    log = run(BASE, tmp_path)
    assert log["status"] == "done"
    assert [e["it"] for e in log["evals"]] == [10, 20]
    assert log["steps"]["it"][-1] == 20
    assert 0.0 <= log["final"]["test_acc"] <= 1.0
    assert log["final"]["sec_per_iter"] > 0
    assert json.loads((tmp_path / "log.json").read_text())["status"] == "done"


def test_resume_mid_epoch(tmp_path):
    first = run(BASE, tmp_path, stop_at=13)
    assert first["status"] == "running" and (tmp_path / "ckpt.pt").exists()
    log = run(BASE, tmp_path)
    assert log["status"] == "done"
    its = [e["it"] for e in log["evals"]]
    assert its == sorted(set(its)) == [10, 20]


def test_finished_run_is_skipped(tmp_path, monkeypatch):
    run(BASE, tmp_path)
    monkeypatch.setattr(train, "resnet_cifar", lambda *a, **k: pytest.fail("retrained"))
    assert run(BASE, tmp_path)["status"] == "done"


def test_config_mismatch_on_resume(tmp_path):
    run(BASE, tmp_path, stop_at=10)
    with pytest.raises(RuntimeError, match="config differs"):
        run({**BASE, "iters": 16}, tmp_path)


def test_data_root_change_allows_resume(tmp_path):
    run(BASE, tmp_path, stop_at=10)
    assert run({**BASE, "data_root": "/elsewhere"}, tmp_path)["status"] == "done"


def test_nan_loss_marks_diverged(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "loss_fn", lambda o, y: F.cross_entropy(o, y) * float("nan"))
    log = run(BASE, tmp_path)
    assert log["status"] == "diverged"
    assert log["final"]["diverged_at"] == 5
    assert read_log(tmp_path)["status"] == "diverged"


def test_batch_larger_than_data(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        run({**BASE, "batch_size": 5000}, tmp_path)
    assert read_log(tmp_path)["status"] == "failed"


def test_loss_decreases(tmp_path):
    log = run({**BASE, "iters": 60, "eval_every": 30, "ckpt_every": 30}, tmp_path)
    loss = log["steps"]["loss"]
    assert all(math.isfinite(v) for v in loss)
    assert min(loss[-3:]) < loss[0]


def test_failure_on_resume_keeps_partial_log(tmp_path):
    run(BASE, tmp_path, stop_at=10)
    with pytest.raises(RuntimeError):
        run({**BASE, "iters": 16}, tmp_path)
    log = read_log(tmp_path)
    assert log["status"] == "failed" and [e["it"] for e in log["evals"]] == [10]


def test_isolated_nonfinite_loss_is_not_divergence(tmp_path, monkeypatch):
    calls = {"n": 0}

    def flaky(o, y):
        calls["n"] += 1
        loss = F.cross_entropy(o, y)
        # Non-finite value with finite gradient, as when AMP overflows and GradScaler skips the step.
        return loss - loss.detach() + float("inf") if calls["n"] == 3 else loss

    monkeypatch.setattr(train, "loss_fn", flaky)
    assert run(BASE, tmp_path)["status"] == "done"


def tb_scalars(log_dir, tag):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(log_dir))
    acc.Reload()
    return [e.step for e in acc.Scalars(tag)] if tag in acc.Tags()["scalars"] else []


def test_tensorboard_events(tmp_path):
    run(BASE, tmp_path / "t")
    tb = tmp_path / "tb" / "t"
    assert tb_scalars(tb, "test/acc") == [10, 20]
    assert tb_scalars(tb, "train/loss") == list(range(1, 21))
    assert tb_scalars(tb, "lr") == list(range(1, 21))


def test_tensorboard_resume_no_duplicates(tmp_path):
    run({**BASE, "eval_every": 5}, tmp_path / "t", stop_at=13)
    run({**BASE, "eval_every": 5}, tmp_path / "t")
    steps = tb_scalars(tmp_path / "tb" / "t", "test/acc")
    assert steps == [5, 10, 15, 20]


def test_runs_without_tensorboard(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
    assert run(BASE, tmp_path / "t")["status"] == "done"
    assert not (tmp_path / "tb").exists()


def test_tensorboard_disabled(tmp_path):
    run({**BASE, "tensorboard": False}, tmp_path / "t")
    assert not (tmp_path / "tb").exists()


def test_tensorboard_fresh_restart_clears_old_events(tmp_path):
    import shutil

    run(BASE, tmp_path / "t")
    shutil.rmtree(tmp_path / "t")
    run(BASE, tmp_path / "t")
    assert tb_scalars(tmp_path / "tb" / "t", "test/acc") == [10, 20]
