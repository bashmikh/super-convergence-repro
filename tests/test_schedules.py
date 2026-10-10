import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from convergence_repro.schedules import linear, make_schedule, onecycle, step


def reference_onecycle(max_lr, total):
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=total, pct_start=0.45, anneal_strategy="linear",
        three_phase=True, cycle_momentum=False, div_factor=max_lr / 0.1, final_div_factor=100,
    )
    lrs = []
    for it in range(total):
        lrs.append(opt.param_groups[0]["lr"])
        if it < total - 1:
            opt.step()
            sched.step()
    return lrs


@pytest.mark.parametrize("max_lr,total", [(3.0, 5000), (1.0, 1000), (2.0, 777)])
def test_onecycle_matches_torch(max_lr, total):
    f = onecycle(max_lr, total)
    ref = reference_onecycle(max_lr, total)
    assert max(abs(f(i) - r) for i, r in enumerate(ref)) < 1e-6


def test_onecycle_key_points():
    f = onecycle(3.0, 5000)
    assert f(0) == pytest.approx(0.1)
    assert max(f(i) for i in range(5000)) == pytest.approx(3.0)
    assert f(4999) == pytest.approx(1e-3)
    assert f(10_000) == pytest.approx(1e-3)


def test_step():
    f = step(0.35, 1000)
    assert f(0) == f(499) == pytest.approx(0.35)
    assert f(500) == f(749) == pytest.approx(0.035)
    assert f(750) == f(999) == pytest.approx(0.0035)


def test_linear():
    f = linear(4.0, 1001)
    assert f(0) == 0.0
    assert f(500) == pytest.approx(2.0)
    assert f(1000) == pytest.approx(4.0)


def test_make_schedule_dispatch():
    base = {"iters": 100, "max_lr": 3.0, "lr0": 0.35, "base_lr": 0.1, "final_lr": 1e-3, "pct": 0.45}
    assert make_schedule({**base, "schedule": "step"})(0) == pytest.approx(0.35)
    assert make_schedule({**base, "schedule": "linear"})(0) == 0.0
    assert make_schedule({**base, "schedule": "onecycle"})(0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        make_schedule({**base, "schedule": "cosine"})


def test_schedule_plot(tmp_path):
    fig, ax = plt.subplots()
    for name, f, n in [("onecycle", onecycle(3.0, 5000), 5000), ("step", step(0.35, 5000), 5000)]:
        ax.plot(range(n), [f(i) for i in range(n)], label=name)
    ax.legend()
    fig.savefig(tmp_path / "schedules.png")
    assert (tmp_path / "schedules.png").stat().st_size > 0


@pytest.mark.parametrize("pct", [0.0, 0.5, 0.6])
def test_onecycle_rejects_bad_pct(pct):
    with pytest.raises(ValueError):
        onecycle(3.0, 1000, pct=pct)
