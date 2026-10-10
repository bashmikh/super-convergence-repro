"""Per-iteration learning-rate schedules as pure functions of the iteration index."""
from typing import Callable

Schedule = Callable[[int], float]


def onecycle(max_lr: float, total: int, base_lr: float = 0.1, final_lr: float = 1e-3,
             pct: float = 0.45) -> Schedule:
    """Linear 1cycle: base -> max -> base, then base -> final.

    Phase boundaries match torch.optim.lr_scheduler.OneCycleLR(three_phase=True).
    """
    if not 0 < pct < 0.5:
        raise ValueError(f"pct must be in (0, 0.5), got {pct}")
    e1 = pct * total - 1
    e2 = 2 * pct * total - 2
    e3 = total - 1

    def lr_at(it: int) -> float:
        if it <= e1:
            return base_lr + (max_lr - base_lr) * it / e1
        if it <= e2:
            return max_lr + (base_lr - max_lr) * (it - e1) / (e2 - e1)
        return base_lr + (final_lr - base_lr) * min(1.0, (it - e2) / (e3 - e2))

    return lr_at


def step(lr0: float, total: int, milestones: tuple[float, ...] = (0.5, 0.75),
         gamma: float = 0.1) -> Schedule:
    """Piecewise-constant schedule; decays by `gamma` at each fraction of `total`."""
    bounds = [m * total for m in milestones]
    return lambda it: lr0 * gamma ** sum(it >= b for b in bounds)


def linear(lr_max: float, total: int) -> Schedule:
    """Linear ramp 0 -> lr_max over `total` iterations (LR range test)."""
    return lambda it: lr_max * it / max(total - 1, 1)


def make_schedule(cfg: dict) -> Schedule:
    name, total = cfg["schedule"], cfg["iters"]
    if name == "onecycle":
        return onecycle(cfg["max_lr"], total, cfg["base_lr"], cfg["final_lr"], cfg["pct"])
    if name == "step":
        return step(cfg["lr0"], total)
    if name == "linear":
        return linear(cfg["max_lr"], total)
    raise ValueError(f"unknown schedule: {name}")
