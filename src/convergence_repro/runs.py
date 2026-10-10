"""Experiment configs: YAML loading, expansion into runs, budget scaling."""
import itertools
from pathlib import Path

import yaml

from .train import DEFAULTS

ORDER = ["E0", "E1", "E3", "E4", "E2", "E4b", "E5", "E6"]
CORE = ["E0", "E1", "E3", "E4", "E2"]
_META = ("id", "seeds", "sweep")
_SCALED = ("iters", "eval_every", "ckpt_every")


_FLOAT_KEYS = ("max_lr", "lr0", "base_lr", "final_lr", "pct", "wd", "momentum", "bn_momentum")


def _coerce(cfg: dict) -> dict:
    """Cast float keys; PyYAML reads `1e-4` (no dot) as a string."""
    out = {k: float(v) if k in _FLOAT_KEYS and v is not None else v for k, v in cfg.items()}
    if out.get("sweep"):
        out["sweep"] = {k: [float(x) for x in v] if k in _FLOAT_KEYS else v
                        for k, v in out["sweep"].items()}
    return out


def load_config(path) -> dict:
    return _coerce(yaml.safe_load(Path(path).read_text()))


def _fmt(v) -> str:
    if isinstance(v, float) and 0 < abs(v) < 0.01:
        mantissa, exp = f"{v:e}".split("e")
        return f"{float(mantissa):g}e{int(exp)}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def expand(exp: dict) -> list[dict]:
    """Cartesian product over `sweep` values and `seeds`; one config per run."""
    base = {k: v for k, v in exp.items() if k not in _META}
    sweep = exp.get("sweep") or {}
    keys = list(sweep)
    runs = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        suffix = "".join(f"_{k}{_fmt(v)}" for k, v in zip(keys, combo))
        for seed in exp.get("seeds", [0]):
            cfg = {**base, **dict(zip(keys, combo)), "seed": seed, "exp": exp["id"]}
            cfg["run_id"] = f"{exp['id']}{suffix}_s{seed}"
            runs.append(cfg)
    return runs


def apply_scale(cfg: dict, scale: float) -> dict:
    """Scale iteration budget and eval/checkpoint intervals proportionally."""
    if scale == 1.0:
        return cfg
    out = dict(cfg)
    for k in _SCALED:
        out[k] = max(1, round(cfg.get(k, DEFAULTS[k]) * scale))
    return out
