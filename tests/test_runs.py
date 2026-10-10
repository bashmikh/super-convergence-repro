from pathlib import Path

import pytest

from convergence_repro.runs import ORDER, apply_scale, expand, load_config
from convergence_repro.train import DEFAULTS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
NUMERIC = ("max_lr", "lr0", "wd", "iters", "eval_every", "eval_subset", "train_subset")


def test_all_configs_exist():
    assert sorted(p.stem for p in CONFIGS.glob("*.yaml")) == sorted(ORDER)


@pytest.mark.parametrize("exp_id", ORDER)
def test_config_types(exp_id):
    for cfg in expand(load_config(CONFIGS / f"{exp_id}.yaml")):
        assert set(cfg) - {"exp", "run_id"} <= set(DEFAULTS)
        for k in NUMERIC:
            if k in cfg and cfg[k] is not None:
                assert isinstance(cfg[k], (int, float)), (exp_id, k, cfg[k])


def test_expand_sweep_and_seeds():
    exp = {"id": "E4", "schedule": "onecycle", "seeds": [0, 1], "sweep": {"wd": [1.0e-3, 1.0e-5]}}
    runs = expand(exp)
    assert [r["run_id"] for r in runs] == ["E4_wd1e-3_s0", "E4_wd1e-3_s1", "E4_wd1e-5_s0", "E4_wd1e-5_s1"]
    assert all(r["exp"] == "E4" and "sweep" not in r and "seeds" not in r for r in runs)


def test_expand_plain_and_string_sweep():
    assert [r["run_id"] for r in expand({"id": "E1", "seeds": [0, 1]})] == ["E1_s0", "E1_s1"]
    runs = expand({"id": "E5", "sweep": {"schedule": ["onecycle", "step"]}})
    assert [r["run_id"] for r in runs] == ["E5_scheduleonecycle_s0", "E5_schedulestep_s0"]
    assert [r["run_id"] for r in expand({"id": "E6", "sweep": {"max_lr": [1.0, 2.0]}})] == [
        "E6_max_lr1_s0", "E6_max_lr2_s0"]


def test_apply_scale():
    cfg = {"iters": 5000, "eval_every": 250}
    s = apply_scale(cfg, 0.8)
    assert s["iters"] == 4000 and s["eval_every"] == 200 and s["ckpt_every"] == 400
    assert apply_scale(cfg, 1.0) == cfg
    assert apply_scale({"iters": 3}, 0.1)["iters"] == 1


def test_yaml_exponent_without_dot(tmp_path):
    p = tmp_path / "E9.yaml"
    p.write_text("id: E9\nwd: 1e-4\nseeds: [0]\nsweep:\n  max_lr: [1e0, 2]\n")
    runs = expand(load_config(p))
    assert runs[0]["wd"] == 1e-4 and isinstance(runs[0]["max_lr"], float)
    assert runs[0]["run_id"] == "E9_max_lr1_s0"
