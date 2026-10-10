import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sh(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_smoke_fake(tmp_path):
    r = sh("scripts/smoke.py", "--out", tmp_path, "--tiny", "--device", "cpu", "--time-iters", "20")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "forecast" in r.stdout.lower()


def test_run_all_fake_only_and_skip(tmp_path):
    args = ("scripts/run_all.py", "--out", tmp_path, "--only", "E1", "--scale", "0.004",
            "--fake", "--device", "cpu")
    r = sh(*args)
    assert r.returncode == 0, r.stdout + r.stderr
    log = json.loads((tmp_path / "E1_s0" / "log.json").read_text())
    assert log["status"] == "done" and log["config"]["iters"] == 20
    r2 = sh(*args)
    assert "skip" in r2.stdout


def test_plots_script(tmp_path):
    r = sh("scripts/plots.py", "--out", tmp_path)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "figures" / "schedules.png").exists()


def test_doctor_cpu(tmp_path):
    r = sh("scripts/doctor.py", "--device", "cpu", "--out", tmp_path / "out", "--data", tmp_path / "data")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "0 failed" in r.stdout


def test_doctor_reports_unwritable_out(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    r = sh("scripts/doctor.py", "--device", "cpu", "--out", blocker / "sub", "--data", tmp_path)
    assert r.returncode == 1 and "FAIL" in r.stdout


def test_fetch_data_help():
    r = sh("scripts/fetch_data.py", "--help")
    assert r.returncode == 0 and "--cache" in r.stdout
