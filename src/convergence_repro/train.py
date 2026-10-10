"""Single training run: SGD + AMP, per-iteration LR, atomic checkpoints, JSON log."""
import json
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import augment, fake_cifar, load_cifar10, normalize
from .models import resnet_cifar
from .schedules import make_schedule

DEFAULTS = {
    "depth": 20,
    "schedule": "onecycle",
    "max_lr": 3.0,
    "lr0": 0.35,
    "base_lr": 0.1,
    "final_lr": 1e-3,
    "pct": 0.45,
    "iters": 5000,
    "batch_size": 1000,
    "wd": 1e-4,
    "momentum": 0.9,
    "bn_momentum": 0.05,
    "seed": 0,
    "train_subset": None,
    "eval_every": 250,
    "eval_subset": None,
    "ckpt_every": 500,
    "log_every": 10,
    "amp": True,
    "fake_data": False,
    "data_root": "data",
    "device": None,
    "tensorboard": True,
}
FINISHED = ("done", "diverged")
_RESUME_IGNORED = ("data_root", "device", "tensorboard")
_EMA = 0.98


def loss_fn(logits, y):
    return F.cross_entropy(logits, y)


def _atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj: dict) -> None:
    _atomic(Path(path), lambda p: p.write_text(json.dumps(obj)))


def read_log(out_dir) -> dict | None:
    path = Path(out_dir) / "log.json"
    return json.loads(path.read_text()) if path.exists() else None


def open_writer(log_dir: Path, purge_step: int | None):
    """TensorBoard SummaryWriter, or None if the tensorboard package is missing."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        print("tensorboard not installed; skipping TensorBoard logging", flush=True)
        return None
    return SummaryWriter(str(log_dir), purge_step=purge_step)


def _comparable(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k not in _RESUME_IGNORED}


@torch.no_grad()
def evaluate(model, x, y, batch_size: int, amp: bool) -> tuple[float, float]:
    """Mean cross-entropy and accuracy without augmentation."""
    model.eval()
    cuda = x.device.type == "cuda"
    loss = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for i in range(0, len(y), batch_size):
        xb = normalize(x[i:i + batch_size])
        if cuda:
            xb = xb.contiguous(memory_format=torch.channels_last)
        yb = y[i:i + batch_size]
        with torch.autocast(x.device.type, dtype=torch.float16, enabled=amp):
            out = model(xb)
        loss += F.cross_entropy(out.float(), yb, reduction="sum")
        correct += (out.argmax(1) == yb).sum()
    model.train()
    return loss.item() / len(y), correct.item() / len(y)


def run(cfg: dict, out_dir, stop_at: int | None = None) -> dict:
    """Train one configuration in `out_dir`; resume from `ckpt.pt` if present."""
    cfg = {**DEFAULTS, **cfg}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path, ckpt_path = out_dir / "log.json", out_dir / "ckpt.pt"
    prev = read_log(out_dir)
    if prev and prev["status"] in FINISHED:
        return prev

    device = torch.device(cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu"))
    cuda = device.type == "cuda"
    amp = cfg["amp"] and cuda
    log = {
        "run_id": cfg.get("run_id", out_dir.name), "config": cfg, "seed": cfg["seed"],
        "status": "running", "steps": {"it": [], "lr": [], "mom": [], "loss": []},
        "evals": [], "final": {},
        "env": {"torch": torch.__version__,
                "device": torch.cuda.get_device_name(device) if cuda else "cpu"},
    }
    tb = None

    try:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        if cuda:
            torch.backends.cudnn.benchmark = True
        data = (fake_cifar(device) if cfg["fake_data"]
                else load_cifar10(cfg["data_root"], device, cfg["train_subset"]))
        n, bs, iters = len(data.y_train), cfg["batch_size"], cfg["iters"]
        if bs > n:
            raise ValueError(f"batch_size {bs} exceeds train size {n}")
        steps_per_epoch = n // bs
        x_test, y_test = data.x_test, data.y_test
        if cfg["eval_subset"]:
            x_test, y_test = x_test[:cfg["eval_subset"]], y_test[:cfg["eval_subset"]]

        model = resnet_cifar(cfg["depth"], cfg["bn_momentum"]).to(device)
        if cuda:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=cfg["momentum"],
                              weight_decay=cfg["wd"], nesterov=False)
        scaler = torch.amp.GradScaler(device.type, enabled=amp)
        lr_at = make_schedule(cfg)
        gen = torch.Generator(device=device)
        gen.manual_seed(cfg["seed"])
        st = {"it": 0, "pos": steps_per_epoch, "wall": 0.0, "ema_loss": None, "ema_acc": None}
        perm = None

        if ckpt_path.exists():
            ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if _comparable(ck["config"]) != _comparable(cfg):
                raise RuntimeError(f"{out_dir}: config differs from checkpoint; "
                                   "delete the directory to restart")
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            scaler.load_state_dict(ck["scaler"])
            st, log = ck["state"], ck["log"]
            log["config"] = cfg
            perm = ck["perm"].to(device) if ck["perm"] is not None else None
            gen.set_state(ck["gen"])
            torch.set_rng_state(ck["torch_rng"])
            np.random.set_state(ck["np_rng"])
            if cuda and ck["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(ck["cuda_rng"])

        if cfg["tensorboard"]:
            tb_dir = out_dir.parent / "tb" / out_dir.name
            if not st["it"]:
                shutil.rmtree(tb_dir, ignore_errors=True)
            # Events after the last checkpoint are written again on resume; purge them.
            tb = open_writer(tb_dir, st["it"] + 1 if st["it"] else None)

        buf: list[torch.Tensor] = []
        t0, wall0 = time.time(), st["wall"]

        def wall() -> float:
            return wall0 + time.time() - t0

        def flush() -> bool:
            """Sync buffered batch stats, update EMA and step log; False on divergence.

            Isolated non-finite losses (fp16 overflow, step skipped by GradScaler) are ignored.
            """
            if not buf:
                return True
            vals = torch.stack(buf).cpu().tolist()
            buf.clear()
            finite = [(l, a) for l, a in vals if math.isfinite(l)]
            if len(finite) < len(vals) and (
                    not finite or not all(torch.isfinite(p).all() for p in model.parameters())):
                return False
            if tb:
                first = st["it"] - len(vals) + 1
                for k, (loss, _) in enumerate(vals):
                    if math.isfinite(loss):
                        tb.add_scalar("train/loss", loss, first + k)
                    tb.add_scalar("lr", lr_at(first + k - 1), first + k)
            for loss, acc in finite:
                st["ema_loss"] = loss if st["ema_loss"] is None else _EMA * st["ema_loss"] + (1 - _EMA) * loss
                st["ema_acc"] = acc if st["ema_acc"] is None else _EMA * st["ema_acc"] + (1 - _EMA) * acc
            s = log["steps"]
            s["it"].append(st["it"])
            s["lr"].append(lr_at(st["it"] - 1))
            s["mom"].append(cfg["momentum"])
            s["loss"].append(vals[-1][0])
            return True

        def do_eval() -> None:
            loss, acc = evaluate(model, x_test, y_test, bs, amp)
            log["evals"].append({
                "it": st["it"], "lr": lr_at(max(st["it"] - 1, 0)),
                "train_loss_ema": st["ema_loss"], "train_acc_ema": st["ema_acc"],
                "test_loss": loss, "test_acc": acc, "wall": wall(),
            })
            write_json(log_path, log)
            print(f"  {log['run_id']}  it {st['it']}/{iters}  lr {lr_at(max(st['it'] - 1, 0)):.4g}  "
                  f"test_acc {acc:.4f}  {wall():.0f}s", flush=True)
            if tb:
                it = st["it"]
                tb.add_scalar("test/loss", loss, it)
                tb.add_scalar("test/acc", acc, it)
                if st["ema_loss"] is not None:
                    tb.add_scalar("train/loss_ema", st["ema_loss"], it)
                    tb.add_scalar("train/acc_ema", st["ema_acc"], it)
                tb.flush()

        def save_ckpt() -> None:
            st["wall"] = wall()
            obj = {
                "model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "state": dict(st), "perm": perm, "gen": gen.get_state(),
                "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if cuda else None,
                "log": log, "config": cfg,
            }
            _atomic(ckpt_path, lambda p: torch.save(obj, p))
            write_json(log_path, log)

        def finish(status: str) -> dict:
            accs = [e["test_acc"] for e in log["evals"]]
            log["status"] = status
            log["final"] = {
                "test_acc": accs[-1], "best_test_acc": max(accs), "total_time": wall(),
                "sec_per_iter": wall() / max(st["it"], 1),
                "diverged_at": st["it"] if status == "diverged" else None,
            }
            write_json(log_path, log)
            return log

        while st["it"] < iters:
            if st["pos"] >= steps_per_epoch:
                perm = torch.randperm(n, device=device, generator=gen)
                st["pos"] = 0
            idx = perm[st["pos"] * bs:(st["pos"] + 1) * bs]
            st["pos"] += 1
            x = augment(normalize(data.x_train[idx]), gen)
            if cuda:
                x = x.contiguous(memory_format=torch.channels_last)
            y = data.y_train[idx]
            lr = lr_at(st["it"])
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            buf.append(torch.stack([loss.detach().float(), (logits.argmax(1) == y).float().mean()]))
            st["it"] += 1
            it = st["it"]

            at_eval = it % cfg["eval_every"] == 0 or it == iters
            at_ckpt = it % cfg["ckpt_every"] == 0 and it < iters
            at_stop = stop_at is not None and it >= stop_at and it < iters
            if it % cfg["log_every"] == 0 or at_eval or at_ckpt or at_stop:
                if not flush():
                    do_eval()
                    return finish("diverged")
            if at_eval:
                do_eval()
            if at_ckpt or at_stop:
                save_ckpt()
            if at_stop:
                return log
        return finish("done")

    except torch.cuda.OutOfMemoryError:
        _write_failure(log_path, log, prev, "oom")
        print(f"[{log['run_id']}] CUDA out of memory at batch_size={cfg['batch_size']}; "
              "reduce --batch-size", flush=True)
        raise
    except Exception:
        _write_failure(log_path, log, prev, "failed")
        raise
    finally:
        if tb:
            tb.close()


def _write_failure(path: Path, log: dict, prev: dict | None, status: str) -> None:
    """Record failure status without discarding evaluations already on disk."""
    if prev and len(prev["evals"]) > len(log["evals"]):
        log = prev
    log["status"] = status
    write_json(path, log)
