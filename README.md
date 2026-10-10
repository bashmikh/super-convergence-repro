# Super-Convergence: reproduction on a single T4 (Colab)

Reproduction of Smith & Topin, *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates* (arXiv:1708.07120v3) with ResNet-20 on CIFAR-10.

## Run in Colab

1. Upload `colab.ipynb` to Colab (File → Upload notebook, or File → Open notebook → GitHub) and select a T4 runtime.
2. In the Config cell set `REPO_URL`; optionally `OUT` (results on Drive), `DATA`, `DATA_CACHE`, `SCALE`.
3. Runtime → Run all. The notebook mounts Drive, clones the repository to `/content/repo`, installs the package, fetches CIFAR-10 (downloaded once, then copied from the archive cached on Drive), runs `doctor.py` and the smoke test, starts TensorBoard, runs the queue and builds figures. All commands use absolute paths, so any cell works after the Config cell regardless of the current directory.
4. If the smoke-test forecast exceeds the available time, set `SCALE` (e.g. 0.7) before the first experiment run and keep it unchanged. After a disconnect or a runtime restart, Run all again: finished runs are skipped, interrupted runs resume from the last checkpoint.

Results are written to `/content/drive/MyDrive/sc-repro`.

## TensorBoard

Each run writes TensorBoard events to `<out>/tb/<run_id>/`: `train/loss` and `lr` every iteration, `train/loss_ema`, `train/acc_ema`, `test/loss`, `test/acc` at each evaluation. The x-axis is the iteration. `log.json` remains the source for figures and the summary table.

- Colab: run the TensorBoard cell before the experiment cell (`%load_ext tensorboard`, `%tensorboard --logdir {OUT}/tb`); the embedded panel updates while the queue runs. Events on Drive can lag by a minute or two.
- Local: `uv run tensorboard --logdir runs/tb`, then open http://localhost:6006.
- Remote machine over SSH: `ssh -L 6006:localhost:6006 <host>`, start TensorBoard there, open http://localhost:6006 locally.

Set `tensorboard: false` in a config to disable. Without the `tensorboard` package the run proceeds and prints a warning (install with `uv pip install -e ".[tensorboard]"`).

## Local

Requires a CUDA GPU supported by the installed torch build; check with `doctor.py`.

```bash
uv sync
uv run python scripts/doctor.py --out runs
uv run python scripts/run_all.py --out runs                 # full queue
uv run python scripts/run_all.py --out runs --only E0,E1    # selected experiments
uv run python scripts/plots.py --out runs --readme README.md
```

## Development

```bash
uv run pytest
```

## Hypotheses

| ID | Claim | Paper | Experiments |
|---|---|---|---|
| H1 | 1cycle with max LR 3 matches piecewise-constant accuracy in several times fewer iterations | Fig. 1, Fig. 4b | E1 vs E2, E3 |
| H2 | Test accuracy stays high for LR up to ~3 in the LR range test | §3, Fig. 2b, Fig. 8b, Table 6 | E0, E6 |
| H3a | Large LR regularizes: train loss rises while test loss falls for LR ≈ 0.3–2 | §A.2, Fig. 8 | E0 |
| H3b | Regularization must be balanced: WD 1e-3 suppresses super-convergence | §A.6, Fig. 12b | E4, E4b |
| H4 | The gain grows as training data shrinks | §5, Table 1 | E5 |

## Experiments

ResNet-20, batch 1000, SGD momentum 0.9, BN momentum 0.05.

| ID | Schedule | Iterations | WD | Other | Seeds |
|---|---|---|---|---|---|
| E0 | linear 0 → 4 | 1000 | 1e-4 | eval every 50 it on 5000 test images | 0 |
| E1 | 1cycle 0.1 → 3 → 0.1 → 0.001 | 5000 | 1e-4 | | 0, 1 |
| E2 | step 0.35, ×0.1 at 50%, 75% | 20000 | 1e-4 | | 0 |
| E3 | step 0.35 | 5000 | 1e-4 | | 0, 1 |
| E4 | 1cycle, max LR 3 | 5000 | 1e-3, 1e-5, 1e-6 | | 0 |
| E4b | step 0.35 | 5000 | 1e-3 | control for H3b | 0 |
| E5 | 1cycle, step | 5000 | 1e-4 | 10 000 training samples | 0 |
| E6 | 1cycle, max LR 1, 2 | 5000 | 1e-4 | | 0 |

E4b, E5, E6 are optional and run last.

## Learning-rate schedules

- **1cycle** (`onecycle`): the paper's policy (§3) — a single LR cycle shorter than the run, followed by an annealing tail. Here: linear 0.1 → `max_lr` over 45% of iterations, `max_lr` → 0.1 over the next 45%, 0.1 → 0.001 over the last 10%. Equivalent to `torch.optim.lr_scheduler.OneCycleLR(three_phase=True, anneal_strategy="linear")`.
- **step** (`step`): piecewise-constant baseline (PC-LR); starts at `lr0`, multiplied by 0.1 at 50% and 75% of iterations.
- **linear** (`linear`): LR range test; linear 0 → `max_lr` over the run.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/doctor.py` | Environment check: dependencies, GPU architecture support, fp16 step, configs, output and data directories. Exit code 1 on any failure. |
| `scripts/fetch_data.py` | Downloads CIFAR-10 into `--data`; with `--cache` reuses a valid archive from the cache directory and stores it there after the first download. |
| `scripts/smoke.py` | Short runs of each schedule on synthetic CIFAR-shaped data with an interrupt/resume check; measures s/iter and prints the queue time forecast. |
| `scripts/run_all.py` | Runs the experiment queue in order E0, E1, E3, E4, E2, E4b, E5, E6. Skips finished runs, resumes interrupted ones. `--only E1,E3` restricts the queue, `--scale 0.8` shrinks all iteration budgets. |
| `scripts/train.py` | Runs a single configuration; overrides such as `--wd`, `--max-lr`, `--iters` take precedence over the config. |
| `scripts/lr_range_test.py` | Runs E0 and saves its figure. |
| `scripts/plots.py` | Builds all available figures, `summary.csv`, `summary.md`; with `--readme` updates the Results section. |

Each run writes `<out>/<run_id>/log.json` (config, step log, evaluations, final metrics, status), `ckpt.pt` and TensorBoard events in `<out>/tb/<run_id>/`, and prints a progress line at each evaluation. Status is one of `running`, `done`, `diverged`, `oom`, `failed`; `done` and `diverged` are final.

## Config fields

`configs/<ID>.yaml` defines one experiment. Fields not set take the defaults below.

| Field | Default | Meaning |
|---|---|---|
| `id` | — | Experiment ID; prefix of run IDs. |
| `seeds` | `[0]` | One run per seed. |
| `sweep` | — | Map of field → list of values; one run per combination (Cartesian product), e.g. `sweep: {wd: [1.0e-3, 1.0e-5]}` gives runs `E4_wd1e-3_s0`, `E4_wd1e-5_s0`. |
| `schedule` | `onecycle` | `onecycle`, `step` or `linear`. |
| `max_lr` | 3.0 | Peak LR for `onecycle`, final LR for `linear`. |
| `base_lr` | 0.1 | Start and end of the `onecycle` cycle. |
| `final_lr` | 0.001 | `onecycle` LR at the last iteration. |
| `pct` | 0.45 | `onecycle` only: fraction of iterations for each cycle phase (rise, then fall); the tail takes `1 − 2·pct` (10% by default). Must be < 0.5. Same meaning as `pct_start` in `OneCycleLR`. |
| `lr0` | 0.35 | Initial LR for `step`. |
| `iters` | 5000 | Training iterations (budget). |
| `batch_size` | 1000 | Batch size. |
| `wd` | 1e-4 | Weight decay (L2, all parameters). |
| `momentum` | 0.9 | SGD momentum. |
| `bn_momentum` | 0.05 | BatchNorm running-statistics momentum. |
| `depth` | 20 | ResNet depth, 6n+2. |
| `train_subset` | all | Fixed random subset of training images (seed 0). |
| `eval_every` | 250 | Test evaluation interval, iterations. |
| `eval_subset` | all | Evaluate on the first N test images. |
| `ckpt_every` | 500 | Checkpoint interval, iterations. |
| `log_every` | 10 | Step-log interval, iterations. |
| `amp` | true | fp16 autocast on CUDA. |
| `tensorboard` | true | Write TensorBoard events to `<out>/tb/<run_id>/`. |

Floats in YAML need a decimal point in the mantissa (`1.0e-4`); values without it are cast on load.

## Differences from the paper

- PyTorch instead of Caffe; one T4 instead of 8 GPUs.
- Budgets reduced: 1cycle to long baseline ratio 1:4 (paper: 10k vs 80k, 1:8).
- Mixed precision (fp16 autocast).
- BN momentum 0.05 (moving_average_fraction 0.95) for both schedules; the paper uses 0.999 for piecewise-constant runs (Table 1). Table 6 suggests 0.85–0.90 for ~5k-iteration runs.
- Constant momentum 0.9; no cyclical momentum.
- 1cycle tail length (10% of iterations, `pct` = 0.45) is our choice; the paper (§3) only requires the cycle to be shorter than the run, with LR then decreasing several orders of magnitude.
- Downsampling shortcut: 3×3 average pooling with padding 1 (paper: padding 0, inconsistent with the 8×8 head).
- Reference ResNet-20 numbers (90.4% CLR vs 88.6% PC-LR, Fig. 4b) come from a ~20k-iteration cycle and are not directly comparable.

## Results

<!-- RESULTS:START -->
Not yet available.
<!-- RESULTS:END -->

Figures: `<out>/figures/`.

## Citation

This repository reproduces:

Leslie N. Smith, Nicholay Topin. *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates.* arXiv:1708.07120v3, 2018. https://arxiv.org/abs/1708.07120

```bibtex
@article{smith2017superconvergence,
  title   = {Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates},
  author  = {Smith, Leslie N. and Topin, Nicholay},
  journal = {arXiv preprint arXiv:1708.07120},
  year    = {2017}
}
```

The code is an independent implementation from the paper's description; no code from the authors is used. Also relevant:

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016, pp. 770–778. — ResNet architecture.
- L. N. Smith. *Cyclical Learning Rates for Training Neural Networks.* WACV 2017. — cyclical learning rates and the LR range test.
