# OMEL — Online Meta-Expert Learning

Code for **Online Meta-Expert Learning (OMEL)**: online meta-learning with a **multi-scale expert ensemble**, **multiplicative expert weight updates**, and **online distillation** back to a global meta-learner.

## Overview

Online meta-learning combines ideas from meta-learning and online learning to enable rapid adaptation to new tasks in a lifelong learning setting. However, most existing methods rely on explicit task boundaries, which are often unavailable in real-world scenarios such as evolving fraud-pattern detection or user-preference prediction. Experiments show that when task boundaries are ambiguous, the accuracy of online meta-learning can drop by up to 28%. Existing approaches primarily address this issue through heuristic boundary detection or threshold-based parameter adjustment, which are often unreliable under non-stationary environments.

We overcome this limitation by proposing an **Online Meta-Expert Learning (OMEL)** framework. More specifically, OMEL dynamically updates a set of multi-scale experts using multiplicative weight updates, enabling the model to adapt to evolving data distributions at different temporal scales. To balance rapid adaptation and robustness, OMEL integrates a global meta-parameter with the expert ensemble and **employs online distillation** to transfer knowledge back to the meta-learner. We show that OMEL achieves a sublinear fixed-window dynamic regret of $\tilde{O}(\sqrt{\tau})$. Experiments on image classification and transaction anomaly detection demonstrate strong empirical performance.

## Repository layout

| Path | Role |
|------|------|
| `OMEL.py` | Unified entry: `--mode` selects `train_MNIST.py` / `train_CIFAR.py` / `train_ECNY.py`; remaining CLI flags are forwarded. |
| `train_MNIST.py` | Rainbow-MNIST image stream. |
| `train_CIFAR.py` | Online CIFAR-10 pair stream (requires `--tasks_pkl`). |
| `train_ECNY.py` | Online ECNY graph / transaction stream. |
| `plot.py` | Plot curves from `pt/omel_results_*.pt`. |
| `pt/` | Default directory for saved `omel_results_*_seed*.pt` (created if missing). |

## Requirements

- **Python** 3.8+ recommended  
- **Core:** PyTorch, torchvision, NumPy, Pillow, Matplotlib (for `plot.py`)  
- **ECNY mode:** PyTorch Geometric and project data under your `--data_dir` (see below)

Install dependencies in your environment (example with conda):

```bash
conda activate your_env
pip install torch torchvision matplotlib numpy pillow
# For ECNY / GNN backends, also install torch-geometric and its dependencies.
```

## Data

- **MNIST (`--mode mnist`):** Rainbow-MNIST task folders; default search uses `Rainbow-MNIST/` next to the repo or `OMEL_MNIST_TASKS_ROOT`.
- **CIFAR (`--mode cifar`):** Online task pickle (`--tasks_pkl`) and CIFAR-10 archive (`--tar_path`).
- **ECNY (`--mode ecny`):** Directory with features / labels / graph files (see `train_ECNY.py` / `stream_aml.py` for expected layout).

Adjust paths to match your machine.

## Quick start (training)

From the repository root:

```bash
# Optional: activate your environment
# source /path/to/anaconda3/bin/activate your_env
cd /path/to/OMEL

# Rainbow-MNIST
python OMEL.py --mode mnist --seed 43 --tasks_root ./Rainbow-MNIST/tasks

# Online CIFAR-10
python OMEL.py --mode cifar --seed 43 \
  --tasks_pkl ./Online-CIFAR-10/online_cifar10_tasks_1200_unique.pkl \
  --tar_path ./Online-CIFAR-10/cifar-10-python.tar.gz

# Online ECNY
python OMEL.py --mode ecny --data_dir ./Online-ECNY
```

Each training script accepts additional hyperparameters; run `python OMEL.py --mode <mnist|cifar|ecny> --help` to see the forwarded script’s options (OMEL itself only documents `--mode` in its own `--help`).

## Plotting

After runs finish, checkpoints live under `./pt` by default. Example plots:

```bash
cd /path/to/OMEL
python3 plot.py --domain mnist --results_dir ./pt --seeds 43 --smooth_window 5
python3 plot.py --domain cifar --results_dir ./pt --seeds 43 --smooth_window 20
python3 plot.py --domain ecny --results_dir ./pt --seeds 43 --smooth_window 8
```

## Outputs

- **`pt/omel_results_<domain>_seed<SEED>.pt`:** Torch checkpoints with histories / metrics (exact keys depend on the domain script).

