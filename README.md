# OMEL — Online Meta-Expert Learning

Online meta-learning combines ideas from meta-learning and online learning to enable rapid adaptation to new tasks in a lifelong learning setting. However, most existing methods rely on explicit task boundaries, which are often unavailable in real-world scenarios such as evolving fraud-pattern detection or user-preference prediction. Experiments show that when task boundaries are ambiguous, the accuracy of online meta-learning can drop by up to 28%. Existing approaches primarily address this issue through heuristic boundary detection or threshold-based parameter adjustment, which are often unreliable under non-stationary environments.

We overcome this limitation by proposing an **Online Meta-Expert Learning (OMEL)** framework. More specifically, OMEL dynamically updates a set of multi-scale experts using multiplicative weight updates, enabling the model to adapt to evolving data distributions at different temporal scales. To balance rapid adaptation and robustness, OMEL integrates a global meta-parameter with the expert ensemble and employs online distillation to transfer knowledge back to the meta-learner. 

## Repository layout

| Path | Role |
|------|------|
| `OMEL.py` | Unified entry: `--mode` selects `train_MNIST.py` / `train_CIFAR.py` / `train_ECNY.py`|
| `train_MNIST.py` | Rainbow-MNIST image stream. |
| `train_CIFAR.py` | Online CIFAR-10 image pair stream (requires `--tasks_pkl`). |
| `train_ECNY.py` | Online ECNY transaction stream. |
| `plot.py` | Plot curves from `pt/omel_results_*.pt`. |
| `pt/` | Default directory for saved `omel_results_*_seed*.pt`. |

## Requirements

- **Python** 3.8+ recommended  
- **Core:** PyTorch, torchvision, NumPy, Pillow, Matplotlib

## Data

Due to their large size, the Rainbow-MNIST, Online-CIFAR-10, and Online-ECNY datasets are hosted separately via anonymous download links:

- **Rainbow-MNIST:** [Rainbow-MNIST](https://www.kaggle.com/datasets/anonymouskaggledata/rainbow-mnist-zip)
- **Online-CIFAR-10:** [Online-CIFAR-10](https://www.kaggle.com/datasets/anonymouskaggledata/online-cifar-10)
- **Online-ECNY：**.[Online-ECNY](https://www.kaggle.com/datasets/anonymouskaggledata/online-ecny).


## Quick start (training)

From the repository root:

```bash
activate your environment
cd /path/to/OMEL

# Rainbow-MNIST
python OMEL.py --mode mnist --seed 43 --tasks_root ./Rainbow-MNIST/tasks

# Online-CIFAR-10
python OMEL.py --mode cifar --seed 43 \
  --tasks_pkl ./Online-CIFAR-10/online_cifar10_tasks_1200_unique.pkl \
  --tar_path ./Online-CIFAR-10/cifar-10-python.tar.gz

# Online-ECNY
python OMEL.py --mode ecny --data_dir ./Online-ECNY
```



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

