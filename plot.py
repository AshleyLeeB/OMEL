from __future__ import annotations
import argparse
import math
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import torch

_PLOT_SCRIPT_DIR = Path(__file__).resolve().parent
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".pdf", ".svg"})


def _resolve_plot_out_dir(out_arg: str, default_dir: Path) -> Path:
    default_dir = default_dir.expanduser().resolve()
    s = (out_arg or "").strip()
    if not s:
        return default_dir
    p = Path(s).expanduser().resolve()
    if p.is_dir():
        return p
    if p.suffix.lower() in _IMAGE_SUFFIXES:
        return p.parent
    return p


def _figure_output_path_png(out_dir: Path, dataset_tag: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (dataset_tag or "plot").strip().lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{tag}_{ts}.png"


def _load_pt(path: Path) -> Dict[str, Any]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        try:
            return torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(str(path), map_location="cpu")


def _moving_average_nan(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return y.astype(float, copy=True)
    half = window // 2
    y = y.astype(float)
    out = np.full_like(y, np.nan, dtype=float)
    n = len(y)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = y[lo:hi]
        if np.all(np.isnan(seg)):
            continue
        out[i] = float(np.nanmean(seg))
    return out


def _moving_tail_avg(values: List[float], window: int) -> List[float]:
    if window <= 1:
        return values[:]
    out: List[float] = []
    for i in range(len(values)):
        left = max(0, i - window + 1)
        chunk = [v for v in values[left : i + 1] if not math.isnan(v)]
        if not chunk:
            out.append(math.nan)
        else:
            out.append(sum(chunk) / len(chunk))
    return out


def _parse_seeds(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _ensure_odd(w: int, label: str) -> int:
    if w > 1 and w % 2 == 0:
        w += 1
        print(f"[plot] {label} adjusted to odd: {w}")
    return w


def _configure_plot_style_mnist() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "legend.title_fontsize": 11,
        }
    )


def _mnist_load_result(path: str) -> Dict[int, Any]:
    data = _load_pt(Path(path))
    return {int(k): v for k, v in data["results"].items()}


def _mnist_get_metric(v: Any, key: str) -> float:
    if isinstance(v, dict):
        raw = v.get(key)
        if raw is None and key == "acc_after_900":
            raw = v.get("acc_test")
        if raw is None:
            return float("nan")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return float("nan")
    if isinstance(v, (float, int)):
        if key in ("acc_after_900", "acc_after_100"):
            return float(v)
        return float("nan")
    return float("nan")


def _mnist_final_classification_error_pct(v: Any) -> float:
    if isinstance(v, dict):
        et = v.get("error_test")
        if et is not None:
            return float(et)
    acc = _mnist_get_metric(v, "acc_after_900")
    if math.isnan(acc):
        return float("nan")
    return (1.0 - acc) * 100.0


def _mean_and_sem(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    valid = ~np.isnan(values)
    counts = valid.sum(axis=0)
    if values.shape[0] <= 1:
        sem = np.zeros_like(mean, dtype=np.float64)
    else:
        std = np.nanstd(values, axis=0, ddof=1)
        sem = std / np.sqrt(np.maximum(counts, 1))
    sem[counts <= 1] = 0.0
    return mean, sem


def _mnist_collect_matrix(
    base_dir: Path,
    base_name: str,
    seeds: List[int],
    task_ids: List[int],
    compute_fn: Callable[[Any], float],
) -> Optional[np.ndarray]:
    run_paths: List[Path] = []
    for s in seeds:
        p = base_dir / f"{base_name}_seed{s}.pt"
        if p.is_file():
            run_paths.append(p)
    if not run_paths:
        legacy = base_dir / f"{base_name}.pt"
        if legacy.is_file():
            run_paths = [legacy]
        else:
            return None
    values = np.full((len(run_paths), len(task_ids)), np.nan, dtype=np.float32)
    for i, p in enumerate(run_paths):
        res = _mnist_load_result(str(p))
        for j, tid in enumerate(task_ids):
            if tid not in res:
                continue
            values[i, j] = compute_fn(res[tid])
    return values


def _mnist_resolve_omel_base(results_dir: Path, seeds: List[int]) -> str:
    for cand in ("omel_results_mnist", "omel_results"):
        for s in seeds:
            if (results_dir / f"{cand}_seed{s}.pt").is_file():
                return cand
    return "omel_results_mnist"


def _mnist_omel_pt_paths(results_dir: Path, omel_base: str, seeds: List[int]) -> List[Path]:
    paths: List[Path] = []
    for s in seeds:
        p = results_dir / f"{omel_base}_seed{s}.pt"
        if p.is_file():
            paths.append(p)
    return paths


def _mnist_infer_task_ids(paths: List[Path], task_ids_arg: str) -> List[int]:
    if task_ids_arg.strip():
        return sorted({int(x.strip()) for x in task_ids_arg.split(",") if x.strip()})
    if not paths:
        raise FileNotFoundError("No OMEL MNIST .pt files found")
    sets: List[set[int]] = []
    for p in paths:
        res = _mnist_load_result(str(p))
        sets.append(set(res.keys()))
    union = set().union(*sets) if sets else set()
    if not union:
        raise ValueError("Empty 'results' in .pt file")
    inter = set.intersection(*sets) if len(sets) >= 1 else set()
    if len(sets) >= 2 and len(union) > len(inter):
        print(
            f"[plot][mnist] results key counts per seed: {[len(s) for s in sets]}; "
            f"union {len(union)} tasks, intersection {len(inter)}. Using union for x-axis (missing seed -> NaN)."
        )
    return sorted(union)


def plot_domain_mnist(args: argparse.Namespace) -> Path:
    _configure_plot_style_mnist()
    results_dir = Path(args.results_dir).expanduser().resolve()
    seeds = _parse_seeds(args.seeds)
    omel_base = _mnist_resolve_omel_base(results_dir, seeds)
    paths = _mnist_omel_pt_paths(results_dir, omel_base, seeds)
    if not paths:
        raise FileNotFoundError(
            f"[plot][mnist] no OMEL .pt in {results_dir} (tried {omel_base}_seed*.pt), seeds={seeds}"
        )
    task_ids = _mnist_infer_task_ids(paths, args.task_ids)
    n_seen = int(args.max_train_seen)

    d0 = _load_pt(paths[0])
    n_st = d0.get("n_stream_tasks")
    if n_st is not None and len(task_ids) < int(n_st):
        print(
            f"[plot][mnist] note: stream has T={int(n_st)} tasks but results has only {len(task_ids)} keys; "
            "this .pt is from a subset-eval run. Re-train with all tasks to plot the full axis."
        )

    plt.figure(figsize=(20, 5))
    sw_all = _ensure_odd(int(args.smooth_window), "smooth_window")
    sw_mid = sw_all if sw_all > 1 else _ensure_odd(int(args.smooth_window_center), "smooth_window_center")

    color = "red"
    name = "OMEL"
    markevery = 1 if len(task_ids) <= 30 else max(1, len(task_ids) // 25)

    def _plot_panel(subplot_idx: int, compute_fn: Callable[[Any], float], ylabel: str, title: str, use_sw: int) -> None:
        plt.subplot(1, 3, subplot_idx)
        values = _mnist_collect_matrix(results_dir, omel_base, seeds, task_ids, compute_fn=compute_fn)
        mean, sem = _mean_and_sem(values)
        if np.all(np.isnan(mean)):
            print(f"[plot][mnist] all metrics NaN, skip: {title}")
            return
        mean_plot = _moving_average_nan(mean, use_sw)
        low_plot = _moving_average_nan(mean - sem, use_sw)
        high_plot = _moving_average_nan(mean + sem, use_sw)
        plt.fill_between(task_ids, low_plot, high_plot, color=color, alpha=0.2, linewidth=0)
        plt.plot(
            task_ids,
            mean_plot,
            label=name if subplot_idx == 3 else "_nolegend_",
            color=color,
            marker="o",
            markevery=markevery,
            linewidth=2,
            markersize=5,
        )
        plt.xlabel("Task index")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, alpha=0.3)

    _plot_panel(
        1,
        lambda v: _mnist_get_metric(v, "samples_for_40"),
        "task data needed for 40% acc.",
        "Task Learning Efficiency",
        sw_all,
    )
    _plot_panel(
        2,
        lambda v: (1.0 - _mnist_get_metric(v, "acc_after_100")) * 100.0,
        "Task classification error (%)",
        "Task Performance after 100 datapoints",
        sw_mid,
    )
    _plot_panel(
        3,
        _mnist_final_classification_error_pct,
        "Final task classification error (%)",
        f"Task End Performance ({n_seen} datapoints)",
        sw_all,
    )

    plt.legend(loc="best")
    plt.tight_layout()

    out_dir = _resolve_plot_out_dir(args.out, _PLOT_SCRIPT_DIR)
    out_path = _figure_output_path_png(out_dir, "mnist")
    plt.savefig(str(out_path), dpi=300)
    plt.close()
    print(f"[plot][mnist] saved: {out_path}")
    return out_path


_SERIF_RC_BASE = {
    "font.family": "serif",
    "font.serif": [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "DejaVu Serif",
    ],
}
_SERIF_RC_CIFAR = dict(_SERIF_RC_BASE)
_TITLE_FS = 20
_AXIS_FS = 16
_TICK_FS = 14
_LEGEND_FS = 13
_LINEWIDTH_RAW = 0.85
_ALPHA_RAW = 0.14
_LINEWIDTH_SMOOTH = 2.4


def _cifar_resolve_stem(results_dir: Path, seeds: List[int], slug: str) -> str:
    if slug and slug != "auto":
        return f"omel_results_{slug}"
    for s in seeds or [43]:
        for sp in ("cifar100", "cifar10", "cifar"):
            stem = f"omel_results_{sp}"
            if (results_dir / f"{stem}_seed{s}.pt").is_file():
                return stem
    return "omel_results_cifar10"


def _cifar_task_curve_from_pt(path: Path) -> Tuple[List[int], List[float]]:
    d = _load_pt(path)
    curve = d.get("task_curve") or []
    xs: List[int] = []
    ys: List[float] = []
    for row in curve:
        xs.append(int(row["task_index"]))
        ys.append(float(row["test_error"]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ys[i] for i in order]


def _cifar_mean_curve_over_seeds(
    results_dir: Path, stem: str, seeds: List[int]
) -> Tuple[List[int], List[float]]:
    dfs: List[Tuple[List[int], List[float]]] = []
    for sid in seeds:
        p = results_dir / f"{stem}_seed{sid}.pt"
        if not p.is_file():
            raise FileNotFoundError(f"[plot][cifar] missing {p}")
        dfs.append(_cifar_task_curve_from_pt(p))
    common_x = set(dfs[0][0])
    for xlist, _ in dfs[1:]:
        common_x &= set(xlist)
    if not common_x:
        raise ValueError("No intersection of task_index across seeds")
    x_sorted = sorted(common_x)
    y_mean: List[float] = []
    for xi in x_sorted:
        vals = []
        for xlist, ylist in dfs:
            j = xlist.index(xi)
            vals.append(ylist[j])
        y_mean.append(float(sum(vals) / len(vals)))
    return x_sorted, y_mean


def plot_domain_cifar(args: argparse.Namespace) -> Path:
    results_dir = Path(args.results_dir).expanduser().resolve()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("[plot][cifar] --seeds is required")
    stem = _cifar_resolve_stem(results_dir, seeds, "auto")
    print(f"[plot][cifar] using stem={stem}, seeds={seeds}")

    xs, ys = _cifar_mean_curve_over_seeds(results_dir, stem, seeds)
    sw = max(1, int(args.smooth_window))
    ys_plot = _moving_tail_avg(ys, sw)

    with plt.rc_context(_SERIF_RC_CIFAR):
        fig, ax = plt.subplots(figsize=(10, 7))
        if sw > 1 and not args.no_raw_underlay:
            ax.plot(
                xs,
                ys,
                color="red",
                linewidth=_LINEWIDTH_RAW,
                alpha=_ALPHA_RAW,
                solid_capstyle="round",
                zorder=2,
                label="_nolegend_",
            )
        ax.plot(
            xs,
            ys_plot,
            color="red",
            linewidth=_LINEWIDTH_SMOOTH,
            alpha=1.0,
            solid_capstyle="round",
            zorder=100,
            label="OMEL",
        )
        ax.set_xlabel("Task Index", fontsize=_AXIS_FS, fontweight="bold")
        ax.set_ylabel("Test Error (%)", fontsize=_AXIS_FS, fontweight="bold")
        ax.set_title(args.title, fontsize=_TITLE_FS, fontweight="bold")
        ax.tick_params(axis="both", which="major", labelsize=_TICK_FS)
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick.set_fontweight("bold")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=4))
        span = max(float(xs[-1] - xs[0]), 1e-9)
        pad = max(0.5, span * 0.02)
        ax.set_xlim(float(xs[0]) - pad, float(xs[-1]) + pad)
        fin = [v for v in ys + ys_plot if not math.isnan(v)]
        if fin:
            lo, hi = min(fin), max(fin)
            span_y = max(hi - lo, 1e-6)
            pad_y = max(span_y * 0.025, 0.35)
            ax.set_ylim(lo - pad_y, hi + pad_y)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            loc="best",
            fontsize=_LEGEND_FS,
            ncol=2,
            labelspacing=0.75,
            handlelength=2.8,
            borderpad=0.9,
        )
        fig.tight_layout()
        dataset_tag = (
            stem.replace("omel_results_", "", 1) if stem.startswith("omel_results_") else stem
        ) or "cifar"
        out_dir = _resolve_plot_out_dir(args.out, _PLOT_SCRIPT_DIR)
        out_path = _figure_output_path_png(out_dir, dataset_tag)
        fig.savefig(str(out_path), dpi=200)
        plt.close(fig)
    print(f"[plot][cifar] saved: {out_path}")
    return out_path


_SERIF_RC_ECNY: Dict[str, Any] = {
    **_SERIF_RC_BASE,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
}


def _format_y_test_error_tick(x: float, _pos: int) -> str:
    p = x * 100.0
    if abs(p - round(p)) < 1e-9:
        return str(int(round(p)))
    return f"{p:g}"


def _rolling_mean_series(y: np.ndarray, window: int) -> np.ndarray:
    w = max(1, int(window))
    if w <= 1:
        return y.astype(float, copy=True)
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        seg = y[lo : i + 1]
        out[i] = float(np.nanmean(seg))
    return out


def _ecny_history_from_pt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    d = _load_pt(path)
    hist = d.get("history") or []
    ts = np.array([int(h["t"]) for h in hist], dtype=np.int64)
    err = np.array([float(h.get("test_error", math.nan)) for h in hist], dtype=np.float64)
    return ts, err


def _ecny_mean_over_seeds(results_dir: Path, seeds: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    stem = "omel_results_ecny"
    per: List[Tuple[np.ndarray, np.ndarray]] = []
    for sid in seeds:
        p = results_dir / f"{stem}_seed{sid}.pt"
        if not p.is_file():
            legacy = results_dir / f"omel_results_seed{sid}.pt"
            if legacy.is_file():
                p = legacy
            else:
                raise FileNotFoundError(f"[plot][ecny] missing {results_dir}/{stem}_seed{sid}.pt")
        per.append(_ecny_history_from_pt(p))
    merged_t = per[0][0]
    for t_arr, _ in per[1:]:
        merged_t = np.array(sorted(set(merged_t.tolist()) & set(t_arr.tolist())), dtype=np.int64)
    if merged_t.size == 0:
        raise ValueError("No intersection of step t across seeds")
    y_stack = []
    for t_arr, e_arr in per:
        mp = {int(t_arr[i]): float(e_arr[i]) for i in range(len(t_arr))}
        y_stack.append(np.array([mp[int(t)] for t in merged_t], dtype=np.float64))
    y_mean = np.mean(np.stack(y_stack, axis=0), axis=0)
    return merged_t, y_mean


def plot_domain_ecny(args: argparse.Namespace) -> Path:
    results_dir = Path(args.results_dir).expanduser().resolve()
    seeds = _parse_seeds(args.seeds)
    if not seeds:
        seeds = [43]
    t, y_raw = _ecny_mean_over_seeds(results_dir, seeds)
    sw = int(args.smooth_window)
    if sw < 0:
        sw = 0
    do_smooth = sw > 1
    y_plot = _rolling_mean_series(y_raw, sw) if do_smooth else y_raw

    with plt.rc_context(_SERIF_RC_ECNY):
        fig, ax = plt.subplots(figsize=(10, 7))
        if do_smooth and not args.no_raw_underlay:
            ax.plot(
                t.astype(float),
                y_raw,
                color="red",
                linewidth=_LINEWIDTH_RAW,
                alpha=_ALPHA_RAW,
                solid_capstyle="round",
                zorder=2,
                label="_nolegend_",
            )
        ax.plot(
            t.astype(float),
            y_plot,
            color="red",
            linewidth=_LINEWIDTH_SMOOTH,
            alpha=1.0,
            solid_capstyle="round",
            zorder=100,
            label="OMEL",
        )
        ax.set_xlabel("Step", fontsize=16, fontweight="bold")
        ax.set_ylabel("Test Error (%)", fontsize=16, fontweight="bold")
        ax.set_title(args.title, fontsize=20, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(_format_y_test_error_tick))
        ax.tick_params(axis="both", which="major", labelsize=14)
        for tick in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            tick.set_fontweight("bold")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, min_n_ticks=4))
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(
            loc="best",
            prop={"family": "serif", "weight": "bold", "size": 13},
            ncol=2,
            labelspacing=0.75,
            handlelength=2.8,
            borderpad=0.9,
        )
        fig.tight_layout()
        out_dir = _resolve_plot_out_dir(args.out, _PLOT_SCRIPT_DIR)
        out_path = _figure_output_path_png(out_dir, "ecny")
        fig.savefig(str(out_path), dpi=int(args.dpi))
        plt.close(fig)
    print(f"[plot][ecny] saved: {out_path}")
    return out_path


def main() -> None:
    default_pt = _PLOT_SCRIPT_DIR / "pt"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        choices=("mnist", "cifar", "ecny"),
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(default_pt),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
    )
    parser.add_argument("--title", type=str, default="")
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--no_raw_underlay",
        action="store_true",
    )
    parser.add_argument(
        "--task_ids",
        type=str,
        default="",
    )
    parser.add_argument(
        "--max_train_seen",
        type=int,
        default=900,
    )
    parser.add_argument(
        "--smooth_window_center",
        type=int,
        default=3,
    )
    args = parser.parse_args()

    if not args.title.strip():
        if args.domain == "cifar":
            args.title = "Online CIFAR (OMEL)"
        elif args.domain == "ecny":
            args.title = "Online-ECNY"

    if args.domain == "mnist":
        plot_domain_mnist(args)
    elif args.domain == "cifar":
        plot_domain_cifar(args)
    else:
        plot_domain_ecny(args)


if __name__ == "__main__":
    main()
