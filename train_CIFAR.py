"""OMEL training on online CIFAR-10 pair tasks (requires --tasks_pkl)."""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from cifar10_online_stream import load_built_stream_from_pickle


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetEncoder(nn.Module):
    """ResNet-18 style encoder for CIFAR-size images."""

    def __init__(self, blocks_per_stage: List[int] | None = None, emb_dim: int = 256) -> None:
        super().__init__()
        if blocks_per_stage is None:
            blocks_per_stage = [2, 2, 2, 2]

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.in_channels = 64
        self.layer1 = self._make_stage(64, blocks_per_stage[0], stride=1)
        self.layer2 = self._make_stage(128, blocks_per_stage[1], stride=2)
        self.layer3 = self._make_stage(256, blocks_per_stage[2], stride=2)
        self.layer4 = self._make_stage(512, blocks_per_stage[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(512, emb_dim)

    def _make_stage(self, out_channels: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        z = self.proj(x)
        return F.normalize(z, dim=1, eps=1e-6)


class ResNetPairNet(nn.Module):
    """
    Shared ResNet encoder + pair head.
    Forward returns logits for BCEWithLogitsLoss (same=1, diff=0).
    """

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.encoder = ResNetEncoder(emb_dim=emb_dim)
        feat_dim = emb_dim * 4  # z1, z2, |z1-z2|, z1*z2
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        feat = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=1)
        return self.head(feat).squeeze(1)


def pair_accuracy_sigmoid(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets.float()).float().mean().item()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return pair_accuracy_sigmoid(logits, targets)


def test_error_percent(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((1.0 - accuracy(logits, targets)) * 100.0)


def _average_precision_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = int(y_true.size)
    pos = int(y_true.sum())
    if pos == 0:
        return float("nan")
    if pos == n:
        return 1.0
    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / pos
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    inds = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[inds + 1] - mrec[inds]) * mpre[inds + 1]))


def eval_pair_acc_and_auprc_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    logits_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()
            logits = model(x1, x2)
            logits_parts.append(logits.reshape(-1).detach().cpu())
            y_parts.append(y.reshape(-1).detach().cpu())
    if not logits_parts:
        return 0.0, float("nan")
    logits_cat = torch.cat(logits_parts, dim=0).numpy().astype(np.float64)
    y_cat = torch.cat(y_parts, dim=0).numpy().astype(np.float64)
    y_bin = (y_cat >= 0.5).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits_cat))
    acc = float(((probs >= 0.5).astype(np.int64) == y_bin).mean())
    ap = _average_precision_binary(y_bin, probs)
    return acc, float(ap)


def eval_pair_acc_and_auprc_on_loader_with_params(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    logits_parts: List[torch.Tensor] = []
    y_parts: List[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for x1, x2, y in loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()
            logits = functional_call(model, (params, buffers), (x1, x2))
            logits_parts.append(logits.reshape(-1).detach().cpu())
            y_parts.append(y.reshape(-1).detach().cpu())
    if not logits_parts:
        return 0.0, float("nan")
    logits_cat = torch.cat(logits_parts, dim=0).numpy().astype(np.float64)
    y_cat = torch.cat(y_parts, dim=0).numpy().astype(np.float64)
    y_bin = (y_cat >= 0.5).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits_cat))
    acc = float(((probs >= 0.5).astype(np.int64) == y_bin).mean())
    ap = _average_precision_binary(y_bin, probs)
    return acc, float(ap)


@dataclass
class Expert:
    start: int
    end: int
    params: Dict[str, torch.Tensor]
    weight: float


def clone_param_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def normalize_weights(weights: Dict[str, float], active_ids: List[str]) -> Dict[str, float]:
    vals = [max(1e-12, float(weights.get(eid, 0.0))) for eid in active_ids]
    s = sum(vals)
    if s <= 0:
        base = 1.0 / max(1, len(active_ids))
        return {eid: base for eid in active_ids}
    return {eid: v / s for eid, v in zip(active_ids, vals)}


def _eta_from_prior_and_regret(pi: float, regret_sq: float) -> float:
    pi = max(1e-12, float(pi))
    regret_sq = max(0.0, float(regret_sq))
    return min(0.5, math.sqrt(max(1e-12, math.log(1.0 / pi)) / (1.0 + regret_sq)))


def adapt_ml_prod(
    prev_weights: Dict[str, float],
    active_ids: List[str],
    losses: Dict[str, float],
    prior_weights: Dict[str, float],
    regret_sq_state: Dict[str, float],
    eta_state: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    if not active_ids:
        return prev_weights, regret_sq_state, eta_state

    p_t = normalize_weights(prev_weights, active_ids)
    m_t = 0.0
    for eid in active_ids:
        m_t += float(p_t[eid]) * float(losses.get(eid, 0.0))

    out = dict(prev_weights)
    new_regret_sq = dict(regret_sq_state)
    new_eta_state = dict(eta_state)

    for eid in active_ids:
        pi_e = max(1e-12, float(prior_weights.get(eid, p_t[eid])))
        old_r2 = float(regret_sq_state.get(eid, 0.0))
        old_eta = float(eta_state.get(eid, _eta_from_prior_and_regret(pi_e, old_r2)))

        l_e = float(losses.get(eid, 0.0))
        r_e = m_t - l_e
        new_r2 = old_r2 + r_e * r_e
        new_eta = _eta_from_prior_and_regret(pi_e, new_r2)

        base = max(1e-12, float(prev_weights.get(eid, pi_e)))
        factor = max(1e-12, 1.0 + old_eta * r_e)
        out[eid] = (base * factor) ** (new_eta / max(old_eta, 1e-12))

        new_regret_sq[eid] = new_r2
        new_eta_state[eid] = new_eta

    norm = normalize_weights(out, active_ids)
    out.update(norm)
    return out, new_regret_sq, new_eta_state

def one_step_adapt(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    x1_sup: torch.Tensor,
    x2_sup: torch.Tensor,
    y_sup: torch.Tensor,
    inner_lr: float,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    loss_fn = nn.BCEWithLogitsLoss()
    names = list(params.keys())
    logits = functional_call(model, (params, buffers), (x1_sup, x2_sup))
    loss = loss_fn(logits, y_sup.float())
    grads = torch.autograd.grad(
        loss,
        [params[n] for n in names],
        create_graph=create_graph,
        allow_unused=False,
    )
    return {n: params[n] - inner_lr * g for n, g in zip(names, grads)}


def k_step_adapt(
    model: nn.Module,
    params: Dict[str, torch.Tensor],
    buffers: Dict[str, torch.Tensor],
    x1_sup: torch.Tensor,
    x2_sup: torch.Tensor,
    y_sup: torch.Tensor,
    inner_lr: float,
    n_steps: int,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    out = params
    steps = max(1, int(n_steps))
    for _ in range(steps):
        out = one_step_adapt(
            model=model,
            params=out,
            buffers=buffers,
            x1_sup=x1_sup,
            x2_sup=x2_sup,
            y_sup=y_sup,
            inner_lr=inner_lr,
            create_graph=create_graph,
        )
    return out


def evaluate_and_record(
    model: nn.Module,
    train_subset: Subset,
    seen_count: int,
    test_subset: Subset,
    device: torch.device,
    eval_batch_size: int,
    inner_lr: float,
    n_adapt_steps: int,
    buffers: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[float, float]:
    seen_eff = max(0, min(int(seen_count), len(train_subset)))
    test_loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False)
    if buffers is None:
        buffers = {name: buf for name, buf in model.named_buffers()}
    if seen_eff <= 0 or int(n_adapt_steps) <= 0:
        return eval_pair_acc_and_auprc_on_loader(model, test_loader, device)

    support_subset = Subset(train_subset.dataset, list(train_subset.indices[:seen_eff]))
    support_loader = DataLoader(support_subset, batch_size=seen_eff, shuffle=False)
    try:
        x1_sup, x2_sup, y_sup = next(iter(support_loader))
    except StopIteration:
        return eval_pair_acc_and_auprc_on_loader(model, test_loader, device)

    x1_sup = x1_sup.to(device)
    x2_sup = x2_sup.to(device)
    y_sup = y_sup.to(device).float()
    params = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    adapted = k_step_adapt(
        model=model,
        params=params,
        buffers=buffers,
        x1_sup=x1_sup,
        x2_sup=x2_sup,
        y_sup=y_sup,
        inner_lr=inner_lr,
        n_steps=int(n_adapt_steps),
        create_graph=False,
    )
    return eval_pair_acc_and_auprc_on_loader_with_params(model, adapted, buffers, test_loader, device)

def supervised_update_on_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x1b: torch.Tensor,
    x2b: torch.Tensor,
    yb: torch.Tensor,
    n_grad_steps: int,
    batch_size: int,
) -> None:
    if n_grad_steps <= 0:
        return
    if x1b.numel() == 0:
        return
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    n = int(x1b.shape[0])
    bs = max(1, min(int(batch_size), n))
    for _ in range(n_grad_steps):
        idx = torch.randint(0, n, (bs,), device=x1b.device)
        x1_m = x1b.index_select(0, idx)
        x2_m = x2b.index_select(0, idx)
        y_m = yb.index_select(0, idx).float()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x1_m, x2_m)
        loss = loss_fn(logits, y_m)
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--tasks_pkl", type=str, default="")
    parser.add_argument("--tar_path", type=str, default="")
    parser.add_argument("--num_tasks", type=int, default=1200)
    parser.add_argument("--task_ids", type=str, default="")
    parser.add_argument("--max_train_seen", type=int, default=400)
    parser.add_argument("--window_size", type=int, default=400)
    parser.add_argument("--step_size", type=int, default=400)
    parser.add_argument("--support_ratio", type=float, default=0.5)
    parser.add_argument("--inner_lr", type=float, default=0.01)
    parser.add_argument("--meta_lr", type=float, default=1e-4)
    parser.add_argument("--lambda_distill", type=float, default=0.4)
    parser.add_argument("--inner_batch_size", type=int, default=16)
    parser.add_argument("--n_supervised_steps", type=int, default=10)
    parser.add_argument("--expert_inner_steps", type=int, default=3)
    parser.add_argument("--meta_inner_steps", type=int, default=2)
    parser.add_argument("--eval_inner_steps", type=int, default=-1)
    parser.add_argument("--distill_warmup_rounds", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--overlap_ratio", type=float, default=0.0)
    parser.add_argument("--results_dir", type=str, default="")
    args = parser.parse_args()

    print_done_every = 10
    task_curve_print_every = 50

    if os.environ.get("OMEL_EXPERT_INNER_STEPS", "").strip():
        args.expert_inner_steps = int(os.environ["OMEL_EXPERT_INNER_STEPS"].strip())
    if os.environ.get("OMEL_META_INNER_STEPS", "").strip():
        args.meta_inner_steps = int(os.environ["OMEL_META_INNER_STEPS"].strip())

    comparison_dir = os.path.dirname(os.path.abspath(__file__))
    if not args.tasks_pkl.strip():
        raise SystemExit("train_CIFAR.py requires --tasks_pkl (path to the online task-stream .pkl).")
    tasks_pkl = os.path.abspath(os.path.expanduser(args.tasks_pkl.strip()))
    print(f"tasks_pkl={tasks_pkl}")
    tar_override = os.path.abspath(os.path.expanduser(args.tar_path.strip())) if args.tar_path.strip() else None
    default_pt_dir = os.path.join(comparison_dir, "pt")
    results_dir = os.path.abspath(args.results_dir.strip()) if args.results_dir.strip() else default_pt_dir
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    built = load_built_stream_from_pickle(
        tasks_pkl,
        tar_path=tar_override,
        max_train_seen=int(args.max_train_seen),
        overlap_ratio=float(args.overlap_ratio),
        blurry_seed=int(args.seed),
    )
    data_cfg = built.bundle["config"]
    data_slug = "cifar10"
    ordered_task_ids_all = built.ordered_task_ids
    total_tasks = len(ordered_task_ids_all)
    if int(args.num_tasks) <= 0:
        running_tasks = total_tasks
    else:
        running_tasks = min(int(args.num_tasks), total_tasks)
    print(f"total_tasks_in_pkl={total_tasks}, running_tasks={running_tasks}")

    ordered_task_ids = ordered_task_ids_all[:running_tasks]
    T = len(ordered_task_ids)
    allowed_task_ids = set(ordered_task_ids)

    stream = [(tid, gidx) for tid, gidx in built.stream_index if tid in allowed_task_ids]
    if args.task_ids.strip():
        requested_indices = {int(x.strip()) for x in args.task_ids.split(",") if x.strip()}
        plot_task_ids = [i for i in sorted(requested_indices) if 1 <= i <= T]
    else:
        plot_task_ids = list(range(1, T + 1))
    index_to_task_id = {i: ordered_task_ids[i - 1] for i in plot_task_ids}
    task_id_to_index = {tid: i for i, tid in enumerate(ordered_task_ids, start=1)}
    if plot_task_ids:
        print(f"T={T}, evaluating {len(plot_task_ids)} tasks, task_index_range=[{plot_task_ids[0]},{plot_task_ids[-1]}]")
    else:
        print(f"T={T}, evaluating 0 tasks")
    print(
        f"OMEL cfg: window(B)={args.window_size}, step={args.step_size}, support_ratio={args.support_ratio}, "
        f"inner_lr={args.inner_lr}, meta_lr={args.meta_lr}, lambda={args.lambda_distill}, "
        f"inner_bs={args.inner_batch_size}, sup_steps={args.n_supervised_steps}, "
        f"expert_inner_steps={args.expert_inner_steps}, meta_inner_steps={args.meta_inner_steps}, "
        f"eval_inner_steps={args.eval_inner_steps}, "
        f"warmup={args.distill_warmup_rounds}, "
        f"overlap={args.overlap_ratio}, eval_int={args.eval_interval}, "
        f"max_train_seen={args.max_train_seen}"
    )
    _con = data_cfg.get("construction", "")
    if _con == "random_5class_sampled_pools":
        _extra = (
            f"construction={_con}, classes_per_task={data_cfg.get('classes_per_task')}, "
            f"pool_per_class={data_cfg.get('pool_per_class')}"
        )
    elif _con:
        _extra = f"construction={_con}, window={data_cfg.get('window_size_classes')}"
    else:
        _extra = f"keep_prev={data_cfg.get('keep_prev')}, new_per_task={data_cfg.get('new_per_task')}"
    print(
        f"data (from pkl): num_tasks={data_cfg.get('num_tasks')}, seed={data_cfg.get('seed')}, "
        f"{_extra}, "
        f"train_pairs=({data_cfg.get('train_same_pairs')},{data_cfg.get('train_diff_pairs')}), "
        f"test_pairs=({data_cfg.get('test_same_pairs')},{data_cfg.get('test_diff_pairs')})"
    )
    print(f"tasks_pkl={tasks_pkl}")
    print(f"tar_path={tar_override or data_cfg.get('tar_path')}")
    print(f"results_dir={results_dir}")
    print(
        f"[train] online_order=list_order_in_pkl | T={T} | stream_len={len(stream)} | "
        f"first_task_ids={ordered_task_ids[:3]}...{ordered_task_ids[-1]}"
    )

    model = ResNetPairNet().to(device)
    model_buffers = dict(model.named_buffers())
    optimizer_meta = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    optimizer_inner = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
    loss_task = nn.BCEWithLogitsLoss()

    experts: Dict[str, Expert] = {}
    weights: Dict[str, float] = {}
    prior_weights: Dict[str, float] = {}
    regret_sq_state: Dict[str, float] = {}
    eta_state: Dict[str, float] = {}
    results: Dict[int, Dict[str, float]] = {}
    seen_by_index: Dict[int, int] = {i: 0 for i in range(1, T + 1)}
    eval_dict_by_index: Dict[int, Dict[int, float]] = {i: {} for i in range(1, T + 1)}
    eval_auprc_dict_by_index: Dict[int, Dict[int, float]] = {i: {} for i in range(1, T + 1)}
    reach_target_acc_by_index: Dict[int, Optional[int]] = {i: None for i in range(1, T + 1)}
    last_eval_acc_by_index: Dict[int, float] = {}
    last_eval_auprc_by_index: Dict[int, float] = {}
    last_eval_seen_by_index: Dict[int, int] = {}

    step = max(1, int(args.step_size))
    B = max(2, int(args.window_size))
    eval_adapt_steps = (
        max(1, int(args.meta_inner_steps)) if args.eval_inner_steps < 0 else int(args.eval_inner_steps)
    )

    last_meta = 0.0
    last_task = 0.0
    last_distill = 0.0
    last_active = 0
    stream_history: List[Dict[str, float]] = []

    t = 1
    cursor = 0
    while cursor < len(stream):
        window = stream[cursor : min(len(stream), cursor + B)]
        cursor += step
        window_task_ids = [tid for tid, _ in window]
        current_task_min = min(window_task_ids) + 1 if window_task_ids else -1
        current_task_max = max(window_task_ids) + 1 if window_task_ids else -1

        x1s: List[torch.Tensor] = []
        x2s: List[torch.Tensor] = []
        ys: List[float] = []
        for tid, gidx in window:
            x1_i, x2_i, y_i = built.task_train[tid].dataset[gidx]
            x1s.append(x1_i)
            x2s.append(x2_i)
            ys.append(float(y_i))

            idx = task_id_to_index[tid]
            seen_by_index[idx] += 1

        if len(x1s) < 2:
            t += 1
            continue

        x1b = torch.stack(x1s, dim=0).to(device)
        x2b = torch.stack(x2s, dim=0).to(device)
        yb = torch.tensor(ys, dtype=torch.float32, device=device)

        n = x1b.shape[0]
        n_support = int(round(n * float(args.support_ratio)))
        n_support = max(1, min(n - 1, n_support))
        x1_sup, x2_sup, y_sup = x1b[:n_support], x2b[:n_support], yb[:n_support]
        x1_qry, x2_qry, y_qry = x1b[n_support:], x2b[n_support:], yb[n_support:]

        model.train()
        buffers = model_buffers

        max_k = int(math.floor(math.log2(max(1, t))))
        for k in range(max_k + 1):
            interval = 2 ** k
            if t % interval == 0:
                start = t
                end = t + interval - 1
                eid = f"{start}_{end}"
                experts[eid] = Expert(
                    start=start,
                    end=end,
                    params=clone_param_dict(model),
                    weight=2.0 ** (-(k + 1)),
                )
                weights[eid] = experts[eid].weight
                prior_weights[eid] = experts[eid].weight
                regret_sq_state[eid] = 0.0
                eta_state[eid] = _eta_from_prior_and_regret(prior_weights[eid], 0.0)

        active_ids = [eid for eid, e in experts.items() if e.start <= t <= e.end]
        active_norm = normalize_weights(weights, active_ids)
        weights.update(active_norm)

        expert_logits: Dict[str, torch.Tensor] = {}
        expert_losses: Dict[str, float] = {}
        for eid in active_ids:
            e = experts[eid]
            p0 = {n: w.detach().clone().requires_grad_(True) for n, w in e.params.items()}
            p1 = k_step_adapt(
                model=model,
                params=p0,
                buffers=buffers,
                x1_sup=x1_sup,
                x2_sup=x2_sup,
                y_sup=y_sup,
                inner_lr=args.inner_lr,
                n_steps=int(args.expert_inner_steps),
                create_graph=False,
            )
            with torch.no_grad():
                logits_q = functional_call(model, (p1, buffers), (x1_qry, x2_qry))
                l_q = loss_task(logits_q, y_qry)
            experts[eid].params = {n: w.detach().clone() for n, w in p1.items()}
            expert_logits[eid] = logits_q.detach()
            expert_losses[eid] = float(l_q.item())

        logits_ens = None
        for eid in active_ids:
            w = float(weights[eid])
            logits_ens = expert_logits[eid] * w if logits_ens is None else logits_ens + expert_logits[eid] * w
        if logits_ens is None:
            with torch.no_grad():
                logits_ens = model(x1_qry, x2_qry)
        stream_err = test_error_percent(logits_ens, y_qry)

        best_eid = min(active_ids, key=lambda eid: expert_losses[eid]) if active_ids else None
        teacher_logits = expert_logits[best_eid] if best_eid is not None else logits_ens.detach()

        theta = {n: p for n, p in model.named_parameters()}
        theta_tilde = k_step_adapt(
            model=model,
            params=theta,
            buffers=buffers,
            x1_sup=x1_sup,
            x2_sup=x2_sup,
            y_sup=y_sup,
            inner_lr=args.inner_lr,
            n_steps=int(args.meta_inner_steps),
            create_graph=True,
        )
        logits_tilde = functional_call(model, (theta_tilde, buffers), (x1_qry, x2_qry))
        l_task = loss_task(logits_tilde, y_qry)
        l_distill = F.binary_cross_entropy_with_logits(logits_tilde, torch.sigmoid(teacher_logits.detach()))
        lambda_eff = args.lambda_distill if t > args.distill_warmup_rounds else 0.0
        l_meta = l_task + lambda_eff * l_distill

        optimizer_meta.zero_grad(set_to_none=True)
        l_meta.backward()
        optimizer_meta.step()

        if args.n_supervised_steps and args.n_supervised_steps > 0:
            supervised_update_on_batch(
                model=model,
                optimizer=optimizer_inner,
                x1b=x1b,
                x2b=x2b,
                yb=yb,
                n_grad_steps=int(args.n_supervised_steps),
                batch_size=int(args.inner_batch_size),
            )

        if active_ids:
            weights, regret_sq_state, eta_state = adapt_ml_prod(
                prev_weights=weights,
                active_ids=active_ids,
                losses=expert_losses,
                prior_weights=prior_weights,
                regret_sq_state=regret_sq_state,
                eta_state=eta_state,
            )

        expired = [eid for eid, e in experts.items() if e.end <= t]
        for eid in expired:
            experts.pop(eid, None)
            weights.pop(eid, None)
            prior_weights.pop(eid, None)
            regret_sq_state.pop(eid, None)
            eta_state.pop(eid, None)

        last_meta = float(l_meta.item())
        last_task = float(l_task.item())
        last_distill = float(l_distill.item())
        last_active = len(active_ids)
        stream_history.append(
            {
                "t": float(t),
                "meta_loss": last_meta,
                "task_loss": last_task,
                "distill_loss": last_distill,
                "test_error": stream_err,
                "n_active_experts": float(last_active),
            }
        )

        for idx in plot_task_ids:
            seen = int(seen_by_index[idx])
            if seen <= 0:
                continue
            seen_eff = min(seen, int(args.max_train_seen))
            interval = max(1, int(args.eval_interval))
            prev_seen = int(last_eval_seen_by_index.get(idx, 0))
            prev_bucket = prev_seen // interval
            curr_bucket = seen_eff // interval
            hit_final = (prev_seen < int(args.max_train_seen)) and (seen_eff >= int(args.max_train_seen))
            do_eval = (curr_bucket > prev_bucket) or hit_final
            if not do_eval:
                continue
            if seen_eff in eval_dict_by_index[idx]:
                continue

            tid = index_to_task_id[idx]
            acc, auprc = evaluate_and_record(
                model,
                built.task_train[tid],
                seen_eff,
                built.task_test[tid],
                device,
                args.eval_batch_size,
                args.inner_lr,
                eval_adapt_steps,
                buffers=model_buffers,
            )
            model.train()
            eval_dict_by_index[idx][seen_eff] = float(acc)
            eval_auprc_dict_by_index[idx][seen_eff] = float(auprc)
            last_eval_acc_by_index[idx] = float(acc)
            last_eval_auprc_by_index[idx] = float(auprc)
            last_eval_seen_by_index[idx] = int(seen_eff)
            if reach_target_acc_by_index[idx] is None and acc >= 0.4:
                reach_target_acc_by_index[idx] = seen_eff

        if t % 50 == 0:
            if last_eval_acc_by_index:
                mean_last = sum(last_eval_acc_by_index.values()) / max(1, len(last_eval_acc_by_index))
                print(
                    f"[OMEL] t={t}, cursor={min(cursor, len(stream))}/{len(stream)}, "
                    f"task_range=[{current_task_min},{current_task_max}], "
                    f"active_experts={last_active}, meta={last_meta:.4f}, test_error={stream_err:.4f}, "
                    f"mean_last_acc={mean_last:.4f}"
                )
            else:
                print(
                    f"[OMEL] t={t}, cursor={min(cursor, len(stream))}/{len(stream)}, "
                    f"task_range=[{current_task_min},{current_task_max}], "
                    f"active_experts={last_active}, meta={last_meta:.4f}, test_error={stream_err:.4f}"
                )

        t += 1

    done_every = max(1, int(print_done_every))
    for idx in plot_task_ids:
        ed = eval_dict_by_index[idx]

        def _closest_acc(target_seen: int) -> float:
            if not ed:
                return float("nan")
            k0 = min(ed.keys(), key=lambda k: abs(k - target_seen))
            return float(ed[k0])

        eda = eval_auprc_dict_by_index[idx]

        def _closest_auprc(target_seen: int) -> float:
            if not eda:
                return float("nan")
            k0 = min(eda.keys(), key=lambda k: abs(k - target_seen))
            return float(eda[k0])

        samples_to_target_acc = float(
            reach_target_acc_by_index[idx]
            if reach_target_acc_by_index[idx] is not None
            else min(seen_by_index[idx], args.max_train_seen)
        )
        acc_at_100_seen = _closest_acc(100)
        acc_at_max_seen = _closest_acc(int(args.max_train_seen))
        acc_test = acc_at_max_seen
        auprc_at_max_seen = _closest_auprc(int(args.max_train_seen))
        auprc_test = auprc_at_max_seen

        results[idx] = {
            "samples_to_target_acc": samples_to_target_acc,
            "target_acc_threshold": 0.4,
            "acc_at_100_seen": float(acc_at_100_seen),
            "acc_at_max_seen": float(acc_at_max_seen),
            "acc_test": float(acc_test),
            "error_test": float((1.0 - acc_test) * 100.0) if acc_test == acc_test else float("nan"),
            "auprc_test": float(auprc_test) if auprc_test == auprc_test else float("nan"),
            "meta_loss": last_meta,
            "task_loss": last_task,
            "distill_loss": last_distill,
            "n_active_experts": float(last_active),
        }
        if idx % done_every == 0 or idx == 1 or idx == T:
            ap_done = f"{auprc_at_max_seen:.4f}" if auprc_at_max_seen == auprc_at_max_seen else "nan"
            print(
                f"[OMEL] done idx={idx}, task_id={index_to_task_id[idx]}, "
                f"seen={seen_by_index[idx]}, acc@{args.max_train_seen}={acc_at_max_seen}, "
                f"auprc@{args.max_train_seen}={ap_done}"
            )

    task_curve: List[Dict[str, float]] = []
    for idx in range(1, T + 1):
        ed = eval_dict_by_index[idx]
        eda = eval_auprc_dict_by_index[idx]
        if ed:
            final_seen = max(ed.keys())
            final_acc = float(ed[final_seen])
            final_err = float((1.0 - final_acc) * 100.0)
            final_auprc = float(eda[final_seen]) if eda and final_seen in eda else float("nan")
        else:
            final_seen = min(int(seen_by_index[idx]), int(args.max_train_seen))
            final_acc = float("nan")
            final_err = float("nan")
            final_auprc = float("nan")
        task_curve.append(
            {
                "task_index": float(idx),
                "task_id": float(ordered_task_ids[idx - 1]),
                "seen": float(final_seen),
                "acc": final_acc,
                "test_error": final_err,
                "binary_auprc": final_auprc,
            }
        )

    pe = max(1, int(task_curve_print_every))
    print("# per-task test error (1 - acc_mid on held-out test), tutorial-style TSV")
    print("task_id\ttest_error")
    for row in task_curve:
        task_idx = int(row["task_index"])
        if task_idx % pe != 0 and task_idx != 1 and task_idx != T:
            continue
        tid = int(row["task_id"])
        acc = float(row["acc"])
        if acc == acc:
            print(f"{tid}\t{(1.0 - acc):.4f}")
        else:
            print(f"{tid}\tnan")

    out_path = os.path.join(results_dir, f"omel_results_{data_slug}_seed{args.seed}.pt")
    torch.save(
        {
            "results": results,
            "task_ids": plot_task_ids,
            "index_to_task_id": index_to_task_id,
            "seed": args.seed,
            "data_name": data_slug,
            "tasks_pkl": tasks_pkl,
            "data_config": dict(data_cfg),
            "overlap_ratio": args.overlap_ratio,
            "window_size": args.window_size,
            "support_ratio": args.support_ratio,
            "step_size": args.step_size,
            "eval_interval": args.eval_interval,
            "cudnn_benchmark": False,
            "expert_inner_steps": int(args.expert_inner_steps),
            "meta_inner_steps": int(args.meta_inner_steps),
            "stream_history": stream_history,
            "task_curve": task_curve,
        },
        out_path,
    )
    print(f"[OMEL] done, saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()

