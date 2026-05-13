from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import transforms
from PIL import Image

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class RainbowMNISTTaskDataset(Dataset):

    def __init__(self, root_dir: str, task_id: int, transform=None):
        self.root_dir = root_dir
        self.task_id = task_id
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        task_dir = os.path.join(root_dir, str(task_id))
        if not os.path.isdir(task_dir):
            raise FileNotFoundError(f"Task directory not found: {task_dir}")
        for digit in range(10):
            digit_dir = os.path.join(task_dir, str(digit))
            if not os.path.isdir(digit_dir):
                continue
            for fname in sorted(os.listdir(digit_dir)):
                if fname.lower().endswith(".png"):
                    path = os.path.join(digit_dir, fname)
                    self.samples.append((path, digit))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for task {task_id} in {task_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(
                (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
                 .view(img.size[1], img.size[0], 3)
                 .numpy().astype("float32") / 255.0)
            ).permute(2, 0, 1)
        return img, label


def build_all_tasks(root_dir: str) -> List[int]:
    task_ids: List[int] = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.isdigit():
            task_ids.append(int(name))
    task_ids.sort()
    return task_ids


class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        c = 32
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(32 * 28 * 28, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def build_task_dataset(tasks_root: str, task_id: int) -> RainbowMNISTTaskDataset:
    tfm = transforms.Compose([transforms.ToTensor()])
    return RainbowMNISTTaskDataset(tasks_root, task_id, transform=tfm)


def split_task_train_test_by_digit(
    dataset: RainbowMNISTTaskDataset,
    test_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    digit_to_indices: Dict[int, List[int]] = {d: [] for d in range(10)}
    for idx, (_, digit) in enumerate(dataset.samples):
        digit_to_indices[int(digit)].append(idx)
    train_indices: List[int] = []
    test_indices: List[int] = []
    for d in range(10):
        indices = digit_to_indices[d]
        if not indices:
            continue
        rng = random.Random(seed + int(dataset.task_id) * 1000 + d)
        indices_shuf = indices[:]
        rng.shuffle(indices_shuf)
        n_test = int(len(indices_shuf) * test_ratio)
        test_d = indices_shuf[:n_test]
        train_d = indices_shuf[n_test:]
        test_indices.extend(test_d)
        train_indices.extend(train_d)
    test_indices.sort()
    rng_train_order = random.Random(seed + int(dataset.task_id) * 10000 + 999)
    train_indices_shuf = train_indices[:]
    rng_train_order.shuffle(train_indices_shuf)
    return Subset(dataset, train_indices_shuf), Subset(dataset, test_indices)


def truncate_subset(subset: Subset, max_n: int) -> Subset:
    return Subset(subset.dataset, subset.indices[:max_n])


def eval_on_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    accs: List[float] = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            accs.append(accuracy(logits, y))
    if not accs:
        return 0.0
    return sum(accs) / len(accs)


def _select_with_cap(indices: List[int], n: int, rng: random.Random) -> List[int]:
    if n <= 0 or not indices:
        return []
    if n >= len(indices):
        out = indices[:]
        rng.shuffle(out)
        return out
    return rng.sample(indices, n)


def make_blurry_subset(
    curr_train: Subset,
    prev_train: Optional[Subset],
    next_train: Optional[Subset],
    overlap_ratio: float,
    seed: int,
    task_pos: int,
) -> Subset:
    rng = random.Random(seed + task_pos * 100003 + 97)
    n_total = len(curr_train.indices)
    if n_total == 0:
        return curr_train
    overlap_ratio = max(0.0, min(0.9, float(overlap_ratio)))
    n_mix = int(round(n_total * overlap_ratio))
    n_curr = max(1, n_total - n_mix)
    n_prev, n_next = 0, 0
    if prev_train is not None and next_train is not None:
        n_prev = n_mix // 2
        n_next = n_mix - n_prev
    elif prev_train is not None:
        n_prev = n_mix
    elif next_train is not None:
        n_next = n_mix
    curr_pick = _select_with_cap(curr_train.indices, n_curr, rng)
    prev_pick = _select_with_cap(prev_train.indices if prev_train is not None else [], n_prev, rng)
    next_pick = _select_with_cap(next_train.indices if next_train is not None else [], n_next, rng)
    curr_ds = curr_train.dataset
    prev_ds = prev_train.dataset if prev_train is not None else curr_ds
    next_ds = next_train.dataset if next_train is not None else curr_ds
    concat_ds = ConcatDataset([curr_ds, prev_ds, next_ds])
    off_curr = 0
    off_prev = len(curr_ds)
    off_next = len(curr_ds) + len(prev_ds)
    merged: List[int] = []
    merged.extend([off_curr + i for i in curr_pick])
    merged.extend([off_prev + i for i in prev_pick])
    merged.extend([off_next + i for i in next_pick])
    rng.shuffle(merged)
    if not merged:
        merged = [off_curr + curr_train.indices[0]]
    return Subset(concat_ds, merged)


@dataclass
class BuiltStream:
    ordered_task_ids: List[int]
    task_train: Dict[int, Subset]
    task_test: Dict[int, Subset]
    stream_index: List[Tuple[int, int]]


def build_task_stream(
    tasks_root: str,
    *,
    seed: int,
    test_ratio: float,
    max_train_seen: int,
    overlap_ratio: float = 0.0,
) -> BuiltStream:
    all_task_ids = build_all_tasks(tasks_root)
    rng = random.Random(seed)
    ordered = all_task_ids[:]
    rng.shuffle(ordered)
    base_train: Dict[int, Subset] = {}
    task_test: Dict[int, Subset] = {}
    for tid in ordered:
        ds = build_task_dataset(tasks_root, tid)
        ds_train, ds_test = split_task_train_test_by_digit(ds, test_ratio=test_ratio, seed=seed)
        base_train[tid] = truncate_subset(ds_train, max_train_seen)
        task_test[tid] = ds_test
    task_train: Dict[int, Subset] = {}
    if overlap_ratio > 0.0:
        for pos, tid in enumerate(ordered, start=1):
            prev_tid = ordered[pos - 2] if pos - 2 >= 0 else None
            next_tid = ordered[pos] if pos < len(ordered) else None
            task_train[tid] = make_blurry_subset(
                curr_train=base_train[tid],
                prev_train=base_train[prev_tid] if prev_tid is not None else None,
                next_train=base_train[next_tid] if next_tid is not None else None,
                overlap_ratio=overlap_ratio,
                seed=seed,
                task_pos=pos,
            )
    else:
        task_train = base_train
    stream_index: List[Tuple[int, int]] = []
    for tid in ordered:
        for gidx in list(task_train[tid].indices):
            stream_index.append((tid, int(gidx)))
    return BuiltStream(
        ordered_task_ids=ordered,
        task_train=task_train,
        task_test=task_test,
        stream_index=stream_index,
    )


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
    x_sup: torch.Tensor,
    y_sup: torch.Tensor,
    inner_lr: float,
    create_graph: bool,
) -> Dict[str, torch.Tensor]:
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    names = list(params.keys())
    logits = functional_call(model, (params, buffers), (x_sup,))
    loss = loss_fn(logits, y_sup)
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
    x_sup: torch.Tensor,
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
            x_sup=x_sup,
            y_sup=y_sup,
            inner_lr=inner_lr,
            create_graph=create_graph,
        )
    return out


def evaluate_and_record(
    model: nn.Module,
    test_subset: Subset,
    device: torch.device,
    eval_batch_size: int,
) -> float:
    loader = DataLoader(test_subset, batch_size=eval_batch_size, shuffle=False)
    return float(eval_on_loader(model, loader, device))


def supervised_update_on_batch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    xb: torch.Tensor,
    yb: torch.Tensor,
    n_grad_steps: int,
    batch_size: int,
) -> None:
    if n_grad_steps <= 0:
        return
    if xb.numel() == 0:
        return
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    n = int(xb.shape[0])
    bs = max(1, min(int(batch_size), n))
    for _ in range(n_grad_steps):
        idx = torch.randint(0, n, (bs,), device=xb.device)
        x_m = xb.index_select(0, idx)
        y_m = yb.index_select(0, idx)
        optimizer.zero_grad()
        logits = model(x_m)
        loss = loss_fn(logits, y_m)
        loss.backward()
        optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--task_ids", type=str, default="")
    parser.add_argument("--test_ratio", type=float, default=0.3)
    parser.add_argument("--max_train_seen", type=int, default=900)
    parser.add_argument("--tasks_root", type=str, default="")
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--support_ratio", type=float, default=0.5)
    parser.add_argument("--inner_lr", type=float, default=0.05)
    parser.add_argument("--meta_lr", type=float, default=3e-4)
    parser.add_argument("--lambda_distill", type=float, default=0.1)
    parser.add_argument("--expert_eta", type=float, default=2.0)
    parser.add_argument("--inner_batch_size", type=int, default=10)
    parser.add_argument("--n_supervised_steps", type=int, default=20)
    parser.add_argument("--expert_inner_steps", type=int, default=1)
    parser.add_argument("--meta_inner_steps", type=int, default=1)
    parser.add_argument("--distill_warmup_rounds", type=int, default=200)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--print_eval", action="store_true")
    parser.add_argument("--overlap_ratio", type=float, default=0.2)
    parser.add_argument("--results_dir", type=str, default="")
    args = parser.parse_args()

    comparison_dir = os.path.dirname(os.path.abspath(__file__))
    if args.tasks_root.strip():
        tasks_root = os.path.normpath(os.path.abspath(os.path.expanduser(args.tasks_root.strip())))
    else:
        env_root = os.environ.get("OMEL_MNIST_TASKS_ROOT", "").strip()
        if env_root:
            tasks_root = os.path.normpath(os.path.abspath(os.path.expanduser(env_root)))
        else:
            tasks_root = os.path.normpath(
                os.path.join(comparison_dir, "Rainbow-MNIST")
            )
    print(f"tasks_root={tasks_root}")
    default_pt_dir = os.path.join(comparison_dir, "pt")
    results_dir = os.path.abspath(args.results_dir.strip()) if args.results_dir.strip() else default_pt_dir
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    requested_indices = {int(x.strip()) for x in args.task_ids.split(",") if x.strip()}
    built = build_task_stream(
        tasks_root,
        seed=args.seed,
        test_ratio=args.test_ratio,
        max_train_seen=args.max_train_seen,
        overlap_ratio=args.overlap_ratio,
    )
    ordered_task_ids = built.ordered_task_ids
    T = len(ordered_task_ids)
    if requested_indices:
        plot_task_ids = [i for i in sorted(requested_indices) if 1 <= i <= T]
    else:
        plot_task_ids = list(range(1, T + 1))
    index_to_task_id = {i: ordered_task_ids[i - 1] for i in plot_task_ids}
    task_id_to_index = {tid: i for i, tid in enumerate(ordered_task_ids, start=1)}
    if len(plot_task_ids) <= 20:
        print(f"T={T}, plot indices={plot_task_ids}")
    else:
        print(f"T={T}, plot indices count={len(plot_task_ids)} (range {plot_task_ids[0]}..{plot_task_ids[-1]})")
    if len(index_to_task_id) <= 20:
        print(f"index->task_id: {index_to_task_id}")
    else:
        items = list(index_to_task_id.items())
        print(f"index->task_id (head): {dict(items[:5])} ... (tail): {dict(items[-3:])}")
    print(
        f"OMEL cfg: window(B)={args.window_size}, step={args.step_size}, support_ratio={args.support_ratio}, "
        f"inner_lr={args.inner_lr}, meta_lr={args.meta_lr}, lambda={args.lambda_distill}, "
        f"eta={args.expert_eta}, inner_bs={args.inner_batch_size}, sup_steps={args.n_supervised_steps}, "
        f"expert_inner_steps={args.expert_inner_steps}, meta_inner_steps={args.meta_inner_steps}, "
        f"warmup={args.distill_warmup_rounds}, overlap={args.overlap_ratio}, eval_int={args.eval_interval}"
    )
    print(f"results_dir={results_dir}")

    model = BaselineCNN().to(device)
    optimizer_meta = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    optimizer_inner = torch.optim.SGD(model.parameters(), lr=args.inner_lr)
    loss_task = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_kldiv = nn.KLDivLoss(reduction="batchmean")

    experts: Dict[str, Expert] = {}
    weights: Dict[str, float] = {}
    prior_weights: Dict[str, float] = {}
    regret_sq_state: Dict[str, float] = {}
    eta_state: Dict[str, float] = {}
    results: Dict[int, Dict[str, float]] = {}

    seen_by_index: Dict[int, int] = {i: 0 for i in range(1, T + 1)}
    eval_dict_by_index: Dict[int, Dict[int, float]] = {i: {} for i in range(1, T + 1)}
    reach_40_by_index: Dict[int, Optional[int]] = {i: None for i in range(1, T + 1)}
    last_eval_acc_by_index: Dict[int, float] = {}
    last_eval_seen_by_index: Dict[int, int] = {}

    stream = built.stream_index
    step = max(1, int(args.step_size))
    B = max(2, int(args.window_size))

    last_meta = 0.0
    last_task = 0.0
    last_distill = 0.0
    last_active = 0

    t = 1
    cursor = 0
    while cursor < len(stream):
        window = stream[cursor : min(len(stream), cursor + B)]
        cursor += step

        xs: List[torch.Tensor] = []
        ys: List[int] = []
        for tid, gidx in window:
            x_i, y_i = built.task_train[tid].dataset[gidx]
            xs.append(x_i)
            ys.append(int(y_i))

            idx = task_id_to_index[tid]
            seen_by_index[idx] += 1

        if len(xs) < 2:
            t += 1
            continue

        xb = torch.stack(xs, dim=0).to(device)
        yb = torch.tensor(ys, dtype=torch.long, device=device)

        n = xb.shape[0]
        n_support = int(round(n * float(args.support_ratio)))
        n_support = max(1, min(n - 1, n_support))
        x_sup, y_sup = xb[:n_support], yb[:n_support]
        x_qry, y_qry = xb[n_support:], yb[n_support:]

        model.train()
        buffers = {n: b for n, b in model.named_buffers()}

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
                x_sup=x_sup,
                y_sup=y_sup,
                inner_lr=args.inner_lr,
                n_steps=int(args.expert_inner_steps),
                create_graph=False,
            )
            with torch.no_grad():
                logits_q = functional_call(model, (p1, buffers), (x_qry,))
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
                logits_ens = model(x_qry)

        best_eid = min(active_ids, key=lambda eid: expert_losses[eid]) if active_ids else None
        teacher_logits = expert_logits[best_eid] if best_eid is not None else logits_ens.detach()

        theta = {n: p for n, p in model.named_parameters()}
        theta_tilde = k_step_adapt(
            model=model,
            params=theta,
            buffers=buffers,
            x_sup=x_sup,
            y_sup=y_sup,
            inner_lr=args.inner_lr,
            n_steps=int(args.meta_inner_steps),
            create_graph=True,
        )
        logits_tilde = functional_call(model, (theta_tilde, buffers), (x_qry,))
        l_task = loss_task(logits_tilde, y_qry)
        l_distill = loss_kldiv(
            F.log_softmax(logits_tilde, dim=1),
            F.softmax(teacher_logits.detach(), dim=1),
        )
        lambda_eff = args.lambda_distill if t > args.distill_warmup_rounds else 0.0
        l_meta = l_task + lambda_eff * l_distill

        optimizer_meta.zero_grad()
        l_meta.backward()
        optimizer_meta.step()

        if args.n_supervised_steps and args.n_supervised_steps > 0:
          
            supervised_update_on_batch(
                model=model,
                optimizer=optimizer_inner,
                xb=xb,
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
            acc = evaluate_and_record(model, built.task_test[tid], device, args.eval_batch_size)
            model.train()
            eval_dict_by_index[idx][seen_eff] = float(acc)
            last_eval_acc_by_index[idx] = float(acc)
            last_eval_seen_by_index[idx] = int(seen_eff)
            if reach_40_by_index[idx] is None and acc >= 0.4:
                reach_40_by_index[idx] = seen_eff
            
            if args.print_eval:
                print(f"[OMEL][eval] t={t} idx={idx} task_id={tid} seen={seen_eff} acc={acc:.4f}")

        if t % 50 == 0:
            
            if last_eval_acc_by_index:
                mean_last = sum(last_eval_acc_by_index.values()) / max(1, len(last_eval_acc_by_index))
                ex_idx = sorted(last_eval_acc_by_index.keys())[0]
                ex_seen = last_eval_seen_by_index.get(ex_idx, -1)
                ex_acc = last_eval_acc_by_index.get(ex_idx, float("nan"))
                print(
                    f"[OMEL] t={t}, cursor={min(cursor, len(stream))}/{len(stream)}, "
                    f"active_experts={last_active}, meta={last_meta:.4f}, "
                    f"mean_last_acc={mean_last:.4f}, example(idx={ex_idx},seen={ex_seen})={ex_acc:.4f}"
                )
            else:
                print(
                    f"[OMEL] t={t}, cursor={min(cursor, len(stream))}/{len(stream)}, "
                    f"active_experts={last_active}, meta={last_meta:.4f}"
                )

        t += 1

   
    for idx in plot_task_ids:
        ed = eval_dict_by_index[idx]

        def _closest_acc(target_seen: int) -> float:
            if not ed:
                return float("nan")
            k0 = min(ed.keys(), key=lambda k: abs(k - target_seen))
            return float(ed[k0])

        samples_for_40 = float(reach_40_by_index[idx] if reach_40_by_index[idx] is not None else min(seen_by_index[idx], args.max_train_seen))
        acc_after_100 = _closest_acc(100)
        acc_after_900 = _closest_acc(int(args.max_train_seen))
        acc_test = acc_after_900

        results[idx] = {
            "samples_for_40": samples_for_40,
            "acc_after_100": float(acc_after_100),
            "acc_after_900": float(acc_after_900),
            "acc_test": float(acc_test),
            "error_test": float((1.0 - acc_test) * 100.0) if acc_test == acc_test else float("nan"),
            "meta_loss": last_meta,
            "task_loss": last_task,
            "distill_loss": last_distill,
            "n_active_experts": float(last_active),
        }
        print(f"[OMEL] done idx={idx}, task_id={index_to_task_id[idx]}, seen={seen_by_index[idx]}, acc@{args.max_train_seen}={acc_after_900}")

    
    out_path = os.path.join(results_dir, f"omel_results_mnist_seed{args.seed}.pt")
    torch.save(
        {
            "results": results,
            "task_ids": plot_task_ids,
            "n_stream_tasks": int(T),
            "index_to_task_id": index_to_task_id,
            "seed": args.seed,
            "data_name": "mnist",
            "overlap_ratio": args.overlap_ratio,
            "window_size": args.window_size,
            "support_ratio": args.support_ratio,
            "step_size": args.step_size,
            "eval_interval": args.eval_interval,
        },
        out_path,
    )
    print(f"[OMEL] Success, results saved to: {out_path}")


if __name__ == "__main__":
    main()

