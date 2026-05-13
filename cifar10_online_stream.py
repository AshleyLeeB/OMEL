from __future__ import annotations
import os
import pickle
import random
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


def _to_image_tensor(x_flat) -> torch.Tensor:
    x = torch.tensor(x_flat, dtype=torch.float32).view(3, 32, 32) / 255.0
    return x


class CIFAR10PairDataset(Dataset):
    """Same/different binary classification from raw CIFAR-10 pixels and pair indices."""

    def __init__(
        self,
        data_flat: Sequence[Sequence[int]],
        pair_indices: Sequence[Sequence[int]],
        pair_labels: Sequence[int],
    ):
        self.data_flat = data_flat
        self.pair_indices = [[int(p[0]), int(p[1])] for p in pair_indices]
        self.pair_labels = [float(y) for y in pair_labels]

    def __len__(self) -> int:
        return len(self.pair_labels)

    def __getitem__(self, idx: int):
        i, j = self.pair_indices[idx]
        x1 = _to_image_tensor(self.data_flat[i])
        x2 = _to_image_tensor(self.data_flat[j])
        return x1, x2, self.pair_labels[idx]


def _find_member(tar: tarfile.TarFile, suffix: str):
    for m in tar.getmembers():
        if m.name.endswith(suffix):
            return m
    return None


def load_cifar10_from_tar(tar_path: str):
    with tarfile.open(tar_path, "r:gz") as tar:
        train_batches = []
        for i in range(1, 6):
            m = _find_member(tar, f"/data_batch_{i}")
            if m is None:
                raise FileNotFoundError(f"data_batch_{i} not found in tar archive")
            fh = tar.extractfile(m)
            if fh is None:
                raise FileNotFoundError(f"Cannot read tar member {m.name}")
            train_batches.append(pickle.load(fh, encoding="bytes"))

        test_m = _find_member(tar, "/test_batch")
        if test_m is None:
            raise FileNotFoundError("test_batch not found in tar archive")
        test_f = tar.extractfile(test_m)
        if test_f is None:
            raise FileNotFoundError("Failed to open test_batch")
        test = pickle.load(test_f, encoding="bytes")

    train_data = torch.cat([torch.as_tensor(b[b"data"]) for b in train_batches], dim=0).numpy()
    train_labels: List[int] = []
    for b in train_batches:
        train_labels.extend(int(x) for x in b[b"labels"])
    train = {b"data": train_data, b"labels": train_labels}
    return train, test


def truncate_subset(subset: Subset, max_n: int) -> Subset:
    return Subset(subset.dataset, subset.indices[:max_n])


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
    bundle: Dict[str, object]


def load_built_stream_from_pickle(
    pkl_path: str,
    *,
    tar_path: Optional[str] = None,
    max_train_seen: Optional[int] = None,
    overlap_ratio: float = 0.0,
    blurry_seed: int = 0,
) -> BuiltStream:
    """
    Load an online task-stream pickle and CIFAR-10 pixels from the tar into a BuiltStream.
    """
    pkl_path = os.path.abspath(os.path.expanduser(pkl_path))
    with open(pkl_path, "rb") as f:
        bundle = pickle.load(f)

    if not isinstance(bundle, dict) or "config" not in bundle or "tasks" not in bundle:
        raise ValueError(f"Invalid pkl (missing config/tasks): {pkl_path}")

    cfg = bundle["config"]
    tasks = bundle["tasks"]
    if not tasks:
        raise ValueError("tasks list in pkl is empty")

    t0 = tasks[0]
    if "train_pair_indices" not in t0 or "test_pair_indices" not in t0:
        raise ValueError(
            "This pkl is legacy or missing train/test pair fields; regenerate with the current "
            "online CIFAR-10 pairing build."
        )

    tar = tar_path.strip() if tar_path and str(tar_path).strip() else str(cfg.get("tar_path", "")).strip()
    if not tar:
        raise ValueError("tar_path not provided and not set in pkl config")
    tar = os.path.abspath(os.path.expanduser(tar))
    if not os.path.isfile(tar):
        # Cross-machine pkl reuse: if config.tar_path is a stale absolute path,
        # fall back to standard archive names beside this pkl or this script.
        candidates = [
            os.path.join(os.path.dirname(pkl_path), "cifar-10-python.tar.gz"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cifar-10-python.tar.gz"),
        ]
        resolved = None
        for cand in candidates:
            c = os.path.abspath(os.path.expanduser(cand))
            if os.path.isfile(c):
                resolved = c
                break
        if resolved is None:
            raise FileNotFoundError(
                f"CIFAR archive not found at {tar}; also not found among candidates: {candidates}"
            )
        tar = resolved

    m = int(cfg["max_train_seen"]) if max_train_seen is None else int(max_train_seen)

    train_raw, test_raw = load_cifar10_from_tar(tar)

    def _row_task_id(row):
        if isinstance(row, dict):
            return int(row["task_id"])
        return int(getattr(row, "task_id"))

    ordered_task_ids: List[int] = []
    for row in tasks:
        ordered_task_ids.append(_row_task_id(row))
    if len(set(ordered_task_ids)) != len(ordered_task_ids):
        raise ValueError("Duplicate task_id in pkl tasks; cannot build stream")

    base_train: Dict[int, Subset] = {}
    task_test: Dict[int, Subset] = {}

    for row in tasks:
        tid = _row_task_id(row)

        train_pairs = row["train_pair_indices"]
        train_pair_labels = row["train_pair_labels"]
        test_pairs = row["test_pair_indices"]
        test_pair_labels = row["test_pair_labels"]

        train_pair_ds = CIFAR10PairDataset(train_raw[b"data"], train_pairs, train_pair_labels)
        test_pair_ds = CIFAR10PairDataset(test_raw[b"data"], test_pairs, test_pair_labels)
        base_train[tid] = truncate_subset(
            Subset(train_pair_ds, list(range(len(train_pair_labels)))),
            m,
        )
        task_test[tid] = Subset(test_pair_ds, list(range(len(test_pair_labels))))

    task_train: Dict[int, Subset]
    if overlap_ratio > 0.0:
        task_train = {}
        for pos, tid in enumerate(ordered_task_ids, start=1):
            prev_tid = ordered_task_ids[pos - 2] if pos - 2 >= 0 else None
            next_tid = ordered_task_ids[pos] if pos < len(ordered_task_ids) else None
            task_train[tid] = make_blurry_subset(
                curr_train=base_train[tid],
                prev_train=base_train[prev_tid] if prev_tid is not None else None,
                next_train=base_train[next_tid] if next_tid is not None else None,
                overlap_ratio=overlap_ratio,
                seed=int(blurry_seed),
                task_pos=pos,
            )
    else:
        task_train = base_train

    stream_index: List[Tuple[int, int]] = []
    for tid in ordered_task_ids:
        for gidx in list(task_train[tid].indices):
            stream_index.append((tid, int(gidx)))

    return BuiltStream(
        ordered_task_ids=ordered_task_ids,
        task_train=task_train,
        task_test=task_test,
        stream_index=stream_index,
        bundle=bundle,
    )
