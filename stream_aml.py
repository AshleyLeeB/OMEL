import importlib.util
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple
import numpy as np


@dataclass
class NodeStreamWindow:
    t: int
    edge_index: np.ndarray
    x: np.ndarray
    S_t_idx: np.ndarray
    S_t_labels: np.ndarray
    Q_t_idx: np.ndarray
    Q_t_labels: np.ndarray
    num_nodes: int
    node_mode: bool = True

def _load_dgl_edge_index(graph) -> Tuple[np.ndarray, int]:
    if importlib.util.find_spec("dgl") is None:
        raise ImportError("Loading new_graph_guiyihua.pkl requires DGL: pip install dgl")
    if hasattr(graph, "edges"):
        u, v = graph.edges()
        u, v = np.asarray(u.numpy(), dtype=np.int64), np.asarray(v.numpy(), dtype=np.int64)
        edge_index = np.stack([u, v], axis=0)
        n_nodes = graph.num_nodes()
        if hasattr(n_nodes, "item"):
            n_nodes = int(n_nodes.item())
        else:
            n_nodes = int(n_nodes)
        return edge_index, n_nodes
    raise TypeError("new_graph_guiyihua.pkl must be a DGL graph or a dict with edge_index")


def _node_labels_array(label_pkl: dict, n_nodes: int) -> np.ndarray:
    out = np.zeros(n_nodes, dtype=np.float32)
    for i in range(n_nodes):
        v = label_pkl.get(i, label_pkl.get(str(i), 0))
        out[i] = float(v)

    unique_vals = set(np.unique(out).tolist())
    if 3.0 in unique_vals or len(unique_vals - {0.0, 1.0, 3.0}) == 0:
        out = np.where(out == 1.0, 1.0, 0.0).astype(np.float32)
    return out


class AMLStream:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        window_size: int = 500,
        support_ratio: float = 0.7,
        overlap: bool = False,
        sampling_mode: str = "sequential",
        target_anomaly_ratio: float = 0.0,
        random_seed: int = 42,
        stratified_split: bool = True,
    ):
        self.data_dir = Path(data_dir or DEFAULT_AMLWORLD_DIR)
        self.window_size = window_size
        self.support_ratio = support_ratio
        self.overlap = overlap
        self.sampling_mode = str(sampling_mode).strip().lower()
        self.target_anomaly_ratio = float(target_anomaly_ratio)
        self.random_seed = int(random_seed)
        self.stratified_split = bool(stratified_split)

        self._x: Optional[np.ndarray] = None
        self._edge_index: Optional[np.ndarray] = None
        self._node_labels: Optional[np.ndarray] = None
        self._n_nodes: Optional[int] = None

    def _load_once(self):
        if self._edge_index is not None:
            return
        feat_path = self.data_dir / "features_5.npy"
        graph_path = self.data_dir / "new_graph_guiyihua.pkl"
        label_path = self.data_dir / "label.pkl"
        if not feat_path.exists():
            raise FileNotFoundError(f"Missing file: {feat_path}")
        if not graph_path.exists():
            raise FileNotFoundError(f"Missing file: {graph_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"Missing file: {label_path}")

        self._x = np.load(str(feat_path), allow_pickle=True).astype(np.float32)
        if self._x.ndim == 1:
            self._x = np.expand_dims(self._x, axis=1)

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        if isinstance(graph, list) and len(graph) > 0:
            graph = graph[0]
        self._edge_index, self._n_nodes = _load_dgl_edge_index(graph)

        with open(label_path, "rb") as f:
            label = pickle.load(f)
        self._node_labels = _node_labels_array(label, self._n_nodes)

    def _split_window(self, n: int) -> Tuple[int, int]:
        n_support = max(1, int(n * self.support_ratio))
        return n_support, n - n_support

    def _sample_balanced_indices(self, rng: np.random.Generator, n: int) -> np.ndarray:
        labels = self._node_labels
        pos_pool = np.where(labels > 0.5)[0]
        neg_pool = np.where(labels <= 0.5)[0]
        if len(pos_pool) == 0 or len(neg_pool) == 0:
            return rng.choice(np.arange(self._n_nodes, dtype=np.int64), size=n, replace=True)

        r = max(0.0, min(1.0, self.target_anomaly_ratio))
        n_pos = int(round(n * r))
        if r > 0.0:
            n_pos = max(1, n_pos)
        n_pos = min(n, n_pos)
        n_neg = n - n_pos
        if n_neg <= 0:
            n_neg = 1
            n_pos = n - n_neg

        pos_idx = rng.choice(pos_pool, size=n_pos, replace=len(pos_pool) < n_pos)
        neg_idx = rng.choice(neg_pool, size=n_neg, replace=len(neg_pool) < n_neg)
        idx = np.concatenate([pos_idx, neg_idx], axis=0).astype(np.int64)
        rng.shuffle(idx)
        return idx

    def _split_indices_stratified(self, idx: np.ndarray, n_support: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        labels = self._node_labels[idx]
        pos = idx[labels > 0.5]
        neg = idx[labels <= 0.5]
        rng.shuffle(pos)
        rng.shuffle(neg)
        total = max(1, len(idx))
        n_pos_support = int(round(n_support * (len(pos) / total)))
        n_pos_support = max(0, min(len(pos), n_pos_support))
        n_neg_support = n_support - n_pos_support
        n_neg_support = max(0, min(len(neg), n_neg_support))

        support = np.concatenate([pos[:n_pos_support], neg[:n_neg_support]], axis=0)
        if len(support) < n_support:
            remain = np.concatenate([pos[n_pos_support:], neg[n_neg_support:]], axis=0)
            if len(remain) > 0:
                take = min(n_support - len(support), len(remain))
                support = np.concatenate([support, remain[:take]], axis=0)
        rng.shuffle(support)

        support_set = set(support.tolist())
        query = np.array([i for i in idx.tolist() if i not in support_set], dtype=np.int64)
        return support.astype(np.int64), query.astype(np.int64)

    def __iter__(self) -> Iterator[NodeStreamWindow]:
        self._load_once()
        N = self._n_nodes
        B = self.window_size
        step = 1 if self.overlap else B
        t = 0
        use_balanced = (self.sampling_mode == "balanced") or (self.target_anomaly_ratio > 0.0)
        n_windows = max(0, (N - B) // step + 1)
        if n_windows <= 0:
            return
        rng = np.random.default_rng(self.random_seed)

        start = 0
        while t < n_windows:
            n_support, _ = self._split_window(B)
            if use_balanced:
                idx = self._sample_balanced_indices(rng, B)
            else:
                end = start + B
                idx = np.arange(start, end, dtype=np.int64)
            if self.stratified_split:
                S_t_idx, Q_t_idx = self._split_indices_stratified(idx, n_support, rng)
            else:
                S_t_idx = idx[:n_support]
                Q_t_idx = idx[n_support:]
            S_t_labels = self._node_labels[S_t_idx]
            Q_t_labels = self._node_labels[Q_t_idx]

            yield NodeStreamWindow(
                t=t,
                edge_index=self._edge_index,
                x=self._x,
                S_t_idx=S_t_idx,
                S_t_labels=S_t_labels,
                Q_t_idx=Q_t_idx,
                Q_t_labels=Q_t_labels,
                num_nodes=N,
                node_mode=True,
            )
            t += 1
            if not use_balanced:
                start += step


def main():
    stream = AMLStream()
    print("AMLWorld node-anomaly stream (features_5.npy, label.pkl, new_graph_guiyihua.pkl)")
    print("window_size B =", stream.window_size, "support_ratio =", stream.support_ratio)
    print("-" * 50)
    for w in stream:
        print(
            f"t={w.t}: |S_t|={len(w.S_t_idx)}, |Q_t|={len(w.Q_t_idx)}, "
            f"num_nodes={w.num_nodes}"
        )
        if w.t >= 2:
            break


if __name__ == "__main__":
    main()
