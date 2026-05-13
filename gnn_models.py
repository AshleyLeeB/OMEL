"""
GCN for transaction anomaly detection.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


def _build_node_features(accounts: pd.DataFrame, node_to_idx: dict) -> np.ndarray:
    n_nodes = len(node_to_idx)
    acc = accounts.copy()
    acc["wallet_id"] = acc["wallet_id"].astype(str)
    feat_cols = ["wallet_type", "wallet_level", "init_balance"]
    available = [c for c in feat_cols if c in acc.columns]
    if not available:
        available = [c for c in acc.columns if c != "wallet_id" and c.startswith("feat_")]
    if not available:
        available = [
            c
            for c in acc.columns
            if c != "wallet_id"
            and (str(acc[c].dtype).startswith("float") or str(acc[c].dtype).startswith("int"))
        ]
    if not available:
        return np.ones((n_nodes, 1), dtype=np.float32)
    X = (
        acc.set_index("wallet_id")
        .reindex(list(node_to_idx.keys()), axis=0)[available]
        .fillna(0)
        .astype(np.float32)
        .values
    )

    if X.shape[1] > 0:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    if X.ndim == 1:
        X = X[:, None]
    return X


def _edges_to_tensors(
    edges: pd.DataFrame,
    node_to_idx: dict,
) -> tuple:
    src = edges["src"].astype(str).map(node_to_idx)
    dst = edges["dst"].astype(str).map(node_to_idx)

    mask = src.notna() & dst.notna()
    src = src[mask].astype(np.int64).values
    dst = dst[mask].astype(np.int64).values
    edge_index = np.stack([src, dst], axis=0)
    if "is_risk" in edges.columns:
        edge_label = edges.loc[mask, "is_risk"].astype(np.float32).values
    else:
        edge_label = np.zeros(len(src), dtype=np.float32)
    return edge_index, edge_label


def _build_graph_from_window(
    S_t: pd.DataFrame,
    Q_t: pd.DataFrame,
    accounts: pd.DataFrame,
) -> Data:
    all_edges = pd.concat([S_t, Q_t], axis=0, ignore_index=True)
    nodes = pd.unique(
        np.concatenate(
            [
                all_edges["src"].astype(str).values,
                all_edges["dst"].astype(str).values,
            ]
        )
    )
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(node_to_idx)
    acc_sub = accounts[accounts["wallet_id"].astype(str).isin(node_to_idx)]
    x = _build_node_features(acc_sub, node_to_idx)
    if x.shape[0] != n_nodes or x.shape[1] == 0:
        x = np.zeros((n_nodes, max(1, x.shape[1] if x.size else 1)), dtype=np.float32)
        if len(acc_sub) > 0:
            x_fill = _build_node_features(acc_sub, node_to_idx)
            x[: x_fill.shape[0], : x_fill.shape[1]] = x_fill
    x = torch.from_numpy(x).float()

    edge_index, _ = _edges_to_tensors(all_edges, node_to_idx)
    edge_index = to_undirected(torch.from_numpy(edge_index).long())

    _, support_labels = _edges_to_tensors(S_t, node_to_idx)
    support_labels = torch.from_numpy(support_labels).float()

    query_edge_index, query_labels = _edges_to_tensors(Q_t, node_to_idx)
    query_edge_index = torch.from_numpy(query_edge_index).long()
    query_labels = torch.from_numpy(query_labels).float()

    data = Data(x=x, edge_index=edge_index, num_nodes=n_nodes)
    data.support_labels = support_labels
    data.support_edge_index = torch.from_numpy(_edges_to_tensors(S_t, node_to_idx)[0]).long()
    data.query_edge_index = query_edge_index
    data.query_labels = query_labels
    data.node_to_idx = node_to_idx
    return data


class _GCNBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class EdgeAnomalyScorer(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = _GCNBackbone(num_node_features, hidden_dim, out_dim, dropout=dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        query_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        h = self.backbone(x, edge_index)
        u, v = query_edge_index[0], query_edge_index[1]
        hu = h[u]
        hv = h[v]
        edge_feat = torch.cat([hu, hv], dim=-1)
        return self.edge_mlp(edge_feat).squeeze(-1)


def get_backbone(
    num_node_features: int,
    hidden_dim: int = 64,
    out_dim: int = 32,
    dropout: float = 0.2,
) -> EdgeAnomalyScorer:
    return EdgeAnomalyScorer(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    )


class NodeAnomalyScorer(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.backbone = _GCNBackbone(num_node_features, hidden_dim, out_dim, dropout=dropout)
        self.node_head = nn.Linear(out_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x, edge_index)
        return self.node_head(h).squeeze(-1)


def get_backbone_node(
    num_node_features: int,
    hidden_dim: int = 64,
    out_dim: int = 32,
    dropout: float = 0.2,
) -> NodeAnomalyScorer:
    return NodeAnomalyScorer(
        num_node_features=num_node_features,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        dropout=dropout,
    )


def build_graph_and_labels(window) -> tuple:
    S_t = window.S_t
    Q_t = window.Q_t
    accounts = window.accounts
    data = _build_graph_from_window(S_t, Q_t, accounts)
    num_node_features = int(data.x.size(1))
    return data, num_node_features


def build_graph_and_labels_node(window) -> tuple:
    x = torch.from_numpy(window.x).float()
    edge_index = to_undirected(torch.from_numpy(window.edge_index).long())
    support_node_idx = torch.from_numpy(window.S_t_idx).long()
    support_labels = torch.from_numpy(window.S_t_labels).float()
    query_node_idx = torch.from_numpy(window.Q_t_idx).long()
    query_labels = torch.from_numpy(window.Q_t_labels).float()
    num_node_features = int(x.size(1))
    data = Data(x=x, edge_index=edge_index, num_nodes=window.num_nodes)
    data.support_node_idx = support_node_idx
    data.support_labels = support_labels
    data.query_node_idx = query_node_idx
    data.query_labels = query_labels
    return data, num_node_features
