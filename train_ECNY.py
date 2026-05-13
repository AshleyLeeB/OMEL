from __future__ import annotations

import math
import csv
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except ImportError:
    roc_auc_score = None
    average_precision_score = None

from gnn_models import (
    build_graph_and_labels,
    build_graph_and_labels_node,
    get_backbone,
    get_backbone_node,
)

def _is_pyg_data(data: Any) -> bool:
    return hasattr(data, "x") and hasattr(data, "edge_index") and hasattr(data, "to")


@dataclass
class Expert:
    interval: tuple[int, int]
    phi: Dict[str, torch.Tensor]

    def id(self) -> tuple[int, int]:
        return self.interval


def _to_device(data: Any, device: torch.device) -> Any:
    if _is_pyg_data(data):
        data = data.to(device)
        return data
    if isinstance(data, dict):
        out = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    return data


def _get_tensors(data: Any, device: torch.device):
    data = _to_device(data, device)
    if _is_pyg_data(data):
        return (
            data.x,
            data.edge_index,
            data.support_edge_index,
            data.support_labels,
            data.query_edge_index,
            data.query_labels,
        )
    return (
        data["x"],
        data["edge_index"],
        data["support_edge_index"],
        data["support_labels"],
        data["query_edge_index"],
        data["query_labels"],
    )


def _get_tensors_node(data: Any, device: torch.device):
    data = _to_device(data, device)
    if _is_pyg_data(data):
        return (
            data.x,
            data.edge_index,
            data.support_node_idx,
            data.support_labels,
            data.query_node_idx,
            data.query_labels,
        )
    return (
        data["x"],
        data["edge_index"],
        data["support_node_idx"],
        data["support_labels"],
        data["query_node_idx"],
        data["query_labels"],
    )


def _compute_auc_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    out = {"auroc": float("nan"), "auprc": float("nan")}
    if roc_auc_score is None or average_precision_score is None:
        return out
    try:
        y_score = logits.detach().cpu().float().numpy()
        y_true = labels.detach().cpu().float().numpy()
        if y_true.min() == y_true.max():
            return out
        out["auroc"] = roc_auc_score(y_true, y_score)
        out["auprc"] = average_precision_score(y_true, y_score)
    except Exception:
        pass
    return out


def _compute_error_rate(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    if labels.numel() == 0:
        return float("nan")
    with torch.no_grad():
        probs = logits.sigmoid()
        preds = (probs >= 0.5).float()
        labels_f = labels.float()
        correct = (preds == labels_f).float().mean().item()
        return float(1.0 - correct)


ExpertId = Tuple[int, int]


def _soft_binary_targets(y: torch.Tensor, eps: float) -> torch.Tensor:
    if eps <= 0.0:
        return y
    return y * (1.0 - eps) + 0.5 * eps


def normalize_weights(weights: Dict[ExpertId, float], active_ids: List[ExpertId]) -> Dict[ExpertId, float]:
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
    prev_weights: Dict[ExpertId, float],
    active_ids: List[ExpertId],
    losses: Dict[ExpertId, float],
    prior_weights: Dict[ExpertId, float],
    regret_sq_state: Dict[ExpertId, float],
    eta_state: Dict[ExpertId, float],
) -> Tuple[Dict[ExpertId, float], Dict[ExpertId, float], Dict[ExpertId, float]]:
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


class OMEL:

    def __init__(
        self,
        stream: Any,
        alpha1: float = 0.01,
        alpha2: float = 0.001,
        lam: float = 1.0,
        ml_prod_eta: float = 0.1,
        expert_inner_steps: int = 3,
        meta_inner_steps: int = 1,
        distill_warmup_rounds: int = 20,
        label_smoothing: float = 0.05,
        online_supervised_steps: int = 0,
        online_supervised_batch_size: int = 256,
        online_pos_weight: float = 1.0,
        hidden_dim: int = 64,
        out_dim: int = 32,
        device: Optional[str] = None,
    ):
        self.stream = stream
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.lam = lam
        self.ml_prod_eta = ml_prod_eta
        self.expert_inner_steps = max(1, int(expert_inner_steps))
        self.meta_inner_steps = max(1, int(meta_inner_steps))
        self.distill_warmup_rounds = max(0, int(distill_warmup_rounds))
        self.label_smoothing = float(label_smoothing)
        self.online_supervised_steps = max(0, int(online_supervised_steps))
        self.online_supervised_batch_size = max(1, int(online_supervised_batch_size))
        self.online_pos_weight = float(max(1e-8, online_pos_weight))
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._model: Optional[Any] = None
        self._num_node_features: Optional[int] = None
        self._node_mode: Optional[bool] = None
        self.optimizer_meta: Optional[torch.optim.Optimizer] = None
        self.experts: list[Expert] = []
        self.p: Dict[ExpertId, float] = {}
        self.prior_weights: Dict[ExpertId, float] = {}
        self.regret_sq_state: Dict[ExpertId, float] = {}
        self.eta_state: Dict[ExpertId, float] = {}
        self._step = 0

    def _online_supervised_update_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
        query_node_idx: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> None:
        if self.online_supervised_steps <= 0:
            return
        assert self.optimizer_meta is not None
        idx_all = torch.cat([support_node_idx, query_node_idx], dim=0)
        y_all = torch.cat([support_labels, query_labels], dim=0)
        y_all = _soft_binary_targets(y_all, self.label_smoothing)
        n = int(idx_all.numel())
        if n <= 0:
            return
        pw = torch.tensor(self.online_pos_weight, device=self.device)
        self._model.train()
        for _ in range(self.online_supervised_steps):
            bs = min(self.online_supervised_batch_size, n)
            choose = torch.randint(0, n, (bs,), device=self.device)
            idx_b = idx_all.index_select(0, choose)
            y_b = y_all.index_select(0, choose)
            logits_all = self._model(x, edge_index)
            logits_b = logits_all[idx_b]
            loss = F.binary_cross_entropy_with_logits(logits_b, y_b, pos_weight=pw)
            self.optimizer_meta.zero_grad()
            loss.backward()
            self.optimizer_meta.step()

    def _online_supervised_update_edge(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        query_edge_index: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> None:
        if self.online_supervised_steps <= 0:
            return
        assert self.optimizer_meta is not None
        edge_all = torch.cat([support_edge_index, query_edge_index], dim=1)
        y_all = torch.cat([support_labels, query_labels], dim=0)
        y_all = _soft_binary_targets(y_all, self.label_smoothing)
        n = int(y_all.numel())
        if n <= 0:
            return
        pw = torch.tensor(self.online_pos_weight, device=self.device)
        self._model.train()
        for _ in range(self.online_supervised_steps):
            bs = min(self.online_supervised_batch_size, n)
            choose = torch.randint(0, n, (bs,), device=self.device)
            edge_b = edge_all.index_select(1, choose)
            y_b = y_all.index_select(0, choose)
            logits_b = self._model(x, edge_index, edge_b)
            loss = F.binary_cross_entropy_with_logits(logits_b, y_b, pos_weight=pw)
            self.optimizer_meta.zero_grad()
            loss.backward()
            self.optimizer_meta.step()

    def pretrain(
        self,
        epochs: int = 1,
        max_steps_per_epoch: Optional[int] = None,
        lr: Optional[float] = None,
        pos_weight: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, float]:
        if epochs <= 0:
            return {"pretrain_loss": float("nan"), "epochs": 0.0, "steps": 0.0}
        if lr is not None and lr > 0:
            assert self.optimizer_meta is not None
            for g in self.optimizer_meta.param_groups:
                g["lr"] = float(lr)

        total_loss = 0.0
        total_steps = 0
        pw = torch.tensor(float(max(1e-8, pos_weight)), device=self.device)

        for ep in range(1, int(epochs) + 1):
            ep_loss = 0.0
            ep_steps = 0
            for i, window in enumerate(self.stream):
                if max_steps_per_epoch is not None and i >= max_steps_per_epoch:
                    break

                node_mode = getattr(window, "node_mode", False)
                if node_mode:
                    data, num_node_features = build_graph_and_labels_node(window)
                    self._ensure_model(num_node_features, node_mode=True)
                    x, edge_index, support_node_idx, support_labels, query_node_idx, query_labels = _get_tensors_node(
                        data, self.device
                    )
                    idx = torch.cat([support_node_idx, query_node_idx], dim=0)
                    y = torch.cat([support_labels, query_labels], dim=0)
                    logits_all = self._model(x, edge_index)
                    logits = logits_all[idx]
                else:
                    data, num_node_features = build_graph_and_labels(window)
                    self._ensure_model(num_node_features, node_mode=False)
                    x, edge_index, support_edge_index, support_labels, query_edge_index, query_labels = _get_tensors(
                        data, self.device
                    )
                    q_edge = torch.cat([support_edge_index, query_edge_index], dim=1)
                    y = torch.cat([support_labels, query_labels], dim=0)
                    logits = self._model(x, edge_index, q_edge)

                y_s = _soft_binary_targets(y, self.label_smoothing)
                loss = F.binary_cross_entropy_with_logits(logits, y_s, pos_weight=pw)
                assert self.optimizer_meta is not None
                self.optimizer_meta.zero_grad()
                loss.backward()
                self.optimizer_meta.step()

                lv = float(loss.item())
                ep_loss += lv
                total_loss += lv
                ep_steps += 1
                total_steps += 1

            if verbose:
                mean_ep = ep_loss / max(1, ep_steps)
                print(f"[pretrain] epoch={ep}/{epochs} steps={ep_steps} loss={mean_ep:.4f}")

        return {
            "pretrain_loss": total_loss / max(1, total_steps),
            "epochs": float(epochs),
            "steps": float(total_steps),
        }

    def _ensure_model(self, num_node_features: int, node_mode: bool = False):
        if self._model is None or self._num_node_features != num_node_features or self._node_mode != node_mode:
            self._num_node_features = num_node_features
            self._node_mode = node_mode
            if node_mode:
                self._model = get_backbone_node(
                    num_node_features,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                ).to(self.device)
            else:
                self._model = get_backbone(
                    num_node_features,
                    hidden_dim=self.hidden_dim,
                    out_dim=self.out_dim,
                ).to(self.device)
            self.optimizer_meta = torch.optim.Adam(self._model.parameters(), lr=self.alpha2)

    def _load_state(self, state: Dict[str, torch.Tensor]):
        self._model.load_state_dict(state, strict=False)

    def _get_theta_copy(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self._model.state_dict().items()}

    def _named_buffers_copy(self) -> Dict[str, torch.Tensor]:
        return {n: b.detach() for n, b in self._model.named_buffers()}

    def _adapt_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._model(x, edge_index, support_edge_index)
        y_s = _soft_binary_targets(support_labels, self.label_smoothing)
        return F.binary_cross_entropy_with_logits(logits, y_s)

    def _adapt_loss_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
    ) -> torch.Tensor:
        logits = self._model(x, edge_index)
        y_s = _soft_binary_targets(support_labels, self.label_smoothing)
        return F.binary_cross_entropy_with_logits(logits[support_node_idx], y_s)

    def _task_loss(
        self,
        logits: torch.Tensor,
        query_labels: torch.Tensor,
    ) -> torch.Tensor:
        y_q = _soft_binary_targets(query_labels, self.label_smoothing)
        return F.binary_cross_entropy_with_logits(logits, y_q)

    def _distill_loss(self, s_tilde: torch.Tensor, s_teacher: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(s_tilde, s_teacher.detach().sigmoid())

    def _inner_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._load_state(state)
        self._model.train()
        self._model.zero_grad()
        loss = self._adapt_loss(x, edge_index, support_edge_index, support_labels)
        loss.backward()
        new_state = {}
        with torch.no_grad():
            for k, v in state.items():
                if k in self._model.state_dict() and self._model.state_dict()[k].grad is not None:
                    g = self._model.state_dict()[k].grad
                    new_state[k] = v - self.alpha1 * g
                else:
                    new_state[k] = v
        return new_state

    def _inner_step_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
        state: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        self._load_state(state)
        self._model.train()
        self._model.zero_grad()
        loss = self._adapt_loss_node(x, edge_index, support_node_idx, support_labels)
        loss.backward()
        new_state = {}
        with torch.no_grad():
            for k, v in state.items():
                if k in self._model.state_dict() and self._model.state_dict()[k].grad is not None:
                    g = self._model.state_dict()[k].grad
                    new_state[k] = v - self.alpha1 * g
                else:
                    new_state[k] = v
        return new_state

    def _one_step_adapt_edge_functional(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        create_graph: bool,
    ) -> Dict[str, torch.Tensor]:
        y_s = _soft_binary_targets(support_labels, self.label_smoothing)
        pb = {**params, **buffers}
        logits = functional_call(self._model, pb, (x, edge_index, support_edge_index))
        loss = F.binary_cross_entropy_with_logits(logits, y_s)
        names = list(params.keys())
        grads = torch.autograd.grad(
            loss,
            [params[n] for n in names],
            create_graph=create_graph,
            allow_unused=False,
        )
        return {n: params[n] - self.alpha1 * g for n, g in zip(names, grads)}

    def _k_step_adapt_edge_functional(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        n_steps: int,
        create_graph: bool,
    ) -> Dict[str, torch.Tensor]:
        out = params
        for _ in range(max(1, int(n_steps))):
            out = self._one_step_adapt_edge_functional(
                out,
                buffers,
                x,
                edge_index,
                support_edge_index,
                support_labels,
                create_graph=create_graph,
            )
        return out

    def _one_step_adapt_node_functional(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
        create_graph: bool,
    ) -> Dict[str, torch.Tensor]:
        y_s = _soft_binary_targets(support_labels, self.label_smoothing)
        pb = {**params, **buffers}
        full = functional_call(self._model, pb, (x, edge_index))
        loss = F.binary_cross_entropy_with_logits(full[support_node_idx], y_s)
        names = list(params.keys())
        grads = torch.autograd.grad(
            loss,
            [params[n] for n in names],
            create_graph=create_graph,
            allow_unused=False,
        )
        return {n: params[n] - self.alpha1 * g for n, g in zip(names, grads)}

    def _k_step_adapt_node_functional(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
        n_steps: int,
        create_graph: bool,
    ) -> Dict[str, torch.Tensor]:
        out = params
        for _ in range(max(1, int(n_steps))):
            out = self._one_step_adapt_node_functional(
                out,
                buffers,
                x,
                edge_index,
                support_node_idx,
                support_labels,
                create_graph=create_graph,
            )
        return out

    def _meta_step(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_edge_index: torch.Tensor,
        support_labels: torch.Tensor,
        query_edge_index: torch.Tensor,
        query_labels: torch.Tensor,
        s_teacher: torch.Tensor,
        t_alg: int,
    ) -> float:
        self._model.train()
        params = {n: p for n, p in self._model.named_parameters()}
        buffers = self._named_buffers_copy()
        cg = True
        tilde_params = self._k_step_adapt_edge_functional(
            params,
            buffers,
            x,
            edge_index,
            support_edge_index,
            support_labels,
            self.meta_inner_steps,
            create_graph=cg,
        )
        pb = {**tilde_params, **buffers}
        logits_q = functional_call(self._model, pb, (x, edge_index, query_edge_index))
        l_task = self._task_loss(logits_q, query_labels)
        l_distill = self._distill_loss(logits_q, s_teacher)
        lambda_eff = self.lam if t_alg > self.distill_warmup_rounds else 0.0
        ell_meta = l_task + lambda_eff * l_distill
        assert self.optimizer_meta is not None
        self.optimizer_meta.zero_grad()
        ell_meta.backward()
        self.optimizer_meta.step()
        return float(ell_meta.item())

    def _meta_step_node(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        support_node_idx: torch.Tensor,
        support_labels: torch.Tensor,
        query_node_idx: torch.Tensor,
        query_labels: torch.Tensor,
        s_teacher: torch.Tensor,
        t_alg: int,
    ) -> float:
        self._model.train()
        params = {n: p for n, p in self._model.named_parameters()}
        buffers = self._named_buffers_copy()
        cg = True
        tilde_params = self._k_step_adapt_node_functional(
            params,
            buffers,
            x,
            edge_index,
            support_node_idx,
            support_labels,
            self.meta_inner_steps,
            create_graph=cg,
        )
        pb = {**tilde_params, **buffers}
        full = functional_call(self._model, pb, (x, edge_index))
        tilde_s = full[query_node_idx]
        l_task = self._task_loss(tilde_s, query_labels)
        l_distill = self._distill_loss(tilde_s, s_teacher)
        lambda_eff = self.lam if t_alg > self.distill_warmup_rounds else 0.0
        ell_meta = l_task + lambda_eff * l_distill
        assert self.optimizer_meta is not None
        self.optimizer_meta.zero_grad()
        ell_meta.backward()
        self.optimizer_meta.step()
        return float(ell_meta.item())

    def run_step(self, window: Any) -> Dict[str, float]:
        node_mode = getattr(window, "node_mode", False)
        if self._node_mode is None:
            self._node_mode = node_mode

        if node_mode:
            return self._run_step_node(window)
        return self._run_step_edge(window)

    def _run_step_edge(self, window: Any) -> Dict[str, float]:
        t_alg = self._step + 1
        data, num_node_features = build_graph_and_labels(window)
        self._ensure_model(num_node_features, node_mode=False)
        x, edge_index, support_edge_index, support_labels, query_edge_index, query_labels = _get_tensors(
            data, self.device
        )

        for k in range(0, int(math.floor(math.log2(t_alg))) + 1):
            if t_alg % (2 ** k) == 0:
                interval = (t_alg, t_alg + (2 ** k) - 1)
                e = Expert(interval=interval, phi=self._get_theta_copy())
                self.experts.append(e)
                prio = float(2.0 ** (-(k + 1)))
                eid = e.id()
                self.p[eid] = prio
                self.prior_weights[eid] = prio
                self.regret_sq_state[eid] = 0.0
                self.eta_state[eid] = _eta_from_prior_and_regret(prio, 0.0)

        E_t = [e for e in self.experts if e.interval[0] <= t_alg <= e.interval[1]]
        if not E_t:
            self._step += 1
            return {"meta_loss": 0.0, "task_loss": 0.0, "n_experts": 0, "auroc": float("nan"), "auprc": float("nan")}

        active_ids: List[ExpertId] = [e.id() for e in E_t]
        total_p = sum(self.p.get(e.id(), 0) for e in E_t)
        if total_p <= 0:
            for e in E_t:
                self.p[e.id()] = 1.0 / len(E_t)
        else:
            for e in E_t:
                self.p[e.id()] = self.p.get(e.id(), 0) / total_p

        meta_base_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}

        scores_per_expert: Dict[ExpertId, torch.Tensor] = {}
        losses_per_expert: Dict[ExpertId, float] = {}
        for e in E_t:
            phi_new = e.phi
            for _ in range(self.expert_inner_steps):
                phi_new = self._inner_step(
                    x, edge_index, support_edge_index, support_labels, phi_new
                )
            e.phi = phi_new
            self._load_state(phi_new)
            self._model.eval()
            with torch.no_grad():
                s_e = self._model(x, edge_index, query_edge_index)
            ell_e = float(self._task_loss(s_e, query_labels).item())
            scores_per_expert[e.id()] = s_e
            losses_per_expert[e.id()] = ell_e

        s_ens = sum(self.p[e.id()] * scores_per_expert[e.id()] for e in E_t)
        auc_metrics = _compute_auc_metrics(s_ens, query_labels)
        test_error = _compute_error_rate(s_ens, query_labels)

        e_star = min(E_t, key=lambda ex: losses_per_expert[ex.id()])
        s_teacher = scores_per_expert[e_star.id()]

        self._load_state(meta_base_state)
        meta_loss_val = self._meta_step(
            x,
            edge_index,
            support_edge_index,
            support_labels,
            query_edge_index,
            query_labels,
            s_teacher,
            t_alg,
        )
        self._online_supervised_update_edge(
            x,
            edge_index,
            support_edge_index,
            support_labels,
            query_edge_index,
            query_labels,
        )

        self.p, self.regret_sq_state, self.eta_state = adapt_ml_prod(
            prev_weights=self.p,
            active_ids=active_ids,
            losses=losses_per_expert,
            prior_weights=self.prior_weights,
            regret_sq_state=self.regret_sq_state,
            eta_state=self.eta_state,
        )

        self.experts = [e for e in self.experts if e.interval[1] != t_alg]
        live = {e.id() for e in self.experts}
        self.p = {k: v for k, v in self.p.items() if k in live}
        self.prior_weights = {k: v for k, v in self.prior_weights.items() if k in live}
        self.regret_sq_state = {k: v for k, v in self.regret_sq_state.items() if k in live}
        self.eta_state = {k: v for k, v in self.eta_state.items() if k in live}

        self._step += 1
        return {
            "meta_loss": meta_loss_val,
            "task_loss": sum(losses_per_expert.values()) / len(losses_per_expert),
            "n_experts": len(E_t),
            "auroc": auc_metrics["auroc"],
            "auprc": auc_metrics["auprc"],
            "test_error": test_error,
        }

    def _run_step_node(self, window: Any) -> Dict[str, float]:
        t_alg = self._step + 1
        data, num_node_features = build_graph_and_labels_node(window)
        self._ensure_model(num_node_features, node_mode=True)
        x, edge_index, support_node_idx, support_labels, query_node_idx, query_labels = _get_tensors_node(
            data, self.device
        )
        for k in range(0, int(math.floor(math.log2(t_alg))) + 1):
            if t_alg % (2 ** k) == 0:
                interval = (t_alg, t_alg + (2 ** k) - 1)
                e = Expert(interval=interval, phi=self._get_theta_copy())
                self.experts.append(e)
                prio = float(2.0 ** (-(k + 1)))
                eid = e.id()
                self.p[eid] = prio
                self.prior_weights[eid] = prio
                self.regret_sq_state[eid] = 0.0
                self.eta_state[eid] = _eta_from_prior_and_regret(prio, 0.0)
        E_t = [e for e in self.experts if e.interval[0] <= t_alg <= e.interval[1]]
        if not E_t:
            self._step += 1
            return {"meta_loss": 0.0, "task_loss": 0.0, "n_experts": 0, "auroc": float("nan"), "auprc": float("nan")}
        active_ids: List[ExpertId] = [e.id() for e in E_t]
        total_p = sum(self.p.get(e.id(), 0) for e in E_t)
        if total_p <= 0:
            for e in E_t:
                self.p[e.id()] = 1.0 / len(E_t)
        else:
            for e in E_t:
                self.p[e.id()] = self.p.get(e.id(), 0) / total_p

        meta_base_state = {k: v.detach().clone() for k, v in self._model.state_dict().items()}

        scores_per_expert: Dict[ExpertId, torch.Tensor] = {}
        losses_per_expert: Dict[ExpertId, float] = {}
        for e in E_t:
            phi_new = e.phi
            for _ in range(self.expert_inner_steps):
                phi_new = self._inner_step_node(
                    x, edge_index, support_node_idx, support_labels, phi_new
                )
            e.phi = phi_new
            self._load_state(phi_new)
            self._model.eval()
            with torch.no_grad():
                s_all = self._model(x, edge_index)
                s_e = s_all[query_node_idx]
            ell_e = float(self._task_loss(s_e, query_labels).item())
            scores_per_expert[e.id()] = s_e
            losses_per_expert[e.id()] = ell_e
        s_ens = sum(self.p[e.id()] * scores_per_expert[e.id()] for e in E_t)
        auc_metrics = _compute_auc_metrics(s_ens, query_labels)
        test_error = _compute_error_rate(s_ens, query_labels)
        e_star = min(E_t, key=lambda ex: losses_per_expert[ex.id()])
        s_teacher = scores_per_expert[e_star.id()]

        self._load_state(meta_base_state)
        meta_loss_val = self._meta_step_node(
            x,
            edge_index,
            support_node_idx,
            support_labels,
            query_node_idx,
            query_labels,
            s_teacher,
            t_alg,
        )
        self._online_supervised_update_node(
            x,
            edge_index,
            support_node_idx,
            support_labels,
            query_node_idx,
            query_labels,
        )
        self.p, self.regret_sq_state, self.eta_state = adapt_ml_prod(
            prev_weights=self.p,
            active_ids=active_ids,
            losses=losses_per_expert,
            prior_weights=self.prior_weights,
            regret_sq_state=self.regret_sq_state,
            eta_state=self.eta_state,
        )
        self.experts = [e for e in self.experts if e.interval[1] != t_alg]
        live = {e.id() for e in self.experts}
        self.p = {k: v for k, v in self.p.items() if k in live}
        self.prior_weights = {k: v for k, v in self.prior_weights.items() if k in live}
        self.regret_sq_state = {k: v for k, v in self.regret_sq_state.items() if k in live}
        self.eta_state = {k: v for k, v in self.eta_state.items() if k in live}
        self._step += 1
        n_pos = float((query_labels > 0.5).sum().item())
        n_neg = float(len(query_labels) - n_pos)
        return {
            "meta_loss": meta_loss_val,
            "task_loss": sum(losses_per_expert.values()) / len(losses_per_expert),
            "n_experts": len(E_t),
            "auroc": auc_metrics["auroc"],
            "auprc": auc_metrics["auprc"],
            "n_pos": n_pos,
            "n_neg": n_neg,
            "test_error": test_error,
        }

    def run(
        self,
        max_steps: Optional[int] = None,
        online_skip_windows: int = 0,
        verbose: bool = True,
        return_history: bool = False,
    ):
        history = []
        skip = max(0, int(online_skip_windows))
        limit = max_steps if (max_steps is not None and int(max_steps) > 0) else None
        online_i = 0
        for i, window in enumerate(self.stream):
            if i < skip:
                continue
            if limit is not None and online_i >= limit:
                break
            metrics = self.run_step(window)
            online_i += 1
            history.append(
                {
                    "t": int(self._step),
                    "meta_loss": float(metrics.get("meta_loss", float("nan"))),
                    "task_loss": float(metrics.get("task_loss", float("nan"))),
                    "n_experts": int(metrics.get("n_experts", 0)),
                    "auroc": float(metrics.get("auroc", float("nan"))),
                    "auprc": float(metrics.get("auprc", float("nan"))),
                    "test_error": float(metrics.get("test_error", float("nan"))),
                    "q_pos": int(metrics["n_pos"]) if metrics.get("n_pos") is not None else -1,
                    "q_neg": int(metrics["n_neg"]) if metrics.get("n_neg") is not None else -1,
                }
            )
            if verbose:
                auroc_s = f"{metrics['auroc']:.4f}" if not math.isnan(metrics['auroc']) else "nan"
                auprc_s = f"{metrics['auprc']:.4f}" if not math.isnan(metrics['auprc']) else "nan"
                err = metrics.get("test_error")
                err_s = f"{err:.4f}" if err is not None and not math.isnan(err) else "nan"
                n_pos = metrics.get("n_pos")
                n_neg = metrics.get("n_neg")
                if n_pos is not None and n_neg is not None:
                    print(
                        f"t={self._step} meta_loss={metrics['meta_loss']:.4f} "
                        f"task_loss={metrics['task_loss']:.4f} "
                        f"AUROC={auroc_s} AUPRC={auprc_s} TestError={err_s} n_experts={metrics['n_experts']} "
                        f"Q_pos={int(n_pos)} Q_neg={int(n_neg)}"
                    )
                else:
                    print(
                        f"t={self._step} meta_loss={metrics['meta_loss']:.4f} "
                        f"task_loss={metrics['task_loss']:.4f} "
                        f"AUROC={auroc_s} AUPRC={auprc_s} TestError={err_s} n_experts={metrics['n_experts']}"
                    )
        return history if return_history else self


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--window_size", type=int, default=500)
    parser.add_argument("--support_ratio", type=float, default=0.7)
    parser.add_argument(
        "--sampling_mode", type=str, default="sequential", choices=["sequential", "balanced"]
    )
    parser.add_argument("--target_anomaly_ratio", type=float, default=0.0)
    parser.add_argument("--stream_seed", type=int, default=42)
    parser.add_argument("--stratified_split", action="store_true")
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--online_skip_windows", type=int, default=-1)
    parser.add_argument("--alpha1", type=float, default=0.01)
    parser.add_argument("--alpha2", type=float, default=0.001)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--expert_inner_steps", type=int, default=3)
    parser.add_argument("--meta_inner_steps", type=int, default=1)
    parser.add_argument("--distill_warmup", type=int, default=20)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--save_pretrained_ckpt", type=str, default="")
    parser.add_argument("--pretrain_epochs", type=int, default=0)
    parser.add_argument("--pretrain_max_steps", type=int, default=30)
    parser.add_argument("--pretrain_lr", type=float, default=0.001)
    parser.add_argument("--pretrain_pos_weight", type=float, default=10.0)
    parser.add_argument("--online_supervised_steps", type=int, default=0)
    parser.add_argument("--online_supervised_batch_size", type=int, default=256)
    parser.add_argument("--online_pos_weight", type=float, default=1.0)
    parser.add_argument("--save_csv", type=str, default="")
    parser.add_argument("--results_dir", type=str, default="")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/lwy/OMEL/Online-ECNY",
    )
    args = parser.parse_args()

    from stream_aml import AMLStream

    _repo = Path(__file__).resolve().parent
    _set_seed(int(args.seed))
    default_pt_dir = _repo / "pt"
    results_dir = (
        Path(args.results_dir.strip()).expanduser().resolve()
        if args.results_dir.strip()
        else default_pt_dir
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir.strip()).expanduser().resolve()

    print(f"data_dir={data_dir}")
    print(f"results_dir={results_dir}")
    stream = AMLStream(
        data_dir=data_dir,
        window_size=args.window_size,
        support_ratio=args.support_ratio,
        overlap=False,
        sampling_mode=args.sampling_mode,
        target_anomaly_ratio=args.target_anomaly_ratio,
        random_seed=args.stream_seed,
        stratified_split=args.stratified_split,
    )

    omel = OMEL(
        stream,
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        lam=args.lam,
        ml_prod_eta=0.1,
        expert_inner_steps=args.expert_inner_steps,
        meta_inner_steps=args.meta_inner_steps,
        distill_warmup_rounds=args.distill_warmup,
        label_smoothing=args.label_smoothing,
        online_supervised_steps=args.online_supervised_steps,
        online_supervised_batch_size=args.online_supervised_batch_size,
        online_pos_weight=args.online_pos_weight,
    )

    first_window = next(iter(stream), None)
    if first_window is not None:
        if getattr(first_window, "node_mode", False):
            _, nfeat = build_graph_and_labels_node(first_window)
            omel._ensure_model(nfeat, node_mode=True)
        else:
            _, nfeat = build_graph_and_labels(first_window)
            omel._ensure_model(nfeat, node_mode=False)

    if args.pretrained_ckpt.strip():
        ckpt_path = Path(args.pretrained_ckpt.strip()).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=omel.device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError(
                "Unsupported pretrained checkpoint format; expected a state_dict or a dict with key 'model_state_dict'."
            )
        omel._model.load_state_dict(state, strict=False)
        print(f"loaded pretrained ckpt: {ckpt_path}")

    if args.pretrain_epochs > 0:
        stats = omel.pretrain(
            epochs=args.pretrain_epochs,
            max_steps_per_epoch=args.pretrain_max_steps,
            lr=args.pretrain_lr,
            pos_weight=args.pretrain_pos_weight,
            verbose=True,
        )
        print(f"pretrain done: loss={stats['pretrain_loss']:.4f}, steps={int(stats['steps'])}")
        if args.save_pretrained_ckpt.strip():
            save_path = Path(args.save_pretrained_ckpt.strip()).expanduser().resolve()
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": omel._model.state_dict()}, str(save_path))
            print(f"saved pretrained ckpt: {save_path}")

    if int(args.online_skip_windows) < 0:
        online_skip = int(args.pretrain_max_steps) if int(args.pretrain_epochs) > 0 else 0
    else:
        online_skip = max(0, int(args.online_skip_windows))

    ms_arg = int(args.max_steps)
    ms_effective = None if ms_arg <= 0 else ms_arg
    print(
        f"[OMEL] online: skip first {online_skip} stream window(s), then "
        f"{'run all remaining windows' if ms_effective is None else f'run at most {ms_effective} window(s)'}"
    )

    history = omel.run(
        max_steps=ms_effective,
        online_skip_windows=online_skip,
        verbose=True,
        return_history=True,
    )

    if args.save_csv.strip():
        csv_path = Path(args.save_csv.strip()).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "meta_loss", "task_loss", "n_experts", "auroc", "auprc", "test_error", "q_pos", "q_neg"])
            for row in history:
                writer.writerow(
                    [
                        int(row["t"]),
                        f"{row['meta_loss']:.6f}",
                        f"{row['task_loss']:.6f}",
                        int(row["n_experts"]),
                        "nan" if math.isnan(row["auroc"]) else f"{row['auroc']:.6f}",
                        "nan" if math.isnan(row["auprc"]) else f"{row['auprc']:.6f}",
                        "nan" if math.isnan(row["test_error"]) else f"{row['test_error']:.6f}",
                        int(row["q_pos"]),
                        int(row["q_neg"]),
                    ]
                )
        print(f"[OMEL] CSV written to: {csv_path}")

    last = history[-1] if history else {}
    results_pt: Dict[int, Dict[str, float]] = {}
    plot_task_ids: List[int] = []
    index_to_task_id: Dict[int, int] = {}
    if history:
        plot_task_ids = [1]
        index_to_task_id = {1: -1}
        results_pt[1] = {
            "stream_windows": float(len(history)),
            "final_meta_loss": float(last.get("meta_loss", 0.0)),
            "final_task_loss": float(last.get("task_loss", 0.0)),
            "final_auroc": float(last.get("auroc", float("nan"))),
            "final_auprc": float(last.get("auprc", float("nan"))),
            "final_test_error": float(last.get("test_error", float("nan"))),
            "final_n_experts": float(last.get("n_experts", 0)),
        }
    out_path = results_dir / f"omel_results_ecny_seed{int(args.seed)}.pt"
    torch.save(
        {
            "results": results_pt,
            "task_ids": plot_task_ids,
            "index_to_task_id": index_to_task_id,
            "seed": int(args.seed),
            "data_name": "ecny",
            "stream_seed": int(args.stream_seed),
            "data_dir": str(data_dir),
            "overlap_ratio": 0.0,
            "window_size": int(args.window_size),
            "support_ratio": float(args.support_ratio),
            "step_size": 0,
            "eval_interval": 0,
            "history": history,
            "backbone": "GCN",
            "alpha1": float(args.alpha1),
            "alpha2": float(args.alpha2),
            "lam": float(args.lam),
            "expert_inner_steps": int(args.expert_inner_steps),
            "meta_inner_steps": int(args.meta_inner_steps),
        },
        str(out_path),
    )
    print(f"[OMEL] done, saved checkpoint to: {out_path}")


if __name__ == "__main__":
    main()
    import os

    os._exit(0)

