"""Small executable equation reference, NOT a server implementation or FC substitute.

The phi supplied by tests is synthetic. Production must use the pinned FrozenFC.phi.
"""
from __future__ import annotations
import math
from typing import Callable
import torch
from torch import nn
from torch.nn import functional as F


def unit(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if not bool(torch.isfinite(x).all()) or bool((n <= 1e-12).any()):
        raise ValueError("Nonfinite/zero representation")
    return x / n


def route(e: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bounded energies; exact zero-rho allowed without log(0)."""
    if e.ndim != 2 or rho.ndim != 1 or e.shape[1] != rho.numel() or not rho.numel():
        raise ValueError("Expected nonempty query-by-group energies and group rho")
    if not bool(torch.isfinite(e).all()) or not bool(torch.isfinite(rho).all()):
        raise ValueError("Nonfinite route input")
    if bool((e.abs() > 4 + 1e-6).any()) or bool(((rho < 0) | (rho > 1)).any()):
        raise ValueError("Outside frozen route range")
    w = e.exp() * rho.unsqueeze(0) / rho.numel()
    den = 1 + w.sum(-1, keepdim=True)
    return w / den, den.reciprocal().squeeze(-1)


def matched_mass(e: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    if e.shape != alpha.shape:
        raise ValueError("Intervention must preserve actual query/group set")
    return alpha.sum(-1, keepdim=True) * torch.softmax(e, -1)


def correct(base: torch.Tensor, values: torch.Tensor, alpha: torch.Tensor, eta: float) -> tuple[torch.Tensor, torch.Tensor]:
    if not 0 <= eta <= .5:
        raise ValueError("eta must lie in[0,.5]")
    delta = eta * (alpha @ values).mean(0)
    # Exact identity at disabled routing; leave a graph-connected zero at eta0.
    if eta == 0 or not bool(alpha.detach().any()):
        return base + delta * 0, delta
    return unit(base + delta), delta


def match_delta_norm(base: torch.Tensor, original: torch.Tensor, control: torch.Tensor,
                     eps: float = 1e-12) -> tuple[torch.Tensor | None, dict]:
    nr, nc = torch.linalg.vector_norm(original), torch.linalg.vector_norm(control)
    if float(nr) <= eps:
        return base, {"status": "ORIGINAL_ZERO", "delta": torch.zeros_like(base)}
    if float(nc) <= eps:
        return None, {"status": "UNDEFINED_ZERO_CONTROL", "delta": None}
    delta = control * (nr / nc)
    return unit(base + delta), {"status": "MATCHED", "delta": delta}


class FGReference(nn.Module):
    """Architecture/equation reference. No classifier, phi weights or MA is owned."""
    def __init__(self, grouping: str = "SURFACE", direct_routing: bool = False,
                 dim: int = 768, hidden: int = 128):
        super().__init__()
        if grouping not in ("VIEW", "SURFACE"):
            raise ValueError("Unknown group mode")
        self.grouping, self.direct_routing = grouping, direct_routing
        self.embed = nn.Sequential(nn.Linear(dim+8, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.quality = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.membership = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.group_ff = nn.Sequential(nn.Linear(hidden, hidden*2), nn.GELU(), nn.Linear(hidden*2, hidden))
        self.group_norm = nn.LayerNorm(hidden)
        self.queries = nn.Parameter(torch.empty(4, hidden))
        nn.init.normal_(self.queries, 0, .02)
        nn.init.zeros_(self.membership[-1].weight)
        nn.init.zeros_(self.membership[-1].bias)

    def forward(self, inputs: dict, phi: Callable, *, completed_steps: int = 0,
                training_forward: bool = False, eta_override: float | None = None,
                rho_override: torch.Tensor | float | None = None) -> dict:
        forbidden = {"class_id", "membership_target", "owner", "family", "scene", "corruption_type"}
        if forbidden.intersection(inputs):
            raise ValueError("Supervision/identity must not enter forward")
        views = torch.nonzero(inputs['view_valid'], as_tuple=False).flatten()
        if not len(views):
            raise ValueError("No real view")
        base = unit(inputs['fc_vectors'][views].detach().mean(0))
        valid = inputs['local_valid'][views].bool()
        raw = torch.where(valid[..., None], inputs['local_raw'][views], 0)
        local = torch.where(valid[..., None], inputs['local_unit'][views], 0)
        meta = torch.where(valid[..., None], inputs['metadata'][views], 0)
        h = self.embed(torch.cat((local, meta), -1))
        q, m = self.quality(h).squeeze(-1), self.membership(h).squeeze(-1)
        out = dict(base=base, embedding=base, local_valid=valid, local_hidden=h,
                   membership_logits=m, quality_logits=q, view_indices=views,
                   group_count=0, fallback=None)
        if not bool(valid.any()):
            out['embedding'] = base + (h.sum()+q.sum()+m.sum())*0
            out['fallback'] = 'NO_LOCAL_SUPPORT_FC_FALLBACK'
            return out
        nv, ns = valid.shape
        ids = (torch.arange(nv, device=raw.device)[:, None].expand(nv, ns) if self.grouping == 'VIEW'
               else torch.arange(ns, device=raw.device)[None, :].expand(nv, ns))
        flat, fv = ids.flatten(), valid.flatten()
        present = torch.unique(flat[fv], sorted=True)
        members = (present[:, None] == flat[None, :]) & fv[None, :]
        a = torch.softmax(q.flatten()[None].expand(len(present), -1).masked_fill(~members, -torch.inf), -1)
        mean = a @ h.reshape(-1, h.shape[-1])
        keys = self.group_norm(mean + self.group_ff(mean))
        u = unit(phi(a @ raw.reshape(-1, raw.shape[-1])))
        values = u - base
        rho = (members.to(m.dtype) @ m.sigmoid().flatten()) / members.sum(-1)
        e = 4*torch.tanh(self.queries @ keys.T / math.sqrt(keys.shape[-1]))
        effective = rho if self.direct_routing else torch.ones_like(rho)
        if rho_override is not None:
            effective = torch.as_tensor(rho_override, dtype=rho.dtype, device=rho.device).expand_as(rho)
        alpha, null = route(e, effective)
        eta = (.5*min(1., completed_steps/200) if training_forward else .5) if eta_override is None else eta_override
        z, delta = correct(base, values, alpha, eta)
        out.update(embedding=z, delta=delta, rho=rho, effective_rho=effective, alpha=alpha,
                   null=null, energies=e, values=values, quality_weights=a, members=members,
                   group_count=len(present), eta=eta)
        return out


def membership_loss(result: dict, targets: torch.Tensor) -> tuple[torch.Tensor, int]:
    y = targets[result['view_indices']]
    valid = result['local_valid'] & (y >= 0)
    if not bool(valid.any()):
        return result['membership_logits'].sum()*0, 0
    return F.binary_cross_entropy_with_logits(result['membership_logits'][valid], y[valid].float()), int(valid.sum())


def qualifies(row: dict, reference: dict) -> bool:
    tol = 1e-10
    return (row['step'] >= 250 and row['A'] >= reference['A']+.005-tol
            and row['M'] >= reference['M']-tol and row['C'] >= reference['C']-tol
            and row['net_corrections'] > 0)


def path_counts(main_nominee: str | None, repeat_same_qualified: bool) -> tuple[int, int, int]:
    if main_nominee is None:
        return 8000, 0, 0
    if not repeat_same_qualified:
        return 16000, 0, 0
    return 16000, 286, 22
