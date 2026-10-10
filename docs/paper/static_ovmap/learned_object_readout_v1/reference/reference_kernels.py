"""Small CPU references for the prompt contracts; not the production model.

These tests cannot validate real ScanNet registration or frozen FC/MaskAdapter
numerical parity. Codex must run the corresponding bounded production checks.
"""
from __future__ import annotations
import hashlib
from typing import Iterable
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def family_id(scan: str) -> str:
    pieces = scan.split('_')
    if len(pieces) != 2 or not pieces[0].startswith('scene') or not pieces[0][5:].isdigit() or not pieces[1].isdigit():
        raise ValueError(f'invalid ScanNet scan ID: {scan}')
    return pieces[0]


def family_split(scans: Iterable[str], denied: Iterable[str]) -> dict[str, list[str]]:
    denied = set(denied); chosen: dict[str, str] = {}
    for scan in scans:
        family = family_id(scan)
        if family in denied:
            continue
        if family not in chosen or int(scan.rsplit('_', 1)[1]) < int(chosen[family].rsplit('_', 1)[1]):
            chosen[family] = scan
    ordered = sorted(chosen, key=lambda f:(hashlib.sha256(('LR1-family-20261009|'+f).encode()).hexdigest(),f))
    if len(ordered)<32:
        raise ValueError('BLOCKED_TRAINING_DATA: need 32 independent families')
    return {'train':[chosen[f] for f in ordered[:24]],'dev':[chosen[f] for f in ordered[24:28]],'holdout':[chosen[f] for f in ordered[28:32]]}


def split_classes(ids: Iterable[int]) -> tuple[list[int], list[int]]:
    ids=list(ids)
    if len(ids)!=200 or len(set(ids))!=200:
        raise ValueError('exact 200 distinct class IDs required')
    ranked=sorted(ids,key=lambda i:(hashlib.sha256(('LR1-class-20261009|'+str(i)).encode()).hexdigest(),i))
    novel=set(ranked[:40])
    return [i for i in ids if i not in novel],[i for i in ids if i in novel]


def frozen_projection(x: torch.Tensor, phi: nn.Module) -> torch.Tensor:
    """Autograd must follow x even if all phi parameters are frozen."""
    return F.normalize(phi(x),dim=-1)


def masked_mean(x: torch.Tensor, valid: torch.Tensor, dim: int = 0) -> torch.Tensor:
    v=valid.to(x.dtype)
    if v.sum()<=0:
        raise ValueError('no real observation')
    return (x*v.unsqueeze(-1)).sum(dim=dim)/v.sum(dim=dim).unsqueeze(-1)


def normalized_feature_grid(uv: np.ndarray, original_hw: tuple[int,int], resized_hw: tuple[int,int], padded_hw: tuple[int,int]) -> np.ndarray:
    uv=np.asarray(uv,np.float64); h,w=original_hw; hr,wr=resized_hw; hp,wp=padded_hw
    if uv.ndim!=2 or uv.shape[1]!=2 or min(h,w,hr,wr,hp,wp)<=0:
        raise ValueError('valid pixel coordinates and image sizes required')
    res=(uv+.5)*np.array([wr/w,hr/h])-.5
    return 2*(res+.5)/np.array([wp,hp])-1


def register_one_depth_point(uv: np.ndarray, z: float, kd: np.ndarray, kc: np.ndarray, e_cd: np.ndarray, t_wd: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    p=z*(np.linalg.inv(kd)@np.r_[uv,1.]); ph=np.r_[p,1.]
    world=(t_wd@ph)[:3]; color=(e_cd@ph)[:3]
    if color[2]<=0:
        raise ValueError('behind color camera')
    image=kc@color
    return world,image[:2]/image[2]


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid=labels>=0
    if not bool(valid.any()):
        return logits.sum()*0.0
    return F.binary_cross_entropy_with_logits(logits[valid],labels[valid].to(logits.dtype))


class GroupReference(nn.Module):
    """A synthetic grouped nonlinear-key/raw-value attention reference."""
    def __init__(self, value_dim: int=5, hidden: int=7):
        super().__init__();self.embed=nn.Linear(value_dim,hidden);self.score=nn.Linear(hidden,1)
        self.ff=nn.Sequential(nn.Linear(hidden,2*hidden),nn.GELU(),nn.Linear(2*hidden,hidden))
        self.norm=nn.LayerNorm(hidden);self.query=nn.Parameter(torch.randn(hidden))
    def forward(self,x: torch.Tensor,groups: torch.Tensor,valid: torch.Tensor) -> torch.Tensor:
        keys=[];values=[]
        h=torch.tanh(self.embed(x))
        for g in torch.unique(groups[valid],sorted=True):
            mask=valid&(groups==g);a=torch.softmax(self.score(h[mask]).squeeze(-1),0)
            mean=(a[:,None]*h[mask]).sum(0)
            keys.append(self.norm(mean+self.ff(mean)));values.append((a[:,None]*x[mask]).sum(0))
        if not keys:
            raise ValueError('no local groups; production falls back to MA')
        k=torch.stack(keys);b=torch.stack(values);a=torch.softmax(k@self.query/(k.shape[-1]**.5),0)
        return (a[:,None]*b).sum(0)


def replace_f(scores: dict[str,np.ndarray|None], temperatures: dict[str,float], new_f: np.ndarray) -> np.ndarray:
    from scipy.special import softmax
    nf=np.asarray(new_f,np.float64)
    if nf.ndim!=1 or not np.isfinite(nf).all():raise ValueError('finite new full F scores required')
    nq=[]
    for name in ('N','Q'):
        if scores.get(name) is not None:
            nq.append(softmax(np.asarray(scores[name],np.float64)/temperatures[name]))
    pf=softmax(nf/temperatures['F']);return (np.mean(nq,axis=0)+pf)/2 if nq else pf


def upgrade_gate(rep: dict, cf: dict, g_rep: dict, g_cf: dict, d_cf: dict, tol: float=1e-10) -> bool:
    keys=('apall','ap50','ap25','miou','macc')
    values=[d[k] for d in (rep,cf,g_rep,g_cf,d_cf) for k in keys]
    if not np.isfinite(values).all():return False
    return all(rep[k]>=g_rep[k]-tol and cf[k]>=g_cf[k]-tol for k in keys) and cf['apall']>d_cf['apall']+tol and cf['ap50']>=d_cf['ap50']-tol
