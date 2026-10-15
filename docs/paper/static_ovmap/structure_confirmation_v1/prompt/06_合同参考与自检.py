"""CPU synthetic contract checks only. No repository, data, models or Git writes.
Run: python 06_合同参考与自检.py --output 07_实际合同检查结果.json
Production must implement the full contracts; this is not a scientific worker.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch import nn


def norm(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if not torch.isfinite(x).all() or bool((n <= 1e-12).any()):
        raise ValueError('Undefined synthetic representation')
    return x / n


def permute_valid(x: dict[str, torch.Tensor], key: str, condition: str) -> dict[str, torch.Tensor]:
    out = {k: v.clone() for k, v in x.items()}
    for view, frame in enumerate(x['frame_ids'].tolist()):
        ix = torch.nonzero(x['local_valid'][view], as_tuple=False).flatten()
        token = json.dumps(['FG2_GROUP_CONTROL_V1', key, condition, int(frame), 9173],
                           ensure_ascii=False, separators=(',', ':')).encode()
        seed = int.from_bytes(hashlib.sha256(token).digest()[:8], 'big')
        order = torch.as_tensor(np.random.default_rng(seed).permutation(len(ix)), dtype=torch.long)
        for k in ('local_raw', 'local_unit', 'metadata'):
            out[k][view, ix] = x[k][view, ix[order]]
    return out


def factors(raw: torch.Tensor, hidden: torch.Tensor, quality: torch.Tensor, valid: torch.Tensor,
            base: torch.Tensor, phi: nn.Module, queries: torch.Tensor,
            inner_learned: bool, outer_learned: bool, pre_project: bool = False,
            grouping: str = 'SURFACE') -> dict[str, torch.Tensor]:
    v, s = valid.shape
    ids = torch.arange(s)[None].expand(v, s) if grouping == 'SURFACE' else torch.arange(v)[:, None].expand(v, s)
    present = torch.unique(ids[valid], sorted=True)
    members = (present[:, None] == ids.flatten()[None]) & valid.flatten()[None]
    if not len(present):
        return {'z': base, 'alpha': base.new_zeros(queries.shape[0], 0), 'a': base.new_zeros(0, v*s)}
    if inner_learned:
        a = torch.softmax(quality.flatten()[None].expand(len(present), -1).masked_fill(~members, -torch.inf), -1)
    else:
        a = members.to(raw.dtype) / members.sum(-1, keepdim=True)
    keys = a @ hidden.reshape(-1, hidden.shape[-1])
    flat = raw.reshape(-1, raw.shape[-1])
    values = a @ phi(flat) if pre_project else phi(a @ flat)
    u = norm(values)
    e = 4 * torch.tanh(queries @ keys.T / np.sqrt(hidden.shape[-1]))
    w = torch.exp(e) / len(present)
    alpha = w / (1 + w.sum(-1, keepdim=True))
    mass = alpha.sum(-1, keepdim=True)
    if not outer_learned:
        alpha = mass.expand_as(alpha) / len(present)
    z = norm(base + .5 * (alpha @ (u - base)).mean(0))
    return {'z': z, 'a': a, 'alpha': alpha, 'mass': mass, 'u': u, 'e': e}


def select_seed17(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # Test fixture has no near ties; production uses tolerance 1e-10.
    return sorted(rows, key=lambda r: (-r['A8'], -r['M8'], -r['C8'], r['step']))[0]


def namespace_matches(matches: dict[str, Any], token: str) -> dict[str, Any]:
    def rename(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: token+'|'+v if k == 'filename' and isinstance(v, str) else rename(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rename(v) for v in x]
        return copy.deepcopy(x)
    return {token+'|'+k: rename(v) for k, v in matches.items()}


def run_checks(spec_path: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    spec = json.loads(spec_path.read_text())
    results: list[dict[str, Any]] = []
    check_count = 0
    def record(group: str, names: list[str]) -> None:
        nonlocal check_count
        check_count += len(names)
        results.append({'group': group, 'status': 'PASS', 'checks': names})

    v, s, c, d, h = 4, 7, 6, 5, 8
    raw = torch.randn(v, s, c, dtype=torch.float64)
    local = torch.randn(v, s, d, dtype=torch.float64)
    meta = torch.randn(v, s, 8, dtype=torch.float64)
    valid = torch.tensor([[1,1,1,1,0,0,1], [1,0,1,1,1,1,1], [0,1,1,0,1,1,0], [1,1,0,1,1,0,1]], dtype=torch.bool)
    x = {'local_raw':raw,'local_unit':local,'metadata':meta,'local_valid':valid,
         'fc_vectors':torch.randn(v,d,dtype=torch.float64),'view_valid':torch.ones(v,dtype=torch.bool),
         'frame_ids':torch.tensor([2,6,11,30])}
    y = permute_valid(x, 'synthetic-object', 'clean')
    for key in ('local_valid','fc_vectors','view_valid','frame_ids'):
        assert torch.equal(x[key],y[key])
    for view in range(v):
        a = torch.cat([x[k][view] for k in ('local_raw','local_unit','metadata')],-1)[valid[view]]
        b = torch.cat([y[k][view] for k in ('local_raw','local_unit','metadata')],-1)[valid[view]]
        assert sorted(map(tuple,a.tolist())) == sorted(map(tuple,b.tolist()))
    small = {k:value[:2] for k,value in x.items()}
    shuffled_small = permute_valid(small,'synthetic-object','clean')
    assert all(torch.equal(y[k][:2],shuffled_small[k]) for k in x)
    assert torch.equal(valid.sum(0),y['local_valid'].sum(0))
    record('T1_INPUT_GROUPING', ['seven_keys_unchanged_nonlocal','joint_token_multiset','per_group_counts','prefix_does_not_change_random_map'])

    phi = nn.Sequential(nn.Linear(c,9), nn.GELU(), nn.Linear(9,d)).double().eval().requires_grad_(False)
    hidden = torch.randn(v,s,h,dtype=torch.float64,requires_grad=True)
    q = torch.randn(v,s,dtype=torch.float64,requires_grad=True)
    queries = torch.randn(4,h,dtype=torch.float64,requires_grad=True)
    base = norm(torch.randn(d,dtype=torch.float64))
    ll = factors(raw,hidden,q,valid,base,phi,queries,True,True)
    lu = factors(raw,hidden,q,valid,base,phi,queries,True,False)
    assert torch.allclose(ll['alpha'].sum(-1),lu['alpha'].sum(-1),atol=1e-14,rtol=0)
    assert torch.allclose(lu['alpha'],lu['alpha'][:,:1].expand_as(lu['alpha']),atol=1e-14,rtol=0)
    target = norm(torch.randn(d,dtype=torch.float64)); loss = -(lu['z']@target)
    loss.backward(retain_graph=True)
    assert q.grad is not None and float(q.grad.norm())>0
    assert queries.grad is not None and float(queries.grad.norm())>0
    assert all(p.grad is None for p in phi.parameters())
    pre = factors(raw,hidden,q,valid,base,phi,queries,True,True,True)
    assert not torch.allclose(pre['u'],ll['u'],atol=1e-10,rtol=0)
    zero = factors(raw,hidden,q,torch.zeros_like(valid),base,phi,queries,True,True)
    assert torch.equal(zero['z'],base)
    pUU,pLU,pUL,pLL=.21,.29,.31,.43
    inner=.5*((pLL-pUL)+(pLU-pUU)); outer=.5*((pLL-pLU)+(pUL-pUU))
    assert abs(inner+outer-(pLL-pUU))<1e-12
    record('T2_ARITHMETIC_GRADIENT', ['outer_mass_preserved','relative_uniformity','quality_gradient_through_frozen_phi',
                                      'own_mass_gradient_not_detached','backbone_no_gradient','projection_order_is_distinct',
                                      'no_local_exact_fallback','factorial_sum_without_double_interaction'])

    rows=[{'step':500,'A8':.42,'M8':.5,'C8':.5},{'step':1000,'A8':.44,'M8':.5,'C8':.5},
          {'step':1500,'A8':.44,'M8':.5,'C8':.5}]
    winner=select_seed17(rows);assert winner['step']==1000
    required=set(spec['train']['checkpoints']);assert required=={250,500,1000,1500,2000}
    assert spec['selection']['seed29_reselect'] is False
    model=nn.Linear(3,2).double();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    sample=torch.randn(4,3,dtype=torch.float64)
    def update(m,o):
        o.zero_grad();value=m(sample).square().mean();value.backward();o.step()
    update(model,optimizer)
    state=copy.deepcopy({'m':model.state_dict(),'o':optimizer.state_dict()})
    update(model,optimizer)
    restored=nn.Linear(3,2).double();restored.load_state_dict(state['m'])
    opt=torch.optim.AdamW(restored.parameters(),lr=1e-3);opt.load_state_dict(state['o']);update(restored,opt)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
    record('T3_SELECTION_RESUME', ['seed17_earliest_tie','all_checkpoint_states_required','seed29_no_reselect','one_update_resume_exact_cpu_fixture'])

    train=spec['train']; nm=len(train['methods']);ns=len(train['seeds'])
    assert nm==8 and ns==2
    assert nm*ns*train['updates_per_arm']==train['max_new_updates']==32000
    assert nm*ns*train['updates_per_arm']*32==spec['limits']['train_head_calls_nominal']
    assert nm*ns*len(train['checkpoints'])*69*4==22080
    slots=nm*ns+len(spec['static_methods']);assert slots==19
    extras=len(spec['confirmation']['extra_matched_step_heads'])*ns
    assert extras==16 and (slots+extras)*12==spec['confirmation']['max_real_AP_scene_rows']==420
    max_confirm=768*4*3*slots+768*3*slots+768*4*extras+768*extras
    assert max_confirm==280320 and max_confirm<=spec['limits']['max_confirmation_head_calls']
    map_slots=len(spec['map_regression']['learned_methods'])*ns+len(spec['map_regression']['static_methods'])
    assert map_slots*2*26==572 and map_slots*2*2==44
    assert slots*16*3*4==3648 and 3648<=spec['limits']['max_cost_head_calls']
    assert len(spec['statistics']['confirmation_pairs'])*1000==4000
    left=namespace_matches({'scan':{'gt':{'pred':[{'filename':'mask.npy'}]}}},'COPY0')
    right=namespace_matches({'scan':{'gt':{'pred':[{'filename':'mask.npy'}]}}},'COPY1')
    assert set(left).isdisjoint(right)
    assert left['COPY0|scan']['gt']['pred'][0]['filename'] != right['COPY1|scan']['gt']['pred'][0]['filename']
    assert spec['confirmation']['replace_missing_scene'] is False
    assert spec['confirmation']['prediction_lock_before_real_GT_parse'] is True
    assert spec['full_external_reproduction'].startswith('NOT_IN_SCOPE')
    record('T4_COUNTS_SCOPE_EVALUATION', ['32000_updates','1024000_training_heads','22080_DEV_heads','19_selected_predictors',
                                         '16_matched_predictors','420_real_AP_rows','280320_confirmation_head_upper_bound',
                                         '572_old_map_rows_44_pools','3648_cost_calls','4000_paired_draw_sets',
                                         'duplicated_scan_and_filename_isolation','no_outcome_scene_replacement',
                                         'real_predictions_before_GT','external_adaptation_scope'])
    return {'status':'PASS','scope':'PACKAGE_SYNTHETIC_CONTRACT_ONLY_NOT_REAL_FC_TRAINING_OR_SCORER',
            'groups':results,'group_count':len(results),'individual_assertion_topics':check_count,
            'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            'torch_version':torch.__version__,'numpy_version':np.__version__,
            'new_real_training_updates':0,'new_RGB_encodings':0,'new_SAM_encodings':0,'new_map_scores':0,
            'Github_publication_performed':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,default=Path(__file__).with_name('02_冻结执行配置.json'))
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('07_实际合同检查结果.json'))
    args=parser.parse_args();result=run_checks(args.spec);args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'groups':result['group_count'],'topics':result['individual_assertion_topics']},ensure_ascii=False))
