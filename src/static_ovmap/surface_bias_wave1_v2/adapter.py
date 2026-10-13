"""Only raw values change. The parent query/key/quality graph is untouched."""
import time
import numpy as np
from .common import FORWARD_KEYS
from .solver import correction

STATS=dict(adapter_calls=0,cpu_solver_seconds=0.,cpu_gpu_copy_seconds=0.,corrected_readout_seconds=0.)

def adapt(head,inputs,phi,method,strength,*,original=None):
    result,audit=_adapt(head,inputs,phi,method,strength,original=original)
    STATS['adapter_calls']+=1
    for output,key in (('cpu_solver_seconds','solver_seconds'),('cpu_gpu_copy_seconds','copy_seconds'),('corrected_readout_seconds','readout_seconds')): STATS[output]+=audit[key]
    return result,audit

def _adapt(head,inputs,phi,method,strength,*,original=None):
    import torch
    if set(inputs)!=FORWARD_KEYS: raise ValueError('Exactly seven tensor inputs required')
    valid=inputs['local_valid'].bool() & inputs['view_valid'].bool()[:,None]
    raw=inputs['local_raw']
    if not bool(torch.isfinite(raw[valid]).all()): raise ValueError('Nonfinite valid original raw is not a fallback')
    if original is None: original=head(inputs,phi)
    audit=dict(method=method,lambda_=float(strength),numeric_fallback=False,active_views=int(valid.any(1).sum()),status='ZERO_LAMBDA',solver_seconds=0.,copy_seconds=0.,readout_seconds=0.,additional_phi_calls=0)
    if strength==0: return original,audit
    start=time.perf_counter(); cpu=raw.detach().cpu().numpy(); support=valid.detach().cpu().numpy()
    audit['copy_seconds']=time.perf_counter()-start
    start=time.perf_counter()
    try: value,estimate=correction(cpu,support,method,strength)
    except FloatingPointError:
        audit.update(status='NUMERIC_CORRECTION_FALLBACK',numeric_fallback=True); return original,audit
    audit.update(solver_seconds=time.perf_counter()-start,estimate=estimate,status=estimate['status'])
    if estimate['status']!='OK': return original,audit
    if not np.isfinite(value[support]).all():
        audit.update(status='NUMERIC_CORRECTION_FALLBACK',numeric_fallback=True); return original,audit
    start=time.perf_counter(); changed=dict(inputs); changed['local_raw']=torch.from_numpy(value).to(raw.device,dtype=raw.dtype)
    if raw.is_cuda: torch.cuda.synchronize()
    audit['copy_seconds']+=time.perf_counter()-start
    start=time.perf_counter()
    try:
        result=head(changed,phi)
    except ValueError as exc:
        if str(exc)!='Nonfinite or zero representation': raise
        audit.update(status='NUMERIC_CORRECTION_FALLBACK',numeric_fallback=True,additional_phi_calls=1); return original,audit
    if raw.is_cuda: torch.cuda.synchronize()
    audit.update(readout_seconds=time.perf_counter()-start,additional_phi_calls=int(result['group_count']>0))
    if not bool(torch.isfinite(result['embedding']).all()):
        audit.update(status='NUMERIC_CORRECTION_FALLBACK',numeric_fallback=True); return original,audit
    for key in ('quality_logits','quality_weights','alpha','null','energies','membership_logits'):
        if key in original and not torch.equal(original[key],result[key]): raise ValueError('Value-only invariant changed: '+key)
    audit['embedding_difference']=float(torch.linalg.vector_norm(result['embedding']-original['embedding']))
    if original['group_count']:
        weights=original['quality_weights']; views=original['view_indices']
        old_raw=torch.where(valid[views,:,None],raw[views],0)
        new_raw=torch.where(valid[views,:,None],changed['local_raw'][views],0)
        old_group=weights@old_raw.reshape(-1,raw.shape[-1])
        new_group=weights@new_raw.reshape(-1,raw.shape[-1])
        audit['group_raw_difference_norm']=float(torch.linalg.vector_norm(new_group-old_group))
        audit['group_raw_norm_change']=float(torch.linalg.vector_norm(new_group)-torch.linalg.vector_norm(old_group))
        audit['group_phi_difference']=float(torch.linalg.vector_norm(result['group_vectors']-original['group_vectors']))
        audit['attention_effect_norm']=float(torch.linalg.vector_norm(weights@(new_raw-old_raw).reshape(-1,raw.shape[-1])))
    return result,audit
