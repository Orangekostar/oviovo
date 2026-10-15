"""Resident-model timings include parent region pooling and nested head time."""
import hashlib
from pathlib import Path
import time
import numpy as np
import torch
from .common import verified,write,immutable_record,unit_key,existing,objects,arrays_record
from .registry import load_registry
from .grouping import method_inputs
from static_ovmap.surface_bias_wave1_v2.data import ObjectLoader

def measure(binding,fc,budget):
    root=Path(binding['output_root']);lock=verified(root/'predictor_lock.json');identity=unit_key('cost',lock['identity'],producers=('cost','registry','models','grouping'),objects=16,warmup=1,repeats=3)
    path=root/'cost/receipt.json';old=existing(path,identity)
    if old:return old
    selected=sorted([o for o in objects(binding,'dev') if o['base']],key=lambda o:(hashlib.sha256(o['key'].encode()).hexdigest(),o['key']))[:16]
    slots,heads,_=load_registry(binding,fc,selected_only=True)
    for head in heads.values():head.cpu()
    loader=ObjectLoader(fc,verified(root/'features/train-dev.json'),maximum_GiB=32);rows=[];start=time.perf_counter()
    with torch.no_grad():
        for slot in slots:
            head=heads[slot['unique_model']].to(fc.device);torch.cuda.synchronize();torch.cuda.empty_cache()
            baseline_allocated=torch.cuda.memory_allocated();baseline_reserved=torch.cuda.memory_reserved()
            for prefix in (2,4,8):
                for obj in selected:
                    for repeat in range(4):
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
                        x,_,_=loader.load(obj,0,prefix);x,_=method_inputs(slot['method'],x,obj['key'],'clean');torch.cuda.synchronize();before=time.perf_counter()
                        budget.consume('cost_head',persist=False);result=head(x,fc.phi);torch.cuda.synchronize();after=time.perf_counter();z=result['embedding'].cpu().numpy();torch.cuda.synchronize();end=time.perf_counter()
                        rows.append(dict(slot=slot['id'],method=slot['method'],seed=slot['seed'],prefix=prefix,key=obj['key'],repeat=repeat,warmup=repeat==0,total_seconds=end-begin,head_seconds=after-before,load_pool_transfer_seconds=(before-begin)+(end-after),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),baseline_allocated_bytes=baseline_allocated,baseline_reserved_bytes=baseline_reserved,views=int(x['view_valid'].sum()),groups=result['group_count']))
                print('COST',slot['id'],prefix,flush=True)
            budget.flush()
            del x,result,z
            head.cpu();torch.cuda.synchronize();torch.cuda.empty_cache()
    summaries=[]
    for slot in slots:
        for prefix in (2,4,8):
            measured=[r for r in rows if r['slot']==slot['id'] and r['prefix']==prefix and not r['warmup']]
            summaries.append(dict(slot=slot['id'],prefix=prefix,n=len(measured),parameters=sum(p.numel() for p in heads[slot['unique_model']].parameters()),**{key+suffix:float(np.quantile([r[key] for r in measured],q)) for key in ('total_seconds','head_seconds','load_pool_transfer_seconds','peak_allocated_bytes','peak_reserved_bytes') for suffix,q in (('_median',.5),('_p90',.9))}))
    if len(rows)!=3648:raise ValueError('Fixed cost3648 invocation budget differs')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,rows=rows,summaries=summaries,object_keys=[o['key'] for o in selected],object_sort='SHA256_UTF8_KEY_ASC_KEY_TIE',head_calls=3648,warmup_included_in_calls=True,head_time_nested_in_total=True,model_loading_excluded=True,resident_models='FROZEN_FC_AND_ONE_MEASURED_HEAD;OTHER_HEADS_ON_CPU',new_RGB_encodings=fc.images,elapsed_seconds=time.perf_counter()-start,boundary='MODEL_RESIDENT_PARENT_CACHED_GRIDS_REGION_POOL_PHI_TRANSFER_AND_HEAD;NOT_END_TO_END_MAPPING_FPS'))
