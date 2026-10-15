"""Exact source-derived phi workload from executed draws and cached validity.

This reads saved inputs and ledgers only. It runs no model and does not present
derived phi counts as a measured kernel timer or as image encodings.
"""
from collections import Counter
from pathlib import Path
import numpy as np
from .common import verified,write,canonical_digest,objects,ConsumptionIndex,METHODS

def summarize(binding):
    root=Path(binding['output_root']);training=verified(root/'training/receipt.json');budget=verified(root/'workload.json');prepare=verified(root/'prepare_cache.json');items=objects(binding,'train');namespace=prepare['sparse_namespace'];cache={};index=ConsumptionIndex(root/'features/verifications.json')
    def layout(oi,condition,prefix):
        key=(int(oi),int(condition));obj=items[int(oi)]
        if key not in cache:
            digest=canonical_digest(dict(namespace=namespace,object_inputs=obj['inputs']['sha256'],object_key=obj['key'],views=obj['views'],condition=int(condition)))
            paths=[p/'features/sparse'/namespace/(digest+'.json') for p in (root,Path(binding['fg_parent_root']))];source=next(p for p in paths if p.is_file());record=verified(source)
            if record['key']!=digest or record['source_inputs']['sha256']!=obj['inputs']['sha256']:raise ValueError('Workload sparse input differs')
            index.identity(record['arrays']['path'],record['arrays'])
            with np.load(record['arrays']['path'],allow_pickle=False) as a:cache[key]=(a['view_valid'].copy(),a['local_valid'].copy())
        views,valid=cache[key];v=valid[:prefix][views[:prefix].astype(bool)]
        return dict(calls=int(bool(v.any())),surface_vectors=int(v.any(0).sum()),view_vectors=int(v.any(1).sum()),pre_vectors=int(v.sum()))
    def count(draws):
        frequencies=Counter()
        for oi,prefix,condition in draws:frequencies[(int(oi),0,int(prefix))]+=1;frequencies[(int(oi),int(condition),int(prefix))]+=1
        total=dict(calls=0,surface_vectors=0,view_vectors=0,pre_vectors=0)
        for (oi,condition,prefix),n in frequencies.items():
            for k,v in layout(oi,condition,prefix).items():total[k]+=n*v
        return total
    rows=[];schedules={}
    for seed in (17,29):
        with np.load(root/f'training/seed{seed}/draws.npz',allow_pickle=False) as a:schedule=a['draws'].copy()
        schedules[seed]=schedule;counts=count(schedule)
        for method in METHODS:
            receipt=verified(root/f'training/seed{seed}/{method}/receipt.json');calls=0 if method=='MV_ATTN' else counts['calls'];vectors=0 if method=='MV_ATTN' else counts['pre_vectors' if method=='S_PRE' else 'view_vectors' if method=='V_LL' else 'surface_vectors']
            rows.append(dict(method=method,seed=seed,completed_updates=receipt['completed_steps'],nominal_head_forwards=64000,nominal_phi_forward_calls=calls,nominal_phi_projected_vectors=vectors,elapsed_seconds=receipt['elapsed_seconds'],peak_allocated_bytes=receipt['peak_allocated_bytes'],peak_reserved_bytes=receipt['peak_reserved_bytes'],parameters=receipt['model_parameters'],trainable_parameters=receipt['trainable_parameters']))
    DEV=[]
    for p in sorted((root/'training').glob('seed*/*/dev_*.json')):
        method=p.parent.name;record=verified(p);coverage=[c for r in record['records'] for c in r['coverage']];calls=0 if method=='MV_ATTN' else sum(c['group_count'] is not None and c['group_count']>0 for c in coverage);vectors=0 if method=='MV_ATTN' else sum(c['local_observations'] if method=='S_PRE' else c['group_count'] or 0 for c in coverage)
        DEV.append(dict(record=str(p),head_calls=record['head_or_base_forward_calls'],phi_forward_calls=calls,phi_projected_vectors=vectors))
    checkpoint_heads=sum(v['head_calls'] for v in DEV);static_heads=sum(verified(p)['head_or_base_forward_calls'] for p in (root/'recognition').glob('dev_B_*.json'));extra=budget['counts']['main_head']-1024000-checkpoint_heads-static_heads
    replay=None
    if extra==3108 and len(list((root/'training').glob('seed*/*/replay_*.json')))==1:
        # Recorded interruption: G seed17 completed97 updates and two micro pairs
        # of update98, before its first DEV. Counters prove all3108 extra calls.
        replay=dict(method='G_LL',seed=17,actual_extra_head_forwards=extra,**count(schedules[17][:97*16+2]),basis='DEV_KEY_REPAIR_FAILURE_LEDGER_AND_MASTER_COUNTER;NO_ADDITIONAL_INFERENCE')
    result=write(root/'training/workload_detail.json',dict(status='COMPLETE',training=training['identity'],rows=rows,DEV=DEV,checkpoint_DEV_head_forwards=checkpoint_heads,static_DEV_head_forwards=static_heads,actual_main_head_forwards=budget['counts']['main_head'],actual_training_head_forwards=budget['counts']['main_head']-checkpoint_heads-static_heads,extra_training_or_replayed_validation_forwards=extra,exact_known_replay_phi=replay,actual_optimizer_updates=budget['counts']['optimizer_updates'],replayed_updates=budget['counts']['replayed_updates'],phi_count_evidence='SOURCE_DERIVED_FROM_ACTUAL_EXECUTED_DRAWS_AND_HASH_VERIFIED_CACHED_VALID_LAYOUT;NOT_KERNEL_TIME_MEASUREMENT',training_phi_kernel_seconds=None,old_source_image_encodings=0,additional_head_calls=0,additional_image_encodings=0))
    index.write_memo(root/'training/accounting_verifications.json');return result

if __name__=='__main__':
    import argparse
    from .binding import load_binding
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True);args=parser.parse_args();print(summarize(load_binding(args.output_root))['identity'])
