"""Explicit one-time lineage for the DEV key fix before the first G evaluation.

Existing S/V computations do not consume object keys. Resume generations are
rebound only after their complete model/optimizer/RNG/sampler payload is proven
unchanged. Original bytes remain outside Git; permanent scientific states and
scores are never rewritten. This is not a general identity bypass.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import torch
from .common import (REPO,TASK,verified,write,canonical_digest,digest_file,
    unit_key,ConsumptionIndex,objects)
from .training import PRODUCERS

def payload_digest(value):
    digest=hashlib.sha256()
    def visit(v):
        digest.update(type(v).__name__.encode()+b'\0')
        if isinstance(v,torch.Tensor):
            a=v.detach().cpu().contiguous().numpy();digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(a.tobytes())
        elif isinstance(v,np.ndarray):
            digest.update(v.dtype.str.encode());digest.update(str(v.shape).encode());digest.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=repr):visit(k);visit(v[k])
        elif isinstance(v,(tuple,list)):
            for item in v:visit(item)
        else:digest.update(repr(v).encode())
    visit(value);return digest.hexdigest()

def run(binding):
    root=Path(binding['output_root']);before=verified(root/'review/dev_key_repair_before.json');folder=REPO/'src/static_ovmap/structure_confirmation_v1';index=ConsumptionIndex()
    old_source=before['source'];new_source=(folder/'recognition.py').read_text()
    if old_source.replace('obj=items[self.position//4]','obj=selected[self.position//4]')!=new_source:raise ValueError('Only the exact DEV key indexing repair is authorized')
    if before['G_DEV_files'] or list((root/'training/seed17/G_LL').glob('dev_*.json')):raise ValueError('This repair cannot silently reuse prior G validation outputs')
    hashes={n:digest_file(folder/(n+'.py')) for n in PRODUCERS}
    if any(hashes[n]!=before['producer_sha256'][n] for n in PRODUCERS if n!='recognition'):raise ValueError('Other frozen training sources changed')
    dev=objects(binding,'dev');base=[o for o in dev if o['base']]
    mismatches=[dict(position=i,legacy_key=dev[i]['key'],correct_key=o['key']) for i,o in enumerate(base) if dev[i]['key']!=o['key']]
    contract=write(root/'review/dev_key_repair_contract.json',dict(status='COMPLETE',before=before['identity'],old_producer=before['producer_digest'],new_producer=canonical_digest(hashes),old_recognition_source=old_source,new_recognition_source=new_source,only_change='Adapter uses selected base-object list for deterministic G key',unaffected_methods=['S_LL','V_LL','B_FC','B_SURFACE_MEAN','B_PFE_ADAPTED'],unaffected_proof='method_inputs returns the identical x and None for every method except G_LL; key and condition have no computational use in that branch. Training sample order, draw schedule, model, loss, optimizer and RNG code are unchanged.',DEV_original_count=len(dev),DEV_base_count=len(base),mismatched_positions=mismatches,G_DEV_before_fix=0,scientific_permanent_states_rewritten=False))
    features=verified(root/'features/train-dev.json');prepare=verified(root/'prepare_cache.json')
    with np.load(root/'training/seed17/draws.npz',allow_pickle=False) as a:schedule=a['draws'].copy()
    inputs=dict(binding=binding['identity'],prepare=prepare['identity'],features=features['identity'],FC=prepare['FC_identity'],draws=canonical_digest(schedule.tolist()))
    transformed=[]
    for method in ('S_LL','V_LL','G_LL'):
        directory=root/'training/seed17'/method;originals=root/'review/dev_key_repair/original/seed17'/method;originals.mkdir(parents=True,exist_ok=True)
        legacy=canonical_digest(dict(task=TASK,schema=1,phase='train_arm',producer=before['producer_digest'],inputs=inputs,parameters=dict(seed=17,method=method)))
        current=unit_key('train_arm',inputs,producers=PRODUCERS,seed=17,method=method)
        for name in ('last.pt','last.previous.pt'):
            source=directory/name
            if not source.exists():continue
            archive=originals/name
            if not archive.exists():shutil.copyfile(source,archive)
            state=torch.load(archive,map_location='cpu',weights_only=False)
            if state['input_identity']!=legacy:raise ValueError('Legacy resume does not bind the recorded producer')
            unchanged={k:v for k,v in state.items() if k!='input_identity'};proof=payload_digest(unchanged)
            current_state=torch.load(source,map_location='cpu',weights_only=False)
            if current_state['input_identity'] not in (legacy,current):raise ValueError('Unexpected resume identity')
            if current_state['input_identity']==legacy:
                state['input_identity']=current;temp=source.with_name(source.name+'.rebind.tmp');torch.save(state,temp);temp.replace(source)
            restored=torch.load(source,map_location='cpu',weights_only=False)
            if payload_digest({k:v for k,v in restored.items() if k!='input_identity'})!=proof:raise ValueError('Rebound resume changed model/optimizer/RNG/sampler content')
            transformed.append(dict(kind='RESUME_METADATA_ONLY',method=method,original=index.identity(archive),current=index.identity(source),payload_sha256=proof,completed_step=state['step'],model_optimizer_RNG_sampler_unchanged=True))
        receipt=directory/'receipt.json'
        if receipt.exists():
            archive=originals/'receipt.json'
            if not archive.exists():shutil.copyfile(receipt,archive)
            original=verified(archive)
            if method not in ('S_LL','V_LL') or original['status']!='COMPLETE' or original['input_identity']!=legacy:raise ValueError('Unexpected completed arm for this bounded repair')
            model_bytes=[index.identity(r['model']['path'],r['model']) for r in original['validations']]
            value={k:v for k,v in original.items() if k!='identity'};value.update(input_identity=current,execution_source_producer=before['producer_digest'],equivalent_current_producer=canonical_digest(hashes),source_repair_contract=contract['identity'],original_execution_receipt=index.identity(archive),execution_kind='FRESH_SAME_TASK;UNMODIFIED_NUMERICAL_RESULT;COMPUTE_EQUIVALENCE_AFTER_DEV_KEY_FIX')
            write(receipt,value)
            transformed.append(dict(kind='UNCHANGED_SAME_TASK_NUMERICAL_RESULT',method=method,original=index.identity(archive),current=index.identity(receipt),permanent_model_bytes=model_bytes,all_five_DEV_files_unchanged=True))
        partial=directory/'failed_partial_update.json'
        if partial.exists():
            archive=originals/partial.name
            if not archive.exists():shutil.copyfile(partial,archive)
            value=verified(archive)
            if value['input_identity']!=legacy:raise ValueError('Unexpected interruption identity')
            value.pop('identity');value.update(input_identity=current,original_failure=index.identity(archive));write(partial,value)
    return write(root/'review/dev_key_repair.json',dict(status='COMPLETE',contract=contract['identity'],transformed=transformed,new_scientific_training_updates=0,new_head_calls=0,originals_preserved=True,G_pre_DEV_replay_accounted_by_existing_resume_logic=True,not_a_parent_checkpoint_warm_start=True))

if __name__=='__main__':
    import argparse
    from .binding import load_binding
    parser=argparse.ArgumentParser();parser.add_argument('--output-root',required=True);args=parser.parse_args()
    print(json.dumps(run(load_binding(args.output_root)),ensure_ascii=False,indent=2))
