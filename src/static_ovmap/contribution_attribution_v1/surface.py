"""Joint token interventions preserve view inputs and their valid layouts."""
import hashlib
import json
import numpy as np
from .common import TASK

def permute_valid(x,key,condition,shuffle_seed):
    import torch
    out={name:value.clone() for name,value in x.items()}
    for view,frame in enumerate(x['frame_ids'].tolist()):
        valid=torch.nonzero(x['local_valid'][view],as_tuple=False).flatten()
        token=json.dumps([TASK,key,condition,int(frame),int(shuffle_seed)],ensure_ascii=False,separators=(',',':'))
        seed=int.from_bytes(hashlib.sha256(token.encode()).digest()[:8],'big')
        rng=np.random.default_rng(seed)
        permutation=torch.as_tensor(rng.permutation(len(valid)),device=valid.device,dtype=torch.long)
        source=valid[permutation]
        for name in ('local_raw','local_unit','metadata'):
            out[name][view,valid]=x[name][view,source]
    return out

def permute_slots(x,indices):
    import torch
    indices=torch.as_tensor(indices,dtype=torch.long,device=x['local_valid'].device)
    slots=x['local_valid'].shape[1]
    if len(indices)!=slots or sorted(indices.tolist())!=list(range(slots)):raise ValueError('Common slot mapping must be a permutation of all slots')
    out={name:value.clone() for name,value in x.items()}
    for name in ('local_raw','local_unit','metadata','local_valid'):
        out[name]=x[name].index_select(1,indices)
    return out

def _historical(binding,index):
    from pathlib import Path
    from .common import read_document,canonical_digest
    from static_ovmap.preservation_routing.selection import summarize
    parent=Path(binding['fg_parent_root']);reference=read_document(parent/'recognition/dev_FC8.json',index)
    expected=sorted((r['key'],r['class_id']) for r in reference['records']);rows=[];selected=[]
    if len(expected)!=69:raise ValueError('DEV must contain exactly 69 base originals')
    for seed in (17,29):
        directory=parent/f'training/seed{seed}'
        draw=read_document(directory/'draw_contract.json',index);index.identity(directory/'draws.npz')
        with np.load(directory/'draws.npz',allow_pickle=False) as arrays:
            if canonical_digest(arrays['draws'].tolist())!=draw['schedule_digest']:raise ValueError('Draw schedule changed')
        initial=[]
        for method in ('FG1_VIEW','FG2_SURFACE'):
            receipt=read_document(directory/method/'receipt.json',index);initial.append(receipt['initial_model_digest'])
            if receipt['input_contract']['draws']!=draw['schedule_digest']:raise ValueError('Arm draws differ')
            path=directory/method/'dev_1000.json'
            if not path.exists():
                rows.append(dict(seed=seed,method=method,status='MISSING_STEP1000_RECEIPT'));continue
            result=read_document(path,index);index.identity(result['arrays']['path'],result['arrays'])
            if sorted((r['key'],r['class_id']) for r in result['records'])!=expected:raise ValueError('Matched DEV key/classes differ')
            with np.load(result['arrays']['path'],allow_pickle=False) as arrays:
                pairs=sorted(zip(arrays['object_keys'].tolist(),arrays['class_ids'].tolist()))
                if pairs!=expected or arrays['full200_cosines'].shape!=(69,4,200):raise ValueError('Matched arrays support differs')
            rows.append(dict(seed=seed,method=method,step=1000,status='COMPLETE',metrics=summarize(result['records'],reference['records']),
                             receipt=result['identity'],arrays=result['arrays'],weight_at_step1000_available='NOT_INFERRED_FROM_RECORDS'))
            step=receipt['selected_step'];saved=read_document(directory/method/f'dev_{step:04d}.json',index)
            selected.append(dict(seed=seed,method=method,step=step,metrics=summarize(saved['records'],reference['records']),receipt=saved['identity']))
        if len(set(initial))!=1:raise ValueError('Same-seed arm initialization differs')
    return reference,dict(status='COMPLETE' if len(rows)==4 and all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL_DEPENDENCY',
                          matched_step=rows,selected=selected,full200_competitors=True,new_training=0)

def _score_record(obj,z,fc):
    import torch
    import torch.nn.functional as F
    score=z@fc.text_tensor.T;t=fc.ids.index(obj['class_id']);other=score.clone();other[t]=-torch.inf
    return dict(prediction=fc.ids[int(score.argmax())],CE=float(F.cross_entropy((z@fc.base_text.T/.07)[None],torch.tensor([fc.base_targets[obj['class_id']]],device=fc.device))),
                NLL=float(-F.log_softmax(score/.07,dim=0)[t]),margin=float(score[t]-other.max())),score

def _records(items,values,mode):
    return [dict(key=o['key'],family=o['family'],class_id=o['class_id'],base=True,
                 predictions=[values[(o['key'],c)][mode]['prediction'] for c in range(4)],
                 CE=[values[(o['key'],c)][mode]['CE'] for c in range(4)],
                 NLL=[values[(o['key'],c)][mode]['NLL'] for c in range(4)],
                 margins=[values[(o['key'],c)][mode]['margin'] for c in range(4)]) for o in items]

def run(root):
    """Only selected frozen heads, DEV intervention and two engineering originals."""
    import os
    import time
    from pathlib import Path
    import torch
    from .binding import load_binding
    from .common import (ConsumptionIndex,PathResolver,read_document,verified,write,unit_identity,
                         cached_record,digest_file,reserve,DependencyError)
    from static_ovmap.frozen_fc_routing.common import objects,arrays_record
    from static_ovmap.frozen_fc_routing.export import load_exported
    from static_ovmap.frozen_fc_routing.resources import queued_gpu_lock
    from static_ovmap.learned_object_readout.features import FrozenFC,execution_config
    from static_ovmap.surface_bias_wave1_v2.data import ObjectLoader
    from static_ovmap.preservation_routing.selection import summarize
    root=Path(root);binding=load_binding(root);spec=binding['task_specification'];index=ConsumptionIndex()
    parent=Path(binding['fg_parent_root']);registry=read_document(parent/'export/weights_registry.json',index)
    reference,history=_historical(binding,index);history=write(root/'surface/history.json',history)
    items=sorted([o for o in objects(binding,'dev') if o['base']],key=lambda o:o['key'])
    engineering=sorted([o for o in objects(binding,'train') if o['base']],key=lambda o:o['key'])[:2]
    if len(items)!=69 or len(engineering)!=2:raise ValueError('Fixed S support differs')
    features=read_document(root/'features/train-dev.json',index);resolver=PathResolver(binding['path_map']);features=resolver.rewrite(features)
    direct={}
    for obj in items+engineering:
        direct[obj['key']]=dict(inputs=index.identity(obj['inputs']['path'],obj['inputs']),frames=[])
        for view in obj['views'][:8]:
            frame=features['frames'][obj['scene']+':'+str(view['frame_id'])]
            direct[obj['key']]['frames'].append(index.identity(frame['arrays']['path'],frame['arrays']))
    identity=unit_identity('surface',dict(binding=binding['identity'],registry=registry['identity'],history=history['identity'],direct=direct),
                           producers=('common','binding','surface'),seeds=spec['recognition']['shuffle_seeds'],role='S_DEV',prefix=8)
    previous=cached_record(root/'surface.json',identity)
    if previous:return previous
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=binding['gpu']:raise ValueError('Single inherited GPU visibility required')
    names=('FG2_SURFACE','S29_FG2_SURFACE');expected_sha=('de7fc82edba5440d9c338f5253ecd3600a340271089600c9ff2d286e2711d963','501a6ad010fd2951c93a7638fce94c599b994c7fe9215adad82054919a84ea77')
    for name,sha,step in zip(names,expected_sha,(1000,2000)):
        row=registry['heads'][name];weights=resolver.rewrite(row['weights'])
        index.identity(weights['path'],weights)
        if weights['sha256']!=sha or row['selected_step']!=step:raise ValueError('Locked selected FG2 head changed')
    modes=['ORIGINAL']+['SHUFFLE_'+str(s) for s in spec['recognition']['shuffle_seeds']]
    with queued_gpu_lock(execution_config(binding)['gpu_lock']):
        start=time.perf_counter();reserve(root);fc=FrozenFC(binding);torch.cuda.reset_peak_memory_stats()
        def forbidden_capture(*args,**kwargs):raise RuntimeError('RGB capture is forbidden by this task')
        fc.capture=forbidden_capture
        if fc.model_key!=registry['FC_identity'] or fc.model_key!='6132b1b1da69a7a3bcd650586f0bc7e9aea2b64d8183a004b1a61c39892c8b44':raise ValueError('Frozen FC model identity changed')
        text=resolver.rewrite(registry['text_identity']);index.identity(text['path'],text)
        if text['sha256']!='8e3a532d4945493f7ecaab63168fab9232c8d9d0693c717893c7f9b73a88c4b4' or text['bytes']!=669136:raise ValueError('Frozen text bytes changed')
        if fc.session.text_identity['sha256']!=text['sha256'] or fc.session.text_identity['bytes']!=text['bytes']:raise ValueError('Runtime text bytes differ')
        runtime=dict(status='COMPLETE',FC_identity=fc.model_key,text_identity=fc.session.text_identity,heads={})
        folder=root/'surface/runtime';folder.mkdir(parents=True,exist_ok=True)
        for name in names:
            row=dict(registry['heads'][name]);weights=resolver.rewrite(row['weights']);link=folder/(name+'.pt')
            if not link.exists():link.symlink_to(weights['path'])
            row['weights']=dict(weights,relative_path=link.name);runtime['heads'][name]=row
        write(folder/'weights_registry.json',runtime)
        heads={n:load_exported(folder/'weights_registry.json',n,fc.device,FC_identity=fc.model_key,text_identity=fc.session.text_identity) for n in names}
        loader=ObjectLoader(fc,features,maximum_GiB=32);phi={'calls':0,'vectors':0};original_phi=fc.phi
        def counted_phi(value):
            phi['calls']+=1;phi['vectors']+=int(value.numel()//value.shape[-1]);return original_phi(value)
        fc.phi=counted_phi;values={name:{} for name in names};new_calls=0;engineering_rows=[]
        with torch.no_grad():
            for obj in engineering:
                x,_,_=loader.load(obj,0,8)
                indices=list(reversed(range(x['local_valid'].shape[1])))
                for name,head in heads.items():
                    a=head(x,fc.phi)['embedding'];b=head(permute_slots(x,indices),fc.phi)['embedding'];c=head({k:v.clone() for k,v in x.items()},fc.phi)['embedding'];new_calls+=3
                    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6);torch.testing.assert_close(a,c,rtol=0,atol=0)
                    engineering_rows.append(dict(key=obj['key'],head=name,slot_permutation_max_abs=float((a-b).abs().max()),unchanged_input_bitwise=bool(torch.equal(a,c)),head_calls=3))
            for obj in items:
                for condition in range(4):
                    path=root/'surface/units'/(obj['key'].replace(':','_')+f'_c{condition}.json')
                    key=unit_identity('surface_unit',dict(parent=identity,key=obj['key'],condition=condition),producers=('common','binding','surface'))
                    unit=cached_record(path,key)
                    if unit:
                        index.identity(unit['arrays']['path'],unit['arrays'])
                    else:
                        x,_,_=loader.load(obj,condition,8);unit_values={};zs=[];ss=[];calls=0
                        for name,head in heads.items():
                            unit_values[name]={}
                            for mode in modes:
                                altered=x if mode=='ORIGINAL' else permute_valid(x,obj['key'],condition,int(mode.split('_')[1]))
                                z=head(altered,fc.phi)['embedding'];r,score=_score_record(obj,z,fc);calls+=1;new_calls+=1
                                unit_values[name][mode]=r;zs.append(z.cpu().numpy());ss.append(score.cpu().numpy())
                        if new_calls>spec['limits']['max_frozen_head_forwards']:raise ValueError('Frozen head hard budget exceeded')
                        arrays=arrays_record(path.with_suffix('.npz'),dict(embeddings=np.asarray(zs,np.float32),full200_cosines=np.asarray(ss,np.float32),valid_ids=np.asarray(fc.ids),heads=np.asarray(names),modes=np.asarray(modes)),index)
                        unit=write(path,dict(status='COMPLETE',input_identity=key,key=obj['key'],condition=condition,values=unit_values,arrays=arrays,head_calls=calls,new_rgb_encodings=0))
                    for name in names:values[name][(obj['key'],condition)]=unit['values'][name]
                print('SURFACE_OBJECT',obj['key'],new_calls,flush=True)
        summaries={};parity={}
        for name,seed,step in zip(names,(17,29),(1000,2000)):
            historical=read_document(parent/f'training/seed{seed}/FG2_SURFACE/dev_{step:04d}.json',index)
            rows={mode:_records(items,values[name],mode) for mode in modes}
            old={r['key']:r for r in historical['records']}
            for r in rows['ORIGINAL']:
                if r['predictions']!=old[r['key']]['predictions']:raise ValueError('Original selected head labels differ from inherited records')
            index.identity(historical['arrays']['path'],historical['arrays'])
            with np.load(historical['arrays']['path'],allow_pickle=False) as arrays:
                old_indices={k:i for i,k in enumerate(arrays['object_keys'].tolist())};max_error=0.
                if arrays['valid_ids'].tolist()!=fc.ids:raise ValueError('Inherited full200 text order differs')
                for obj in items:
                    for condition in range(4):
                        unit=verified(root/'surface/units'/(obj['key'].replace(':','_')+f'_c{condition}.json'))
                        with np.load(unit['arrays']['path'],allow_pickle=False) as a:
                            pos=names.index(name)*len(modes);actual=a['embeddings'][pos];saved=arrays['embeddings'][old_indices[obj['key']],condition]
                            np.testing.assert_allclose(actual,saved,rtol=1e-5,atol=1e-6);max_error=max(max_error,float(np.max(np.abs(actual-saved))))
            metrics={mode:summarize(r,reference['records']) for mode,r in rows.items()}
            summaries[name]=dict(seed=seed,selected_step=step,metrics=metrics,records=rows,
                                shuffle_mean={k:float(np.mean([metrics[m][k] for m in modes[1:]])) for k in ('A','M','C','net_corrections')},
                                interpretation='OOD_DEPENDENCE_NOT_RANDOM_GROUPING_RETRAINING_EVIDENCE')
            parity[name]=dict(historical_receipt=historical['identity'],embedding_max_abs=max_error,all_labels_equal=True,full200_order_equal=True)
        torch.cuda.synchronize();work=dict(loader.workload,total_phi_calls=phi['calls'],total_phi_vectors=phi['vectors'],
             group_phi_calls=phi['calls']-loader.workload['local_phi_calls'],host_cache_limit_bytes=loader.limit,host_cache_final_bytes=loader.bytes,
             new_head_calls_this_invocation=new_calls,logical_DEV_head_calls=3312,engineering_head_calls=12,new_rgb_encodings=fc.images,new_optimizer_updates=0,
             new_sam_proposals=0,new_map_construction=0,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),wall_seconds=time.perf_counter()-start)
        if fc.images or new_calls>4000:raise ValueError('Frozen-only inference budget violated')
        index.write_memo(root/'surface/input_verifications.json')
        return write(root/'surface.json',dict(status='COMPLETE',input_identity=identity,history=history,intervention=summaries,
                     engineering=engineering_rows,selected_parity=parity,workload=work,interpretation='EXPOSED_DEV_DEPENDENCE_ONLY'))
