"""One streaming native pass shared by A0/A1/A2, never persist local U banks."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from static_ovmap.frozen_fc_routing.models import FrozenFCBase,unit
from static_ovmap.preservation_routing.selection import summarize
from .common import (items,verified,write,arrays_record,ConsumptionIndex,canonical_digest,cached_unit,unit_identity,CONDITIONS,PREFIXES)
from .subsets import validate_inputs,state_key
from .decomposition import compose,effects,local_statistics
from .action_bank import build_actions
from .features import action_features

def load_prefix(producer,obj,condition,prefix):
    x,_,_=producer(obj,condition,prefix);validate_inputs(x)
    if len(x['view_valid'])>prefix:raise ValueError('Producer exposed future views')
    expected=[v['frame_id'] for v in obj.get('views',[])[:prefix]]
    if expected and x['frame_ids'].tolist()!=expected:raise ValueError('Actual frame IDs differ from the requested prefix')
    return x

def assert_history(old_z,new_z,old_labels,new_labels,old_metrics,new_metrics):
    if not np.allclose(old_z,new_z,rtol=1e-5,atol=1e-6):raise ValueError('Historical embeddings differ')
    if old_labels!=new_labels:raise ValueError('Historical labels differ')
    if any(old_metrics[k]!=new_metrics[k] for k in old_metrics):raise ValueError('Historical AMC differs')

class NativeLedger:
    def __init__(self,root):
        self.path=Path(root)/'native_forwards.jsonl';self.path.parent.mkdir(parents=True,exist_ok=True)
        self.count=len(self.path.read_text().splitlines()) if self.path.exists() else 0;self.stream=self.path.open('a');self.seconds=0.;self.subset_count=0;self.full_count=0
    def call(self,head,x,phi,context,subset=False):
        if self.count>=85000:raise RuntimeError('Hard85000 pre-map native-head budget exhausted')
        self.count+=1;self.subset_count+=int(subset);self.full_count+=int(not subset)
        self.stream.write(json.dumps(dict(attempt=self.count,**context,subset=subset,frame_ids=x['frame_ids'].tolist()),sort_keys=True)+'\n');self.stream.flush()
        begin=time.perf_counter();result=head(x,phi);torch.cuda.synchronize() if x['fc_vectors'].is_cuda else None;self.seconds+=time.perf_counter()-begin
        return result
    def close(self):self.stream.close()

def stats(result):
    m,g,null,j=local_statistics(result)
    return dict(route_mass=m,null=null,group_count=j,weighted_sum_norm=float(torch.linalg.vector_norm(g)))

def score_embedding(z,fc,class_id):
    scores=z@fc.text_tensor.T;target=fc.ids.index(class_id);other=scores.clone();other[target]=-torch.inf
    ce=F.cross_entropy((z@fc.base_text.T/.07)[None],torch.tensor([fc.base_targets[class_id]],device=z.device))
    return dict(prediction=fc.ids[int(scores.argmax())],CE=float(ce),NLL=float(-F.log_softmax(scores/.07,0)[target]),margin=float(scores[target]-other.max()),correct=fc.ids[int(scores.argmax())]==class_id),scores.detach().cpu().numpy()

def static_bases(x,mask_areas):
    views=torch.nonzero(x['view_valid'],as_tuple=False).flatten();vectors=x['fc_vectors'][views];areas=torch.as_tensor(mask_areas,dtype=vectors.dtype,device=vectors.device)[views]
    if len(areas)!=len(vectors) or not bool(torch.isfinite(areas).all()) or bool((areas<0).any()) or float(areas.sum())<=0:raise ValueError('Original mask-pixel area is unavailable')
    normalized=unit(vectors.detach());pair=normalized@normalized.T;n=len(vectors)
    medoid=pair.masked_fill(torch.eye(n,dtype=torch.bool,device=pair.device),0).sum(-1)/(n-1) if n>1 else pair.new_ones(1)
    # Actual frame ID ascending resolves equal medoid values.
    chosen=sorted(range(n),key=lambda j:(-float(medoid[j]),int(x['frame_ids'][views[j]])))[0]
    return dict(MEAN=FrozenFCBase()(x),MASK_AREA=unit((vectors*(areas/areas.sum())[:,None]).sum(0)),COSINE_MEDOID=unit(vectors[chosen]))

def mask_areas(obj,condition,prefix,fc):
    fc.index.identity(obj['inputs']['path'],obj['inputs'])
    with np.load(obj['inputs']['path'],allow_pickle=False) as a:
        if 'masks' not in a:raise ValueError('Original mask pixels missing; no grid-area fallback')
        masks=a['masks'][condition,:prefix]
        return masks.reshape(len(masks),-1).sum(1,dtype=np.float64)

def compatible_cached_unit(path,identity,binding,role,obj,condition,feature_identity):
    path=Path(path)
    if not path.exists():return None
    old=verified(path)
    if old.get('input_identity')==identity:return cached_unit(path,identity)
    receipt=verified(Path(binding['output_root'])/'engineering/medoid_repair_compatibility.json')
    if role!='train' or receipt['only_completed_role']!='train':raise ValueError('Static repair cannot reuse changed DEV candidates')
    from .common import REPO,digest_file
    for name,sha in receipt['new_files'].items():
        if digest_file(REPO/'src/static_ovmap/branch_utility_wave2_v1'/(name+'.py'))!=sha:raise ValueError('Compatibility applies only to the reviewed exact engineering patch')
    expected=canonical_digest(dict(binding=binding['identity'],sources=receipt['old_source_digest'],phase='capture',role=role,inputs=obj['inputs']['sha256'],object_key=obj['key'],condition=condition,heads={str(s):binding['new_spec']['heads'][str(s)]['sha256'] for s in (17,29)},features=feature_identity))
    if old.get('input_identity')!=expected:raise ValueError('Prior native unit lineage differs from the preserved producer')
    return cached_unit(path,expected)

def run(binding,fc,loader,heads):
    root=Path(binding['output_root']);ledger=NativeLedger(root);index=fc.index;units=[];start=time.perf_counter();parity_count=0
    try:
        with torch.no_grad():
            for role in ('train','dev'):
                for oi,obj in enumerate(items(binding,role)):
                    for condition in range(4):
                        path=root/'capture/units'/role/(obj['key'].replace(':','_')+f'_c{condition}.json')
                        identity=unit_identity(binding,'capture',role=role,inputs=obj['inputs']['sha256'],object_key=obj['key'],condition=condition,heads={str(s):binding['new_spec']['heads'][str(s)]['sha256'] for s in (17,29)},features=loader.features)
                        old=compatible_cached_unit(path,identity,binding,role,obj,condition,loader.features)
                        if old:
                            units.append(dict(role=role,path=str(path),identity=old['identity']));continue
                        xs={};fulls={};cache={17:{},29:{}};rows=[];embeddings=[];cosines=[];features=[];groups=[];factorials=[];fc_indices={};actual={}
                        def append(z,info,feature=None):
                            row,score=score_embedding(z,fc,obj['class_id']);row.update(info);i=len(rows);rows.append(row);embeddings.append(z.detach().cpu().numpy());cosines.append(score);features.append(np.zeros(24,np.float32) if feature is None else feature);return i
                        # Current-prefix tensors are loaded before any prefix statistics.
                        for prefix in PREFIXES:
                            x=load_prefix(loader.load,obj,condition,prefix);xs[prefix]=x;actual[str(prefix)]=int(x['view_valid'].sum());base=FrozenFCBase()(x)
                            fc_indices[str(prefix)]=append(base,dict(kind='FC_BASE',seed=None,prefix=prefix,actual_views=actual[str(prefix)],edited=False))
                            for seed in (17,29):
                                context=dict(role=role,object_key=obj['key'],condition=condition,seed=seed,prefix=prefix);key=state_key(x,condition)
                                if key not in cache[seed]:cache[seed][key]=ledger.call(heads[seed],x,fc.phi,context)
                                full=cache[seed][key];fulls[seed,prefix]=full
                                torch.testing.assert_close(compose(full['base'],full),full['embedding'],rtol=0,atol=0);parity_count+=1
                                def evaluate_subset(sub):
                                    k=state_key(sub,condition)
                                    if k not in cache[seed]:cache[seed][k]=ledger.call(heads[seed],sub,fc.phi,context,subset=True)
                                    return cache[seed][k]
                                actions=build_actions(x,full,evaluate_subset) if prefix in (4,8) else [dict(kind='KEEP',frame_id=-1,view_index=None,edited=False,noop_proof=None,embedding=full['embedding'],base=full['base'],local=full)]
                                indices=[];keep_ce=None
                                for action in actions:
                                    feature=action_features(x,full,action,action['view_index']) if action['kind']!='KEEP' else None
                                    info=dict(kind=action['kind'],seed=seed,prefix=prefix,actual_views=actual[str(prefix)],frame_id=action['frame_id'],edited=action['edited'],noop_proof=action['noop_proof'],**stats(action['local']))
                                    i=append(action['embedding'],info,feature)
                                    if action['kind']=='KEEP':keep_ce=rows[i]['CE']
                                    rows[i]['utility']=0. if action['noop_proof'] else keep_ce-rows[i]['CE'];indices.append(i)
                                static={}
                                if role=='dev' and prefix in (4,8):
                                    for base_name,bb in static_bases(x,mask_areas(obj,condition,prefix,fc)).items():
                                        for gamma in (0.,.5,1.):
                                            z=(full['embedding'] if base_name=='MEAN' and gamma==1 else bb if gamma==0 else compose(bb,full,.5*gamma))
                                            static[f'{base_name}:{gamma:g}']=append(z,dict(kind='STATIC',seed=seed,prefix=prefix,actual_views=actual[str(prefix)],base=base_name,gamma=gamma,edited=False))
                                groups.append(dict(seed=seed,prefix=prefix,indices=indices,static=static,frame_ids=x['frame_ids'].tolist(),support_identity=key))
                        for seed in (17,29):
                            for p,q in ((2,4),(4,8)):
                                r0,r1=fulls[seed,p],fulls[seed,q];b0,b1=r0['base'],r1['base'];_,g0,_,_=local_statistics(r0);_,g1,_,_=local_statistics(r1)
                                zs=(r0['embedding'],compose(b1,r0),compose(b0,r1),r1['embedding']);scored=[score_embedding(z,fc,obj['class_id'])[0] for z in zs];h=[v['margin'] for v in scored];c=[float(v['correct']) for v in scored]
                                eff=effects(*h);corr=effects(*c)
                                if abs(eff['base']+eff['local']-eff['total'])>1e-6:raise ValueError('Factorial symmetric effects failed additivity')
                                factorials.append(dict(seed=seed,old_prefix=p,new_prefix=q,old_actual=actual[str(p)],new_actual=actual[str(q)],alias_actual_support=actual[str(p)]==actual[str(q)],frame_ids_old=xs[p]['frame_ids'].tolist(),frame_ids_new=xs[q]['frame_ids'].tolist(),cells={name:value for name,value in zip(('00','10','01','11'),scored)},effects=eff,correctness_effects=corr,old_local=stats(r0),new_local=stats(r1),base_direction_change=float(torch.linalg.vector_norm(b1-b0)),local_weighted_direction_change=float(torch.linalg.vector_norm(g1-g0))))
                        array=arrays_record(path.with_suffix('.npz'),dict(embeddings=np.asarray(embeddings,np.float32),cosines=np.asarray(cosines,np.float32),features=np.asarray(features,np.float32)),index)
                        saved=write(path,dict(status='COMPLETE',input_identity=identity,object={k:obj[k] for k in ('key','scene','family','class_id','base')},role=role,condition=condition,condition_name=CONDITIONS[condition],actual_views=actual,rows=rows,groups=groups,FC_indices=fc_indices,factorials=factorials,arrays=array,GT_used_in_forward=False,GT_used_in_features=False,local_U_persisted=False))
                        units.append(dict(role=role,path=str(path),identity=saved['identity']))
                        del xs,fulls,cache,rows,embeddings,cosines,features,actions,action,full,r0,r1,g0,g1
                    if (oi+1)%10==0:print(f'capture {role} originals={oi+1} native_attempts={ledger.count}',flush=True)
        if len(units)!=(622+69)*4:raise ValueError('Incomplete original/condition capture')
        capture=write(root/'capture.json',dict(status='COMPLETE',units=units,actual_pre_map_native_attempts=ledger.count,head_budget=85000,logical_forecast=82920,head_wall_seconds_this_invocation=ledger.seconds,new_full_head_calls_this_invocation=ledger.full_count,new_subset_head_calls_this_invocation=ledger.subset_count,real_FP32_compose_diagonals_checked_this_invocation=parity_count,shared_capture_for=['baseline','factorial','action_bank','STATIC_DEV'],group_U_saved=False,elapsed_seconds=time.perf_counter()-start))
        return baseline_summary(binding,fc,capture)
    finally:ledger.close()

def recognition_records(units,prefix,seed,method='FG2',static=None,selected=None,include_arrays=False):
    grouped={};zs={};cs={};edit=[];types={k:0 for k in ('KEEP','G','L','B')}
    for entry in units:
        unit=verified(entry['path']);obj=unit['object'];condition=unit['condition'];g=next(g for g in unit['groups'] if g['seed']==seed and g['prefix']==prefix)
        index=unit['FC_indices'][str(prefix)] if method=='FC_BASE' else g['static'][static] if static else selected.get((obj['key'],condition),g['indices'][0]) if selected else g['indices'][0]
        row=unit['rows'][index];r=grouped.setdefault(obj['key'],dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=True,predictions=[None]*4,CE=[None]*4,NLL=[None]*4,margins=[None]*4))
        for field,source in [('predictions','prediction'),('CE','CE'),('NLL','NLL'),('margins','margin')]:r[field][condition]=row[source]
        edit.append(int(row['edited']));types[row['kind'] if row['kind'] in types else 'KEEP']+=1
        if include_arrays:
            with np.load(unit['arrays']['path'],allow_pickle=False) as a:
                zs.setdefault(obj['key'],[None]*4)[condition]=a['embeddings'][index].copy();cs.setdefault(obj['key'],[None]*4)[condition]=a['cosines'][index].copy()
    records=[grouped[k] for k in sorted(grouped)]
    if any(None in r['predictions'] for r in records):raise ValueError('Recognition condition coverage incomplete')
    return records,dict(mean_edits=float(np.mean(edit)),edit_types=types,embeddings=np.asarray([zs[k] for k in sorted(zs)],np.float32) if include_arrays else None,cosines=np.asarray([cs[k] for k in sorted(cs)],np.float32) if include_arrays else None)

def baseline_summary(binding,fc,capture):
    root=Path(binding['output_root']);summaries={};parity=[]
    for role in ('train','dev'):
        units=[u for u in capture['units'] if u['role']==role]
        for prefix in PREFIXES:
            for method,seed in [('FC_BASE',17),('FG2',17),('FG2',29)]:
                records,extra=recognition_records(units,prefix,seed,method,include_arrays=True);metrics=summarize(records);key=f'{role}:{method}:{seed if method=="FG2" else 0}:p{prefix}'
                path=root/'recognition'/f'{role}_{method}_s{seed if method=="FG2" else 0}_p{prefix}.json'
                array=arrays_record(path.with_suffix('.npz'),dict(embeddings=extra['embeddings'],full200_cosines=extra['cosines'],object_keys=np.asarray([r['key'] for r in records]),valid_ids=np.asarray(fc.ids),class_ids=np.asarray([r['class_id'] for r in records])),fc.index)
                measured=write(path,dict(status='COMPLETE',records=records,metrics=metrics,arrays=array,role=role,prefix=prefix,seed=seed,method=method));summaries[key]=dict(path=str(path),metrics=metrics,identity=measured['identity'])
                if role=='dev':
                    if method=='FG2':oldpath=Path(binding['previous_root'])/'recognition'/f'FG2_s{seed}_l0_p{prefix}.json'
                    else:oldpath=Path(binding['fg_parent_root'])/'recognition/dev/FC8.json' if prefix==8 else None
                    if oldpath and oldpath.exists():
                        old=verified(oldpath);oldrows={r['key']:r for r in old['records'] if r['base']};ordered=[oldrows[r['key']] for r in records];oldmetric=summarize(ordered);descriptor=fc.index.identity(old['arrays']['path'],old['arrays'])
                        with np.load(descriptor['path'],allow_pickle=False) as a:
                            positions={str(k):i for i,k in enumerate(a['object_keys'])};zz=np.asarray([a['embeddings'][positions[r['key']]] for r in records])
                        assert_history(zz,extra['embeddings'],[r['predictions'] for r in ordered],[r['predictions'] for r in records],{k:oldmetric[k] for k in ('A','M','C')},{k:metrics[k] for k in ('A','M','C')});parity.append(dict(path=str(oldpath),identity=old['identity'],method=method,seed=seed,prefix=prefix,labels_and_AMC_exact=True,embedding_rtol=1e-5,embedding_atol=1e-6))
    return write(root/'baseline.json',dict(status='COMPLETE',summaries=summaries,history_parity=parity,FC_computed_once_shared_by_heads=True,FC_missing_prefixes=[2,4],real_diagonal_proof=capture['real_FP32_compose_diagonals_checked_this_invocation'],actual_pre_map_native_attempts=capture['actual_pre_map_native_attempts'],capture_identity=capture['identity'],original_denominators=dict(train=622,dev=69)))

def factorial(root):
    root=Path(root);capture=verified(root/'capture.json');rows=[]
    for entry in capture['units']:
        unit=verified(entry['path'])
        for row in unit['factorials']:rows.append(dict(row,role=unit['role'],key=unit['object']['key'],family=unit['object']['family'],condition=unit['condition_name']))
    groups={}
    for r in rows:
        k=f'{r["role"]}:s{r["seed"]}:{r["old_prefix"]}-{r["new_prefix"]}';groups.setdefault(k,[]).append(r)
    summaries=[]
    for k,bank in sorted(groups.items()):
        summaries.append(dict(group=k,records=len(bank),originals=len({r['key'] for r in bank}),families=len({r['family'] for r in bank}),exact_support_aliases=sum(r['alias_actual_support'] for r in bank),**{f'mean_{name}':float(np.mean([r['effects'][name] for r in bank])) for name in ('base','local','interaction','total')}))
    examples={}
    for category in ('GLOBAL_HARMFUL_ONLY','LOCAL_HARMFUL_ONLY','COMPLEMENTARY'):
        def match(r):
            b,l=r['effects']['base'],r['effects']['local'];return b<-1e-10 and l>=-1e-10 if category=='GLOBAL_HARMFUL_ONLY' else l<-1e-10 and b>=-1e-10 if category=='LOCAL_HARMFUL_ONLY' else b>1e-10 and l>1e-10
        matched=[r for r in rows if match(r)];examples[category]=min(matched,key=lambda r:(r['key'],r['role'],r['seed'],r['old_prefix'],r['condition'])) if matched else None
    return write(root/'factorial.json',dict(status='COMPLETE',rows=rows,summaries=summaries,examples=examples,selection_rule='FIXED_EFFECT_SIGNS_EARLIEST_KEY',effects_additive=True,interaction_added_to_total=False,GT_margin='FULL200_OFFLINE_ONLY',capture_identity=capture['identity'],new_native_forwards=0))

def action_bank(root):
    root=Path(root);capture=verified(root/'capture.json');rows=[];actual={};coverage={}
    for entry in capture['units']:
        unit=verified(entry['path']);obj=unit['object'];actual.setdefault(unit['role'],{}).setdefault(obj['key'],unit['actual_views'])
        for g in unit['groups']:
            if g['prefix'] not in (4,8):continue
            actions=[unit['rows'][i] for i in g['indices']];keep=actions[0];best_ce=max(actions,key=lambda a:a['utility']);best_top=max(actions,key=lambda a:(int(a['correct']),a['margin']))
            rows.append(dict(role=unit['role'],key=obj['key'],family=obj['family'],condition=unit['condition_name'],seed=g['seed'],prefix=g['prefix'],actual_views=keep['actual_views'],logical_actions=len(actions),proved_noops=sum(bool(a.get('noop_proof')) for a in actions),best_CE_utility=best_ce['utility'],best_CE_kind=best_ce['kind'],best_CE_top1_correct=best_ce['correct'],best_Top1_correct=best_top['correct'],best_Top1_kind=best_top['kind'],KEEP_correct=keep['correct']))
    for role,objects in actual.items():coverage[role]=dict(originals=len(objects),actual_n4=sum(v['4']==4 for v in objects.values()),actual_n8=sum(v['8']==8 for v in objects.values()),short_banks=sum(v['8']<8 for v in objects.values()),denominator='ALL_ORIGINALS')
    return write(root/'action_bank.json',dict(status='COMPLETE',units=capture['units'],coverage=coverage,diagnostic_GT_best_rows=rows,GT_best_is_not_deployable=True,best_CE_is_not_best_Top1=True,utility_competition=160,utility_temperature=.07,diagnostic_competition=200,new_native_forwards=0))
