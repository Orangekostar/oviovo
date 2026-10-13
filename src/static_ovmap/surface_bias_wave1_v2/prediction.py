"""Finite object readouts and fixed-owner N/Q/F map replacements."""
from collections import OrderedDict
from pathlib import Path
import time
import numpy as np
from .common import METHODS, LAMBDAS, REFERENCES, verified, write, arrays_record, unit_identity, cached_unit, PathResolver, _array_digest, canonical_digest, require
from .adapter import adapt
from .selection import summarize, lock_selection, activate

class PrefixLoader:
    def __init__(self,fc,features):
        from .data import ObjectLoader
        self.parent=ObjectLoader(fc,features,maximum_GiB=30); self.fc=fc; self.features=features['identity']; self.cache=OrderedDict(); self.bytes=0; self.limit=2*2**30; self.seconds=0.; self.calls=0

    def load(self,obj,condition=0,requested_views=8):
        if requested_views not in (2,4,8): raise ValueError('Only frozen2/4/8 prefixes are supported')
        self.fc.index.identity(obj['inputs']['path'],obj['inputs'])
        identity=canonical_digest(dict(inputs=obj['inputs']['sha256'],FC=self.fc.model_key,features=self.features,condition=condition,prefix=requested_views))
        start=time.perf_counter(); self.calls+=1
        if identity in self.cache:
            cpu,audit=self.cache.pop(identity); self.cache[identity]=(cpu,audit); x={k:v.to(self.fc.device) for k,v in cpu.items()}
        else:
            x,_,audit=self.parent.load(obj,condition,requested_views); cpu={k:v.detach().cpu().clone() for k,v in x.items()}
            size=sum(v.numel()*v.element_size() for v in cpu.values())
            while self.cache and self.bytes+size>self.limit:
                _,(old,_)=self.cache.popitem(last=False); self.bytes-=sum(v.numel()*v.element_size() for v in old.values())
            if size<=self.limit: self.cache[identity]=(cpu,audit); self.bytes+=size
        self.seconds+=time.perf_counter()-start
        return x,x['fc_vectors'],audit

def items(binding,role='dev'):
    from static_ovmap.frozen_fc_routing.common import objects
    return sorted([o for o in objects(binding,role) if o['base']],key=lambda o:o['key'])

def recognition(binding,fc,loader,head,seed,method,strength=0,prefix=8,reference=None):
    import torch
    from torch.nn import functional as F
    from static_ovmap.frozen_fc_routing.models import FrozenFCBase
    root=Path(binding['output_root']); schema=f'{method}_s{seed}_l{strength:g}_p{prefix}'
    path=root/'recognition'/f'{schema}.json'; identity=unit_identity(binding,'recognition',method=method,seed=seed,lambda_=strength,prefix=prefix,features=loader.features)
    cached=cached_unit(path,identity)
    if cached is not None:
        for obj in items(binding): fc.index.identity(obj['inputs']['path'],obj['inputs'])
        return cached
    rows=[]; zs=[]; ss=[]; numeric=0; started=time.perf_counter()
    sample_keys={}
    for obj in items(binding): sample_keys.setdefault(obj['family'],obj['key'])
    with torch.no_grad():
        for obj in items(binding):
            predictions=[]; CE=[]; NLL=[]; margins=[]; gaps=[]; audits=[]; embeddings=[]; scores=[]
            for condition in range(4):
                unit_path=root/'recognition/units'/schema/(obj['key'].replace(':','_')+f'_c{condition}.json')
                key=unit_identity(binding,'readout',inputs=obj['inputs']['sha256'],FC=fc.model_key,features=loader.features,method=method,seed=seed,lambda_=strength,prefix=prefix,condition=condition)
                row=cached_unit(unit_path,key)
                if row is None:
                    x,_,_=loader.load(obj,condition,prefix)
                    if method=='FC8': result=dict(embedding=FrozenFCBase()(x)); audit=dict(status='FROZEN_REFERENCE',numeric_fallback=False)
                    elif method=='FG2': result=head(x,fc.phi); audit=dict(status='SAME_SEED_PARENT_FG2',numeric_fallback=False)
                    else: result,audit=adapt(head,x,fc.phi,method,strength)
                    z=result['embedding']; score=z@fc.text_tensor.T; target=fc.ids.index(obj['class_id']); other=score.clone(); other[target]=-torch.inf
                    pair=score.topk(2).values
                    values=dict(embedding=z.cpu().numpy(),cosines=score.cpu().numpy())
                    if obj['key']==sample_keys[obj['family']] and 'quality_weights' in result: values['quality_weights']=result['quality_weights'].cpu().numpy()
                    array=arrays_record(unit_path.with_suffix('.npz'),values,fc.index)
                    row=write(unit_path,dict(status='COMPLETE',input_identity=key,arrays=array,prediction=fc.ids[int(score.argmax())],CE=float(F.cross_entropy((z@fc.base_text.T/.07)[None],torch.tensor([fc.base_targets[obj['class_id']]],device=fc.device))),NLL=float(-F.log_softmax(score/.07,0)[target]),GT_margin=float(score[target]-other.max()),gap=float(pair[0]-pair[1]),audit=audit,actual_views=int(x['view_valid'].sum()),GT_used_for_forward=False))
                fc.index.identity(row['arrays']['path'],row['arrays'])
                with np.load(row['arrays']['path'],allow_pickle=False) as array: embeddings.append(array['embedding'].copy()); scores.append(array['cosines'].copy())
                predictions.append(row['prediction']); CE.append(row['CE']); NLL.append(row['NLL']); margins.append(row['GT_margin']); gaps.append(row['gap']); audits.append(row['audit']); numeric+=int(row['audit'].get('numeric_fallback',False))
            rows.append(dict(key=obj['key'],family=obj['family'],class_id=obj['class_id'],base=True,predictions=predictions,CE=CE,NLL=NLL,margins=margins,gaps=gaps,audits=audits)); zs.append(embeddings); ss.append(scores)
    if len(rows)!=69: raise ValueError('Fixed69 DEV originals required')
    metrics=summarize(rows,reference); metrics.update(method=method,lambda_=strength,numeric_fallback_rate=numeric/276,seed=seed,prefix=prefix)
    arrays=arrays_record(path.with_suffix('.npz'),dict(embeddings=np.asarray(zs,np.float32),full200_cosines=np.asarray(ss,np.float32),object_keys=np.asarray([r['key'] for r in rows]),valid_ids=np.asarray(fc.ids),class_ids=np.asarray([r['class_id'] for r in rows])),fc.index)
    return write(path,dict(status='COMPLETE',input_identity=identity,metrics=metrics,records=rows,arrays=arrays,elapsed_seconds=time.perf_counter()-started,original_objects=69,variant_records=276,new_rgb_encodings=0))

def baseline(binding,fc,loader,heads,seed,prefix=8):
    return recognition(binding,fc,loader,heads[seed],seed,'FG2',prefix=prefix)

def parity(binding,fc,loader,heads):
    import torch
    resolver=PathResolver(binding['path_map']); root=Path(binding['output_root']); parent=Path(binding['fg_parent_root']); count=0
    with torch.no_grad():
        for obj in items(binding,'train')[:2]:
            for condition in range(4):
                for prefix in (2,4,8):
                    x,_,_=loader.load(obj,condition,prefix)
                    for seed in (17,29):
                        original=heads[seed](x,fc.phi); zero,_=adapt(heads[seed],x,fc.phi,METHODS[2],0,original=original)
                        torch.testing.assert_close(zero['embedding'],original['embedding'],rtol=0,atol=0)
                        corrected,_=adapt(heads[seed],x,fc.phi,METHODS[2],.25,original=original)
                        for key in ('quality_logits','alpha','null'):
                            if key in original: torch.testing.assert_close(corrected[key],original[key],rtol=0,atol=0)
                        unsupported=dict(x,local_valid=torch.zeros_like(x['local_valid'])); old=heads[seed](unsupported,fc.phi); new,_=adapt(heads[seed],unsupported,fc.phi,METHODS[2],.25,original=old)
                        torch.testing.assert_close(old['embedding'],new['embedding'],rtol=0,atol=0); count+=1
        for seed in (17,29):
            measured=baseline(binding,fc,loader,heads,seed); key=binding['new_spec']['heads'][str(seed)]['registry_key']; compare_history(parent/'recognition/dev'/f'{key}.json',measured,resolver,fc)
        fc8=recognition(binding,fc,loader,None,0,'FC8'); compare_history(parent/'recognition/dev/FC8.json',fc8,resolver,fc)
    return write(root/'parity.json',dict(status='COMPLETE',sample_objects=2,engineering_cells=count,real_FC_used=True,bitwise_value_only=True,bitwise_shortcuts=True,history_embedding_rtol=1e-5,history_embedding_atol=1e-6,history_labels_and_AMC_exact=True,dev_originals=69))

def compare_history(path,measured,resolver,fc):
    historical=verified(path); old={r['key']:r for r in historical['records'] if r['base']}
    rows=measured['records']
    if set(old)!={r['key'] for r in rows}: raise ValueError('Historical base-object keys differ')
    for row in rows:
        if old[row['key']]['predictions']!=row['predictions'] or old[row['key']]['class_id']!=row['class_id']: raise ValueError('Historical label changed; investigate environment before selection')
    metrics=summarize([old[r['key']] for r in rows])
    for metric in ('A','M','C'):
        if metrics[metric]!=measured['metrics'][metric]: raise ValueError('Historical AMC differs')
    descriptor=resolver.rewrite(historical['arrays']); fc.index.identity(descriptor['path'],descriptor)
    with np.load(descriptor['path'],allow_pickle=False) as arr, np.load(measured['arrays']['path'],allow_pickle=False) as new:
        positions={str(key):i for i,key in enumerate(arr['object_keys'])}
        if not np.array_equal(arr['valid_ids'],fc.ids): raise ValueError('Historical category order changed')
        expected=np.stack([arr['embeddings'][positions[r['key']]] for r in rows])
        np.testing.assert_allclose(new['embeddings'],expected,rtol=1e-5,atol=1e-6)

def select(binding,fc,loader,heads):
    root=Path(binding['output_root']); require([root/'parity.json'])
    if verified(root/'parity.json')['status']!='COMPLETE': raise ValueError('Real parity required before lambda selection')
    original=baseline(binding,fc,loader,heads,17); candidates={}
    for method in METHODS:
        candidates[method]=[{**summarize(original['records'],original['records']), 'method':method,'lambda_':0,'numeric_fallback_rate':0,'baseline_alias':original['identity']}]
        for strength in LAMBDAS[1:]: candidates[method].append(recognition(binding,fc,loader,heads[17],17,method,strength,reference=original['records'])['metrics'])
    return lock_selection(root,candidates)

def repeat(binding,fc,loader,heads):
    root=Path(binding['output_root']); require([root/'selection_seed17.json']); lock=verified(root/'selection_seed17.json'); original=baseline(binding,fc,loader,heads,29); rows={}
    for method in METHODS:
        strength=lock['selected'][method]['lambda_']
        row=recognition(binding,fc,loader,heads[29],29,method,strength,reference=original['records']) if strength else dict(metrics={**summarize(original['records'],original['records']),'method':method,'lambda_':0,'numeric_fallback_rate':0,'baseline_alias':original['identity']})
        rows[method]=row['metrics']
    write(root/'repeat_seed29.json',dict(status='COMPLETE',selection_identity=lock['identity'],rows=rows,reselected=False))
    first=baseline(binding,fc,loader,heads,17); result=activate(lock,rows,first['metrics'],original['metrics'])
    return write(root/'activation.json',dict(result,selection_identity=lock['identity']))

def prefix(binding,fc,loader,heads):
    root=Path(binding['output_root']); require([root/'activation.json']); lock=verified(root/'selection_seed17.json'); rows=[]
    for seed in (17,29):
        for number in (2,4):
            original=baseline(binding,fc,loader,heads,seed,number); rows.append(dict(original['metrics'],method='FG2',seed=seed,prefix=number))
            for method in (METHODS[2],lock['C_STAR']):
                strength=lock['selected'][method]['lambda_']
                row=recognition(binding,fc,loader,heads[seed],seed,method,strength,number,original['records']) if strength else dict(metrics={**summarize(original['records'],original['records']),'method':method,'lambda_':0,'numeric_fallback_rate':0,'baseline_alias':original['identity']})
                rows.append(dict(row['metrics'],seed=seed,prefix=number))
    return write(root/'prefix.json',dict(status='COMPLETE',selection_identity=lock['identity'],rows=rows,conditions=4,reselected=False))

def predict(binding,fc,heads):
    import torch
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    root=Path(binding['output_root']); parent=Path(binding['fg_parent_root']); lr=Path(binding['lr_parent_root']); resolver=PathResolver(binding['path_map'])
    require([root/'activation.json']); activation=verified(root/'activation.json')
    if not activation['expanded']: return write(root/'predictions/summary.json',dict(status='NOT_ACTIVATED_BY_FIXED_RULE',methods=[],new_methods=[],logical_new_records=0))
    require([lr/'regression/manifest.json',lr/'features/regression.json',parent/'recognition/regression.json'])
    link=root/'regression'
    if not link.exists(): link.symlink_to(lr/'regression',target_is_directory=True)
    features=resolver.rewrite(verified(lr/'features/regression.json')); loader=PrefixLoader(fc,features)
    historical_scores=resolver.rewrite(verified(parent/'recognition/regression.json'))
    arms={f'{m}_seed{s}':dict(method=m,seed=s,lambda_=activation['locked_lambdas'][m]) for m in activation['expanded'] for s in (17,29)}
    methods=list(REFERENCES)+list(arms); locks={}; total=time.perf_counter()
    with torch.no_grad():
        for names in binding['cohorts'].values():
            for scene in names:
                require([lr/'regression'/scene/'manifest.json',parent/'predictions'/scene/'receipt.json'])
                manifest=resolver.rewrite(verified(lr/'regression'/scene/'manifest.json')); old=resolver.rewrite(verified(parent/'predictions'/scene/'receipt.json')); saved={}
                p,original=load_scene(binding,scene)
                for name,old_name in zip(REFERENCES,('REF_D2','REF_G1','FG0_FC8','FG2_SURFACE','S29_FG2_SURFACE')):
                    descriptor=old['methods'][old_name]; payload=load_prediction(descriptor['manifest'])
                    if name not in ('REF_D2',) and not np.array_equal(payload.owner_ids,original.g1.owner_ids): raise ValueError('Parent fixed-owner reference changed')
                    saved[name]=dict(descriptor,source_kind='EXACT_FG_PARENT_REFERENCE',record_key=payload.record_key,prediction_key=payload.prediction_key)
                context=resolver.rewrite(verified(binding['scenes'][scene]['context']['path'])); fc.index.identity(context['FC_text']['path'],context['FC_text'])
                with np.load(context['FC_text']['path'],allow_pickle=False) as arr: text=arr['text_embeddings'].copy(); ids=arr['valid_ids'].copy()
                if context['FC_physical_model_identity']!=fc.model_key or not np.array_equal(ids,p.valid_ids): raise ValueError('Map FC/text class order changed')
                for name,arm in arms.items():
                    if arm['lambda_']==0:
                        saved[name]=dict(saved['FG2_seed'+str(arm['seed'])],source_kind='ZERO_LAMBDA_SAME_SEED_FG2_ALIAS'); continue
                    unit_path=root/'predictions'/scene/(name+'_receipt.json'); identity=unit_identity(binding,'map_predict',activation=activation['identity'],manifest=manifest['identity'],text=context['FC_text'],arm=arm)
                    previous=cached_unit(unit_path,identity)
                    if previous:
                        saved[name]=previous['prediction']; continue
                    relabels={}; decisions={}; acquired={int(o['owner']):o for o in manifest['objects']}
                    for owner in manifest['eligible_owners']:
                        obj=acquired[int(owner)]; x,_,_=loader.load(obj,0,8); original_result=heads[arm['seed']](x,fc.phi)
                        historical=historical_scores['records'][obj['key']]['arrays']; fc.index.identity(historical['path'],historical)
                        with np.load(historical['path'],allow_pickle=False) as values:
                            key=binding['new_spec']['heads'][str(arm['seed'])]['registry_key']; position=list(values['methods']).index(key)
                            np.testing.assert_allclose(original_result['embedding'].cpu().numpy(),values['embeddings'][position],rtol=1e-5,atol=1e-6)
                        base_F=text@original_result['embedding'].cpu().numpy()
                        result,audit=adapt(heads[arm['seed']],x,fc.phi,arm['method'],arm['lambda_'],original=original_result); new=text@result['embedding'].cpu().numpy()
                        source={k:np.asarray(p.sources[k]['objects'][str(owner)]['scores'],np.float64) if p.sources[k]['objects'][str(owner)]['available'] else None for k in ('N','Q','F')}; old_F=source['F']; source['F']=new
                        base=pool(source,p.temperatures,('N','Q','F')); final=simple_dependence(source,p.temperatures,base,'PE_D2_GROUPED')
                        if final is None or not np.isfinite(final).all(): raise ValueError('Required N/Q/F path failed')
                        label=p.valid_ids[int(final.argmax())]; relabels[owner]=label
                        reference_source=dict(source,F=base_F); reference_pool=pool(reference_source,p.temperatures,('N','Q','F')); reference_final=simple_dependence(reference_source,p.temperatures,reference_pool,'PE_D2_GROUPED')
                        decisions[str(owner)]=dict(old_class=obj['old_class'],new_class=label,historical_F_top1=p.valid_ids[int(old_F.argmax())] if old_F is not None else None,same_seed_FG2_F_top1=p.valid_ids[int(base_F.argmax())],same_seed_FG2_fused_class=p.valid_ids[int(reference_final.argmax())],new_F_top1=p.valid_ids[int(new.argmax())],F_difference_norm=float(np.linalg.norm(new-base_F)),new_F_cosines=new.tolist(),final_probabilities=final.tolist(),audit=audit)
                    payload=build_payload(original.g1,original.d2,original.raw,relabels,manifest['eligible_owners'],original.nearest,original.matched,name,{},dict(fixed_geometry='G1',historical_F_replaced_only=True))
                    if not np.array_equal(payload.owner_ids,original.g1.owner_ids) or payload.geometry!=original.g1.geometry: raise ValueError('Final owner geometry changed')
                    alias=next((r for r in saved.values() if r['prediction_key']==payload.prediction_key),None)
                    if alias: prediction=dict(alias,source_kind='EXACT_OWNER_SEMANTICS_CURRENT_RANK_ALIAS',alias_proof=dict(owners=_array_digest(payload.owner_ids),semantics=_array_digest(payload.semantic_labels),ranks=canonical_digest(payload.instance_ranks)))
                    else:
                        file=save_prediction(payload,root/'predictions'/scene/name); prediction=dict(status='PREDICTION_LOCKED',manifest=str(file),prediction_key=payload.prediction_key,record_key=payload.record_key,source_kind='REAL_VALUE_ONLY_F_REPLACEMENT')
                    write(root/'decisions'/scene/(name+'.json'),dict(status='COMPLETE',owners=decisions,GT_used=False)); write(unit_path,dict(status='COMPLETE',input_identity=identity,prediction=prediction)); saved[name]=prediction
                locks[scene]=write(root/'predictions'/scene/'receipt.json',dict(status='PREDICTIONS_LOCKED',scene=scene,methods=saved,owner_partition_unchanged=True))['identity']
    expected=sum(map(len,binding['cohorts'].values()))
    if len(locks)!=expected or expected!=26: raise ValueError('Full prediction coverage required before scoring')
    return write(root/'predictions/summary.json',dict(status='ALL_PREDICTIONS_LOCKED',scenes=locks,methods=methods,new_methods=list(arms),arms=arms,cohorts=binding['cohorts'],logical_new_records=len(arms)*expected,logical_reference_records=len(REFERENCES)*expected,activation_identity=activation['identity'],payload_and_readout_wall_seconds=time.perf_counter()-total,GT_used=False))
