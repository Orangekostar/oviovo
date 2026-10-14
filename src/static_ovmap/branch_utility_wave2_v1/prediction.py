"""Locked branch policies replace F only, with original N/Q/F and rank path."""
from pathlib import Path
import time
import numpy as np
import torch
from .common import (verified,write,PathResolver,require,REFERENCES,unit_identity,cached_unit,_array_digest,canonical_digest,ConsumptionIndex)
from .capture import static_bases,mask_areas
from .action_bank import build_actions
from .features import action_features
from .selection import select_action
from .utility_model import UtilityRegressor,Normalizer
from .training import ARMS
from .decomposition import compose
from static_ovmap.surface_bias_wave1_v2.prediction import PrefixLoader

def replace_f(source,new_f,temperatures):
    from static_ovmap.paired_evidence_study.residuals import pool,simple_dependence
    changed=dict(source,F=new_f);return simple_dependence(changed,temperatures,pool(changed,temperatures,('N','Q','F')),'PE_D2_GROUPED')

def policies(binding,activation):
    root=Path(binding['output_root']);profiles=verified(root/'selection_lock.json')['profiles'];training=verified(root/'training.json');out={}
    for method in activation['methods']:
        for seed in (17,29):
            profile=profiles[method];entry=dict(profile=profile,seed=seed)
            if method!='STATIC':
                arm=ARMS[1] if method=='NO_NEW_CORRESPONDENCE' else ARMS[0];normal=verified(root/'training'/f'seed{seed}'/'normalizer.json');entry['normalizer']=Normalizer(normal['mean'],normal['std']);desc=training['arms'][f'{seed}:{arm}']['checkpoints'][str(profile['step'])];ConsumptionIndex().identity(desc['path'],desc)
                model=UtilityRegressor(seed).to('cuda');checkpoint=torch.load(desc['path'],map_location='cuda',weights_only=False)
                if checkpoint['normalizer']!=normal['identity']:raise ValueError('Map normalizer differs from locked controller')
                model.load_state_dict(checkpoint['model']);model.eval();entry['model']=model
            out[f'{method}_seed{seed}']=entry
    return out

def run(binding,fc,heads):
    from static_ovmap.frozen_fc_routing.binding import load_scene
    from static_ovmap.disagreement_query.outputs import build_payload
    from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
    root=Path(binding['output_root']);parent=Path(binding['fg_parent_root']);lr=Path(binding['lr_parent_root']);resolver=PathResolver(binding['path_map']);activation=verified(root/'activation.json')
    if not activation['methods']:return write(root/'predictions/summary.json',dict(status='NOT_ACTIVATED_BY_FIXED_RULE',reason=activation['reason'],methods=[],new_methods=[],logical_new_records=0))
    require([lr/'regression/manifest.json',lr/'features/regression.json',parent/'recognition/regression.json'])
    link=root/'regression'
    if not link.exists():link.symlink_to(lr/'regression',target_is_directory=True)
    loader=PrefixLoader(fc,resolver.rewrite(verified(lr/'features/regression.json')));historical=resolver.rewrite(verified(parent/'recognition/regression.json'));active=policies(binding,activation);arms={k:dict(method=v['profile']['method'],seed=v['seed'],profile=v['profile']) for k,v in active.items()};methods=list(REFERENCES)+list(arms);locks={};start=time.perf_counter();native=actions_count=transfers=mlp_calls=objects_count=0;head_seconds=feature_seconds=mlp_seconds=transfer_seconds=0.
    with torch.no_grad():
        for cohort,scenes in binding['cohorts'].items():
            for scene in scenes:
                require([lr/'regression'/scene/'manifest.json',parent/'predictions'/scene/'receipt.json']);manifest=resolver.rewrite(verified(lr/'regression'/scene/'manifest.json'));old=resolver.rewrite(verified(parent/'predictions'/scene/'receipt.json'));saved={};p,original=load_scene(binding,scene)
                for name,oldname in zip(REFERENCES,('REF_D2','REF_G1','FG0_FC8','FG2_SURFACE','S29_FG2_SURFACE')):
                    descriptor=old['methods'][oldname];payload=load_prediction(descriptor['manifest'])
                    if name!='REF_D2' and (not np.array_equal(payload.owner_ids,original.g1.owner_ids) or payload.geometry!=original.g1.geometry):raise ValueError('Inherited G1 owner geometry differs')
                    saved[name]=dict(descriptor,source_kind='EXACT_FG_PARENT_REFERENCE',record_key=payload.record_key,prediction_key=payload.prediction_key)
                context=resolver.rewrite(verified(binding['scenes'][scene]['context']['path']));fc.index.identity(context['FC_text']['path'],context['FC_text'])
                with np.load(context['FC_text']['path'],allow_pickle=False) as a:text=a['text_embeddings'].copy();ids=a['valid_ids'].copy()
                if context['FC_physical_model_identity']!=fc.model_key or not np.array_equal(ids,p.valid_ids):raise ValueError('Map FC/text identity or class order differs')
                path=root/'predictions'/scene/'branch_capture.json';identity=unit_identity(binding,'map_predict',activation=activation['identity'],manifest=manifest['identity'],text=context['FC_text'],arms=arms);cached=cached_unit(path,identity)
                if cached:
                    saved.update(cached['predictions']);locks[scene]=write(root/'predictions'/scene/'receipt.json',dict(status='PREDICTIONS_LOCKED',scene=scene,methods=saved,owner_partition_unchanged=True))['identity'];continue
                relabels={name:{} for name in active};decisions={name:{} for name in active};acquired={int(o['owner']):o for o in manifest['objects']}
                for owner in manifest['eligible_owners']:
                    obj=acquired[int(owner)];x,_,_=loader.load(obj,0,8);objects_count+=1
                    oldarray=historical['records'][obj['key']]['arrays'];fc.index.identity(oldarray['path'],oldarray)
                    with np.load(oldarray['path'],allow_pickle=False) as a:history={seed:(a['embeddings'][list(a['methods']).index(binding['new_spec']['heads'][str(seed)]['registry_key'])].copy(),a['cosines'][list(a['methods']).index(binding['new_spec']['heads'][str(seed)]['registry_key'])].copy()) for seed in (17,29)}
                    source={k:np.asarray(p.sources[k]['objects'][str(owner)]['scores'],np.float64) if p.sources[k]['objects'][str(owner)]['available'] else None for k in ('N','Q','F')}
                    for seed in (17,29):
                        begin=time.perf_counter();full=heads[seed](x,fc.phi);torch.cuda.synchronize();head_seconds+=time.perf_counter()-begin;native+=1
                        begin=time.perf_counter();embedding=full['embedding'].cpu().numpy();transfer_seconds+=time.perf_counter()-begin;transfers+=1
                        np.testing.assert_allclose(embedding,history[seed][0],rtol=1e-5,atol=1e-6);base_F=history[seed][1]
                        selected=[(name,value) for name,value in active.items() if value['seed']==seed];need_actions=any(v['profile']['method']!='STATIC' for _,v in selected);bank=None;fx=None
                        if need_actions:
                            def evaluate(sub):
                                nonlocal native,head_seconds
                                b=time.perf_counter();r=heads[seed](sub,fc.phi);torch.cuda.synchronize();head_seconds+=time.perf_counter()-b;native+=1;return r
                            bank=build_actions(x,full,evaluate);actions_count+=len(bank);begin=time.perf_counter();fx=np.asarray([action_features(x,full,a,a['view_index']) for a in bank[1:]],np.float32).reshape(-1,24);feature_seconds+=time.perf_counter()-begin
                        for name,value in selected:
                            profile=value['profile'];method=profile['method'];kind='KEEP';frame=-1;edited=False
                            if method=='STATIC':
                                bases=static_bases(x,mask_areas(obj,0,8,fc));base=bases[profile['base']];gamma=profile['gamma'];z=full['embedding'] if profile['base']=='MEAN' and gamma==1 else base if gamma==0 else compose(base,full,.5*gamma);unchanged=profile['base']=='MEAN' and gamma==1
                            else:
                                begin=time.perf_counter();tensor=torch.from_numpy(value['normalizer'].transform(fx,method=='NO_NEW_CORRESPONDENCE')).to('cuda');torch.cuda.synchronize();transfer_seconds+=time.perf_counter()-begin;transfers+=1
                                begin=time.perf_counter();scores=value['model'](tensor).cpu().numpy();mlp_seconds+=time.perf_counter()-begin;mlp_calls+=1;choice=select_action(bank,[0.]+scores.tolist(),profile['tau'],method=='COUPLED');action=bank[choice];z=action['embedding'];kind=action['kind'];frame=action['frame_id'];edited=action['edited'];unchanged=not edited
                            begin=time.perf_counter();new=base_F.copy() if unchanged else text@z.detach().cpu().numpy();transfer_seconds+=time.perf_counter()-begin;transfers+=int(not unchanged);final=replace_f(source,new,p.temperatures)
                            if final is None or not np.isfinite(final).all():raise ValueError('Native N/Q/F replacement failed')
                            reference_final=replace_f(source,base_F,p.temperatures);label=p.valid_ids[int(final.argmax())];relabels[name][owner]=label
                            decisions[name][str(owner)]=dict(old_class=obj['old_class'],new_class=label,same_seed_FG2_F_top1=p.valid_ids[int(base_F.argmax())],same_seed_FG2_fused_class=p.valid_ids[int(reference_final.argmax())],new_F_top1=p.valid_ids[int(new.argmax())],F_difference_norm=float(np.linalg.norm(new-base_F)),new_F_cosines=new.tolist(),final_probabilities=final.tolist(),kind=kind,frame_id=frame,edited=edited,actual_views=int(x['view_valid'].sum()),logical_actions=len(bank) if bank is not None else 1,GT_used=False)
                newpred={}
                for name in active:
                    payload=build_payload(original.g1,original.d2,original.raw,relabels[name],manifest['eligible_owners'],original.nearest,original.matched,name,{},dict(fixed_geometry='G1',historical_F_replaced_only=True))
                    if not np.array_equal(payload.owner_ids,original.g1.owner_ids) or payload.geometry!=original.g1.geometry:raise ValueError('G1 geometry changed')
                    alias=next((r for r in saved.values() if r['prediction_key']==payload.prediction_key),None)
                    if alias:item=dict(alias,source_kind='EXACT_OWNER_SEMANTICS_CURRENT_RANK_ALIAS',alias_proof=dict(owners=_array_digest(payload.owner_ids),semantics=_array_digest(payload.semantic_labels),ranks=canonical_digest(payload.instance_ranks)))
                    else:
                        file=save_prediction(payload,root/'predictions'/scene/name);item=dict(status='PREDICTION_LOCKED',manifest=str(file),prediction_key=payload.prediction_key,record_key=payload.record_key,source_kind='REAL_BRANCH_ADMISSION_F_REPLACEMENT')
                    write(root/'decisions'/scene/(name+'.json'),dict(status='COMPLETE',owners=decisions[name],GT_used=False));saved[name]=item;newpred[name]=item
                write(path,dict(status='COMPLETE',input_identity=identity,predictions=newpred));locks[scene]=write(root/'predictions'/scene/'receipt.json',dict(status='PREDICTIONS_LOCKED',scene=scene,methods=saved,owner_partition_unchanged=True))['identity'];print('MAP_LOCK',scene,len(manifest['eligible_owners']),flush=True)
    if len(locks)!=26 or len(arms)>8:raise ValueError('Dynamic whole-map prediction coverage differs')
    cost=dict(native_head_forwards=native,logical_enumerated_actions=actions_count,host_device_transfers=transfers,MLP_calls=mlp_calls,original_map_objects=objects_count,native_head_seconds=head_seconds,feature_seconds=feature_seconds,MLP_seconds=mlp_seconds,transfer_seconds=transfer_seconds,loader_seconds=loader.seconds,nested_timers_not_additive=True,boundary='CACHED_NATIVE_MODEL_RESIDENT_ALL_SUBSET_ENUMERATION_NOT_MLP_ONLY')
    write(root/'map_cost.json',dict(status='COMPLETE',**cost))
    return write(root/'predictions/summary.json',dict(status='ALL_PREDICTIONS_LOCKED',scenes=locks,methods=methods,new_methods=list(arms),arms=arms,cohorts=binding['cohorts'],logical_new_records=len(arms)*26,logical_reference_records=len(REFERENCES)*26,activation_identity=activation['identity'],elapsed_seconds=time.perf_counter()-start,GT_used=False,cost=cost))
