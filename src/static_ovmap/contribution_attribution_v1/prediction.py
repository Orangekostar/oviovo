"""Probability-only outputs lock globally before any new-role GT scoring."""
from pathlib import Path
import time
import numpy as np
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.module_validation.scannet_study import load_prediction,save_prediction
from .common import (verified,write,read_document,ConsumptionIndex,PathResolver,unit_identity,cached_record,
                     canonical_digest,DependencyError,reserve,_array_digest)
from .binding import load_binding
from .inventory import scene_sources
from .sources import source_probabilities,combine,classify,parent_d2
from .payloads import fixed_owner_output,validate_eligible_replacement

def _projection(data,index):
    config=read_document(data['config'],index,sealed=False)
    scene=data.get('scene') or next(iter(config['scenes']))
    # Caller always supplies its exact scene; never infer among multiple scenes.
    path=Path(config['scenes'][scene]['projection']).with_name('projection.npz')
    manifest=read_document(path.with_name('manifest.json'),index,sealed=False)
    index.identity(path,manifest)
    with np.load(path,allow_pickle=False) as a:return a['nearest'].copy(),a['matched'].copy()

def _saved_payload(root,payload,references,index):
    for path in references:
        if not Path(path).exists():continue
        manifest=read_document(path,index,sealed=False)
        if manifest['prediction_key']!=payload.prediction_key:continue
        original=load_prediction(path)
        if original.geometry!=payload.geometry or not np.array_equal(original.owner_ids,payload.owner_ids) or not np.array_equal(original.semantic_labels,payload.semantic_labels) or original.instance_ranks!=payload.instance_ranks:
            raise ValueError('Prediction key matched but source payload identity differed')
        index.identity(Path(path).parent/manifest['arrays']['path'],manifest['arrays'])
        return dict(manifest=str(path),prediction_key=payload.prediction_key,record_key=original.record_key,measurement_kind='REUSED_EXACT',alias_candidate='EXACT_PAYLOAD_SCORER_CONTEXT_STILL_REQUIRES_VERIFICATION')
    root.mkdir(parents=True,exist_ok=True);reserve(root)
    path=save_prediction(payload,root)
    return dict(manifest=str(path),prediction_key=payload.prediction_key,record_key=payload.record_key,measurement_kind='NEW_CACHED_ANALYSIS',alias_candidate=None)

def _J_readouts(binding,scene,index):
    resolver=PathResolver(binding['path_map']);lr=Path(binding['lr_parent_root']);fg=Path(binding['fg_parent_root'])
    parent=resolver.rewrite(read_document(lr/'recognition/regression.json',index));learned=resolver.rewrite(read_document(fg/'recognition/regression.json',index))
    manifest=resolver.rewrite(read_document(lr/'regression'/scene/'manifest.json',index));eligible=list(map(int,manifest['eligible_owners']))
    lock=resolver.rewrite(read_document(fg/'predictions'/scene/'receipt.json',index))
    baseline=load_prediction(lock['methods']['FG0_FC8']['manifest'])
    bank={name:{} for name in ('J0_FC8_FONLY','J1_FG2_S17_FONLY','J2_FG2_S29_FONLY')};ids=None
    for owner in eligible:
        key=f'{scene}:{owner}'
        for source,name,method in ((parent,'J0_FC8_FONLY','LR04_FC_8'),(learned,'J1_FG2_S17_FONLY','FG2_SURFACE'),(learned,'J2_FG2_S29_FONLY','S29_FG2_SURFACE')):
            if key not in source['records']:raise DependencyError('Missing eligible source array: '+key+' '+method)
            descriptor=source['records'][key]['arrays'];index.identity(descriptor['path'],descriptor)
            with np.load(descriptor['path'],allow_pickle=False) as a:
                methods=list(a['methods'])
                if method not in methods:raise DependencyError('Missing exact source method array: '+key+' '+method)
                vector=a['cosines'][methods.index(method)].copy();vocabulary=a['valid_ids'].copy()
            if ids is not None and not np.array_equal(ids,vocabulary):raise ValueError('J eligible source ordered vocabulary differs')
            ids=vocabulary
            if vector.shape!=vocabulary.shape or not np.isfinite(vector).all():raise ValueError('Invalid available J F scores')
            bank[name][owner]=vector
    if ids is None:
        # Empty eligible domain is genuine; use the original context vocabulary.
        context=resolver.rewrite(read_document(binding['scenes'][scene]['context']['path'],index))
        ids=np.asarray(context['models']['native']['valid_ids'],np.int64)
    return baseline,eligible,ids,bank,lock

def predict(root,role):
    root=Path(root);binding=load_binding(root);spec=binding['task_specification'];inventory=verified(root/'inventory.json')
    phase='fusion_dev_predict' if role=='FUSION_DEV' else 'main_predict';folder=root/phase;index=ConsumptionIndex(folder/'verifications.json')
    started=time.perf_counter()
    if role=='FUSION_DEV':
        scenes=spec['fusion_dev'];methods=[f'W{i:02d}' for i in range(15)];definitions=[dict(id=m,sources=['N','Q','F'],operator='DEV_SELECTED_WEIGHT',weight=w) for m,w in zip(methods,spec['weights']['grid'],strict=True)]
        if not inventory['fusion_dev_available']:
            return write(root/(phase+'.json'),dict(status='NOT_SELECTED_NO_FUSION_DEV',role=role,reason='INCOMPLETE_FIXED_FOUR_SCENE_FUSION_DEV',records=[],scenes=scenes,methods=[],new_predictions=0))
        weight_lock=None
    else:
        weight_lock=verified(root/'weight_lock.json')
        scenes=[s for names in spec['cohorts'].values() for s in names]
        definitions=[dict(m) for m in spec['new_main_methods'] if not m.get('conditional') or weight_lock['weight'] is not None]
        for definition in definitions:
            if definition['id']=='E8':definition['weight']=weight_lock['weight']
        methods=[m['id'] for m in definitions]
    identity=unit_identity(phase,dict(inventory=inventory['identity'],weight_lock=weight_lock['identity'] if weight_lock else None),producers=('common','sources','payloads','prediction'),role=role,definitions=definitions)
    previous=cached_record(root/(phase+'.json'),identity)
    if previous:return previous
    records=[];scene_locks={};missing=[]
    for scene in scenes:
        key=unit_identity(phase+'_scene',dict(inventory=inventory['identity'],weight_lock=weight_lock['identity'] if weight_lock else None),producers=('common','sources','payloads','prediction'),scene=scene,role=role,definitions=definitions)
        target=folder/scene/'receipt.json';old=cached_record(target,key)
        if old:
            records+=old['records'];scene_locks[scene]=old['identity'];continue
        try:
            data,sources=scene_sources(binding,scene,role,index);data=dict(data,scene=scene)
            native=load_prediction(data['predictions']['NATIVE_READOUT']);incumbents=owner_labels(native);ids=sources['N']['valid_ids']
            if set(map(int,sources['N']['objects']))!=set(incumbents):raise ValueError('Actual A0/source incumbent registry differs')
            nearest,matched=_projection(data,index)
            references=list(data['predictions'].values())
            if role=='REGRESSION':
                a=read_document(Path(binding['area_root'])/'predictions'/scene/'receipt.json',index)
                references+=list(a['predictions'].values())
            probabilities={owner:source_probabilities(sources,data['temperatures'],ids,owner) for owner in incumbents}
            readouts={};scene_records=[]
            for definition in definitions:
                method=definition['id']
                if method.startswith('J'):continue
                labels={};audit={}
                for owner,native_label in incumbents.items():
                    if definition['operator']=='PARENT_D2':label,probability=parent_d2(sources,data['temperatures'],ids,owner,native_label)
                    else:
                        probability=combine(probabilities[owner],definition['sources'],definition['operator'],definition.get('weight'))
                        label=classify(probability,ids,native_label,definition['operator'])
                    labels[owner]=label;audit[str(owner)]=dict(label=label,probabilities=None if probability is None else probability.tolist(),availability={n:p is not None for n,p in probabilities[owner].items()})
                payload=fixed_owner_output(native,labels,ids,nearest,matched,method)
                saved=_saved_payload(folder/scene/method,payload,references,index)
                if saved['manifest'] not in references:references.append(saved['manifest'])
                scene_records.append(dict(scene=scene,method=method,role=role,definition=definition,data=data,**saved,geometry_identity=payload.geometry.to_dict(),owner_partition_digest=_array_digest(payload.owner_ids),source_identities={n:s['identity'] for n,s in sources.items()}))
                readouts[method]=audit
            if role=='REGRESSION':
                try:
                    base,eligible,jids,bank,parent_lock=_J_readouts(binding,scene,index);base_labels=owner_labels(base)
                    if list(map(int,jids))!=list(map(int,ids)):raise ValueError('J original context text order differs')
                    for definition in definitions:
                        method=definition['id']
                        if method not in bank:continue
                        labels=dict(base_labels);audit={}
                        for owner,vector in bank[method].items():labels[owner]=int(jids[int(vector.argmax())]);audit[str(owner)]=dict(label=labels[owner],cosines=vector.tolist(),eligible=True)
                        validate_eligible_replacement(base_labels,labels,eligible)
                        payload=fixed_owner_output(base,labels,jids,nearest,matched,method)
                        saved=_saved_payload(folder/scene/method,payload,[m['manifest'] for m in parent_lock['methods'].values()],index)
                        scene_records.append(dict(scene=scene,method=method,role=role,definition=definition,data=data,eligible=eligible,**saved,geometry_identity=payload.geometry.to_dict(),fixed_noneligible_reference=parent_lock['methods']['FG0_FC8']['prediction_key']))
                        readouts[method]=audit
                except (DependencyError,FileNotFoundError) as exc:
                    missing.append(dict(scene=scene,methods=[d['id'] for d in definitions if d['id'].startswith('J')],reason=str(exc)))
            write(folder/scene/'readouts.json',dict(status='COMPLETE',scene=scene,role=role,valid_ids=ids,methods=readouts,source_availability={str(o):{n:p is not None for n,p in probabilities[o].items()} for o in incumbents},temperatures=data['temperatures'],source_descriptors=data['sources'],GT_used=False))
            result=write(target,dict(status='ALL_PREDICTIONS_LOCKED',input_identity=key,scene=scene,role=role,records=scene_records,methods=[r['method'] for r in scene_records],GT_used=False,direct_inputs=index.entries()))
            scene_locks[scene]=result['identity'];records+=scene_records
            print('PREDICTIONS_LOCKED',role,scene,len(scene_records),flush=True)
        except (DependencyError,FileNotFoundError) as exc:
            missing.append(dict(scene=scene,methods=methods,reason=str(exc)))
    complete={(r['scene'],r['method']) for r in records}=={(s,m) for s in scenes for m in methods}
    result=write(root/(phase+'.json'),dict(status='ALL_PREDICTIONS_LOCKED' if complete else 'PARTIAL_DEPENDENCY',input_identity=identity,role=role,scenes=scenes,methods=methods,records=records,scene_locks=scene_locks,missing=missing,logical_records=len(records),new_optimizer_updates=0,new_rgb_encodings=0,new_head_forwards=0,new_sam_proposals=0,new_map_construction=0,GT_used=False,wall_seconds=time.perf_counter()-started))
    index.write_memo(folder/'verifications.json');return result
