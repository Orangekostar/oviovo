"""Real fixed-G1 outputs through the unchanged parent partition builder."""

import numpy as np
from pathlib import Path

from static_ovmap.minimal_instance_repair.outputs import construct_partition
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import save_prediction,load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,read

from .binding import load_scene,seal,plain,free_space
from .decisions import scene_kernel


def same_output(a,b):
    return (a.geometry==b.geometry and np.array_equal(a.owner_ids,b.owner_ids)
        and np.array_equal(a.semantic_labels,b.semantic_labels)
        and a.instance_ranks==b.instance_ranks)


def relabel_g1(inputs,method,relabels,*,eligible):
    for owner,label in relabels.items():
        mask=inputs.g1.owner_ids==int(owner)
        before=int(inputs.g1.semantic_labels[mask][0]) if mask.any() else None
        protected=bool(np.any(inputs.raw[mask]==0))
        if int(owner) not in eligible and label!=before and not protected:
            raise ValueError('class change outside the frozen common domain')
    payload,audit=construct_partition(inputs,method,[],relabels=relabels)
    if not np.array_equal(payload.owner_ids,inputs.g1.owner_ids):raise ValueError('G1 partition changed')
    original=inputs.d2.owner_ids>0
    if not np.array_equal(payload.semantic_labels[~original],inputs.g1.semantic_labels[~original]):
        raise ValueError('recovered/unknown G1 classes changed')
    allowed=np.isin(inputs.g1.owner_ids,list(eligible))
    if not np.array_equal(payload.semantic_labels[~allowed],inputs.g1.semantic_labels[~allowed]):
        raise ValueError('ineligible/unselected support changed')
    return payload,audit


def predict_scene(binding,scene,*,require_frozen=True):
    if require_frozen:
        from .orchestration import require_freeze
        require_freeze(binding)
    root=Path(binding['output_root']);manifest=read(root/'evidence'/scene/'manifest.json')
    eligibility=read(root/'eligibility'/(scene+'.json'))
    if eligibility['evidence_identity']!=manifest['identity']:raise ValueError('common domain evidence changed')
    frames=[read(root/'coarse'/scene/fid/'receipt.json') for fid in manifest['required_frames']]
    if eligibility['coarse_frames']!={str(r['frame_id']):r['identity'] for r in frames}:
        raise ValueError('sealed common domain C evidence changed')
    coarse={rid:r for f in frames for rid,r in f['records'].items()}
    index=ConsumptionIndex(root/'inputs'/scene/'verifications.json')
    key=canonical_digest({'evidence':manifest['identity'],'domain':eligibility['identity'],
        'producer':[index.identity(__file__),index.identity(Path(__file__).with_name('decisions.py'))]})
    dest=root/'predictions'/scene/'receipt.json'
    if dest.exists():
        old=read(dest);_verified_identity(old)
        if old['input_identity']==key:
            for row in old['methods'].values():
                payload=load_prediction(row['manifest'])
                if payload.prediction_key!=row['prediction_key']:raise ValueError('locked prediction alias changed')
            return old
        from .resume import invalidate_descendants
        invalidate_descendants(binding,scene,'predict','changed evidence/domain/decision producer')
    free_space(root);inputs=load_scene(binding,scene);eligible={int(o) for o,r in eligibility['incumbents'].items() if r['eligible']}
    methods=[r['id'] for r in binding['specification']['methods']];results={};known=[];all_decisions={}
    for method in methods:
        decisions=scene_kernel(manifest,eligibility,coarse,method);all_decisions[method]=decisions
        decision=seal(plain({'status':'DECISIONS_LOCKED','scene':scene,'method':method,'input_identity':key,
            'evidence_identity':manifest['identity'],'domain_identity':eligibility['identity'],
            'incumbents':decisions,'GT_used':False}))
        atomic_write_json(root/'decisions'/scene/(method+'.json'),decision)
        if method in ('SU00_D2','SU01_G1'):
            old='IR00_D2' if method=='SU00_D2' else 'IR01_G1';item=binding['scenes'][scene]['parent_predictions'][old]
            payload=inputs.d2 if old=='IR00_D2' else inputs.g1
            result={'status':'PREDICTION_LOCKED','manifest':item['manifest'],'prediction_key':payload.prediction_key,
                'record_key':payload.record_key,'source_kind':'EXACT_PARENT_BASELINE','decision_identity':decision['identity']}
        else:
            relabels={int(o):r['applied_class'] for o,r in decisions.items() if r['changed']}
            payload,audit=relabel_g1(inputs,method,relabels,eligible=eligible)
            alias=next((row for other,row in known if same_output(payload,other)),None)
            if alias is not None:
                result={**alias,'source_kind':'EXACT_LOCAL_OWNER_SEMANTIC_RANK_ALIAS','decision_identity':decision['identity'],
                    'audit':audit,'alias_proof':{'owner_digest':_array_digest(payload.owner_ids),
                        'semantic_digest':_array_digest(payload.semantic_labels),'rank_digest':canonical_digest(payload.instance_ranks)}}
            else:
                saved=save_prediction(payload,root/'predictions'/scene/method)
                result={'status':'PREDICTION_LOCKED','manifest':str(saved),'prediction_key':payload.prediction_key,
                    'record_key':payload.record_key,'source_kind':'REAL_FIXED_G1_RELABEL','decision_identity':decision['identity'],'audit':audit}
        results[method]=result;known.append((payload,result))
    def posterior_equal(a,b):
        pairs=[(all_decisions[a][o].get('probability'),all_decisions[b][o].get('probability'))
               for o in all_decisions[a]]
        if any(x is None or y is None for x,y in pairs):return None
        return all(np.array_equal(x,y) for x,y in pairs)
    equality={a+'='+b:{'selected_labels_equal':{o:r['applied_class'] for o,r in all_decisions[a].items()}==
                      {o:r['applied_class'] for o,r in all_decisions[b].items()},
        'full_posteriors_equal':posterior_equal(a,b),
        'posterior_null_reason':'literal baseline or matched class-only control has no updated posterior',
        'output_equal':results[a]['prediction_key']==results[b]['prediction_key']}
        for i,a in enumerate(methods) for b in methods[i+1:]}
    receipt=seal({'status':'PREDICTIONS_LOCKED','scene':scene,'input_identity':key,'methods':results,
        'evidence_identity':manifest['identity'],'domain_identity':eligibility['identity'],'equality_flags':equality,
        'owner_partition_unchanged':True,'recovered_classes_unchanged':True,'new_maps':0,'GT_used':False})
    atomic_write_json(dest,receipt);print('PREDICTED',scene,'9 locked conditions',flush=True);return receipt


def predict(binding,*,scenes=None,require_frozen=True):
    names=scenes or [s for values in binding['cohorts'].values() for s in values];receipts=[];blocked=[]
    for scene in names:
        try:receipts.append(predict_scene(binding,scene,require_frozen=require_frozen))
        except (OSError,ValueError,KeyError) as exc:
            block=seal({'scene':scene,'status':'BLOCKED_PREDICTION','reason':type(exc).__name__+': '+str(exc)})
            atomic_write_json(Path(binding['output_root'])/'predictions'/scene/'blocked.json',block);blocked.append(block)
    result=seal({'status':'PREDICTION_COMPLETE' if len(receipts)==len(names) else 'PARTIAL_DEPENDENCY_BLOCK',
        'scenes':{r['scene']:r['identity'] for r in receipts},'blocked':blocked})
    return result
