"""Measured engineering and stage costs; full cold timing is conditional."""
from pathlib import Path
import os
import time

import numpy as np

from static_ovmap.module_validation.scannet_study import save_prediction

from .acquisition import acquire,EvidenceAccess
from .binding import load_scene
from .common import canonical_digest,plain,verified,write
from .outputs import build_payload


def conditional_timing(binding):
    root = Path(binding['output_root'])
    regression = verified(root/'regression.json')
    if regression.get('strict_target_pass') is not True:
        return write(root/'timing.json', {'status': 'NOT_RUN_NO_FULL_TARGET_PASS',
            'reason': regression['status'], 'regression_identity': regression['identity'],
            'cold_calls': 0, 'new_FC_images': 0, 'new_pool_heads': 0,
            'standalone_incremental_seconds': None, 'online_FPS_claim': False})
    from static_ovmap.backbone_wave1.runtime import execute
    from .common import REPO
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=binding['gpu'], PYTHONPATH=str(REPO/'src')+':'+str(REPO),
               OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = execute([binding['FC_python'], '-u', '-m', 'static_ovmap.disagreement_query.cold_worker',
                       '--root', root], REPO, root/'cold/run.log', env=env)
    result = verified(root/'timing.json')
    write(root/'cold/execution.json', {'command': command, 'result_identity': result['identity']})
    return result


def engineering(binding):
    root=Path(binding['output_root']);path=root/'engineering/receipt.json';plans={};requests={};objects=[]
    for scene in binding['specification']['engineering_scenes']:
        plan=verified(root/'plans'/scene/'manifest.json');selected=next((o for o in plan['selected_owner_ids']
            if plan['owners'][str(o)]['query_eligible']),None)
        if selected is None:objects.append({'scene':scene,'owner':None,'status':'NOT_APPLICABLE_NO_ELIGIBLE_INCUMBENT'});continue
        row=plan['owners'][str(selected)];rid=f"{selected}:{row['anchor']}:FULL"
        plans[scene]=plan;requests[scene]=[rid,*row['probe_region_ids']]
        objects.append({'scene':scene,'owner':selected,'anchor_region_id':rid,'probe_region_ids':row['probe_region_ids'],
            'deterministic_rule':'FIRST_QUERY_ELIGIBLE_IN_FROZEN_MARGIN_JS_OWNER_SELECTION_ORDER'})
    fc=acquire(binding,'engineering_anchor',plans,requests,category='ENGINEERING')
    counts=fc['counts']
    if counts.get('FULL_pool_attempts',0)>4 or counts.get('PROBE_pool_attempts',0)>8 or counts.get('FC_encoding_attempts',0)>4:
        raise RuntimeError('engineering envelope exceeded')
    outputs=[]
    for item in objects:
        if item['owner'] is None:continue
        scene=item['scene'];p,e=load_scene(binding,scene)
        zero=build_payload(e.g1,e.d2,e.raw,{},set(),e.nearest,e.matched,'ENGINEERING_ZERO_UPDATE',{})
        identical=zero.prediction_key==e.g1.prediction_key
        if not identical:raise ValueError('real zero-update payload differs from exact G1')
        saved=save_prediction(zero,root/'engineering/payloads'/scene)
        records=fc['records'][scene];view=EvidenceAccess(records,[item['anchor_region_id']])
        allowed=view.get(item['anchor_region_id']);simple=EvidenceAccess(records,[]);denied=False
        try:simple.get(item['anchor_region_id'])
        except ValueError:denied=True
        if not denied:raise ValueError('simple query accessed cached anchor semantic scores')
        outputs.append({'scene':scene,'owner':item['owner'],'payload':str(saved),'zero_update_exact_G1':identical,
            'owner_arrays_exact':bool(np.array_equal(zero.owner_ids,e.g1.owner_ids)),
            'semantic_arrays_exact':bool(np.array_equal(zero.semantic_labels,e.g1.semantic_labels)),
            'rank_tuples_exact':zero.instance_ranks==e.g1.instance_ranks,
            'adapter_parity':records[item['anchor_region_id']]['engineering_adapter_parity'],
            'real_cached_evidence_denied_to_simple_policy':denied,'QD_anchor_allowed':allowed['content_key'],
            'future_candidate_regions_requested':False})
    return write(path,plain({'status':'ENGINEERING_PAYLOADS_LOCKED','objects':objects,'checks':outputs,
        'FC_acquisition_identity':fc['identity'],'counts':counts,'real_object_count':len(outputs),
        'original_ordered_scorer_parity':'PENDING_AFTER_SCIENTIFIC_STAGE_LOCK',
        'no_GT_object_substitution':True,'GT_used':False}))
