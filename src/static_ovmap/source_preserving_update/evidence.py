"""Exact saved FULL observations, historical replay and shared applicability."""

from pathlib import Path
from collections import Counter

import numpy as np

from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.minimal_instance_repair.reread import reread_decision
from static_ovmap.minimal_instance_repair.recognition_plan import unpack_region
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import read,PathResolver,ConsumptionIndex

from .binding import load_scene,seal,plain,free_space
from .decisions import source_components,vector
from .outputs import relabel_g1,same_output


def paired_means(views,coarse,ids,coarse_ids):
    if list(ids)!=list(coarse_ids):raise ValueError('paired A/C class order differs')
    if len(views) not in (1,2) or len(views)!=len(coarse):raise ValueError('one or two complete paired FULL views required')
    if len({(r['frame_id'],r['mask_digest']) for r in views})!=len(views):raise ValueError('duplicate paired observation')
    if any((a['frame_id'],a['mask_digest'])!=(c['frame_id'],c['mask_digest']) for a,c in zip(views,coarse)):
        raise ValueError('coarse differs from the exact FULL observation')
    return (np.mean([vector(r['scores'],len(ids)) for r in rows],axis=0) for rows in (views,coarse))


def replay_decision(old_class,ids,observations,cfg):
    return reread_decision(old_class,ids,observations,
        margin=cfg['aggregate_margin_over_incumbent'],minimum_regions=cfg['stable_min_distinct_regions'],
        minimum_extra=cfg['stable_min_extra_regions'],vote_fraction=cfg['stable_vote_fraction'],
        full_margin=cfg['stable_full_view_margin_min'],median_margin=cfg['stable_median_margin_min'])


def potential_domain(row):
    reasons=[]
    if not row['selected']:reasons.append('NOT_PARENT_SELECTED')
    if row['p0'] is None:reasons.append('D2_UNAVAILABLE')
    if row['scores'].get('F') is None:reasons.append('HISTORICAL_F_UNAVAILABLE')
    if not row['full_views']:reasons.append('NO_PARENT_SUCCESSFUL_FULL')
    if row['protected_raw_zero_count']:reasons.append('PROTECTED_RAW_ZERO')
    return not reasons,reasons


def common_domain(row,coarse):
    potential,reasons=potential_domain(row)
    if not potential:return False,reasons
    for view in row['full_views']:
        if view['region_id'] not in coarse:raise ValueError('required coarse execution is incomplete')
        item=coarse[view['region_id']]
        if not item['available']:
            if item.get('reason') not in ('EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'):
                raise ValueError('execution failure is not scientific C unavailability')
            reasons.append('COARSE_UNAVAILABLE:'+item['reason'])
    return not reasons,reasons


def prepare_scene(binding,scene):
    root=Path(binding['output_root']);free_space(root);row=binding['scenes'][scene]
    path=root/'evidence'/scene/'manifest.json'
    index=ConsumptionIndex(root/'inputs'/scene/'verifications.json')
    producers=[index.identity(Path(__file__).with_name(name+'.py'))
               for name in ('evidence','binding','decisions','outputs')]
    # Successful input-identity reuse is content validated with the inherited memo.
    key=canonical_digest({'binding':binding['identity'],'plan':row['parent_plan'],
        'decisions':row['parent_decisions'],'producer':producers})
    if path.exists():
        old=read(path);_verified_identity(old)
        if old['input_identity']==key:
            for item in old['dependencies']:index.identity(item['path'],item)
            index.write_memo(root/'inputs'/scene/'verifications.json');return old
        from .resume import invalidate_descendants
        invalidate_descendants(binding,scene,'prepare','changed producer or evidence input identity')
    inputs=load_scene(binding,scene);index=inputs.index;resolver=PathResolver(binding['path_map'])
    dependencies=[]
    def doc(item,*,verified=True):
        p=Path(resolver.resolve(item['path'] if isinstance(item,dict) else item))
        dependencies.append(index.identity(p,item if isinstance(item,dict) else None));value=read(p)
        if verified:_verified_identity(value)
        return value
    plan=doc(row['parent_plan']);saved=doc(row['parent_decisions'])
    if (plan['status']!='REGIONS_LOCKED' or saved['status']!='RECOGNITION_COMPLETE'
            or saved['plan_identity']!=plan['identity'] or len(plan['incumbents'])>16):
        raise ValueError('exact parent selection/recognition is incomplete')
    if {str(r['owner']) for r in plan['incumbents']}!=set(saved['incumbents']):raise ValueError('parent selected list changed')
    frames={};historical={};incumbents={};required={}
    for selected in plan['incumbents']:
        owner=str(selected['owner']);observations=[];views=[];banks=set();seen=set()
        if selected!=saved['incumbents'][owner]['selection']:raise ValueError('parent selection identity changed')
        for rid in selected['regions']:
            region=resolver.rewrite(plan['regions'][rid]);fid=str(region['frame_id'])
            if fid not in frames:
                frame=doc(Path(binding['parent_root'])/'recognition'/scene/'frames'/(fid+'.json'))
                if frame['status']!='COMPLETE' or saved['frames'][fid]!=frame['identity']:
                    raise ValueError('successful parent AnyUp frame is unavailable or changed')
                frames[fid]=resolver.rewrite(frame)
            frame=frames[fid]
            if rid not in frame['records']:raise ValueError('successful parent region record missing')
            result=frame['records'][rid]
            obs={k:region[k] for k in ('frame_id','bank','type','mask_digest')}
            obs.update(scores=result.get('scores'),reason=result.get('reason',result.get('fallback_reason')),
                       feature_content_key=result['feature_content_key'])
            observations.append(obs)
            if region['type']!='FULL' or not result['available']:continue
            vector(result['scores'],len(inputs.valid_ids))
            identity=(region['frame_id'],region['mask_digest'])
            if identity in seen:continue
            if region['bank'] in banks:raise ValueError('more than one successful FULL observation per parent bank')
            seen.add(identity);banks.add(region['bank'])
            group=resolver.rewrite(plan['frames'][fid]);mask=unpack_region(region,index)
            dependencies.extend([index.identity(region['mask']['path'],region['mask']),index.identity(group['rgb']['path'],group['rgb'])])
            y,x=np.nonzero(mask);bbox=[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]
            if bbox!=region['bbox'] or list(mask.shape)!=region['original_size']:
                raise ValueError('original FULL mask/bbox/shape changed')
            index.identity(frame['vectors']['path'],frame['vectors']);dependencies.append(index.identity(frame['vectors']['path'],frame['vectors']))
            with np.load(frame['vectors']['path'],allow_pickle=False) as vectors:
                if _array_digest(vectors[rid])!=result['vector_sha256']:raise ValueError('parent AnyUp vector changed')
            view={**region,'scores':result['scores'],'feature_content_key':result['feature_content_key'],
                'RGB':group['rgb'],'input_tensor_key':frame['input_tensor_key'],
                'image_content_key':frame['image_content_key'],'actual_dense_shape':frame['actual_dense_shape'],
                'parent_dense_receipt':frame['dense_receipt'],'parent_dense_arrays':frame['dense_arrays'],
                'parent_vector':frame['vectors'],'parent_frame_identity':frame['identity'],
                'AnyUp_operator_identity':frame['operator_identity']}
            views.append(view)
        if len(views)>2:raise ValueError('parent FULL budget exceeded')
        historical[owner]=replay_decision(selected['old_class'],inputs.valid_ids,observations,binding['parent_reread_settings'])
        old=saved['incumbents'][owner]
        for field in ('IR06_class','IR07_class','IR06_reason','IR07_reason','successful_regions','successful_full_views','tests'):
            if historical[owner][field]!=old[field]:raise ValueError('historical reread replay mismatch: '+field)
        if old.get('aggregate_scores')!=historical[owner].get('aggregate_scores'):raise ValueError('historical full cosine aggregation differs')
        scores={name:(source['objects'][owner]['scores'] if source['objects'][owner]['available'] else None)
                for name,source in inputs.sources.items()}
        p0,q,f,w=source_components(scores,inputs.data['temperatures']);bound=inputs.probabilities[owner]['probabilities']
        if (p0 is None)!=(bound is None) or (p0 is not None and np.max(np.abs(p0-np.asarray(bound)))>1e-12):
            raise ValueError('new source component adapter differs from original D2')
        protected=int(np.count_nonzero(inputs.raw[inputs.g1.owner_ids==int(owner)]==0))
        item={'selected':True,'selection':selected,'old_class':selected['old_class'],'scores':scores,
            'p0':None if p0 is None else p0.tolist(),'protected_raw_zero_count':protected,
            'full_views':views,'historical':historical[owner]}
        potential,reasons=potential_domain(item);item.update(potential_common_domain=potential,potential_exclusions=reasons)
        incumbents[owner]=item
        if potential:
            for view in views:
                required.setdefault(str(view['frame_id']),[]).append(view['region_id'])
    history_checks={}
    for method,field in [('IR06_ANYUP_REREAD','IR06_class'),('IR07_BOUNDARY_STABLE','IR07_class')]:
        relabels={int(o):d[field] for o,d in historical.items() if d[field]!=d['old_class']}
        reconstructed,audit=relabel_g1(inputs,method,relabels,eligible={int(o) for o in historical})
        parent_payload=doc(row['parent_predictions'][method]['manifest'],verified=False)
        arrays=Path(row['parent_predictions'][method]['manifest']).parent/parent_payload['arrays']['path']
        dependencies.append(index.identity(arrays,parent_payload['arrays']))
        actual=load_prediction(row['parent_predictions'][method]['manifest'])
        if not same_output(reconstructed,actual) or reconstructed.prediction_key!=actual.prediction_key:
            raise ValueError('actual historical owner/semantic/rank payload replay differs')
        history_checks[method]={'prediction_key':actual.prediction_key,'actual_manifest':row['parent_predictions'][method]['manifest'],
            'owner_digest':_array_digest(actual.owner_ids),'semantic_digest':_array_digest(actual.semantic_labels),
            'rank_digest':canonical_digest(actual.instance_ranks),'array_parity':'EXACT','audit':audit}
    result=seal(plain({'status':'PAIRED_EVIDENCE_LOCKED','scene':scene,'input_identity':key,'cohort':row['cohort'],
        'valid_ids':inputs.valid_ids,'temperatures':inputs.data['temperatures'],'FC_model':inputs.data['FC_physical_model_identity'],
        'FC_text':inputs.data['FC_text'],'parent_plan_identity':plan['identity'],'parent_decision_identity':saved['identity'],
        'incumbents':incumbents,'required_frames':required,'required_regions':sum(map(len,required.values())),
        'selected_count':len(incumbents),'potential_common_count':sum(r['potential_common_domain'] for r in incumbents.values()),
        'd2_probability_parity':inputs.d2_probability_parity,'historical_payload_parity':history_checks,
        'dependencies':list({r['path']:r for r in [*dependencies,*index.cache.values()]}.values()),
        'producer':producers,'GT_used':False,'new_AnyUp_QK':0,
        'new_projections':0,'new_frontend_inference':0,'new_NQ_inference':0}))
    atomic_write_json(path,result);index.write_memo(root/'inputs'/scene/'verifications.json')
    print('PREPARED',scene,'selected',len(incumbents),'potential',result['potential_common_count'],'coarse regions',result['required_regions'],flush=True)
    return result


def prepare(binding):
    receipts=[];blocked=[]
    for names in binding['cohorts'].values():
        for scene in names:
            try:receipts.append(prepare_scene(binding,scene))
            except (OSError,ValueError,KeyError) as exc:
                item=seal({'scene':scene,'status':'BLOCKED_DEPENDENCY','reason':type(exc).__name__+': '+str(exc)})
                atomic_write_json(Path(binding['output_root'])/'evidence'/scene/'blocked.json',item);blocked.append(item)
    image_ids={(r['FC_model'],v['input_tensor_key']) for r in receipts for item in r['incumbents'].values()
               if item['potential_common_domain'] for v in item['full_views']}
    regions=sum(r['required_regions'] for r in receipts)
    if regions>832 or len(image_ids)>832:raise ValueError('actual inventory exceeds fixed acquisition bound')
    result=seal({'status':'EVIDENCE_COMPLETE' if len(receipts)==26 else 'PARTIAL_DEPENDENCY_BLOCK',
        'scenes':{r['scene']:r['identity'] for r in receipts},'blocked':blocked,
        'actual_coarse_region_bound':regions,'actual_unique_FC_image_bound':len(image_ids),
        'nominal_bound':832,'new_AnyUp_QK':0,'new_projections':0})
    atomic_write_json(Path(binding['output_root'])/'evidence_manifest.json',result)
    return result
