"""Actual frozen FP32 FC/ordinary AnyUp, sharing each selected image and Q/K."""

import argparse
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.cvpr_compact.area_fallback import region_vector
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.evidence_exploration.acquisition import project_raw, cosine_record
from static_ovmap.evidence_exploration.anyup_adapter import aligned_inputs, encode_qkv
from static_ovmap.evidence_exploration.timing import stream_selected
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest, _write_npz
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read
from static_ovmap.recovery_wave2.recovery_fc_worker import ContentCache
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity

from .binding import load_binding, seal
from .recognition_plan import unpack_region, region_identity
from .reread import reread_decision


def load_anyup(assets):
    sys.path.insert(0,assets['checkout'])
    from anyup.model import AnyUp
    model = AnyUp(use_natten=False)
    model.load_state_dict(torch.load(assets['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
    return model.eval().requires_grad_(False).to(device='cuda',dtype=torch.float32)


@torch.inference_mode()
def validate_ordinary(root, model, session, guidance, dense, regions, pooled, dependency):
    """One real new-mask check against official AnyUp, with extra work recorded."""
    path = Path(root)/'pilots/ordinary_anyup_parity.json'
    if path.exists():
        old = read(path)
        if old['operator_identity']!=dependency:
            raise ValueError('ordinary AnyUp operator changed after its real parity fixture')
        return old
    begin = time.perf_counter()
    first = next(iter(regions.values()))
    output_hw = tuple(first['output_hw'])
    attempts = []
    for chunk in (256,128,64):
        start = time.perf_counter()
        try:
            official = model(guidance,dense,output_size=output_hw,q_chunk_size=chunk)
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({'status':'TRUE_CUDA_OOM','chunk':chunk,'seconds':time.perf_counter()-start,'reason':str(exc)})
            torch.cuda.empty_cache()
            if chunk==64:
                raise
        else:
            attempts.append({'status':'COMPLETE','chunk':chunk,'seconds':time.perf_counter()-start})
            break
    checks = {}
    for key,region in regions.items():
        expected = (official[0]*region['weights'].reshape(1,*output_hw)).sum((1,2))/(region['weights'].sum()+1e-8)
        torch.testing.assert_close(pooled[key],expected,atol=1e-5,rtol=1e-5)
        a,b = project_raw(session,pooled[key],region['signed']),project_raw(session,expected,region['signed'])
        if a is None or b is None:
            raise ValueError('real ordinary parity fixture has an unusable frozen head representation')
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5)
        ac,bc = cosine_record(session,a,region['pixels']),cosine_record(session,b,region['pixels'])
        if ac['class']!=bc['class']:
            raise ValueError('ordinary streaming changes the official new-mask class')
        checks[key] = {'class':ac['class'],'mask_digest':region['mask_digest'],
            'max_raw_error':float((pooled[key]-expected).abs().max()),
            'max_projected_error':float(np.max(np.abs(a-b)))}
    result = seal({'status':'REAL_ORDINARY_ANYUP_PARITY_VERIFIED','operator_identity':dependency,
        'atol':1e-5,'rtol':1e-5,'actual_new_masks':checks,'attempts':attempts,
        'extra_successful_pilot_FC_inputs':0,'extra_successful_pilot_AnyUp_QK':1,
        'extra_physical_pilot_AnyUp_QK_attempts':len(attempts),'seconds':time.perf_counter()-begin,
        'shape':list(official.shape),'GT_used':False})
    atomic_write_json(path,result)
    return result


@torch.inference_mode()
def acquire_frame(binding, plan, fid, session, model, cache, assets, *, validate=False):
    root = Path(binding['output_root'])
    group = plan['frames'][str(fid)]
    index = ConsumptionIndex(root/'recognition'/plan['scene']/'verifications.json')
    definitions = {r:plan['regions'][r] for r in group['regions']}
    producer = [index.identity(__file__),index.identity(Path(__file__).parents[1]/'evidence_exploration/timing.py'),
                index.identity(Path(__file__).parents[1]/'evidence_exploration/anyup_adapter.py'),
                index.identity(Path(__file__).parents[1]/'cvpr_compact/area_fallback.py')]
    operator_identity = canonical_digest({'producer':producer,'assets':assets['identity'],'model':session.model_key})
    key = canonical_digest({'group':group,'regions':definitions,'operators':operator_identity,
                            'text':session.text_identity})
    path = root/'recognition'/plan['scene']/'frames'/(str(fid)+'.json')
    if path.exists():
        old = read(path)
        if old.get('status')=='COMPLETE':
            if old['input_identity']!=key:
                raise ValueError('completed recognition frame changed; invalidate only this leaf and descendants')
            index.identity(old['vectors']['path'],old['vectors'])
            return old
        path.rename(path.with_name(path.stem+'.failed_'+str(time.time_ns())+'.json'))
    begin = time.perf_counter()
    counts,attempts = Counter(),[]
    result = {'status':'RUNNING','scene':plan['scene'],'frame_id':int(fid),'input_identity':key,
        'operator_identity':operator_identity,'records':{},'counts':{},'attempts':attempts}
    atomic_write_json(path,result)
    try:
        index.identity(group['rgb']['path'],group['rgb'])
        image = cv2.imread(group['rgb']['path'],cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise ValueError('actual selected RGB failed to decode as original uint8 color')
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        tensor,size = image_tensor(rgb,'cuda')
        tensor_key = canonical_digest({'model':session.model_key,'tensor':_array_digest(tensor.cpu().numpy())})
        image_key = fc_image_identity({'image_sha256':group['rgb']['sha256']},session.model_key)
        dense_row = cache.lookup('dense',tensor_key)
        if dense_row is not None:
            if dense_row['image_content_key']!=image_key:
                raise ValueError('bound dense tensor refers to a different selected RGB')
            with np.load(dense_row['arrays']['path'],allow_pickle=False) as arrays:
                dense = torch.from_numpy(arrays['dense'].copy()).to('cuda')
            counts['parent_dense_cache_hits' if dense_row['cache_origin']=='PARENT' else 'task_dense_cache_hits'] += 1
        else:
            counts['FC_frame_encoding_attempts'] += 1
            dense = session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model),tensor)['clip_vis_dense']
            counts['new_FC_frame_inputs'] += 1
            dense_row = cache.write('dense',tensor_key,{'dense':dense.cpu().numpy()},
                                   {'input_tensor_key':tensor_key,'image_content_key':image_key})
        if dense.dtype!=torch.float32 or dense.ndim!=4 or dense.shape[0]!=1 or not torch.isfinite(dense).all():
            raise ValueError('actual dense FC is not the inherited finite batch-one FP32 representation')
        signed,anyup_regions,vectors = {},{},{}
        for rkey,item in definitions.items():
            mask = unpack_region(item,index)
            signed[rkey],_,_ = signed_mask(mask,size,tensor.shape[-2:],dense.shape[-2:],'cuda')
            feature_key = region_identity(item['support_hash'],item,item['type'],session.model_key,
                session.text_identity['sha256'],list(dense.shape[-2:]),item['output_hw'],item['representation'])
            if item['role']=='UNION_FC':
                try:
                    counts['union_FC_region_pool_attempts'] += 1
                    vector,audit = region_vector(session.model,session.operators,dense,signed[rkey])
                except ValueError as exc:
                    if str(exc) not in {'EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'}:
                        raise
                    result['records'][rkey] = {'available':False,'reason':str(exc),'feature_content_key':feature_key}
                    continue
                vector = vector.cpu().numpy().copy()
                record = cosine_record(session,vector,item['pixels'])
                vectors[rkey] = vector
                result['records'][rkey] = {**record,'available':record['feature_available'],
                                          'pooling':audit,'feature_content_key':feature_key}
            else:
                weights = F.interpolate((signed[rkey]>0).float(),size=tuple(item['output_hw']),mode='area').flatten()
                if not torch.isfinite(weights).all():
                    raise ValueError('nonfinite new-mask fine area weights')
                if float(weights.sum())<=0:
                    result['records'][rkey] = {'available':False,'reason':'EMPTY_FINE_AREA_SUPPORT','feature_content_key':feature_key}
                    continue
                anyup_regions[rkey] = {**item,'weights':weights,'signed':signed[rkey],
                                        'feature_content_key':feature_key}
        if anyup_regions:
            index.identity(group['depth']['path'],group['depth'])
            with np.load(group['depth']['path'],allow_pickle=False) as arrays:
                depth = arrays['depth_m']
            # Original guidance preparation is retained. Category/depth outputs
            # are discarded; ordinary attention has no geometry factors.
            guidance,_,_,_ = aligned_inputs(rgb,np.zeros(rgb.shape[:2],np.int32),depth,size,
                                             tensor.shape[-2:],device='cuda')
            output_hw = tuple(next(iter(anyup_regions.values()))['output_hw'])
            counts['AnyUp_QK_attempts'] += 1
            qkv = encode_qkv(model,guidance,dense,output_hw)
            counts['AnyUp_QK_computations'] += 1
            pooled,attempts = stream_selected(model,qkv,dense.shape[-2:],output_hw,anyup_regions,None,'anyup',counts)
            if validate:
                validate_ordinary(root,model,session,guidance,dense,anyup_regions,pooled,operator_identity)
            for rkey,raw in pooled.items():
                vector = project_raw(session,raw,signed[rkey])
                item = anyup_regions[rkey]
                record = cosine_record(session,vector,item['pixels'])
                result['records'][rkey] = {**record,'available':record['feature_available'],
                                          'feature_content_key':item['feature_content_key']}
                if vector is not None:
                    vectors[rkey] = vector
            counts['AnyUp_region_pools'] += len(pooled)
            del qkv,guidance,pooled
        torch.cuda.synchronize()
        vector_path = path.with_suffix('.npz')
        _write_npz(vector_path,vectors)
        result.update(status='COMPLETE',vectors=index.identity(vector_path),
            actual_dense_shape=list(dense.shape),input_tensor_key=tensor_key,image_content_key=image_key,
            dense_receipt=dense_row['receipt_path'],dense_arrays=dense_row['arrays'],
            region_count=len(definitions),attempts=attempts,
            ordinary_attention_only=True,new_segmentation_inference=0,new_NQ_inference=0)
    except BaseException as exc:
        result.update(status='FAILED',reason=type(exc).__name__+': '+str(exc),attempts=attempts)
        raise
    finally:
        result.update(counts=dict(counts),elapsed_seconds=time.perf_counter()-begin)
        atomic_write_json(path,seal(result))
        index.write_memo(root/'recognition'/plan['scene']/'verifications.json')
    print('RECOGNIZED',plan['scene'],fid,'regions',len(definitions),dict(counts),flush=True)
    return seal(result)


def assemble_decisions(binding, plan, records):
    all_regions = {r:value for frame in records for r,value in frame['records'].items()}
    if set(all_regions)!=set(plan['regions']) or any(r['status']!='COMPLETE' for r in records):
        raise ValueError('unfinished recognition is an execution block, not a scientific KEEP')
    incumbents = {}
    cfg = binding['specification']['reread']
    for item in plan['incumbents']:
        observations = []
        for key in item['regions']:
            region,result = plan['regions'][key],all_regions[key]
            observations.append({k:region[k] for k in ('frame_id','bank','type','mask_digest')} |
                {'scores':result.get('scores'),'reason':result.get('reason',result.get('fallback_reason')),
                 'feature_content_key':result['feature_content_key']})
        decision = reread_decision(item['old_class'],binding['scenes'][plan['scene']].get('valid_ids',
            read(binding['scenes'][plan['scene']]['context']['path'])['models']['native']['valid_ids']),observations,
            margin=cfg['aggregate_margin_over_incumbent'],minimum_regions=cfg['stable_min_distinct_regions'],
            minimum_extra=cfg['stable_min_extra_regions'],vote_fraction=cfg['stable_vote_fraction'],
            full_margin=cfg['stable_full_view_margin_min'],median_margin=cfg['stable_median_margin_min'])
        incumbents[str(item['owner'])] = {**decision,'selection':item,'unavailable_types':item['missing']}
    unions = {}
    for digest,item in plan['unions'].items():
        result = all_regions[item['region']] if item['region'] else {'available':False,'reason':item['status']}
        unions[digest] = {**result,'requested_by':item['requested_by'],'region_id':item['region'],
                         'support_hash':item['hypothesis']['support_hash']}
    result = seal({'status':'RECOGNITION_COMPLETE','scene':plan['scene'],'plan_identity':plan['identity'],
        'frames':{str(r['frame_id']):r['identity'] for r in records},'incumbents':incumbents,'unions':unions,
        'GT_used':False,'structural_decisions_changed_by_class_scores':False,'blocked':plan['blocked']})
    atomic_write_json(Path(binding['output_root'])/'recognition'/plan['scene']/'decisions.json',result)
    return result


def run(root, phase):
    binding = load_binding(root)
    assets = read(Path(root)/'assets.json')
    scenes = binding['specification']['pilots'] if phase=='pilot' else [s for names in binding['cohorts'].values() for s in names]
    torch.set_num_threads(4)
    cfg = assets['execution_config']
    index = ConsumptionIndex(Path(root)/'recognition/model_verifications.json')
    session,model,decisions = None,None,[]
    with exclusive_lock(Path(cfg['gpu_lock'])),torch.inference_mode():
        for scene in scenes:
            data = read(binding['scenes'][scene]['context']['path'])
            text_session = FCSession(cfg,data,index,cache=None)
            if session is None:
                session = text_session
                session.load_model()
                model = load_anyup(assets)
                atomic_write_json(Path(root)/'recognition/model_load.json',
                    seal({'FC_model':session.model_key,'FC_model_load_seconds':session.model_load_seconds,
                          'actual_worker_python':sys.executable,'actual_torch':torch.__version__,
                          'FC_weight_audit':session.weight_audit,'models_required':['FC_FROZEN','ORIGINAL_ANYUP']}))
            elif session.model_key!=text_session.model_key:
                raise ValueError('same worker cannot silently change the physical FC model')
            session.text,session.ids,session.text_identity = text_session.text,text_session.ids,text_session.text_identity
            cache = ContentCache(assets['read_only_dense_cache_roots'],Path(root)/'content_cache/fc',session.model_key,
                                 index=index,resolver=PathResolver(binding['path_map']))
            plan = read(Path(root)/'recognition'/scene/'plan.json')
            records = [acquire_frame(binding,plan,fid,session,model,cache,assets,
                validate=phase=='pilot' and not (Path(root)/'pilots/ordinary_anyup_parity.json').exists())
                for fid in sorted(map(int,plan['frames']))]
            decisions.append(assemble_decisions(binding,plan,records))
        index.write_memo(Path(root)/'recognition/model_verifications.json')
    if phase=='pilot':
        from .outputs import predict_study
        from .evaluation import evaluate_scene
        predictions = predict_study(binding,scenes=scenes)
        scored = [evaluate_scene((binding,r['scene']),require_frozen=False) for r in predictions]
        parity = read(Path(root)/'pilots/ordinary_anyup_parity.json')
        result = seal({'status':'INTEGRATED_PILOTS_VERIFIED','scenes':{r['scene']:r['identity'] for r in scored},
            'recognition':{r['scene']:r['identity'] for r in decisions},'ordinary_parity':parity['identity'],
            'real_partitions_scored':sum(len(r['rows']) for r in scored),
            'successful_pilot_science_leaves_reusable':True})
        atomic_write_json(Path(root)/'pilots/summary.json',result)
    else:
        result = seal({'status':'RECOGNITION_COMPLETE','scenes':{r['scene']:r['identity'] for r in decisions},
                        'scene_count':len(decisions)})
        atomic_write_json(Path(root)/'recognition/summary.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--phase',choices=('pilot','recognize'),required=True)
    args = parser.parse_args()
    run(args.root,args.phase)
