"""Fresh G1/observer/proposals/regions per call; only required models resident."""

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import gc
import os
from pathlib import Path
import platform
import subprocess
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.backbone_wave1.runtime import exclusive_lock
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask
from static_ovmap.cvpr_compact.area_fallback import region_vector
from static_ovmap.cvpr_compact.region_worker import FCSession
from static_ovmap.evidence_exploration.acquisition import project_raw, cosine_record
from static_ovmap.evidence_exploration.anyup_adapter import aligned_inputs, encode_qkv
from static_ovmap.evidence_exploration.timing import stream_selected
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.module_validation.evaluation import PredictionPayload
from static_ovmap.module_validation.scannet_study import native_ranks, load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.runtime_parity.instrumentation import Stages
from static_ovmap.runtime_parity.runner import load_common
from static_ovmap.runtime_parity.views import build_views, CallLoader

from .binding import load_binding, seal
from .observations import SourceRowProjector, pose_banks, all_unit_evidence, support_mask, load_panoptic
from .outputs import construct_partition
from .proposals import spatial_neighbors, bank_rates, build_library
from .recognition_plan import choose_view
from .recognition_worker import load_anyup
from .reread import select_incumbents, recognition_masks, reread_decision
from .support_units import build_units
from .verification import structural_decisions


def load_snapshot():
    """Observe shared load; do not alter processes belonging to other users."""
    fields = ('model name', 'processor')
    cpu = [line.strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
           if line.split(':')[0].strip() in fields]
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.used,utilization.gpu',
        '--format=csv,noheader'],capture_output=True,text=True)
    processes = subprocess.run(['ps','-eo','pid,ppid,user,comm,pcpu,pmem'],capture_output=True,text=True)
    return {'load_average':os.getloadavg(),'CPU_description':cpu[:2],
        'GPU_status':gpu.stdout.strip(),'GPU_query_exit':gpu.returncode,
        'processes':processes.stdout.splitlines(),'other_processes_modified':False}


class Timer:
    def __init__(self):
        self.seconds = dict.fromkeys(('base_G1_recovery','additional_observer_proposals',
            'repair_verification','additional_FC_AnyUp','final_payload'),0.)
        self.events,self.counts = {},Counter()

    @contextmanager
    def span(self, name, *, cuda=False):
        begin = time.perf_counter()
        if cuda:
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
        try:
            yield
        finally:
            if cuda:
                b.record()
                self.events.setdefault(name,[]).append((a,b))
            self.seconds[name] += time.perf_counter()-begin


def base_payload(common, d2_labels, recovered, method):
    owners,semantics = common.baseline.owner_ids.copy(),common.baseline.semantic_labels.copy()
    labels = dict(d2_labels)
    for owner,label in labels.items():
        semantics[owners==owner] = label
    for owner,label in recovered.items():
        support = (common.raw==owner)&(common.baseline.owner_ids==0)
        owners[support],semantics[support] = owner,label
        labels[owner] = label
    return PredictionPayload(method,'COMBO',common.scene,common.baseline.geometry,owners,semantics,
        native_ranks(owners,labels,common.nearest,common.matched),{}, {'cold_recomputed':True})


def observe_cold(inputs, binding, timer):
    cfg = binding['specification']['observer']
    frames,_ = pose_banks(inputs.capture['frames'],limit=cfg['max_frames'],
                         translation=cfg['pose_translation_m'],degrees=cfg['pose_rotation_degrees'])
    projector = SourceRowProjector(inputs.xyz,inputs.faces,batch_pixels=cfg['ray_batch_max'])
    observed,maps = [],{}
    for frame in frames:
        path = inputs.capture_path.parent/frame['depth_path']
        with np.load(path,allow_pickle=False) as arrays:
            depth = arrays['depth_m']
        panoptic,_ = load_panoptic(inputs.capture_path.parent,frame,inputs.index,binding['path_map'])
        source,valid = projector.project(frame['intrinsics'],frame['pose_c2w'],depth,
            absolute=cfg['depth_absolute_tolerance_m'],relative=cfg['depth_relative_tolerance'],minimum=cfg['minimum_depth_m'])
        stats = all_unit_evidence(inputs.units.units,len(inputs.raw),source,valid,panoptic,settings=cfg)
        fid = int(frame['frame_id'])
        observed.append({'frame_id':fid,'bank':frame['bank'],'panoptic_available':True,'units':stats})
        maps[fid] = {'source_rows':source,'valid':valid,'panoptic':panoptic,'depth':depth,'bank':frame['bank']}
        timer.counts.update(observer_frames=1,observer_rays=depth.size)
    return {'frames':observed},maps


def cold_regions(inputs, maps, operations, method, spec):
    """Same GT-free rules; masks are transient and are never imported from science."""
    reread = method in ('IR06_ANYUP_REREAD','IR07_BOUNDARY_STABLE','IR08_COMBINATION')
    cfg,repair = spec['reread'],spec['repair']
    incumbents = select_incumbents(inputs.units,inputs.probabilities,limit=cfg['max_incumbents']) if reread else []
    unions = [h for h in operations if h['host'] is None]
    supports = {r['unit']:inputs.units.units[r['unit']].rows for r in incumbents}
    supports.update({h['digest']:np.unique(np.concatenate([inputs.units.units[n].rows for n in h['units']])) for h in unions})
    best = {}
    for fid,frame in maps.items():
        for name,rows in supports.items():
            mask = support_mask(frame['source_rows'],frame['valid'],rows)
            view = choose_view([{'frame_id':fid,'bank':frame['bank'],'mask':mask}],
                minimum_pixels=cfg['minimum_full_pixels'] if name.startswith('I:') else repair['classification_min_pixels'],
                minimum_extent=repair['classification_min_bbox_extent'])
            if view is None:
                continue
            key = (name,frame['bank'] if name.startswith('I:') else 'union')
            previous = best.get(key)
            if previous is None or (-view['pixels'],fid,view['mask_digest'])<(-previous['pixels'],previous['frame_id'],previous['mask_digest']):
                best[key] = view
    regions,by_frame = {},{}
    def add(view,role,owner,kind,key):
        rid = role+':'+str(owner)+':'+str(view['frame_id'])+':'+kind
        regions[rid] = {**view,'role':role,'owner':owner,'type':kind,'hypothesis':key}
        by_frame.setdefault(view['frame_id'],[]).append(rid)
        return rid
    for item in incumbents:
        item['regions'] = []
        for bank in ('proposal','verification'):
            view = best.get((item['unit'],bank))
            if view is None:
                continue
            defs,_ = recognition_masks(view['mask'],maps[view['frame_id']]['panoptic'],
                core_min_fraction=cfg['core_min_fraction'],intersection_min_fraction=cfg['frontend_intersection_min_fraction'],
                minimum_pixels=cfg['minimum_subregion_pixels'])
            for region in defs:
                if method=='IR06_ANYUP_REREAD' and region['type']!='FULL':
                    continue
                v = {**view,'mask':region['mask'],'mask_digest':region['mask_digest'],'pixels':region['pixels']}
                item['regions'].append(add(v,'REREAD',item['owner'],region['type'],None))
    for item in unions:
        view = best.get((item['digest'],'union'))
        if view is not None:
            add(view,'UNION',item['output_owner'],'UNION_FULL',item['digest'])
    return incumbents,regions,by_frame


@torch.inference_mode()
def cold_call(binding, common, session, anyup, call):
    root = Path(binding['output_root'])
    scene,method = call['scene'],call['method']
    dest = root/'timing/calls'/(str(call['repeat'])+'_'+scene+'_'+method)
    expected = read(root/'predictions'/scene/'receipt.json')['methods'][method]
    path = dest/'receipt.json'
    if path.exists():
        previous = read(path)
        if previous['status']=='COLD_CALL_PARITY_VERIFIED' and previous['prediction_key']==expected['prediction_key']:
            return previous
        raise ValueError('existing cold leaf cannot silently switch its scientific reference')
    if dest.exists() and any(dest.iterdir()):
        from .resume import archive_paths
        archive_paths(root,[str(dest.relative_to(root))],'preserved unsuccessful cold call before corrected execution')
    timer,index,stages = Timer(),ConsumptionIndex(),Stages()
    config = binding['specification']
    d2_labels,probabilities = fuse_readout(common.sources,common.data['temperatures'],'D2',common.valid_ids,owner_labels(common.baseline))
    d2 = base_payload(common,d2_labels,{},'IR00_D2')
    d2.lock()
    parent_keys = {key:binding['scenes'][scene]['predictions'][name]['prediction_key'] for key,name in [('D2','IR00_D2'),('G1','IR01_G1')]}
    torch.cuda.empty_cache()
    start_allocated,start_reserved = torch.cuda.memory_allocated(),torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    result = {'status':'RUNNING','scene':scene,'method':method,'repeat':call['repeat']}
    try:
        with timer.span('base_G1_recovery'):
            registry,manifest = build_views(common,dest/'views','R2_ROI_EXACT',index,stages)
            loader = CallLoader(dest/'views/manifest.json',index,stages,reuse=True)
            base_groups = {}
            for selected in manifest['g1'].values():
                if selected:
                    rid = selected[0]
                    base_groups.setdefault(int(manifest['requests'][rid]['frame_id']),[]).append(rid)
        inputs = SimpleNamespace(**vars(common),d2=d2,g1=None,index=index,probabilities=probabilities)
        operations,regions,new_groups,incumbents,maps = [],{}, {},[],{}
        if method!='IR01_G1':
            with timer.span('additional_observer_proposals'):
                inputs.capture_path = Path(common.data['capture_manifest'])
                inputs.capture = read(inputs.capture_path)
                inputs.units = build_units(common.xyz,common.faces,common.raw,d2.owner_ids,d2.geometry,
                                           config['support'],inherited_registry=registry)
                semantic_only = method in ('IR06_ANYUP_REREAD','IR07_BOUNDARY_STABLE')
                if method!='IR02_NEAREST_ATTACH':
                    observation,maps = observe_cold(inputs,binding,timer)
                if not semantic_only:
                    directed,neighbors,edges = spatial_neighbors(inputs,config['support'])
                    if method=='IR02_NEAREST_ATTACH':
                        proposal = verification = {e:{'n':0,'same':None,'separate':None} for e in edges}
                        library = []
                    else:
                        proposal = bank_rates(observation,edges,'proposal',separation_dominance=config['observer']['separation_dominance'])
                        verification = bank_rates(observation,edges,'verification',separation_dominance=config['observer']['separation_dominance'])
                        repair = config['repair']
                        library = build_library(inputs.units.units,inputs.units.seeds,neighbors,proposal,
                            max_parent_id=int(max(common.raw.max(initial=0),d2.owner_ids.max(initial=0))),
                            minimum_views=repair['minimum_proposal_views_per_pair'],minimum_same=repair['minimum_same_rate'],
                            max_units=config['support']['max_units_per_hypothesis'],cap=config['support']['max_group_hypotheses'],
                            separation=repair['separation_penalty'],penalty=repair['edit_penalty'])
            with timer.span('repair_verification'):
                if not semantic_only:
                    slate = structural_decisions(inputs.units,directed,proposal,verification,library,config['support'],config['repair'])
                    structure_method = 'IR05_VERIFIED_REPAIR' if method=='IR08_COMBINATION' else method
                    operations = slate[structure_method]['selected']
                incumbents,regions,new_groups = cold_regions(inputs,maps,operations,method,config)
        captures = {int(f['frame_id']):f for f in read(common.data['capture_manifest'])['frames']}
        recovered,records,union_classes = {},{},{}
        for fid in sorted(base_groups.keys()|new_groups.keys()):
            # No tensor from the previous frame may remain live when encoding
            # the next frame. The region dictionaries otherwise retain it.
            signed = vector = weights = raw = item = None
            rids = base_groups.get(fid,[])
            stage = 'base_G1_recovery' if rids else 'additional_FC_AnyUp'
            with timer.span(stage,cuda=True):
                values = {rid:loader(rid) for rid in rids}
                if values:
                    rgb = values[rids[0]]['image']
                else:
                    rgb = cv2.cvtColor(cv2.imread(str(Path(common.data['capture_manifest']).parent/captures[fid]['rgb_path']),cv2.IMREAD_UNCHANGED),cv2.COLOR_BGR2RGB)
                image,size = image_tensor(rgb,'cuda')
                dense = session.operators['extract_features_convnext'](SimpleNamespace(clip_model=session.model),image)['clip_vis_dense']
                if dense.dtype!=torch.float32 or not torch.isfinite(dense).all():
                    raise ValueError('invalid cold original FP32 dense FC')
                timer.counts['FC_frame_inputs'] += 1
                for rid,value in values.items():
                    signed,_,_ = signed_mask(value['target'],size,image.shape[-2:],dense.shape[-2:],'cuda')
                    vector,_ = region_vector(session.model,session.operators,dense,signed)
                    record = cosine_record(session,vector.cpu().numpy().copy(),int(value['target'].sum()))
                    recovered[int(manifest['requests'][rid]['raw_owner'])] = record['class']
                    timer.counts['G1_region_pools'] += 1
            with timer.span('additional_FC_AnyUp',cuda=True) if new_groups.get(fid) else nullcontext():
                fine = {}
                output_hw = tuple(n//4 for n in image.shape[-2:])
                for rkey in new_groups.get(fid,[]):
                    item = regions[rkey]
                    signed,_,_ = signed_mask(item['mask'],size,image.shape[-2:],dense.shape[-2:],'cuda')
                    if item['role']=='UNION':
                        try:
                            vector,_ = region_vector(session.model,session.operators,dense,signed)
                        except ValueError as exc:
                            if str(exc) not in {'EMPTY_AREA_MASK_SUPPORT','INVALID_REGION_FEATURE'}:
                                raise
                            continue
                        record = cosine_record(session,vector.cpu().numpy().copy(),item['pixels'])
                        union_classes[item['hypothesis']] = record['class']
                        timer.counts['union_FC_region_pools'] += 1
                    else:
                        weights = F.interpolate((signed>0).float(),size=output_hw,mode='area').flatten()
                        if float(weights.sum())>0:
                            fine[rkey] = {**item,'signed':signed,'weights':weights}
                if fine:
                    guidance,_,_,_ = aligned_inputs(rgb,np.zeros(rgb.shape[:2],np.int32),maps[fid]['depth'],size,image.shape[-2:],device='cuda')
                    qkv = encode_qkv(anyup,guidance,dense,output_hw)
                    timer.counts['AnyUp_QK_computations'] += 1
                    pooled,attempts = stream_selected(anyup,qkv,dense.shape[-2:],output_hw,fine,None,'anyup',timer.counts)
                    for rkey,raw in pooled.items():
                        item = fine[rkey]
                        record = cosine_record(session,project_raw(session,raw,item['signed']),item['pixels'])
                        records[rkey] = {k:item[k] for k in ('frame_id','bank','type','mask_digest')} | {'scores':record.get('scores')}
                    timer.counts['AnyUp_region_pools'] += len(pooled)
                    del qkv,guidance,pooled
                fine.clear()
            signed = vector = weights = raw = item = None
            loader.images.clear()
            del dense,image,values
        with timer.span('repair_verification'):
            relabels = {}
            cfg = config['reread']
            for item in incumbents:
                decision = reread_decision(item['old_class'],common.valid_ids,[records[r] for r in item['regions'] if r in records],
                    margin=cfg['aggregate_margin_over_incumbent'],minimum_regions=cfg['stable_min_distinct_regions'],
                    minimum_extra=cfg['stable_min_extra_regions'],vote_fraction=cfg['stable_vote_fraction'],
                    full_margin=cfg['stable_full_view_margin_min'],median_margin=cfg['stable_median_margin_min'])
                name = 'IR06_class' if method=='IR06_ANYUP_REREAD' else 'IR07_class'
                if decision[name]!=item['old_class']:
                    relabels[item['owner']] = decision[name]
        with timer.span('final_payload'):
            inputs.g1 = base_payload(common,d2_labels,recovered,'IR01_G1')
            if method=='IR01_G1':
                payload = inputs.g1
            else:
                payload,_ = construct_partition(inputs,method,operations,union_classes=union_classes,relabels=relabels,
                    lock=False,parent_prediction_keys=parent_keys)
        torch.cuda.synchronize()
        seconds = time.perf_counter()-begin
        allocated,reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
        # All payload hashing and comparisons are outside the synchronized timer.
        payload.lock()
        if payload.prediction_key!=expected['prediction_key']:
            raise ValueError('cold output differs from its actual scientific output: '+scene+'/'+method)
        residual = seconds-sum(timer.seconds.values())
        if residual < -1e-8:
            raise ValueError('exclusive stage sum exceeds synchronized call wall')
        result.update(status='COLD_CALL_PARITY_VERIFIED',seconds_per_scene=seconds,
            exclusive_host_seconds=timer.seconds,unaccounted_wall_seconds=residual,
            nonadditive_CUDA_seconds={s:sum(a.elapsed_time(b) for a,b in es)/1000 for s,es in timer.events.items()},
            peak_cuda_allocated_bytes=allocated,peak_cuda_reserved_bytes=reserved,
            resident_cuda_allocated_bytes=start_allocated,resident_cuda_reserved_bytes=start_reserved,
            incremental_peak_cuda_allocated_bytes=allocated-start_allocated,
            incremental_peak_cuda_reserved_bytes=reserved-start_reserved,
            prediction_key=payload.prediction_key,correctness_checked_after_timer=True,
            persistent_feature_view_result_cache_hits=0,OS_page_cache='UNCONTROLLED_NOT_CLEARED',
            models_required=['FC_FROZEN','ORIGINAL_ANYUP'] if anyup is not None else ['FC_FROZEN'])
    except BaseException as exc:
        result.update(status='FAILED',reason=type(exc).__name__+': '+str(exc),
            observed_failed_wall_seconds=time.perf_counter()-begin)
        raise
    finally:
        result.update(counts=dict(timer.counts))
        atomic_write_json(path,seal(result))
    print('COLD_CALL',scene,method,call['repeat'],seconds,'PARITY_VERIFIED',flush=True)
    return seal(result)


def run(root):
    binding = load_binding(root)
    assets,plan = read(Path(root)/'assets.json'),read(Path(root)/'timing/binding.json')
    torch.set_num_threads(4)
    index = ConsumptionIndex(Path(root)/'timing/model_verifications.json')
    common = {s:load_common(binding['reference'],s,Path(root)/'timing/common') for s in binding['cohorts']['replica8']}
    cfg = assets['execution_config']
    session,anyup,calls = None,None,[]
    loads = []
    load_before = load_snapshot()
    with exclusive_lock(Path(cfg['gpu_lock'])):
        for call in plan['calls']:
            current = FCSession(cfg,common[call['scene']].data,index,cache=None)
            if session is None:
                session = current
                session.load_model()
                loads.append({'model':'FC_FROZEN','seconds':session.model_load_seconds})
            session.text,session.ids,session.text_identity = current.text,current.ids,current.text_identity
            requires = call['method'] in ('IR06_ANYUP_REREAD','IR07_BOUNDARY_STABLE','IR08_COMBINATION')
            if requires and anyup is None:
                begin = time.perf_counter()
                anyup = load_anyup(assets)
                torch.cuda.synchronize()
                loads.append({'model':'ORIGINAL_ANYUP','seconds':time.perf_counter()-begin,'call':call})
            elif not requires and anyup is not None:
                anyup = None
                gc.collect()
                torch.cuda.empty_cache()
            calls.append(cold_call(binding,common[call['scene']],session,anyup,call))
    summary = []
    for method in plan['methods']:
        rows = [r for r in calls if r['method']==method]
        counts = Counter()
        for r in rows:
            counts.update(r['counts'])
        summary.append({'method':method,'calls':len(rows),'mean_seconds_per_scene':float(np.mean([r['seconds_per_scene'] for r in rows])),
            'median_seconds_per_scene':float(np.median([r['seconds_per_scene'] for r in rows])),
            'peak_cuda_allocated_bytes':max(r['peak_cuda_allocated_bytes'] for r in rows),
            'peak_cuda_reserved_bytes':max(r['peak_cuda_reserved_bytes'] for r in rows),
            'incremental_peak_cuda_allocated_bytes':max(r['incremental_peak_cuda_allocated_bytes'] for r in rows),
            'incremental_peak_cuda_reserved_bytes':max(r['incremental_peak_cuda_reserved_bytes'] for r in rows),
            'counts':dict(counts),'mean_exclusive_host_seconds':{s:float(np.mean([r['exclusive_host_seconds'][s] for r in rows])) for s in rows[0]['exclusive_host_seconds']}})
    result = seal({'status':'COLD_TIMING_COMPLETE','binding_identity':plan['identity'],'summary':summary,
        'calls':[{'scene':r['scene'],'method':r['method'],'repeat':r['repeat'],'identity':r['identity']} for r in calls],
        'call_count':len(calls),'model_loads':loads,'all_valid_repeats_included':True,'online_30FPS_validated':False,
        'shared_system_load_before':load_before,'shared_system_load_after':load_snapshot(),
        'hardware':{'CPU':platform.processor(),'platform':platform.platform(),'torch':torch.__version__,
            'GPU':torch.cuda.get_device_name(),'gpu_id':binding['gpu'],'precision':'FP32',
            'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
            'BLAS_threads':4,'raycast_threads':4,'OS_page_cache':'UNCONTROLLED','remaining_load_average':os.getloadavg()}})
    atomic_write_json(Path(root)/'timing/summary.json',result)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    run(parser.parse_args().root)
