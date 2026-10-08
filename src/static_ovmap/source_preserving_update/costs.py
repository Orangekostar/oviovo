"""Resident CPU decision timing and separate actual coarse-acquisition work."""

from collections import Counter
from pathlib import Path
import os
import platform
import statistics
import time

from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,read
from .binding import seal
from .decisions import scene_kernel


def measure(binding):
    from .orchestration import require_freeze
    freeze=require_freeze(binding);root=Path(binding['output_root']);cfg=binding['specification']['timing']
    names=binding['cohorts']['replica8'];methods=[m['id'] for m in binding['specification']['methods']]
    resident={};deps=[];index=ConsumptionIndex(root/'costs/verifications.json')
    for scene in names:
        mp=root/'evidence'/scene/'manifest.json';dp=root/'eligibility'/(scene+'.json')
        manifest,domain=read(mp),read(dp);_verified_identity(manifest);_verified_identity(domain)
        if domain['evidence_identity']!=manifest['identity']:raise ValueError('timing common domain changed')
        coarse={};deps.extend([index.identity(mp),index.identity(dp)])
        for fid in manifest['required_frames']:
            p=root/'coarse'/scene/fid/'receipt.json';frame=read(p);_verified_identity(frame)
            if frame['identity']!=domain['coarse_frames'][fid]:raise ValueError('timing coarse identity changed')
            coarse.update(frame['records']);deps.append(index.identity(p))
        resident[scene]=(manifest,domain,coarse)
    key=canonical_digest({'freeze':freeze['identity'],'timing':cfg,'dependencies':deps,
                          'producer':index.identity(__file__)})
    target=root/'costs/summary.json';previous=read(target) if target.exists() else None
    if previous and previous['input_identity']==key:
        _verified_identity(previous)
        # Resume adds observed verification overhead, while scientific samples stay exact.
        invocations=[read(p) for p in sorted((root/'coarse/invocations').glob('*.json'))]
        if invocations!=previous['acquisition']['worker_invocations']:
            acquisition={**previous['acquisition'],'worker_invocations':invocations,
                'worker_wall_seconds_including_resume_invocations':sum(r['elapsed_seconds'] for r in invocations)}
            previous=seal({**previous,'acquisition':acquisition})
            atomic_write_json(target,previous)
        return previous
    samples=[]
    # Exactly one untimed invocation for each resident scene/method before timed rounds.
    for scene in names:
        for method in methods:scene_kernel(*resident[scene],method)
    repeat=cfg['batched_repeats_per_sample']
    for round_number in (1,2):
        ordered_names=names if round_number==1 else list(reversed(names))
        ordered_methods=methods if round_number==1 else list(reversed(methods))
        for scene in ordered_names:
            inputs=resident[scene]
            for method in ordered_methods:
                start=time.perf_counter_ns()
                for _ in range(repeat):scene_kernel(*inputs,method)
                ns=time.perf_counter_ns()-start
                samples.append({'round':round_number,'scene':scene,'method':method,'repeats':repeat,
                                'milliseconds_per_scene':ns/1e6/repeat})
    summary={}
    for method in methods:
        values=[s for s in samples if s['method']==method]
        if len(values)!=cfg['samples_per_method']:raise ValueError('conditional timing sample coverage differs')
        summary[method]={'samples':len(values),'mean_ms_per_scene':statistics.mean(s['milliseconds_per_scene'] for s in values),
            'per_scene_ms':{s:statistics.mean(v['milliseconds_per_scene'] for v in values if v['scene']==s) for s in names},
            'baseline_action':'LITERAL_CLASS_PASS_THROUGH_NO_UPDATE' if method in ('SU00_D2','SU01_G1') else None}
    complete_counts=Counter();failed_counts=Counter();successful_wall=failed_wall=0.;frames=[];failures=[]
    for p in sorted((root/'coarse').glob('*/*/receipt*.json')):
        item=read(p);_verified_identity(item)
        if item['status']=='COMPLETE':
            complete_counts.update(item['counts']);successful_wall+=item['elapsed_seconds'];frames.append(index.identity(p))
        elif item['status']=='FAILED':
            failed_counts.update(item['counts']);failed_wall+=item['elapsed_seconds'];failures.append(index.identity(p))
    loads=[read(p) for p in sorted((root/'coarse/model_loads').glob('*.json'))]
    invocations=[read(p) for p in sorted((root/'coarse/invocations').glob('*.json'))]
    coarse_summary=read(root/'coarse/summary.json');inventory=read(root/'evidence_manifest.json')
    if complete_counts['successful_coarse_pools']+complete_counts['unavailable_coarse_pools']!=inventory['actual_coarse_region_bound']:
        raise ValueError('completed acquisition pools do not cover the locked bound')
    acquisition={'status':coarse_summary['status'],'successful_frame_count':len(frames),'failed_attempt_count':len(failures),
        'successful_counts':dict(complete_counts),'failed_counts':dict(failed_counts),
        'successful_leaf_wall_seconds_including_first_model_load':successful_wall,'failed_leaf_wall_seconds':failed_wall,
        'model_loads':loads,'model_load_seconds':sum(r['FC_model_load_seconds'] for r in loads),
        'worker_invocations':invocations,'worker_wall_seconds_including_resume_invocations':sum(r['elapsed_seconds'] for r in invocations),
        'frame_receipts':frames,'failed_attempt_receipts':failures,
        'actual_region_bound':inventory['actual_coarse_region_bound'],'actual_unique_image_bound':inventory['actual_unique_FC_image_bound'],
        'new_AnyUp_QK':0,'new_NQ_inference':0,'new_frontend_inference':0,'new_projection':0,
        'new_end_to_end_cold_calls':0,'new_GPU_peak_comparison':False,'A_and_historical_sources':'REUSED_NOT_FREE_AT_DEPLOYMENT'}
    result=seal({'status':'COSTS_COMPLETE' if coarse_summary['status']=='COARSE_COMPLETE' else 'PARTIAL_DEPENDENCY_BLOCK',
        'input_identity':key,'timing_mode':cfg['mode'],'unit':'milliseconds_per_scene','samples':samples,'methods':summary,
        'acquisition':acquisition,'excludes':cfg['excludes'],'warmup':'one per resident scene/method, untimed',
        'round2':'reverse scene and method order','scientific_replication':False,
        'environment':{'python':platform.python_version(),'CPU':platform.processor(),
            'BLAS_threads':{n:os.environ.get(n) for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}},
        'dependencies':deps,'standalone_latency':'NOT_MEASURED_IN_THIS_TASK'})
    atomic_write_json(target,result);index.write_memo(root/'costs/verifications.json');return result
