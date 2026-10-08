"""Bound new-task phases, scientific source freeze and real terminal receipts."""

from datetime import datetime,timezone
from pathlib import Path
import argparse
import subprocess
import time
import traceback

from .binding import bind,load_binding,assets
from .common import REPO,ConsumptionIndex,archive,read,verified,write

PHASES=('bind','assets','plan','pilot','segment','lift','readout','predict','evaluate','diagnose','cost','report')


def source_files():
    return sorted([*list((REPO/'src/static_ovmap/samv_local_probe').glob('*.py')),
        REPO/'scripts/evaluation/run_ovimap_samv_local_probe.py',REPO/'configs/static_ovmap/samv_local_probe_v1.json',
        REPO/'tests/evaluation/test_samv_local_probe.py'])


def freeze(binding):
    root=Path(binding['output_root']);pilot=verified(root/'pilots/summary.json')
    required=['binding','query_plan','geometry','samv_worker','sam2_worker','worker_io','segmentation','lifting',
              'readout','fc_worker','outputs','evaluation','diagnostics','selection','costs','reporting','runner']
    if any(not (REPO/'src/static_ovmap/samv_local_probe'/(name+'.py')).is_file() for name in required):
        raise ValueError('every prescribed implementation phase must exist before freeze')
    if pilot['status'] not in ('REAL_ENGINEERING_PILOTS_VERIFIED','NO_QUERIABLE_TARGET_IN_COHORT') or pilot['new_AP_scored']:
        raise ValueError('real GT-free engineering pilots must precede freeze')
    tests=verified(root/'focused_tests.json')
    if tests['exit_code']!=0 or '12 passed' not in Path(tests['log']['path']).read_text():raise ValueError('focused production tests did not pass')
    index=ConsumptionIndex(root/'freeze_verifications.json');files=[index.identity(p) for p in source_files()]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    for p in source_files():
        relative=str(p.relative_to(REPO));tracked=subprocess.run(['git','show',commit+':'+relative],cwd=REPO,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if tracked.returncode or tracked.stdout!=p.read_bytes():raise ValueError('complete implementation must be committed before scientific freeze: '+relative)
    result=write(root/'freeze.json',{'status':'SOURCE_CONFIG_PROFILE_FROZEN','commit':commit,
        'binding_identity':binding['identity'],'files':files,'pilots_identity':pilot['identity'],
        'assets_identity':verified(root/'model_assets.json')['identity'],'resource_profile':verified(root/'resource_profile.json'),
        'focused_tests':tests,'all_implementation_present':True,'before_remaining_scientific_acquisition_and_new_AP':True})
    index.write_memo(root/'freeze_verifications.json');return result


def require_freeze(binding):
    root=Path(binding['output_root']);record=verified(root/'freeze.json')
    if record['status']!='SOURCE_CONFIG_PROFILE_FROZEN' or record['binding_identity']!=binding['identity']:
        raise ValueError('actual scientific source/configuration freeze required')
    index=ConsumptionIndex(root/'freeze_verifications.json')
    for file in record['files']:index.identity(file['path'],file)
    if verified(root/'model_assets.json')['identity']!=record['assets_identity'] or verified(root/'resource_profile.json')!=record['resource_profile']:
        raise ValueError('model or global resource profile changed after freeze')
    return record


def invalidate(binding,phase,scene,reason):
    """Journal a primary-reviewed correction without touching inherited roots."""
    root=Path(binding['output_root']);graph={
        'plan':['query_plan','segmentation','lifted','semantic_masks','readout','predictions','evaluation','diagnostics'],
        'segment':['segmentation','lifted','semantic_masks','readout','predictions','evaluation','diagnostics'],
        'lift':['lifted','predictions','evaluation','diagnostics'],
        'readout':['semantic_masks','readout','predictions','evaluation','diagnostics'],
        'predict':['predictions','evaluation','diagnostics'],'evaluate':['evaluation','diagnostics'],
        'diagnose':['diagnostics']}
    if phase not in graph or scene not in binding['scenes'] or not reason:raise ValueError('explicit scoped correction required')
    paths=[root/d/scene for d in graph[phase]]
    paths+=[root/d/'summary.json' for d in graph[phase]]
    paths +=[root/'pools',root/'result_store.json',root/'selection.json',root/'costs.json',root/'report_receipt.json',root/'requirement_audit.json']
    if phase in ('plan','segment'):paths+=[root/'timings']
    return archive(root,paths,{'phase':phase,'scene':scene,'reviewed_correction':reason})


def run(args):
    root=Path(args.storage_root or args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    invocation=root/'execution/invocations'/str(time.time_ns());invocation.mkdir(parents=True,exist_ok=True)
    results={};failure=None;binding=None;model_assets=None
    for phase in PHASES if args.phase=='all' else (args.phase,):
        begin=time.perf_counter();print('PHASE',phase,flush=True)
        try:
            if phase=='bind':
                binding=bind(args.spec,args.parent_root,args.output_root,storage_root=args.storage_root,path_map=args.path_map,gpu=args.gpu)
                result=binding
            else:
                binding=binding or load_binding(root)
                if phase=='assets':
                    model_assets=assets(binding,args.assets_root,args.samv_python,args.sam2_python);result=model_assets
                elif phase=='plan':
                    from .query_plan import plan
                    count=verified(root/'resource_profile.json')['window_length'] if (root/'resource_profile.json').exists() else 6
                    result=write(root/'query_plan/summary.json',{'status':'ALL_QUERIES_LOCKED','scenes':{s:r['identity'] for s,r in plan(binding,count).items()}})
                elif phase=='pilot':
                    from .segmentation import pilot
                    result=pilot(binding,model_assets or assets(binding,args.assets_root,args.samv_python,args.sam2_python))
                elif phase=='segment':
                    from .segmentation import segment
                    if not (root/'freeze.json').exists():freeze(binding)
                    result=segment(binding,model_assets or assets(binding,args.assets_root,args.samv_python,args.sam2_python))
                elif phase=='lift':
                    from .lifting import lift
                    require_freeze(binding);result=lift(binding)
                elif phase=='readout':
                    from .readout import acquire
                    require_freeze(binding);result=acquire(binding)
                elif phase=='predict':
                    from .outputs import predict
                    result=predict(binding)
                elif phase=='evaluate':
                    from .evaluation import evaluate
                    result=evaluate(binding)
                elif phase=='diagnose':
                    from .diagnostics import analyze
                    result=analyze(binding)
                elif phase=='cost':
                    from .costs import collect
                    result=collect(binding,model_assets or assets(binding,args.assets_root,args.samv_python,args.sam2_python))
                elif phase=='report':
                    from .reporting import report
                    result=report(binding)
                else:raise ValueError('unknown phase')
            results[phase]=result
        except Exception as exc:
            failure={'phase':phase,'error':type(exc).__name__+': '+str(exc),'traceback':traceback.format_exc()}
            print('PHASE_FAILED',phase,failure['error'],flush=True)
            result={'status':'PARTIAL_DEPENDENCY_BLOCK','failure':failure}
        record=write(invocation/(phase+'.json'),{'phase':phase,'result_status':result['status'],
            'result_identity':result.get('identity'),'elapsed_wall_seconds':time.perf_counter()-begin,
            'UTC':datetime.now(timezone.utc).isoformat(),'failure':failure})
        write(root/'execution'/(phase+'.json'),record)
        if failure:
            if binding:
                from .reporting import partial
                partial(binding,failure)
            break
    complete=failure is None
    if args.phase=='all':complete=complete and len(results)==len(PHASES) and results['evaluate']['status']=='SCIENCE_COMPLETE' and results['report']['status']=='REPORT_COMPLETE'
    terminal=write(root/'execution'/(args.phase+'_terminal.json'),{'status':'COMPLETE' if complete else 'PARTIAL_DEPENDENCY_BLOCK',
        'phase':args.phase,'exit_code':0 if complete else 1,'failure':failure,'phase_statuses':{k:v['status'] for k,v in results.items()},
        'resume_requested':args.resume,'successful_leaves_content_verified':True,'invocation':str(invocation),
        'UTC':datetime.now(timezone.utc).isoformat(),'publication_is_separate':True})
    print('TERMINAL',terminal['status'],'exit',terminal['exit_code'],flush=True);return terminal['exit_code']


def main():
    parser=argparse.ArgumentParser(description='Fixed four-map SAM2/SAM-V probe; no dataset expansion')
    parser.add_argument('--spec',required=True);parser.add_argument('--parent-root',required=True);parser.add_argument('--output-root',required=True)
    parser.add_argument('--storage-root');parser.add_argument('--path-map');parser.add_argument('--gpu')
    parser.add_argument('--phase',choices=(*PHASES,'all'),default='all');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--samv-python');parser.add_argument('--sam2-python');parser.add_argument('--assets-root')
    return run(parser.parse_args())
