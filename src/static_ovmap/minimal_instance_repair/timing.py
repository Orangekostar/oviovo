"""Prescribed new same-boundary latency series, independent of old denominators."""

import os
from pathlib import Path
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .binding import REPO, seal


def make_plan(spec, selected):
    cfg = spec['timing']
    methods = [cfg['baseline']]
    if selected!=cfg['baseline']:
        methods.append(selected)
    for method in cfg['priority_after_winner']:
        if method not in methods and len(methods)<cfg['methods_max']:
            methods.append(method)
    if len(methods)!=cfg['methods_max'] or not set(methods)<={r['id'] for r in spec['methods']}:
        raise ValueError('timing selection left the fixed four-arm slate')
    scenes = spec['cohorts'][cfg['cohort']]
    calls = []
    for repeat in range(1,cfg['repeat_count']+1):
        for scene in scenes if repeat==1 else list(reversed(scenes)):
            for method in methods if repeat==1 else list(reversed(methods)):
                calls.append({'repeat':repeat,'scene':scene,'method':method})
    return {'methods':methods,'calls':calls,'call_count':len(calls),'selected':selected,
        'boundary':cfg['boundary'],'old_timing_denominator_used':False,
        'models_resident':'ONLY_MODELS_REQUIRED_BY_ARM','parent_recovery_cache_allowed':False,
        'observer_hypothesis_region_caches_allowed':False,'correctness_checked_after_timer':True}


def time_study(binding):
    from .recognition import require_freeze
    require_freeze(binding)
    root = Path(binding['output_root'])
    selection = read(root/'selection.json')
    store = read(root/'result_store.json')
    science_identity = store.get('science_identity',store['identity'])
    if store['status']!='SCIENCE_COMPLETE' or selection['result_store_identity']!=science_identity:
        raise ValueError('paired timing must follow actual full scientific evaluation and selection')
    plan = seal({**make_plan(binding['specification'],selection['selected']),
        'selection_identity':selection['identity'],'science_identity':science_identity,
        'binding_identity':binding['identity']})
    path = root/'timing/binding.json'
    if path.exists() and read(path)!=plan:
        raise ValueError('completed timing selection changed; preserve affected calls and invalidate descendants')
    atomic_write_json(path,plan)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(binding['gpu']),OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (root/'timing_execution.log').open('a') as stream:
        run = subprocess.run([binding['FC_python'],'-m','static_ovmap.minimal_instance_repair.timing_worker',
                              '--root',str(root)],cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError('cold timing failed; inspect '+str(root/'timing_execution.log'))
    return read(root/'timing/summary.json')
