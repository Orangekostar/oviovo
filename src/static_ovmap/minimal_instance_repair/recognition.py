"""One inherited GPU worker, prescribed pilots and implementation freeze."""

import json
import os
from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .binding import REPO, seal
from .recognition_plan import prepare_scene


def validate_assets(binding):
    root = Path(binding['output_root'])
    index = ConsumptionIndex(root/'assets_verifications.json')
    parent_path = Path(binding['parent_root'])/'assets.json'
    index.identity(parent_path)
    parent = read(parent_path)
    _verified_identity(parent)
    spec = binding['specification']['anyup']
    checkout = Path(parent['checkout'])
    revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if revision!=spec['commit'] or subprocess.check_output(['git','diff','HEAD','--'],cwd=checkout):
        raise ValueError('AnyUp checkout does not reproduce the inherited pinned source')
    checkpoint = index.identity(parent['checkpoint']['path'],{'sha256':spec['sha256'],'bytes':spec['bytes']})
    for item in parent['sources'].values():
        index.identity(item['path'],item)
    if (parent['status']!='ASSETS_AVAILABLE_STRICT_LOAD_VERIFIED' or not parent['load']['strict']
            or parent['load']['python']!=binding['FC_python'] or checkpoint!=parent['checkpoint']):
        raise ValueError('inherited strict load does not bind the actual worker/assets')
    from static_ovmap.runtime_parity.runner import execution_config
    config = execution_config(binding['reference'])
    source_binding = read(binding['reference']['parent_binding_file']['path'])
    caches = source_binding['parent_fc_cache_roots']
    own_parent = Path(source_binding['output_root'])/'content_cache/fc'
    if own_parent.exists():
        caches = [str(own_parent),*caches]
    result = seal({'status':'ASSETS_AVAILABLE_STRICT_LOAD_VERIFIED','checkpoint':checkpoint,
        'checkout':str(checkout),'sources':parent['sources'],'commit':revision,
        'strict_load_reuse_identity':parent['identity'],'strict_load_receipt':str(parent_path),
        'FC_python':binding['FC_python'],'execution_config':config,'read_only_dense_cache_roots':caches,
        'environment_modified':False,'weights_published':False,'license':spec['license']})
    target = root/'assets.json'
    if target.exists() and read(target)!=result:
        raise ValueError('new-task bound assets changed')
    atomic_write_json(target,result)
    index.write_memo(root/'assets_verifications.json')
    return result


def require_freeze(binding):
    path = Path(binding['output_root'])/'implementation_freeze.json'
    freeze = read(path)
    _verified_identity(freeze)
    if freeze['status']!='IMPLEMENTATION_FROZEN' or freeze['spec_identity']!=binding['spec']:
        raise ValueError('main science requires the complete frozen implementation and numeric protocol')
    index = ConsumptionIndex()
    for item in freeze['files']:
        index.identity(item['path'],item)
    subprocess.run(['git','merge-base','--is-ancestor',freeze['commit'],'HEAD'],cwd=REPO,check=True)
    return freeze


def freeze(binding):
    root = Path(binding['output_root'])
    if (root/'implementation_freeze.json').exists():
        return require_freeze(binding)
    pilot = read(root/'pilots/summary.json')
    if pilot['status']!='INTEGRATED_PILOTS_VERIFIED' or list(pilot['scenes'])!=binding['specification']['pilots']:
        raise ValueError('freeze requires both prescribed actual pilots')
    required = ['binding','observations','support_units','proposals','verification','recognition',
                'reread','outputs','evaluation','diagnostics','selection','timing','reporting']
    # Import every delivered phase before committing; absent responsibilities
    # cannot be hidden by a passing narrow pilot.
    import importlib
    for module in required:
        importlib.import_module('static_ovmap.minimal_instance_repair.'+module)
    env = dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    validation = subprocess.run([binding['specification']['controller_python'],'-m','pytest',
        'tests/static_ovmap/test_minimal_instance_repair.py','-q','--tb=short'],
        cwd=REPO,env=env,capture_output=True,text=True)
    atomic_write_json(root/'tests/freeze_validation.json',seal({'returncode':validation.returncode,
        'stdout':validation.stdout,'stderr':validation.stderr,'suite':'NEW_PRODUCTION_BOUNDARIES_ONLY'}))
    if validation.returncode:
        raise RuntimeError('new-boundary validation failed before implementation freeze')
    files = [*sorted((REPO/'src/static_ovmap/minimal_instance_repair').glob('*.py')),
        REPO/'scripts/evaluation/run_ovimap_minimal_instance_repair.py',
        REPO/'tests/static_ovmap/test_minimal_instance_repair.py',
        REPO/'configs/static_ovmap/minimal_instance_repair_v1.json',
        *sorted((REPO/'docs/paper/static_ovmap/minimal_instance_repair_v1/spec').rglob('*'))]
    files = [p for p in files if p.is_file()]
    subprocess.run(['git','add','--',*[str(p.relative_to(REPO)) for p in files],
                    'docs/superpowers/plans/2026-10-08-minimal-instance-repair.md'],cwd=REPO,check=True)
    subprocess.run(['git','commit','-m','Implement frozen minimal instance repair and incumbent rereading study'],cwd=REPO,check=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    index = ConsumptionIndex()
    result = seal({'status':'IMPLEMENTATION_FROZEN','commit':commit,'spec_identity':binding['spec'],
        'files':[index.identity(p) for p in files],'pilots_identity':pilot['identity'],
        'thresholds_changed_from_supplied_spec':False,'main_recognition_started':False})
    atomic_write_json(root/'implementation_freeze.json',result)
    return result


def run_recognition_phase(binding, phase, *, resume):
    root = Path(binding['output_root'])
    diagnostic = read(root/'diagnostics/summary.json')
    if diagnostic['status']!='COMPLETE' or diagnostic['scene_count']!=26:
        raise ValueError('all 26 fixed-library opportunity diagnoses must precede model work')
    if phase=='assets':
        return validate_assets(binding)
    if phase=='recognize':
        require_freeze(binding)
    scenes = binding['specification']['pilots'] if phase=='pilot' else [s for names in binding['cohorts'].values() for s in names]
    for scene in scenes:
        prepare_scene(binding,scene)
    env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(binding['gpu']),
               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    command = [binding['FC_python'],'-m','static_ovmap.minimal_instance_repair.recognition_pipeline',
               '--root',str(root),'--phase',phase]
    log = root/(phase+'_execution.log')
    with log.open('a') as stream:
        completed = subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError('real recognition worker failed; inspect '+str(log))
    if phase=='pilot':
        from .outputs import predict_study
        from .evaluation import evaluate_scene
        predictions = predict_study(binding,scenes=scenes)
        scored = [evaluate_scene((binding,row['scene']),require_frozen=False) for row in predictions]
        parity = read(root/'pilots/ordinary_anyup_parity.json')
        acquired = read(root/'recognition/pilot_acquisition.json')
        result = seal({'status':'INTEGRATED_PILOTS_VERIFIED','scenes':{r['scene']:r['identity'] for r in scored},
            'recognition':acquired['scenes'],'ordinary_parity':parity['identity'],
            'real_partitions_scored':sum(len(r['rows']) for r in scored),
            'successful_pilot_science_leaves_reusable':True,
            'evaluation_python':os.sys.executable,'model_python':binding['FC_python']})
        atomic_write_json(root/'pilots/summary.json',result)
    return read(root/('pilots/summary.json' if phase=='pilot' else 'recognition/summary.json'))
