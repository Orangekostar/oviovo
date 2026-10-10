"""Task-owned full-parent binding; old branch/count checks remain unchanged."""
import os
from pathlib import Path
import shutil
import subprocess

from static_ovmap.source_preserving_update.binding import load_scene as parent_scene
from static_ovmap.disagreement_query.binding import predictor_namespace
from .common import BASELINES, REPO, ConsumptionIndex, PathResolver, fixed_spec, read, verified, write


def bind(spec_path, parent_root, observation_root, output_root, *, storage_root=None,
         path_map=None, gpu=None, data_roots=()):
    spec = fixed_spec(spec_path)
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    if branch != spec['branch']:
        raise ValueError('Incorrect task branch')
    subprocess.run(['git', 'merge-base', '--is-ancestor', spec['base_commit'], 'HEAD'], cwd=REPO, check=True)
    remote = subprocess.check_output(['git', 'remote', 'get-url', 'origin'], cwd=REPO, text=True).strip()
    if remote not in ('git@github.com:Orangekostar/oviovo.git', 'https://github.com/Orangekostar/oviovo.git'):
        raise ValueError('Incorrect publication repository')
    mapping = read(path_map) if isinstance(path_map, (str, Path)) else (path_map or {})
    resolver = PathResolver(mapping)
    parent = Path(resolver.resolve(parent_root)).resolve()
    observations = Path(resolver.resolve(observation_root)).resolve()
    logical = Path(output_root).absolute()
    root = Path(storage_root).resolve() if storage_root else logical.resolve()
    if root == parent or root.is_relative_to(parent) or root == observations or root.is_relative_to(observations):
        raise ValueError('Parent and observation assets are read-only')
    root.mkdir(parents=True, exist_ok=True)
    if storage_root and logical.resolve() != root:
        if logical.exists() or logical.is_symlink():
            raise ValueError('Existing output relocation differs')
        logical.parent.mkdir(parents=True, exist_ok=True)
        logical.symlink_to(root, target_is_directory=True)
    index = ConsumptionIndex(root / 'input_verifications.json')
    source, store, publication = [verified(parent / name) for name in
                                 ('source_binding.json', 'result_store.json', 'publication/final.json')]
    for name in ('source_binding.json', 'result_store.json', 'publication/final.json'):
        index.identity(parent / name)
    if (publication['status'] != 'PUSH_VERIFIED' or publication['local_HEAD'] != publication['remote_HEAD']
            or publication['full_resume_exit_code'] != 0 or publication['result_store_identity'] != store['identity']):
        raise ValueError('Full source-update publication not verified')
    if store['status'] != 'SCIENCE_COMPLETE' or store['scene_method_coverage'] != 234 or store['full_cohort_pool_coverage'] != 18:
        raise ValueError('Full 26-scene source-update parent is incomplete')
    inherited = resolver.rewrite(source)
    rows = {method: {} for method in BASELINES.values()}
    scenes, discovered = {}, []
    for cohort, names in spec['cohorts'].items():
        if inherited['cohorts'][cohort] != names:
            raise ValueError('Ordered complete cohort differs from parent')
        for scene in names:
            lock_path = parent / 'predictions' / scene / 'receipt.json'
            index.identity(lock_path)
            lock = verified(lock_path)
            if lock['status'] != 'PREDICTIONS_LOCKED':
                raise ValueError('Parent output not locked')
            scenes[scene] = dict(inherited['scenes'][scene], cohort=cohort)
            item = scenes[scene]['context']
            index.identity(item['path'], item)
            context = resolver.rewrite(verified(item['path']))
            if context.get('runtime', {}).get('data_root'):
                discovered.append(context['runtime']['data_root'])
            for method in rows:
                row = next(r for r in store['scene_metrics'] if r['scene'] == scene and r['method'] == method)
                if row['status'] != 'COMPLETE' or row['prediction_identity'] != lock['methods'][method]['prediction_key']:
                    raise ValueError('Actual parent baseline score/output differs')
                rows[method][scene] = resolver.rewrite(row)
    # These immediate dataset roots are declared by the inherited acquisition code.
    prior_root = Path('/mnt/shared/ww/ovimap-module-validation-v1/data/scannet')
    authority = REPO / 'artifacts/static_ovmap/module_validation_v1/scannet_execution_20260922.json'
    index.identity(authority)
    authorization = read(authority)
    if authorization.get('authorization') != 'USER_EXPLICITLY_CONFIRMED':
        raise PermissionError('Prior ScanNet authority is not verified')
    roots = list(map(str, data_roots))
    roots += [os.environ[k] for k in ('SCANNET_ROOT', 'SCANNET_DATA_ROOT') if os.environ.get(k)]
    roots += discovered + [str(prior_root), '/mnt/shared/ww/datasets', '/home/ww/data']
    roots = list(dict.fromkeys(str(Path(resolver.resolve(p)).resolve()) for p in roots))
    if gpu is None and (root / 'source_binding.json').exists():
        gpu = verified(root / 'source_binding.json')['gpu']
    if gpu is None:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.used', '--format=csv,noheader,nounits'], text=True)
        devices = [(int(row.split(',')[2]), int(row.split(',')[0])) for row in result.splitlines() if 'A40' in row]
        if not devices:
            raise RuntimeError('BLOCKED_RUNTIME: no A40 available')
        gpu = min(devices)[1]
    value = {k: inherited[k] for k in ('reference', 'assets', 'FC_python')}
    value.update(status='BOUND_ACTUAL_PARENT', protocol=spec['task_id'], specification=spec,
                 spec=index.identity(spec_path), branch=branch, base_commit=spec['base_commit'],
                 path_map=mapping, parent_root=str(parent), observation_root=str(observations),
                 output_root=str(root), logical_root=str(logical), minimal_root=inherited['parent_root'],
                 scenes=scenes, cohorts=spec['cohorts'], baseline_rows=rows,
                 baseline_lineage={'LR00_D2': ['SU00_D2', 'IR00_D2'], 'LR01_G1': ['SU01_G1', 'IR01_G1']},
                 parent_identity=source['identity'], parent_store_identity=store['identity'],
                 parent_release_commit=publication['local_HEAD'], data_roots=roots, prior_data_root=str(prior_root),
                 scan_authority=index.identity(authority), gpu=str(gpu), deployment='N0_UNCHANGED',
                 new_maps=0, GT_used_by_predictor=False)
    destination = root / 'source_binding.json'
    if destination.exists():
        old = verified(destination)
        if {k: v for k, v in old.items() if k != 'identity'} != value:
            raise ValueError('Bound task inputs changed')
        binding = old
    else:
        free = shutil.disk_usage(root).free / 2**30
        if free < spec['resources']['free_space_min_GiB']:
            raise OSError('BLOCKED_RUNTIME: at least40 GiB initial free space required')
        write(root / 'storage_preflight.json', dict(free_GiB=free, minimum_GiB=40, root=str(root)))
        binding = write(destination, value)
    index.write_memo(root / 'input_verifications.json')
    write(root / 'dependency_manifest.json', dict(status='BOUND', source_binding=binding['identity'],
          FC_python=binding['FC_python'], FC_assets=binding['reference']['fc'],
          controller_python=spec['controller_python_hint'], data_roots=roots,
          parent_release_commit=binding['parent_release_commit'], data_authority=binding['scan_authority'],
          numerical_sources=str(root / 'external/upstream_sources/manifest.json'),
          official_mapping_reference=str(root / 'external/upstream_sources/additional_mapping_reference.json')))
    return binding


def load_scene(binding, scene):
    if scene not in binding['scenes']:
        raise ValueError('Outside the fixed full cohort')
    original = parent_scene(binding, scene)
    return predictor_namespace(original), original
