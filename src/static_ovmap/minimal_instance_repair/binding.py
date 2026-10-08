"""Actual parent/v2 lineage and separate prediction versus evaluation inputs."""

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, PathResolver, read

from .support_units import build_units


REPO = Path(__file__).resolve().parents[3]
BASELINES = {'IR00_D2':'EV00_D2', 'IR01_G1':'EV01_G1_V2'}


def seal(value):
    return {**value, 'identity':canonical_digest({k:v for k,v in value.items() if k!='identity'})}


def validate_baseline(payload, row, receipt):
    labels = owner_labels(payload)
    scoring = canonical_digest({'context':receipt['context'], 'labels':labels,
                                'view':receipt['view'], 'rank_mode':receipt['rank_mode']})
    if (not payload.locked or row['prediction_identity'] != payload.prediction_key
            or row['record_identity'] != payload.record_key
            or row['evaluation_identity'] != receipt['identity'] or scoring != receipt['identity']
            or any(row['metrics'][m] != receipt['metrics'][m] for m in ('apall','ap50','ap25','miou','macc'))
            or row['status'] != 'COMPLETE'
            or receipt['status'] != 'COMPLETE' or receipt['rank_mode'] != 'OFFICIAL_CURRENT_CLASS'
            or receipt['context']['native_geometry'] != payload.geometry.to_dict()):
        raise ValueError('baseline payload or released scoring identity differs')
    for owner, item in receipt['view'].items():
        if labels[int(owner)] != item['label']:
            raise ValueError('baseline scorer class differs from the actual payload')
    return {'prediction_key':payload.prediction_key,'record_key':payload.record_key,
            'evaluation_identity':scoring,'geometry':payload.geometry.to_dict()}


def bind(spec_path, parent_root, output_root, *, storage_root=None, gpu=None,
         path_map=None, parent_reference=None):
    spec_path, parent_root = Path(spec_path).resolve(), Path(parent_root).resolve()
    logical = Path(output_root).absolute()
    root = Path(storage_root).resolve() if storage_root else logical.resolve()
    if root == parent_root or root.is_relative_to(parent_root):
        raise ValueError('new output cannot modify the immutable parent')
    if storage_root and logical.resolve() != root:
        logical.parent.mkdir(parents=True,exist_ok=True)
        if logical.exists() or logical.is_symlink():
            raise ValueError('explicit storage root differs from an existing output mapping')
        logical.symlink_to(root,target_is_directory=True)
    root.mkdir(parents=True,exist_ok=True)
    resolver = PathResolver(path_map or {})
    index = ConsumptionIndex(root/'input_verifications.json')
    def doc(path, expected=None, *, verified=True):
        path = Path(resolver.resolve(path))
        index.identity(path,expected)
        value = read(path)
        if verified:
            _verified_identity(value)
        return value
    spec = doc(spec_path,verified=False)
    authoritative = REPO/'docs/paper/static_ovmap/minimal_instance_repair_v1/spec/PROTOCOL_SPEC.json'
    if spec_path.read_bytes() != authoritative.read_bytes():
        raise ValueError('numeric protocol must be copied verbatim from the supplied package')
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch != spec['branch']:
        raise ValueError('new worktree branch differs from the specification')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    publication = doc(parent_root/'publication/final.json')
    if (publication['status'] != 'PUSH_VERIFIED' or publication['local_sha'] != spec['base_commit']
            or publication['remote_sha'] != spec['base_commit']):
        raise ValueError('parent publication does not attest the prescribed base')
    source_path = parent_root/'source_binding.json'
    parent = doc(source_path)
    store = doc(parent_root/'result_store.json')
    if (parent['cohorts'] != spec['cohorts'] or store['scene_method_coverage'] != 234
            or store['full_cohort_pool_coverage'] != 18 or store['status'] != 'SCIENCE_COMPLETE'):
        raise ValueError('parent does not contain the complete prescribed exposed-cohort study')
    reference_path = Path(parent_reference or parent['parent_reference']['path'])
    reference = doc(reference_path,parent['parent_reference'])
    if reference['identity'] != parent['parent_reference_identity']:
        raise ValueError('actual v2 runtime reference differs from the published parent')
    experiment = doc(Path(reference['v2_results_root'])/'experiment.json')
    if experiment['identity'] != reference['v2_experiment_identity']:
        raise ValueError('actual v2 lineage changed')
    scenes, baseline_rows = {}, {}
    for cohort, names in spec['cohorts'].items():
        for scene in names:
            inherited = parent['scenes'][scene]
            context = doc(inherited['context']['path'],inherited['context'])
            capture = doc(context['capture_manifest'])
            if capture['artifact_type'] != 'OVIMAP_NATIVE_CAPTURE' or capture['scene_id'] != scene:
                raise ValueError('new observer requires the actual same-scene native capture')
            ids = [int(f['frame_id']) for f in capture['frames']]
            if (ids != sorted(set(ids)) or ids != capture['completed_frame_ids']
                    or not set(ids) <= set(capture['scheduled_frame_ids'])):
                raise ValueError('original completed camera schedule changed')
            registry = doc(inherited['registry'])
            views = doc(inherited['view_manifest'])
            g1_source = doc(inherited['G1_source']['path'],inherited['G1_source'])
            regions = doc(inherited['region_receipt'])
            prediction_lock = doc(Path(reference['v2_results_root'])/'predictions'/scene/'receipt.json')
            if (regions['identity'] != inherited['region_identity']
                    or prediction_lock['regions_identity'] != regions['identity']
                    or prediction_lock['experiment_identity'] != experiment['identity']
                    or g1_source['registry_identity'] != registry['identity']
                    or g1_source['request_manifest_identity'] != views['identity']
                    or g1_source['model_identity'] != context['FC_physical_model_identity']):
                raise ValueError('actual v2 support/view/FC/prediction lineage changed')
            records = {}
            for name, old_name in BASELINES.items():
                old = inherited['baseline_rows'][old_name]
                row = doc(old['metric']['receipt_path'])
                scorer = doc(row['evaluation_receipt'],verified=False)
                path = Path(resolver.resolve(old['prediction_manifest']))
                manifest = doc(path,verified=False)
                index.identity(path.parent/manifest['arrays']['path'],manifest['arrays'])
                payload = load_prediction(path)
                checked = validate_baseline(payload,row,scorer)
                if row['identity'] != old['metric']['receipt_identity']:
                    raise ValueError('baseline scene receipt differs from imported metric lineage')
                for field in ('evaluator','exporter','projection','projection_arrays','annotation','gt'):
                    file = scorer['context'][field]
                    index.identity(resolver.resolve(file['path']),file)
                index.identity(scorer['manifest'])
                records[name] = {'prediction_manifest':str(path), **checked}
                baseline_rows.setdefault(name,{})[scene] = {**row,'method':name,
                    'source_method':old_name,'source_row_identity':row['identity'],
                    'reuse_kind':'EXACT_PARENT_PAYLOAD_AND_RELEASED_SCORING'}
                del payload
            scenes[scene] = {'cohort':cohort,'context':inherited['context'],
                'capture_manifest':str(Path(context['capture_manifest'])),
                'capture_identity':capture['identity'],'registry':inherited['registry'],
                'registry_identity':registry['identity'],'G1_source':inherited['G1_source'],
                'view_manifest':inherited['view_manifest'],'view_identity':views['identity'],
                'predictions':records}
            print('BOUND_SCENE',scene,flush=True)
    pools = {}
    for cohort, names in spec['cohorts'].items():
        pools[cohort] = {}
        for name, old_name in BASELINES.items():
            imported = parent['baseline_pools'][cohort][old_name]
            pool = doc(imported['receipt_path'])
            if (pool['identity'] != imported['receipt_identity'] or pool['metrics'] != imported['metrics']
                    or pool['scene_order'] != names or pool['status'] != 'COMPLETE'
                    or pool['aggregation'] != 'RELEASED_DATASET_POOL'):
                raise ValueError('full ordered baseline pool differs')
            pools[cohort][name] = {**pool,'method':name,'source_method':old_name,
                                  'source_receipt':imported['receipt_path']}
    provenance = REPO/'src/static_ovmap/module_validation/native_capture.py'
    caller = Path('/home/ww/crove/ovimap-backbone-wave1-upstream/scripts/panoptic_mapping_.py')
    caller_text = caller.read_text()
    before = caller_text.index('study_capture.before_insertion(')
    insert = caller_text.index('for segment in segment_list:',before)
    if 'panoptic_raster=inst_seg' not in caller_text[before:insert]:
        raise ValueError('capture caller does not pass the pre-insertion 2D raster')
    result = seal({'status':'BOUND_ACTUAL_PARENT_V2','protocol':'OVIMAP_MINIMAL_INSTANCE_REPAIR_V1',
        'spec':index.identity(spec_path),'specification':spec,'cohorts':spec['cohorts'],
        'parent_root':str(parent_root),'parent_source_binding':index.identity(source_path),
        'parent_identity':parent['identity'],'parent_store_identity':store['identity'],
        'reference':reference,'parent_reference':index.identity(reference_path),
        'gpu':str(gpu if gpu is not None else parent['gpu']),'FC_python':parent['FC_python'],
        'logical_root':str(logical),'output_root':str(root),'path_map':path_map or {},
        'scenes':scenes,'baseline_rows':baseline_rows,'baseline_pools':pools,
        'capture_producer':index.identity(provenance),'capture_caller':index.identity(caller),
        'capture_boundary':'PRE_INSERTION_2D_FRONTEND_RASTER',
        'baseline_scene_rows':52,'baseline_full_pools':4,'new_maps':0,
        'deployment':'N0_UNCHANGED','GT_used_by_predictor':False})
    path = root/'source_binding.json'
    if path.exists():
        previous = read(path)
        _verified_identity(previous)
        if previous != result:
            raise ValueError('completed input binding changed')
    else:
        atomic_write_json(path,result)
    index.write_memo(root/'input_verifications.json')
    print('BOUND',result['identity'],'52 baseline rows / 4 pools',flush=True)
    return result


def load_binding(root):
    root = Path(root).resolve()
    binding = read(root/'source_binding.json')
    _verified_identity(binding)
    index = ConsumptionIndex(root/'input_verifications.json')
    for field in ('spec','parent_source_binding','parent_reference'):
        item = binding[field]
        index.identity(item['path'],item)
    return binding


@dataclass
class RepairScene:
    scene: str
    data: dict
    capture: dict
    capture_path: Path
    xyz: np.ndarray
    faces: np.ndarray
    raw: np.ndarray
    d2: object
    g1: object
    sources: dict
    probabilities: dict
    valid_ids: list
    nearest: np.ndarray
    matched: np.ndarray
    units: object
    index: object


def load_scene(binding, scene):
    row = binding['scenes'][scene]
    root = Path(binding['output_root'])
    index = ConsumptionIndex(root/'inputs'/scene/'verifications.json')
    index.identity(row['context']['path'],row['context'])
    data = read(row['context']['path'])
    _verified_identity(data)
    capture_path = Path(data['capture_manifest'])
    index.identity(capture_path)
    capture = read(capture_path)
    _verified_identity(capture)
    if capture['identity'] != row['capture_identity']:
        raise ValueError('same-scene capture identity changed')
    surface = capture_path.parent/capture['surface']['path']
    index.identity(surface,capture['surface'])
    with np.load(surface,allow_pickle=False) as arrays:
        xyz, faces, raw = arrays['surface_xyz'],arrays['surface_faces'],arrays['original_owner']
    d2, g1 = [load_prediction(row['predictions'][name]['prediction_manifest']) for name in BASELINES]
    if (d2.geometry != g1.geometry or _array_digest(xyz) != d2.geometry.xyz_sha256
            or _array_digest(faces) != d2.geometry.faces_sha256
            or capture['tsdf']['sha256'] != d2.geometry.tsdf_sha256):
        raise ValueError('actual surface arrays differ from the unchanged parent geometry')
    active = d2.owner_ids > 0
    if not np.array_equal(g1.owner_ids[active],d2.owner_ids[active]) or not np.array_equal(g1.semantic_labels[active],d2.semantic_labels[active]):
        raise ValueError('parent G1 changed original D2 incumbents')
    ids = list(map(int,data['models']['native']['valid_ids']))
    sources = {}
    for name,item in data['sources'].items():
        index.identity(item['path'])
        value = read(item['path'])
        _verified_identity(value)
        if value['identity'] != item['identity'] or value['valid_ids'] != ids:
            raise ValueError('frozen N/Q/F or text category order changed')
        sources[name] = value
    labels, probabilities = fuse_readout(sources,data['temperatures'],'D2',ids,owner_labels(d2))
    if labels != owner_labels(d2):
        raise ValueError('unchanged D2 fusion does not reproduce the bound incumbent classes')
    index.identity(row['registry'])
    units = build_units(xyz,faces,raw,d2.owner_ids,d2.geometry,binding['specification']['support'],
                        inherited_registry=read(row['registry']))
    resolver = PathResolver({**binding['reference'].get('path_map',{}),**binding['path_map']})
    index.identity(data['config'])
    config = resolver.rewrite(read(data['config']))
    projection_path = Path(config['scenes'][scene]['projection'])
    index.identity(projection_path)
    projection = read(projection_path)
    if projection['identity'] != d2.geometry.projection_identity:
        raise ValueError('original target projection changed')
    path = projection_path.with_name('projection.npz')
    index.identity(path,{'sha256':projection['sha256']})
    with np.load(path,allow_pickle=False) as arrays:
        nearest,matched = arrays['nearest'],arrays['matched']
    index.write_memo(root/'inputs'/scene/'verifications.json')
    return RepairScene(scene,data,capture,capture_path,xyz,faces,raw,d2,g1,sources,
                       probabilities,ids,nearest,matched,units,index)
