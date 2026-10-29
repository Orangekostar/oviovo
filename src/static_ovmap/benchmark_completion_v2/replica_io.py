"""Score-only author Replica tuple conversion; source point order is mandatory."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from .common import (ConsumedInputs, StageFailure, verified, write,
                     canonical_digest, digest_file, source_identity)


def _tuple_path(binding, scene):
    root=Path(binding['output_root'])
    config=binding['asset_config'].get('datasets',{}).get('replica48',{})
    supplied=config.get('gt_root') or config.get('preprocessed_root')
    options=[Path(supplied)/(scene+'.pth'),Path(supplied)/'replica_3d'/(scene+'.pth')] if supplied else []
    options.append(root/'assets/datasets/replica48/replica_3d'/(scene+'.pth'))
    found=next((p for p in options if p.is_file()),None)
    if found is None:
        raise StageFailure('Author Replica48 full four-field tuple is missing',missing=list(map(str,options)))
    return found


def prepare_geometry(binding, scene):
    """Export coordinate-only sidecar; no semantic/instance values are consulted."""
    config=binding['asset_config'].get('datasets',{}).get('replica48',{})
    if config.get('evaluation_mesh_root'):
        supplied=Path(config['evaluation_mesh_root'])/(scene+'.ply')
        if supplied.is_file():
            return supplied
    import torch
    from plyfile import PlyData, PlyElement
    root=Path(binding['output_root'])
    source=_tuple_path(binding,scene)
    index=ConsumedInputs(root)
    original=index.identity(source)
    value=torch.load(source,map_location='cpu',weights_only=False)
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise StageFailure('Replica geometry requires author four-field instance tuple; semantic-only OpenScene is insufficient',
                           missing=[str(source)])
    coords=np.asarray(value[0],dtype=np.float32)
    if coords.ndim!=2 or coords.shape[1]!=3 or not len(coords) or not np.isfinite(coords).all():
        raise ValueError('Original Replica coordinate sidecar is invalid')
    folder=root/'assets/datasets/replica48/geometry'
    path=folder/(scene+'.ply')
    receipt=folder/(scene+'.json')
    identity=canonical_digest(dict(original=original['sha256'],coords=__import__('hashlib').sha256(coords.tobytes()).hexdigest(),
                                   rule='ORIGINAL_ORDER_FP32_COORDINATES_ONLY_V1'))
    if receipt.is_file():
        old=verified(receipt)
        if old['input_identity']!=identity or digest_file(path)!=old['geometry']['sha256']:
            raise ValueError('Frozen Replica coordinate order or source tuple changed')
        return path
    if path.exists():
        raise ValueError('An unverified Replica geometry already occupies the author sidecar path')
    folder.mkdir(parents=True,exist_ok=True)
    vertices=np.empty(len(coords),dtype=[('x','<f4'),('y','<f4'),('z','<f4')])
    for i,key in enumerate(('x','y','z')):
        vertices[key]=coords[:,i]
    PlyData([PlyElement.describe(vertices,'vertex')],text=False,byte_order='<').write(str(path))
    write(receipt,dict(status='DONE',input_identity=identity,source=original,
        geometry=index.identity(path),target_points=len(coords),GT_label_values_read=False,
        original_vertex_order_preserved=True,producer=source_identity(('replica_io',))))
    index.save()
    return path


def encode_labels(semantic, instances, author_source):
    tree = ast.parse(Path(author_source).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScanNetEval')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'assign_instances_for_scan')
    statements = []
    for node in method.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'gt_instances' for n in node.targets):
            break
        statements.append(node)
    if not statements or not any(isinstance(n, ast.Name) and n.id == 'gts' for n in ast.walk(ast.Module(body=statements, type_ignores=[]))):
        raise ValueError('Pinned Replica author GT encoding boundary changed')
    namespace = dict(np=np, self=SimpleNamespace(dataset_name='replica', encode_value=1000),
                     gts_sem=np.asarray(semantic).copy(), gts_ins=np.asarray(instances).copy())
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(author_source), 'exec'), namespace)
    return np.asarray(namespace['gts'], np.int64)


def export_gt(binding, scene, bundle, folder, *, prediction_lock=None):
    """Call only after this scene's complete available predictions are frozen."""
    if prediction_lock is None:
        raise ValueError('A complete available prediction freeze is required before Replica GT access')
    freeze=verified(prediction_lock)
    if (freeze['scene']!=scene or freeze['benchmark']!='replica48'
            or not freeze['all_available_predictions_frozen_before_new_GT_scoring']
            or freeze['bundle']!=bundle['identity']):
        raise ValueError('Replica prediction freeze does not bind this original geometry')
    import torch
    root = Path(binding['output_root'])
    source = _tuple_path(binding,scene)
    protocol=verified(root/'protocol_lock.json')
    author = Path(protocol['official_sources']['evaluator_replica']['path'])
    index = ConsumedInputs(root)
    inputs = [index.identity(source), index.identity(author,protocol['official_sources']['evaluator_replica'])]
    value = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError('Author Replica tuple must contain original coords, colors, semantics, instances')
    coords, _, semantic, instances = [np.asarray(v) for v in value]
    with np.load(bundle['arrays']['geometry']['path'], allow_pickle=False) as arrays:
        xyz = arrays['target_xyz']
        if coords.shape != xyz.shape or not np.array_equal(coords.astype(np.float32), xyz):
            raise ValueError('Replica evaluation geometry does not preserve the author original point order')
    if (semantic.shape != (len(coords),) or instances.shape != semantic.shape
            or not np.issubdtype(semantic.dtype, np.integer) or not np.issubdtype(instances.dtype, np.integer)):
        raise ValueError('Author Replica label axes are malformed')
    values = encode_labels(semantic, instances, author)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (scene + '.txt')
    np.savetxt(path, values, fmt='%d')
    index.save()
    return path, values, inputs, 'UNCHANGED_AUTHOR_OPEN3DIS_REPLICA_GT_PREFIX;EXACT_FP32_COORDINATE_ORDER;FULL_TUPLE'


def prepare_ground_truth(binding, scene):
    """Create the normal common-scoring GT receipt after the actual scene freeze."""
    root=Path(binding['output_root'])
    freeze=root/'benchmark/predictions/replica48'/scene/'receipt.json'
    if not freeze.is_file():
        raise StageFailure('Replica scoring waits for an actual complete available prediction freeze',missing=[str(freeze)])
    bundle=verified(root/'benchmark/bundles/replica48'/scene/'bundle.json')
    folder=root/'benchmark/GT/replica48'/scene
    path=folder/'receipt.json'
    if path.is_file():
        from .standard_score import ground_truth
        return ground_truth(binding,'replica48',scene,prediction_lock=freeze)
    gt,values,inputs,definition=export_gt(binding,scene,bundle,folder,prediction_lock=freeze)
    index=ConsumedInputs(root)
    result=write(path,dict(status='DONE',benchmark='replica48',scene=scene,
        GT=index.identity(gt),inputs=inputs,target_points=len(values),prediction_freeze=verified(freeze)['identity'],
        geometry=bundle['evaluation_geometry'],full_GT_denominator=True,GT_filtered_by_prediction=False,
        definition=definition,converter_producer=source_identity(('replica_io',))))
    index.save()
    return result
