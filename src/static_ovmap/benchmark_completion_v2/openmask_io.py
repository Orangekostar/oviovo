"""Replica camera/PLY/vocabulary I/O around pinned original OpenMask3D code."""
import ast
import hashlib
import inspect
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import ConsumedInputs, StageFailure, digest_file, source_identity, verified, write
from .external import PINS


def checkout(binding):
    assets = binding['asset_config'].get('external', {}).get('OpenMask3D', {})
    repo = Path(assets.get('checkout') or Path(binding['output_root']) / 'assets/external/OpenMask3D')
    if not (repo / '.git').exists():
        raise StageFailure('Pinned OpenMask3D checkout is missing', missing=[str(repo)])
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip() != PINS['OpenMask3D'][1]:
        raise ValueError('OpenMask3D checkout differs from the frozen source pin')
    return repo, assets


def source(binding, relative):
    repo, _ = checkout(binding)
    path = repo / relative
    original = subprocess.check_output(['git', 'show', PINS['OpenMask3D'][1] + ':' + relative], cwd=repo)
    if digest_file(path) != hashlib.sha256(original).hexdigest():
        raise ValueError('Original OpenMask3D source changed: ' + relative)
    return path


def ply_options(binding, scene):
    root = Path(binding['output_root'])
    config = binding['asset_config'].get('datasets', {}).get('replica48', {})
    options = []
    if config.get('evaluation_mesh_root'):
        options.append(Path(config['evaluation_mesh_root']) / (scene + '.ply'))
    if config.get('preprocessed_root'):
        options.append(Path(config['preprocessed_root']) / 'replica_plys' / (scene + '.ply'))
    options.append(root / 'assets/datasets/replica48/replica_plys' / (scene + '.ply'))
    return options


def geometry(binding, scene):
    """Keep the author's real RGB PLY; never synthesize missing proposal colors."""
    path = next((p for p in ply_options(binding, scene) if p.is_file()), None)
    if path is None:
        raise StageFailure('OpenMask3D needs the original colored evaluation PLY',
                           missing=list(map(str, ply_options(binding, scene))))
    import open3d as o3d
    points = o3d.io.read_point_cloud(str(path))
    xyz, colors = np.asarray(points.points), np.asarray(points.colors)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError('Original Replica evaluation PLY coordinates are malformed')
    if colors.shape != xyz.shape or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
        raise StageFailure('OpenMask3D proposal input is missing real RGB colors', missing=[str(path)])
    root = Path(binding['output_root'])
    index = ConsumedInputs(root)
    original = index.identity(path)
    record = write(root / 'external/OpenMask3D/geometry/replica48' / (scene + '.json'), dict(
        status='DONE', scene=scene, geometry=original, point_count=len(xyz),
        point_order_rule='UNCHANGED_ORIGINAL_PLY_VERTEX_ORDER',
        xyz_FP32_sha256=hashlib.sha256(xyz.astype('<f4').tobytes()).hexdigest(),
        RGB_reader='ORIGINAL_OPEN3D_PLY_READER', color_values_synthesized=False,
        geometry_rewritten=False, GT_labels_read=False, formal_G3_verified=False,
        producer=source_identity(('openmask_io',))))
    index.save()
    return record


def camera(binding, scene):
    from .replica_layout import prepare
    import imageio
    layout = prepare(binding, scene)
    path = source(binding, 'openmask3d/data/load.py')
    names = {'Camera', 'get_number_of_images'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError('Original OpenMask3D camera boundary changed')
    namespace = dict(np=np, os=os, math=math, imageio=imageio)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    folder = Path(layout['layout_root'])
    value = namespace['Camera'](intrinsic_path=str(folder / 'intrinsic/intrinsic_color.txt'),
        intrinsic_resolution=[layout['camera']['h'], layout['camera']['w']],
        poses_path=str(folder / 'pose'), depths_path=str(folder / 'depth'),
        extension_depth='.png', depth_scale=layout['camera']['scale'])
    count = namespace['get_number_of_images'](str(folder / 'pose'))
    if count != layout['original_frame_count']:
        raise ValueError('OpenMask3D camera must retain every original Replica pose')
    return value, np.arange(0, count, step=10).tolist()


def _query_class(binding, *, with_clip=False):
    import torch
    path = source(binding, 'openmask3d/evaluation/run_eval_close_vocab_inst_seg.py')
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'InstSegEvaluator')
    namespace = dict(np=np, torch=torch)
    if with_clip:
        import clip
        namespace['clip'] = clip
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['InstSegEvaluator']


def vocabulary(binding):
    scope = verified(Path(binding['output_root']) / 'protocol_lock.json')['scopes']['replica48']
    names, ids = scope['text_names'], scope['text_ids']
    if ids != list(range(1, 49)) or len(names) != 48 or len(set(names)) != 48:
        raise ValueError('OpenMask3D Replica needs the locked48 names and original semantic IDs')
    return names, ids


def text_embeddings(binding, model, device):
    """Use the author's exact prompt loop and text normalization."""
    names, _ = vocabulary(binding)
    cls = _query_class(binding, with_clip=True)
    value = cls.__new__(cls)
    value.device, value.clip_model, value.feature_size = device, model, 768
    value.query_sentences = ['a {} in a scene'.format(name) for name in names]
    return value.get_text_query_embeddings().numpy()


def query(binding, *, masks_path, features_path, text_path):
    """Call the unchanged author constant-score query, with only a48-ID mapper."""
    import torch
    _, ids = vocabulary(binding)
    cls = _query_class(binding)
    value = cls.__new__(cls)
    text = np.load(text_path, allow_pickle=False)
    features = np.load(features_path, allow_pickle=False)
    kwargs = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = False
    masks = torch.load(masks_path, **kwargs)
    if (text.shape != (48, 768) or not np.isfinite(text).all() or masks.ndim != 2
            or features.shape != (masks.shape[1], 768)):
        raise ValueError('Original OpenMask3D masks/features/text axes differ')
    value.text_query_embeddings = text
    value.label_mapper = np.vectorize(dict(enumerate(ids)).get)
    # Keep original normalization, invalid-normalized-feature zeroing, tie order,
    # mask columns and constant1 confidence, including empty feature/mask slots.
    with np.errstate(divide='ignore', invalid='ignore'):
        pred_masks, labels, scores = value.compute_classes_per_mask_diff_scores(
            str(masks_path), str(features_path), keep_first=None)
    matrix = np.asarray(pred_masks.cpu() if hasattr(pred_masks, 'cpu') else pred_masks)
    if not np.isin(matrix, [0, 1]).all() or not np.isin(labels, ids).all() or not np.isfinite(scores).all():
        raise ValueError('Original OpenMask3D query output is incompatible with the frozen axes')
    return dict(pred_masks=matrix, pred_classes=labels, pred_scores=scores)
