"""Replica paths/camera/vocabulary port; original author reader methods retained."""
import ast
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from .common import ConsumedInputs, StageFailure, digest_file, verified
from .external import PINS


def _author_reader(repo):
    """Load only the exact pinned class, without importing other dataset drivers."""
    pin = PINS['MaskClustering'][1]
    if not (repo / '.git').exists():
        raise StageFailure('Pinned MaskClustering checkout is absent', missing=[str(repo)])
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    source = repo / 'dataset/scannet.py'
    expected = subprocess.check_output(['git', 'show', pin + ':dataset/scannet.py'], cwd=repo)
    if actual != pin or digest_file(source) != hashlib.sha256(expected).hexdigest():
        raise ValueError('MaskClustering original dataset reader differs from the pinned source')
    import cv2
    import open3d as o3d
    namespace = dict(np=np, os=os, cv2=cv2, o3d=o3d)
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ScanNetDataset')
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['ScanNetDataset']


def make_dataset(binding, scene, *, text_features_path):
    from .replica_layout import prepare
    from .replica_io import prepare_geometry
    root = Path(binding['output_root'])
    scope = verified(root / 'protocol_lock.json')['scopes']['replica48']
    ids, names = scope['text_ids'], scope['text_names']
    if ids != list(range(1, 49)) or len(names) != 48 or len(set(names)) != 48:
        raise ValueError('Replica dataset port requires the locked48-class vocabulary')
    assets = binding['asset_config'].get('external', {}).get('MaskClustering', {})
    repo = Path(assets.get('checkout') or root / 'assets/external/MaskClustering')
    original = _author_reader(repo)
    layout = prepare(binding, scene)
    geometry = prepare_geometry(binding, scene)
    index = ConsumedInputs(root)
    index.identity(geometry)
    index.save()

    class Replica48Dataset(original):
        def __init__(self):
            super().__init__(scene)
            self.root = layout['layout_root']
            self.rgb_dir = str(Path(self.root) / 'color')
            self.depth_dir = str(Path(self.root) / 'depth')
            self.segmentation_dir = str(Path(self.root) / 'output/mask')
            self.object_dict_dir = str(Path(self.root) / 'output/object')
            self.point_cloud_path = self.mesh_path = str(geometry)
            self.extrinsics_dir = str(Path(self.root) / 'pose')
            self.intrinsic_dir = str(Path(self.root) / 'intrinsic')
            self.depth_scale = layout['camera']['scale']
            self.image_size = (layout['camera']['w'], layout['camera']['h'])

        def get_label_features(self):
            path = Path(text_features_path)
            if not path.is_file():
                raise StageFailure('Original CLIP-H14 Replica text features are missing', missing=[str(path)])
            consumed = ConsumedInputs(root)
            consumed.identity(path)
            value = np.load(path, allow_pickle=True).item()
            if not isinstance(value, dict) or list(value) != names:
                raise ValueError('MaskClustering Replica text feature vocabulary/order differs')
            vectors = np.stack(list(value.values()))
            if vectors.shape != (48, 1024) or not np.isfinite(vectors).all():
                raise ValueError('MaskClustering original CLIP-H14 text feature dimensions differ')
            consumed.save()
            return value

        def get_label_id(self):
            self.class_id, self.class_label = ids, names
            self.label2id = dict(zip(names, ids, strict=True))
            self.id2label = dict(zip(ids, names, strict=True))
            return self.label2id, self.id2label

    return Replica48Dataset()
