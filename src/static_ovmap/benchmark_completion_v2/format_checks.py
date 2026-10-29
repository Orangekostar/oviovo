"""First-formal-scene checks through actual unchanged author format writers."""
import ast
from pathlib import Path
import numpy as np

from .common import verified, write, ConsumedInputs, source_identity


def ovi_author_format(binding, benchmark, scene):
    from plyfile import PlyData, PlyElement
    from os.path import join
    from .external import decode_text_manifest
    from .standard_export import official_prediction
    root = Path(binding['output_root'])
    manifest = verified(root / 'external/OVI-MAP/predictions' / benchmark / scene / 'manifest.json')
    folder = root / 'review/G3_OVI_author_format' / benchmark / scene
    folder.mkdir(parents=True, exist_ok=True)
    text = folder / 'pred_inst_sem_mapping.txt'
    source = root / 'assets/external/OVI-MAP/scripts/eval_sem_seg.py'
    if not text.is_file():
        with np.load(manifest['arrays']['path'], allow_pickle=False) as arrays:
            owners = np.zeros(manifest['target_points'], np.uint16)
            labels = np.zeros_like(owners)
            for i, owner in enumerate(arrays['owners']):
                mask = arrays['indices'][arrays['indptr'][i]:arrays['indptr'][i + 1]]
                owners[mask] = owner
                labels[mask] = arrays['labels'][i]
        for name, data in (('instances.ply', owners), ('semantics.ply', labels)):
            container = np.empty(len(data), dtype=[('label', '<u2')])
            container['label'] = data
            PlyData([PlyElement.describe(container, 'vertex')], text=False).write(str(folder / name))
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'map_pred_mesh')
        namespace = dict(np=np, PlyData=PlyData, pjoin=join)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        actual = namespace['map_pred_mesh'](dict(res_folder=str(folder),
            inst_mesh_f=str(folder / 'instances.ply'), sem_mesh_f=str(folder / 'semantics.ply')))
        if Path(actual) != text:
            raise ValueError('Author native mask manifest path changed')
    decoded = decode_text_manifest(text, target_points=manifest['target_points'])
    active = np.isin(decoded['pred_classes'], manifest.get('valid_instance_ids',
        verified(root / 'protocol_lock.json')['scopes'][benchmark]['instance_ids']))
    decoded = {key: value[:, active] if key == 'pred_masks' else value[active] for key, value in decoded.items()}
    common = official_prediction(manifest)
    for key in ('pred_masks', 'pred_classes', 'pred_scores'):
        if not np.array_equal(decoded[key], common[key]):
            raise ValueError('Pinned native author format and common bridge disagree: ' + key)
    index = ConsumedInputs(root)
    result = write(folder / 'receipt.json', dict(status='DONE', benchmark=benchmark, scene=scene,
        synthetic=False, prediction=manifest['identity'], real_full_target_points=manifest['target_points'],
        source=index.identity(source), author_manifest=index.identity(text),
        actual_author_writer='UNCHANGED_map_pred_mesh_AST',
        author_masks_shape='(N,1); original saved dtype and vertex order',
        pred_masks_exact=True, pred_classes_exact=True, author_confidence_exact=True,
        author_classifier_checked_objects=manifest['actual_author_classifier_checked_objects'],
        model_GPU_calls_added=0, GT_labels_read=False,
        label_container='REAL_FROZEN_NATIVE_PREDICTIONS;NO_GROUND_TRUTH_LABELS',
        producer=source_identity(('format_checks', 'external'))))
    index.save()
    return result
