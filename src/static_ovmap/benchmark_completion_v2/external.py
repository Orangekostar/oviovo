"""Pinned full-method recipes and lossless author prediction-format adapters."""
from pathlib import Path
import subprocess
import numpy as np

from .common import (verified, read, write, save_arrays, digest_file,
                     canonical_digest, ConsumedInputs, StageFailure, source_identity)

PINS = {
    'OVI-MAP': ('https://github.com/OVI-MAP/OVI-MAP.git', '6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9'),
    'OpenMask3D': ('https://github.com/OpenMask3D/openmask3d.git', '3bc3fc52693b25668d0e91d55a2ea714544a4749'),
    'Open3DIS': ('https://github.com/VinAIResearch/Open3DIS.git', 'bc2a3b3ee7161f5754ad7a7fe77202e9a66adc8e'),
    'MaskClustering': ('https://github.com/PKU-EPIC/MaskClustering.git', 'eb2d41c2267cde21966f4340c37b8cf94b05c23c'),
}


def decode_open3dis(path, *, target_points, class_ids):
    """Native masks/classes and the pinned author's constant-1 evaluation score."""
    import torch
    data = torch.load(path, map_location='cpu', weights_only=False)
    masks = []
    for mask in data['ins']:
        if isinstance(mask, dict):
            # Author RLE is one-based start positions and run lengths.
            length = int(mask['length'])
            counts = mask['counts']
            values = np.asarray([int(x) for x in counts.split()] if isinstance(counts, str) else counts, np.int64)
            starts, runs = values[::2] - 1, values[1::2]
            decoded = np.zeros(length, np.uint8)
            for start, run in zip(starts, runs, strict=True):
                if start < 0 or start + run > length:
                    raise ValueError('Author RLE leaves its original point domain')
                decoded[start:start + run] = 1
            masks.append(decoded)
        else:
            masks.append(np.asarray(mask.cpu() if hasattr(mask, 'cpu') else mask).reshape(-1) != 0)
    matrix = np.stack(masks, axis=1) if masks else np.empty((target_points, 0), bool)
    indices = np.asarray(data['final_class'], dtype=np.int64)
    if matrix.shape != (target_points, len(indices)) or (indices < 0).any() or (indices >= len(class_ids)).any():
        raise ValueError('Author Open3DIS class/point ordering is incompatible')
    scores = np.asarray(data['conf'], dtype=np.float64)
    if scores.shape != indices.shape or not np.isfinite(scores).all():
        raise ValueError('Author Open3DIS confidence values are invalid')
    # eval.py ignores the archive conf field and assigns 1.0 to every proposal.
    # Validate the archived scores above; preserve their original bytes on disk.
    return dict(pred_masks=matrix, pred_classes=np.asarray(class_ids)[indices],
                pred_scores=np.ones_like(scores))


def decode_text_manifest(path, *, target_points):
    """OVI/MaskClustering official mask-path, semanticID, confidence triples."""
    path = Path(path)
    masks, labels, scores = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        mask_name, label, score = line.split()
        mask_path = path.parent / mask_name
        mask = np.load(mask_path, allow_pickle=False) if mask_path.suffix == '.npy' else np.loadtxt(mask_path, dtype=np.int64)
        # Pinned OVI map_pred_mesh writes each independent mask as (N,1).
        if mask.shape == (target_points, 1):
            mask = mask[:, 0]
        if mask.shape != (target_points,):
            raise ValueError('Author independent mask has a different original point domain')
        masks.append(mask != 0)
        labels.append(int(label))
        scores.append(float(score))
    matrix = np.stack(masks, axis=1) if masks else np.empty((target_points, 0), bool)
    if not np.isfinite(scores).all():
        raise ValueError('Author confidence contains a nonfinite value')
    return dict(pred_masks=matrix, pred_classes=np.asarray(labels, np.int64), pred_scores=np.asarray(scores))


def save_prediction(folder, prediction, *, method, benchmark, scene, provenance):
    masks, labels, scores = (np.asarray(prediction[key]) for key in ('pred_masks', 'pred_classes', 'pred_scores'))
    if (masks.ndim != 2 or labels.shape != (masks.shape[1],) or scores.shape != labels.shape
            or not np.isfinite(scores).all() or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('Complete-method official mask arrays have incompatible dimensions')
    folder = Path(folder)
    indices = [np.flatnonzero(masks[:, i]) for i in range(masks.shape[1])]
    indptr = np.r_[0, np.cumsum([len(row) for row in indices])].astype(np.int64)
    combined = np.concatenate(indices).astype(np.int64) if indices else np.empty(0, np.int64)
    identity = canonical_digest(dict(method=method, benchmark=benchmark, scene=scene,
        masks=__import__('hashlib').sha256(np.asarray(masks != 0).tobytes()).hexdigest(),
        labels=labels, scores=scores, provenance=provenance, producer=source_identity(('external',))))
    receipt = folder / 'manifest.json'
    if receipt.exists():
        old = verified(receipt)
        if old['input_identity'] != identity or digest_file(old['arrays']['path']) != old['arrays']['sha256']:
            raise ValueError('Completed author prediction export changed')
        return old
    array_path = folder / 'masks.npz'
    save_arrays(array_path, owners=np.arange(1, len(labels) + 1), indptr=indptr, indices=combined,
        labels=labels, confidence=scores, exported=np.ones(len(labels), bool))
    return write(receipt, dict(status='DONE', method=method, benchmark=benchmark, scene=scene,
        input_identity=identity, prediction_identity=identity, target_points=masks.shape[0],
        arrays=dict(path=str(array_path), bytes=array_path.stat().st_size, sha256=digest_file(array_path)),
        confidence='UNCHANGED_AUTHOR_VALUES', overlap_preserved=True, provenance=provenance))


def recipes(binding):
    """Concrete full pipelines, separate dependencies, no reduced-method fallback."""
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    assets = binding['asset_config'].get('external', {})
    result = {}
    contracts = {
        'OVI-MAP': dict(variant='AUTHOR_ORIGIN_IMPL_FULL_VOXEL_.01',
            commands=[['$PY_MAP', 'scripts/panoptic_mapping_.py', '--dataset', '$AUTHOR_DATASET', '--scene_num', '$SCENE',
                       '--data_folder', '$RGBD_ROOT', '--result_folder', '$SCENE_OUT', '--temp_panoptics_folder', '$CROPFORMER_ROOT'],
                      ['$PY_SEMANTIC', 'scripts/eval_inst_seg.py', '--help']],
            required=['python_mapping', 'python_semantic', 'native_extension', 'cropformer_checkpoint', 'siglip_model', 'raw_root'],
            invariant='origin_impl numerical mapping/readout; author frame policy; .01m voxel; no FC/MV/Q_GAIN substitution'),
        'OpenMask3D': dict(variant='FULL_VAL_MASK3D_SAM_VITH_CLIP_L14_336',
            commands=[['$PYTHON', 'openmask3d/class_agnostic_mask_computation/get_masks_scannet200.py',
                'general.checkpoint=$MASK3D_CHECKPOINT', 'model.num_queries=150', 'general.use_dbscan=true',
                'general.dbscan_eps=0.95', 'general.train_mode=false', 'data.test_dataset.data_dir=$PROCESSED_ROOT',
                'data.validation_dataset.data_dir=$PROCESSED_ROOT', 'data.train_dataset.data_dir=$PROCESSED_ROOT',
                'general.mask_save_dir=$MASK_DIR'],
                ['$PYTHON', 'openmask3d/compute_features_scannet200.py', 'data.scans_path=$RAW_ROOT',
                 'data.masks.masks_path=$MASK_DIR', 'external.sam_checkpoint=$SAM_CHECKPOINT',
                 'output.output_directory=$FEATURE_DIR', 'gpu.optimize_gpu_usage=false'],
                ['$PYTHON', 'openmask3d/evaluation/run_eval_close_vocab_inst_seg.py', '--gt_dir=$GT_ROOT',
                 '--mask_pred_dir=$MASK_DIR', '--mask_features_dir=$FEATURE_DIR']],
            required=['python', 'mask3d_checkpoint', 'sam_checkpoint', 'clip_checkpoint', 'processed_root', 'raw_root'],
            invariant='full-val segmented Mask3D proposals; SAM crops and CLIP; no single-scene demo replacement'),
        'Open3DIS': dict(variant='ISBNET_GROUNDED_SAM_REFINED_SECOND_3D_PASS',
            commands=[['$PYTHON', 'tools/' + name + '.py', '--config', '$LOCKED_CONFIG']
                      for name in ('grounding_2d', 'generate_3d_inst', 'refine_grounding_feat', 'generate_3d_inst')],
            required=['python', 'grounding_dino_checkpoint', 'sam_checkpoint', 'clip_checkpoint',
                      'isbnet_proposals', 'isbnet_deep_features', 'superpoints', 'raw_root'],
            invariant='Grounded-SAM, ISBNet proposals/deep features, refinement then second3D output; p2d,p3d,refined=True'),
        'MaskClustering': dict(variant='CROPFORMER_FULL_GRAPH_CLIP_H14_LAION2B',
            commands=[['$PYTHON', 'run.py', '--config', 'scannet']],
            required=['python', 'cropformer_python', 'cropformer_checkpoint', 'clip_h14_checkpoint', 'processed_root'],
            invariant='author step10; full CropFormer, mask graph clustering, CLIP-H14; original native confidence'),
    }
    for method, contract in contracts.items():
        repo = root / 'assets/external' / method
        if not (repo / '.git').exists():
            raise StageFailure('Pinned external repository is missing', missing=[str(repo)])
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        if actual != PINS[method][1]:
            raise ValueError('External full-method checkout changed: ' + method)
        supplied = assets.get(method, {})
        missing = [dict(asset=name, expected=supplied.get(name)) for name in contract['required']
                   if not supplied.get(name) or not Path(supplied[name]).exists()]
        entry_sources = [repo / 'README.md']
        if method == 'Open3DIS':
            entry_sources.extend([repo / 'docs/DATA.md', repo / 'docs/RUN.md', repo / 'configs/scannet200.yaml', repo / 'configs/replica.yaml'])
        elif method == 'OpenMask3D':
            entry_sources.append(repo / 'run_openmask3d_scannet200_eval.sh')
        elif method == 'MaskClustering':
            entry_sources.extend([repo / 'run.py', repo / 'configs/scannet.json'])
        result[method] = write(root / 'external' / method / 'recipe.json', dict(status='BLOCKED_DEPENDENCY' if missing else 'DONE',
            method=method, repository=PINS[method][0], commit=actual, **contract,
            sources=[dict(path=str(p), sha256=digest_file(p)) for p in entry_sources],
            missing=missing, assets=supplied,
            scene_jobs={benchmark: list(scope['scenes']) for benchmark, scope in protocol['scopes'].items()},
            expected_scene_jobs=sum(len(scope['scenes']) for scope in protocol['scopes'].values()),
            full_method_not_run=bool(missing), author_prediction_preferred=True,
            IO_only_Replica_port_required=method in ('OVI-MAP', 'OpenMask3D', 'MaskClustering'),
            reduced_pipeline_allowed=False, missing_method_does_not_block_others=True))
    return write(root / 'external/recipes.json', dict(status='BLOCKED_DEPENDENCY' if any(
        row['status'] != 'DONE' for row in result.values()) else 'DONE', methods=result,
        expected_full_method_scene_jobs=4 * sum(len(s['scenes']) for s in protocol['scopes'].values())))
