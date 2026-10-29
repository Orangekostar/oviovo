"""Full OpenMask3D prediction freeze and independent official Replica scoring."""
from pathlib import Path

import numpy as np

from .common import ConsumedInputs, REPO, source_identity, verified
from .replica_baseline_score import _xyz


def _folder(binding):
    return Path(binding['output_root']) / 'external/OpenMask3D'


def freeze(binding, scene):
    from .openmask_full import STAGES, paths, locked_configuration
    root = Path(binding['output_root'])
    pipeline_path = _folder(binding) / 'pipeline/replica48' / scene / 'receipt.json'
    pipeline = verified(pipeline_path)
    if (pipeline['status'] != 'DONE' or pipeline['scene'] != scene
            or not pipeline.get('full_original_inference_stages')
            or set(pipeline.get('actual_jobs', {})) != set(STAGES)):
        raise ValueError('All four original OpenMask3D stages are required before export')
    if pipeline.get('GT_labels_used_for_prediction') is not False or pipeline.get('new_training_updates') != 0:
        raise ValueError('Full OpenMask3D inference must be label-free without new training')
    if pipeline['original_numeric_configuration'] != locked_configuration(binding):
        raise ValueError('Full OpenMask3D configuration changed before prediction freeze')
    files = paths(binding, scene)
    index = ConsumedInputs(root)
    for stage in STAGES:
        name = 'shared' if stage == 'text' else scene
        job = verified(root / 'jobs/external/OpenMask3D/replica48' / name / stage / 'receipt.json')
        if (job['identity'] != pipeline['actual_jobs'][stage] or job['status'] != 'DONE'
                or job.get('actual_child_exit') != 0 or not job.get('attempts') or not job.get('outputs')):
            raise ValueError('Actual original OpenMask3D job is incomplete: ' + stage)
        for record in job['outputs']:
            index.identity(record['path'], record)
        if not any(record['path'] == str(files[stage]) for record in job['outputs']):
            raise ValueError('Full OpenMask3D stage does not bind its actual native output: ' + stage)
        if stage == 'query' and not any(record['path'] == pipeline['prediction']['path']
                and record['sha256'] == pipeline['prediction']['sha256'] for record in job['outputs']):
            raise ValueError('Full pipeline prediction differs from its original query output')
    from .openmask_io import geometry
    current_geometry = geometry(binding, scene)['geometry']
    index.identity(current_geometry['path'], pipeline['geometry'])
    index.identity(pipeline['prediction']['path'], pipeline['prediction'])
    scope = verified(root / 'protocol_lock.json')['scopes']['replica48']
    if scene not in scope['scenes']:
        raise ValueError('OpenMask3D scene is outside the frozen Replica scope')
    with np.load(pipeline['prediction']['path'], allow_pickle=False) as data:
        prediction = {key: data[key].copy() for key in ('pred_masks', 'pred_classes', 'pred_scores')}
    masks, labels, scores = (prediction[key] for key in ('pred_masks', 'pred_classes', 'pred_scores'))
    if (masks.ndim != 2 or masks.shape[0] != len(_xyz(current_geometry['path']))
            or labels.shape != (masks.shape[1],) or scores.shape != labels.shape
            or not np.isin(masks, [0, 1]).all() or not np.issubdtype(labels.dtype, np.integer)
            or not np.isin(labels, scope['instance_ids']).all() or not np.equal(scores, 1).all()):
        raise ValueError('OpenMask3D native query differs from its point/class/constant-score contract')
    from .external import save_prediction
    result = save_prediction(_folder(binding) / 'predictions/replica48' / scene, prediction,
        method='OpenMask3D', benchmark='replica48', scene=scene,
        provenance=dict(pipeline=dict(path=str(pipeline_path), identity=pipeline['identity']),
            geometry=pipeline['geometry'], native_prediction=pipeline['prediction'],
            original_numeric_configuration=pipeline['original_numeric_configuration'],
            full_original_inference_stages=True, GT_labels_used_for_prediction=False,
            synthetic=False, producer=source_identity(('openmask_score', 'openmask_full', 'openmask_io'))))
    index.save()
    return result


def _adapter():
    return dict(method='OpenMask3D', variant='OpenMask3D (Replica dataset port)',
                producer='openmask_score', freeze=freeze)


def score_prediction(binding, scene, manifest_path):
    from .replica_baseline_score import score_prediction as score
    return score(binding, scene, manifest_path, _adapter())


def pool(binding):
    from .replica_baseline_score import pool as collect
    return collect(binding, _adapter())


def run_scene(binding, scene):
    import os
    import sys
    from .openmask_full import run_scene as infer
    from .replica_io import _tuple_path
    from .runtime import execute
    infer(binding, scene)
    manifest = freeze(binding, scene)
    root = Path(binding['output_root'])
    index = ConsumedInputs(root)
    inputs = [dict(prediction=manifest['identity'], protocol=verified(root / 'protocol_lock.json')['identity'],
        producer=source_identity(('openmask_score', 'replica_baseline_score', 'score_compat', 'replica_validation'))),
        index.identity(_tuple_path(binding, scene))]
    index.save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    output = _folder(binding) / 'evaluation/replica48' / scene / 'receipt.json'
    execute(binding, 'external/OpenMask3D/replica48/' + scene + '/score',
        [sys.executable, '-m', 'static_ovmap.benchmark_completion_v2.openmask_score',
         '--output-root', str(root), '--scene', scene], cwd=REPO, env=env, inputs=inputs,
        outputs=[output, output.with_name('G3_official_entry_agreement.json'),
                 output.with_name('OpenMask3D_matches.json.gz')], gpu=False)
    return verified(output)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--scene', required=True)
    args = parser.parse_args(argv)
    binding = verified(Path(args.output_root) / 'binding.json')
    score_prediction(binding, args.scene, _folder(binding) / 'predictions/replica48' / args.scene / 'manifest.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
