"""Full MaskClustering prediction freeze and shared independent Replica scoring."""
from pathlib import Path

from .common import ConsumedInputs, REPO, source_identity, verified, write
from .replica_baseline_score import _xyz


def _folder(binding):
    return Path(binding['output_root']) / 'external/MaskClustering'


def freeze(binding, scene):
    """Only actual successful original stages can create a full-method export."""
    from .mask_clustering_full import STAGES
    root = Path(binding['output_root'])
    pipeline_path = _folder(binding) / 'pipeline/replica48' / scene / 'receipt.json'
    pipeline = verified(pipeline_path)
    if (pipeline['status'] != 'DONE' or pipeline['scene'] != scene
            or not pipeline.get('full_original_inference_stages')
            or set(pipeline.get('actual_jobs', {})) != set(STAGES)):
        raise ValueError('All five original MaskClustering stages are required before export')
    if pipeline.get('GT_labels_used_for_prediction') is not False or pipeline.get('new_training_updates') != 0:
        raise ValueError('Full MaskClustering inference must be label-free without new training')
    index = ConsumedInputs(root)
    for stage in STAGES:
        name = 'shared' if stage == 'text' else scene
        job = verified(root / 'jobs/external/MaskClustering/replica48' / name / stage / 'receipt.json')
        if (job['identity'] != pipeline['actual_jobs'][stage] or job['status'] != 'DONE'
                or job.get('actual_child_exit') != 0 or not job.get('attempts') or not job.get('outputs')):
            raise ValueError('Actual original MaskClustering job is incomplete: ' + stage)
        for record in job['outputs']:
            index.identity(record['path'], record)
        if stage == 'query' and not any(row['sha256'] == pipeline['prediction']['sha256']
                                        and row['path'] == pipeline['prediction']['path'] for row in job['outputs']):
            raise ValueError('Full pipeline prediction is not the original semantic query output')
    index.identity(pipeline['geometry']['path'], pipeline['geometry'])
    index.identity(pipeline['prediction']['path'], pipeline['prediction'])
    scope = verified(root / 'protocol_lock.json')['scopes']['replica48']
    if scene not in scope['scenes']:
        raise ValueError('MaskClustering scene is outside the frozen Replica scope')
    from .baseline_io import decode_maskclustering
    from .external import save_prediction
    prediction = decode_maskclustering(pipeline['prediction']['path'],
        target_points=len(_xyz(pipeline['geometry']['path'])), class_ids=scope['instance_ids'])
    result = save_prediction(_folder(binding) / 'predictions/replica48' / scene, prediction,
        method='MaskClustering', benchmark='replica48', scene=scene,
        provenance=dict(pipeline=dict(path=str(pipeline_path), identity=pipeline['identity']),
            geometry=pipeline['geometry'], native_prediction=pipeline['prediction'],
            original_numeric_configuration=pipeline['original_numeric_configuration'],
            full_original_inference_stages=True, GT_labels_used_for_prediction=False,
            synthetic=False, producer=source_identity(('mask_clustering_score', 'baseline_io'))))
    index.save()
    return result


def _adapter():
    return dict(method='MaskClustering', variant='MaskClustering (Replica dataset port)',
                producer='mask_clustering_score', freeze=freeze)


def score_prediction(binding, scene, manifest_path):
    from .replica_baseline_score import score_prediction as score
    return score(binding, scene, manifest_path, _adapter())


def pool(binding):
    from .replica_baseline_score import pool as collect
    return collect(binding, _adapter())


def run_scene(binding, scene):
    from .mask_clustering_full import run_scene as infer
    from .replica_io import _tuple_path
    from .runtime import execute
    import os
    import sys
    infer(binding, scene)
    manifest = freeze(binding, scene)
    root = Path(binding['output_root'])
    index = ConsumedInputs(root)
    inputs = [dict(prediction=manifest['identity'], protocol=verified(root / 'protocol_lock.json')['identity'],
                   producer=source_identity(('mask_clustering_score', 'replica_baseline_score', 'score_compat', 'replica_validation'))),
              index.identity(_tuple_path(binding, scene))]
    index.save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    output = _folder(binding) / 'evaluation/replica48' / scene / 'receipt.json'
    execute(binding, 'external/MaskClustering/replica48/' + scene + '/score',
        [sys.executable, '-m', 'static_ovmap.benchmark_completion_v2.mask_clustering_score',
         '--output-root', str(root), '--scene', scene], cwd=REPO, env=env, inputs=inputs,
        outputs=[output, output.with_name('G3_official_entry_agreement.json'),
                 output.with_name('MaskClustering_matches.json.gz')], gpu=False)
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
