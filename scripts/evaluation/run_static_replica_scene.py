#!/usr/bin/env python3
"""Execute the validated static readout/evaluation stages for one completed Replica scene."""
import argparse
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.baselines.ovimap import parse_instance_color_log, remap_instance_colors
from src.static_ovmap.cache_io import sha256_file

SCENES = ('office0', 'office1', 'office2', 'office3', 'office4', 'room0', 'room1', 'room2')


def verify_native_manifest(path):
    x = json.loads(Path(path).read_text())
    if (x.get('status') != 'PASS' or x.get('scene') not in SCENES or x.get('dataset') != 'replica'
            or x.get('audit', {}).get('status') != 'PASS'):
        raise ValueError('completed native Replica manifest required')
    if x.get('frame_ids') != list(range(0, 2000, 10)) or x['audit'].get('frame_count') != 200:
        raise ValueError('native frame protocol differs from fixed Replica200')
    for key in ('instance_mesh', 'semantic_features', 'instance_color_log'):
        r = x['artifacts'][key]
        if sha256_file(r['path']) != r['sha256']:
            raise ValueError('native artifact changed: '+key)
    return x


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-manifest', 'text-cache', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    build = Path('/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native')
    p.add_argument('--evaluator-root', type=Path, default=build/'OVI-MAP')
    p.add_argument('--model', type=Path, default=build/'siglip-large-patch16-384')
    p.add_argument('--data-root', type=Path, default=Path('/home/ww/vv/dataset/Replica'))
    p.add_argument('--python', type=Path, default=Path('/home/ww/miniconda3/envs/ovimap-map/bin/python'))
    p.add_argument('--gpu', default='2')
    p.add_argument('--include-full-history', action='store_true')
    args = p.parse_args()
    native = verify_native_manifest(args.native_manifest)
    scene = native['scene']
    mesh, cache, color_log = [Path(native['artifacts'][k]['path']) for k in
                             ('instance_mesh', 'semantic_features', 'instance_color_log')]
    mapper = args.evaluator_root/'scripts/panoptic_mapping_.py'
    text_metadata = json.loads(args.text_cache.with_suffix('.json').read_text())
    identity = text_metadata['feature_space_identity']
    if (identity['image_encoder_source_sha256'] != sha256_file(args.evaluator_root/'scripts/vl_models.py')
            or identity['model_sha256'] != sha256_file(args.model/'model.safetensors')
            or identity['preprocessor_sha256'] != sha256_file(args.model/'preprocessor_config.json')
            or text_metadata['semantic_const_sha256'] != sha256_file(args.evaluator_root/'scripts/utils/semantic_const.py')
            or text_metadata['output_sha256'] != sha256_file(args.text_cache)
            or text_metadata['dataset'] != 'Replica'
            or native['preflight']['sources']['mapper']['sha256'] != sha256_file(mapper)):
        raise ValueError('native/text/model/source identity mismatch')
    history = args.native_manifest.parent/'query_history'
    if args.include_full_history:
        for key, name in [('full_query_cache', 'full_query_cache.pkl'), ('query_history_receipt', 'history_receipt.json')]:
            if (key not in native['artifacts'] or
                    sha256_file(history/name) != native['artifacts'][key]['sha256']):
                raise ValueError('completed full-history artifact binding required')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'logs').mkdir()
    job = {'status': 'RUNNING', 'scene': scene, 'native_manifest': str(args.native_manifest),
           'native_manifest_sha256': sha256_file(args.native_manifest), 'frame_ids': native['frame_ids'],
           'baseline_role': 'rebuilt B1; internal B0 names mean original readout of this native run',
           'command': sys.argv, 'include_full_history': args.include_full_history,
           'source_sha256': sha256_file(__file__), 'stages': []}
    job['evaluation_input_sha256'] = {str(f): sha256_file(f) for f in
        (args.data_root/(scene+'_mesh.ply'),
         args.evaluator_root/'scripts/datasets/replica_gt_semantics'/f'semantic_labels_{scene}.txt',
         args.evaluator_root/'scripts/datasets/replica_gt_instances'/f'instance_labels_{scene}.txt')}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               OVIMAP_SIGLIP_MODEL=str(args.model), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               PYTHONPATH=str(ROOT))

    def run(name, argv, cwd=ROOT):
        command = [str(args.python), *map(str, argv)]
        stage = {'name': name, 'argv': command, 'cwd': str(cwd), 'status': 'RUNNING'}
        job['stages'].append(stage); write(args.output/'scene_pipeline.json', job)
        start = time.perf_counter()
        with (args.output/'logs'/f'{name}.log').open('w') as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        stage.update(exit_code=result.returncode, seconds=time.perf_counter()-start,
                     status='COMPLETE' if result.returncode == 0 else 'FAILED')
        write(args.output/'scene_pipeline.json', job)
        if result.returncode:
            raise RuntimeError('stage failed: '+name)

    def script(name, values):
        argv = [ROOT/'scripts/evaluation'/name]
        for key, value in values.items():
            argv.append('--'+key)
            argv.extend(value if isinstance(value, list) else [value])
        return argv

    started = time.perf_counter()
    try:
        binding = {'native_cache_sha256': sha256_file(cache), 'source_config_sha256': sha256_file(mapper),
                   'feature_space_id': text_metadata['feature_space_id'], 'feature_dim': identity['feature_dim'],
                   'feature_space_identity': identity, 'evidence_scope': 'fresh manifest-bound native run'}
        write(args.output/'native_binding.json', binding)
        run('enrichment', script('enrich_static_ovmap_observations.py', {
            'native-cache': cache, 'native-mesh': mesh, 'rgbd-root': args.data_root/scene/'results',
            'output': args.output/'enrichment.json'}))
        readout_args = {'native-cache': cache, 'native-binding': args.output/'native_binding.json',
            'text-cache': args.text_cache, 'scene': scene, 'source-config': mapper,
            'history-scope': 'retained_native_top10', 'enrichment': args.output/'enrichment.json',
            'conditions': ['B0', 'S1a', 'S1b', 'S1c', 'C1', 'RANDOM8', 'QUALITY8', 'ALL_VIEWS'],
            'seed': '0', 'output': args.output/'readout'}
        if args.include_full_history:
            run('full_enrichment', script('enrich_static_ovmap_observations.py', {
                'native-cache': history/'full_query_cache.pkl', 'native-mesh': mesh,
                'rgbd-root': args.data_root/scene/'results', 'output': args.output/'full_enrichment.json'}))
            readout_args.update({'full-query-cache': history/'full_query_cache.pkl',
                'history-receipt': history/'history_receipt.json', 'full-enrichment': args.output/'full_enrichment.json'})
            readout_args['conditions'] += ['S1a_FULL', 'RANDOM8_FULL', 'QUALITY8_FULL', 'ALL_VIEWS_FULL']
        run('readout', script('run_static_ovmap_readout.py', readout_args))
        layout = args.output/'layout'
        scene_dir = layout/scene
        crop = scene_dir/'cropformer_inst'; crop.mkdir(parents=True)
        (crop/mesh.name).symlink_to(mesh.resolve())
        colors = parse_instance_color_log(color_log)
        with cache.open('rb') as f: raw = pickle.load(f)
        staged = remap_instance_colors({int(k): v for k, v in raw.items() if k in colors}, colors)
        with (crop/cache.name).open('wb') as f: pickle.dump(staged, f)
        write(args.output/'staging.json', {'stale_ids': sorted(map(int, set(raw)-set(colors))),
              'staged_cache_sha256': sha256_file(crop/cache.name), 'raw_cache_sha256': sha256_file(cache)})
        dataset = args.output/'dataset/Replica'; (dataset/scene).mkdir(parents=True)
        (dataset/(scene+'_mesh.ply')).symlink_to((args.data_root/(scene+'_mesh.ply')).resolve())
        run('gt_preprocess', ['-m', 'scripts.datasets.preprocess_gt_mesh', '--scene_num', scene,
            '--data_folder', dataset, '--result_folder', layout,
            '--gt_sem_folder', args.evaluator_root/'scripts/datasets/replica_gt_semantics',
            '--gt_inst_folder', args.evaluator_root/'scripts/datasets/replica_gt_instances'], args.evaluator_root)
        run('native_postprocess', ['-m', 'scripts.utils.mesh_postprocess_utils', '--scene_num', scene,
                                   '--result_folder', layout], args.evaluator_root)
        original_instance = crop/'instance_map_gt_200.ply'; original_semantic = crop/'semantic_map_gt_200.ply'
        run('semantic', script('evaluate_static_ovmap_readout.py', {'readout': args.output/'readout',
            'projected-instance-map': original_instance, 'original-semantic-map': original_semantic,
            'gt-semantic-map': scene_dir/'gt_semantic_mesh.ply', 'text-cache': args.text_cache,
            'output': args.output/'semantic'}))
        export_code = """import sys
from pathlib import Path
from scripts.evaluation.evaluate_static_ovmap_instances import load_exports
p=Path(sys.argv[2]); out=p/'cropformer_inst/eval'; out.mkdir()
e=load_exports(Path(sys.argv[1])/'scripts/eval_sem_seg.py')
e['map_gt_mesh']({'inst_mesh_f':str(p/'gt_instance_mesh.ply'),'sem_mesh_f':str(p/'gt_semantic_mesh.ply'),'res_folder':str(out)})
e['map_pred_mesh']({'inst_mesh_f':str(p/'cropformer_inst/instance_map_gt_200.ply'),'sem_mesh_f':str(p/'cropformer_inst/semantic_map_gt_200.ply'),'res_folder':str(out)})
"""
        run('reference_export', ['-c', export_code, args.evaluator_root, scene_dir])
        run('instances', script('evaluate_static_ovmap_instances.py', {'evaluator-root': args.evaluator_root,
            'semantic-results': args.output/'semantic', 'projected-instance-map': original_instance,
            'original-semantic-map': original_semantic, 'gt-instance-map': scene_dir/'gt_instance_mesh.ply',
            'gt-semantic-map': scene_dir/'gt_semantic_mesh.ply',
            'reference-pred-mapping': crop/'eval/pred_inst_sem_mapping.txt',
            'reference-gt-ids': crop/'eval/gt_sem_inst_id.npy', 'output': args.output/'instances'}))
        run('projection_audit', script('audit_static_ovmap_projection.py', {'native-mesh': mesh,
            'native-cache': cache, 'baseline-readout': args.output/'readout/B0.json',
            'original-instance-map': original_instance, 'original-semantic-map': original_semantic,
            'output': args.output/'projection_audit'}))
        if json.loads((args.output/'projection_audit/projection_audit.json').read_text())['status'] != 'PASS':
            raise ValueError('independent projection parity failed')
        run('full_geometry', script('evaluate_static_native_geometry.py', {'native-mesh': mesh,
            'native-cache': cache, 'color-log': color_log, 'gt-instance-map': scene_dir/'gt_instance_mesh.ply',
            'scene': scene, 'output': args.output/'geometry'}))
        job.update(status='COMPLETE', wall_seconds=time.perf_counter()-started,
                   cost_scope='warm native features; evaluation-inclusive wall time, not native E2E FPS')
        write(args.output/'scene_pipeline.json', job)
        print(args.output/'scene_pipeline.json')
    except Exception as error:
        job.update(status='FAILED', error=str(error), wall_seconds=time.perf_counter()-started)
        write(args.output/'scene_pipeline.json', job)
        raise


if __name__ == '__main__':
    main()
