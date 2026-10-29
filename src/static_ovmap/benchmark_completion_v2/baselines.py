"""Full-method contracts and independent local/reference coverage, per scope."""
from pathlib import Path
import subprocess

from .common import REPO, StageFailure, verified, write, digest_file, source_identity
from .external import PINS


def contracts(benchmark):
    return {
        'OVI-MAP':dict(variant='ORIGIN_IMPLEMENTATION_STANDARD_PROTOCOL_PORT',
            required=['reconstruction_python','perception_python','cropformer_checkpoint','original_config'],
            commands=[['$PY_CONTROL',str(REPO/'scripts/evaluation/run_ovmap_benchmark_completion_v2.py'),
                '--output-root','$OUT','--resume','--phase','baselines']],
            port='ovi_adapter.predict/score; original author classifier, painted masks and confidence; cached original N only',
            source_files=['README.md','scripts/panoptic_mapping_.py','scripts/utils/mesh_postprocess_utils.py','scripts/eval_sem_seg.py']),
        'OpenMask3D':dict(variant='FULL_MASK3D_SAM_VITH_CLIP_L14_336',
            required=['python','mask_ckpt_scannet200' if benchmark=='scannet200_val' else 'mask_ckpt_replica',
                      'sam_checkpoint','clip_assets','original_config'],
            commands=[['$PYTHON','openmask3d/class_agnostic_mask_computation/get_masks_scannet200.py',
                'general.checkpoint=$MASK_CKPT_SCANNET200','model.num_queries=150','general.use_dbscan=true',
                'general.dbscan_eps=0.95','general.train_mode=false',
                'data.test_dataset.data_dir=$PROCESSED_ROOT','data.validation_dataset.data_dir=$PROCESSED_ROOT',
                'data.train_dataset.data_dir=$PROCESSED_ROOT','general.mask_save_dir=$MASK_DIR'],
                ['$PYTHON','openmask3d/compute_features_scannet200.py','data.scans_path=$RAW_ROOT',
                'data.masks.masks_path=$MASK_DIR','external.sam_checkpoint=$SAM_CHECKPOINT',
                'output.output_directory=$FEATURE_DIR','gpu.optimize_gpu_usage=false'],
                ['$PYTHON','openmask3d/evaluation/run_eval_close_vocab_inst_seg.py',
                '--gt_dir=$GT_ROOT','--mask_pred_dir=$MASK_DIR','--mask_features_dir=$FEATURE_DIR']]
                if benchmark=='scannet200_val' else
                [['$PY_CONTROL','-m','static_ovmap.benchmark_completion_v2.openmask_full',
                  '--output-root','$OUT','--scene','$SCENE'],
                 ['$PY_CONTROL','-m','static_ovmap.benchmark_completion_v2.openmask_score',
                  '--output-root','$OUT','--scene','$SCENE']],
            port='Replica uses openmask_full/openmask_score with arbitrary-scene checkpoint, original120-query proposal and explicit48-class I/O; real full execution/G3 remain required',
            source_files=['README.md','run_openmask3d_scannet200_eval.sh','run_openmask3d_single_scene.sh',
                          'openmask3d/evaluation/run_eval_close_vocab_inst_seg.py']),
        'Open3DIS':dict(variant='ISBNET_GSAM_REFINED_SECOND_3D_PASS',
            required=['python','checkpoints','original_config'],
            commands=[['$PYTHON','tools/'+name+'.py','--config','$LOCKED_CONFIG'] for name in
                      ('grounding_2d','generate_3d_inst','refine_grounding_feat','generate_3d_inst')],
            port='Author predictions preferred; config must contain genuine ISBNet proposals/deep features/superpoints, p2d+p3d+refined',
            source_files=['README.md','docs/DATA.md','docs/RUN.md','configs/scannet200.yaml','configs/replica.yaml',
                          'scripts/grounding_2d.sh','scripts/generate_3d_inst.sh','scripts/refine_grounding_feat.sh']),
        'MaskClustering':dict(variant='FULL_CROPFORMER_GRAPH_CLIP_H14_LAION2B',
            required=['python','cropformer_checkpoint','clip_assets','original_config'],
            commands=[['$PYTHON','run.py','--config','scannet']] if benchmark=='scannet200_val' else
                [['$PY_CONTROL','-m','static_ovmap.benchmark_completion_v2.mask_clustering_full',
                  '--output-root','$OUT','--scene','$SCENE'],
                 ['$PY_CONTROL','-m','static_ovmap.benchmark_completion_v2.mask_clustering_score',
                  '--output-root','$OUT','--scene','$SCENE']],
            port='ScanNet full author script requires inherited single-device/path-only deployment adaptation; Replica dataset reader is a distinct I/O port, not an author-reported method',
            source_files=['README.md','run.py','configs/scannet.json','utils/config.py','dataset/scannet.py',
                          'semantics/get_open-voc_features.py','semantics/open-voc_query.py'])}


def _present(value):
    if isinstance(value,dict):
        return bool(value) and all(_present(v) for v in value.values())
    return isinstance(value,str) and Path(value).exists()


def _status(rows):
    for status in ('FAILED_RUNTIME', 'BLOCKED_RESOURCE', 'BLOCKED_DEPENDENCY'):
        if any(row['status'] == status for row in rows):
            return status
    return 'DONE'


def _replica_method(binding, method, dependencies, run_scene, pool, producers):
    root = Path(binding['output_root'])
    scenes = verified(root / 'protocol_lock.json')['scopes']['replica48']['scenes']
    rows, halt = {}, None
    for scene in scenes:
        if halt:
            rows[scene] = dict(status='BLOCKED_DEPENDENCY', reason='First formal full pipeline must pass before later GPU scenes',
                               missing=[halt])
            continue
        try:
            availability = dependencies(binding, scene)
            write(root / 'external' / method / 'pipeline/replica48' / scene / 'availability.json', availability)
            if availability['missing']:
                rows[scene] = availability
                continue
            score = run_scene(binding, scene)
            if score.get('status') != 'DONE' or not score.get('formal_G3_verified') or score.get('synthetic'):
                raise ValueError('A real complete first-scene author agreement is required before later GPU scenes')
            rows[scene] = dict(status='DONE', score=score['identity'], formal_G3_verified=score['formal_G3_verified'])
        except StageFailure as exc:
            rows[scene] = dict(status=exc.status, reason=str(exc), missing=exc.missing)
            halt = str(root / 'external' / method / 'pipeline/replica48' / scene)
        except ValueError as exc:
            rows[scene] = dict(status='FAILED_RUNTIME', reason=type(exc).__name__ + ': ' + str(exc), missing=[])
            halt = str(root / 'external' / method / 'pipeline/replica48' / scene)
    return write(root / 'external' / method / 'execution_replica48.json', dict(
        status=_status(rows.values()), scenes=rows, pool=pool(binding),
        author_stage_hyperparameters_unchanged=True, first_formal_G3_required=True,
        source=source_identity(('baselines', 'replica_baseline_score', *producers))))


def maskclustering_replica(binding):
    from .mask_clustering_full import dependencies
    from .mask_clustering_score import run_scene, pool
    return _replica_method(binding, 'MaskClustering', dependencies, run_scene, pool,
                           ('mask_clustering_full', 'mask_clustering_score'))


def openmask_replica(binding):
    from .openmask_full import dependencies
    from .openmask_score import run_scene, pool
    return _replica_method(binding, 'OpenMask3D', dependencies, run_scene, pool,
                           ('openmask_full', 'openmask_score', 'openmask_io'))


def run(binding):
    from .author_results import build
    from .external_pooling import ovi
    from .open3dis_score import pool as open3dis_pool
    from .open3dis_replica_import import pool as open3dis_replica_pool
    root=Path(binding['output_root'])
    protocol=verified(root/'protocol_lock.json')
    authors=build(binding)
    native=ovi(binding)
    supplied=binding['asset_config'].get('external',{})
    methods={}
    for method,(_,pin) in PINS.items():
        assets=supplied.get(method) or {}
        repo=Path(assets.get('checkout') or root/'assets/external'/method)
        repo_missing=not (repo/'.git').exists()
        if not repo_missing:
            actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
            if actual!=pin or (assets.get('locked_commit') and assets['locked_commit']!=pin):
                raise ValueError('Full-method checkout differs from pinned source: '+method)
        scopes={}
        for benchmark,scope in protocol['scopes'].items():
            contract=contracts(benchmark)[method]
            reference=authors['results'].get(method+':'+benchmark)
            compatible=bool(reference and reference.get('compatible_frozen_variant',True))
            local=native['benchmarks'][benchmark] if method=='OVI-MAP' else None
            if method == 'Open3DIS' and benchmark == 'scannet200_val':
                local = open3dis_pool(binding)
            if method == 'Open3DIS' and benchmark == 'replica48':
                local = open3dis_replica_pool(binding)
                if local is not None:
                    contract = dict(contract, variant=local['variant'],
                        port='Completed Replica port: original ISBNet, Grounded SAM-HQ, author SPP and all four original stages')
            execution = None
            if method == 'MaskClustering' and benchmark == 'replica48' and not repo_missing:
                execution = maskclustering_replica(binding)
                local = execution['pool']
            if method == 'OpenMask3D' and benchmark == 'replica48' and not repo_missing:
                execution = openmask_replica(binding)
                local = execution['pool']
            missing=[dict(asset=key,expected=assets.get(key)) for key in contract['required'] if not _present(assets.get(key))]
            if execution is not None:
                missing = [dict(scene=scene, **item) for scene, row in execution['scenes'].items()
                           for item in row.get('missing', []) if isinstance(item, dict)]
                missing += [dict(scene=scene, reason=row.get('reason'), expected=row.get('missing'))
                            for scene, row in execution['scenes'].items()
                            if row['status'] != 'DONE' and row.get('reason')]
            complete=compatible or bool(local and local['full_scene_coverage'])
            scopes[benchmark]=dict(status='DONE' if complete else execution['status'] if execution else 'BLOCKED_DEPENDENCY',
                coverage='LOCAL_FULL' if local and local['full_scene_coverage'] else 'AUTHOR_REPORTED_COMPATIBLE' if compatible else 'INCOMPLETE',
                local_prediction_adapter='ovi_adapter' if method=='OVI-MAP' else
                    'open3dis_replica_import (validated completed full Replica port)'
                    if method == 'Open3DIS' and benchmark == 'replica48' and local else
                    'open3dis_score (locally scored complete author predictions)'
                    if method == 'Open3DIS' and local else
                    {'MaskClustering':'mask_clustering_score','OpenMask3D':'openmask_score'}[method]
                    if execution else 'external native format decoder',
                local_Replica_dataset_port_verified=bool(local and local.get('formal_G3_verified')) if benchmark=='replica48' and method in ('OpenMask3D','MaskClustering','Open3DIS') else None,
                author_reference=reference,local_pool=local,expected_scenes=scope['scenes'],
                local_execution=None if execution is None else execution['identity'],
                observed_local_scenes=[] if local is None else local['ordered_scenes'],
                locally_scored_author_predictions=bool(local and local.get('locally_scored_author_predictions')),
                locally_inferred=True if method == 'Open3DIS' and benchmark == 'replica48' and local
                    else None if local is None else local.get('locally_inferred'),
                missing=[] if complete else [dict(asset='pinned_checkout',expected=str(repo))] if repo_missing else missing,
                no_new_inference_needed_for_compatible_author_reference=compatible,
                assets_ready_does_not_imply_inference_completed=True,**contract)
        sources=[]
        if not repo_missing:
            for name in contracts('scannet200_val')[method]['source_files']:
                path=repo/name
                sources.append(dict(path=str(path),sha256=digest_file(path)))
        methods[method]=write(root/'external'/method/'recipe.json',dict(
            status=_status(scopes.values()),
            method=method,commit=pin,repository=PINS[method][0],sources=sources,scopes=scopes,
            required_fields_follow_prompt07=True,source=source_identity(('baselines',)),
            full_method_not_run=not any(s['observed_local_scenes'] and not s['locally_scored_author_predictions']
                                        for s in scopes.values()),
            locally_scored_author_predictions=any(s['locally_scored_author_predictions'] for s in scopes.values()),
            reduced_pipeline_allowed=False,
            expected_scene_jobs=sum(len(s['scenes']) for s in protocol['scopes'].values())))
    return write(root/'external/recipes.json',dict(status=_status(methods.values()),methods=methods,compatible_references_not_local_execution=True,
        expected_full_method_scene_jobs=4*sum(len(s['scenes']) for s in protocol['scopes'].values())))
