"""Finite official scenes: shared real sources, then immutable GLOBAL four cells."""
import os
from pathlib import Path
import pickle
import time
import numpy as np

from .common import (REPO, SEEDS, verified, read, write, save_arrays, ConsumedInputs,
                     Resolver, canonical_digest, digest_file, source_identity, StageFailure)


def template_context(binding, benchmark):
    cohort = 'replica8' if benchmark == 'replica48' else 'scannet_cf18'
    scene = binding['inherited']['cohorts'][cohort][0]
    resolver = Resolver(binding['path_map'])
    return resolver.rewrite(verified(resolver.resolve(binding['inherited']['scenes'][scene]['context']['path'])))


def worker_env(binding):
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(binding['gpu']), PYTHONPATH=str(REPO / 'src') + ':' + str(REPO),
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')


def native_query(binding, bundle):
    from .runtime import execute
    from .protocol import normalize_name
    from static_ovmap.module_validation.boundary_jobs import file_identity
    from static_ovmap.module_validation.scannet_study import native_readout
    from static_ovmap.backbone_wave1 import semantic_readout
    import torch

    root = Path(binding['output_root'])
    folder = root / 'benchmark/NQ' / bundle['benchmark'] / bundle['scene']
    context = template_context(binding, bundle['benchmark'])
    index = ConsumedInputs(root)
    model = dict(context['models']['native'])
    index.identity(model['text']['path'], model['text'])
    positions = {normalize_name(name): i for i, name in enumerate(model['class_names'])}
    rows = [positions[normalize_name(name)] for name in bundle['text_names']]
    with np.load(model['text']['path'], allow_pickle=False) as arrays:
        text = arrays['text_embeddings'][rows].copy()
        canonical = arrays['canonical_embeddings'].copy()
    text_path = folder / 'target_text.npz'
    if not text_path.is_file():
        save_arrays(text_path, text_embeddings=text, canonical_embeddings=canonical,
                    valid_ids=np.asarray(bundle['text_ids'], np.int64))
    else:
        with np.load(text_path, allow_pickle=False) as arrays:
            if not np.array_equal(arrays['text_embeddings'], text) or not np.array_equal(arrays['valid_ids'], bundle['text_ids']):
                raise ValueError('Native/Q target text changed')
    model.update(text=file_identity(text_path), valid_ids=bundle['text_ids'], class_names=bundle['text_names'])
    upstream = Path(context['actual_capture_command']['command'][1]).parent.parent
    job = dict(output_root=str(folder), map_id='NATIVE_BENCHMARK_' + bundle['scene'],
        capture_manifest=bundle['capture']['path'], deferred_metadata=bundle['deferred_metadata']['path'],
        model=model, checkpoint=context['checkpoint'], upstream=str(upstream),
        gpu_lock=f"/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-{binding['gpu']}.lock",
        encoder_cache_root=str(root / 'content_cache/native'),
        original_worker_sha256=digest_file(semantic_readout.__file__), bundle=bundle['identity'])
    job_path = folder / 'job.json'
    write(job_path, job)
    python = context['runtime']['semantic_python']
    env = worker_env(binding)
    env.update(PATH=str(Path(python).parent) + ':' + os.environ.get('PATH', ''),
        LD_LIBRARY_PATH=str(Path(python).parents[1] / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    # The frozen parent owns this same inherited GPU lock internally.
    child = execute(binding, f"benchmark/{bundle['benchmark']}/{bundle['scene']}/native_query",
        [python, '-u', '-m', 'static_ovmap.benchmark_completion_v2.nq_worker', '--job', job_path],
        cwd=REPO, env=env, inputs=[index.identity(job_path), index.identity(Path(__file__).with_name('nq_worker.py'))],
        outputs=[folder / name for name in ('native_features.pkl', 'query_scores.npz', 'native_query_receipt.json',
                                             'query_decisions.json', 'boundary_adaptation.json')],
        gpu=True, child_gpu_lock=True)
    receipt = read(folder / 'native_query_receipt.json')
    if receipt['status'] != 'COMPLETE':
        raise ValueError('Native/Q child did not produce complete real model outputs')
    with (folder / 'native_features.pkl').open('rb') as stream:
        saved = pickle.load(stream)
    native = native_readout(saved, text, canonical, tuple(bundle['text_ids']))
    ids = np.asarray(bundle['text_ids'], np.int64)
    sources = {'N': {}, 'Q': {}}
    for obj in bundle['objects']:
        owner = obj['owner']
        row = native.get(owner)
        available = row is not None and row['status'] == 'AVAILABLE'
        # Preserve the original N cosine operation in its own SigLIP space.
        score = torch.nn.functional.cosine_similarity(torch.as_tensor(row['feature'], dtype=torch.float32),
            torch.as_tensor(text, dtype=torch.float32), dim=-1).numpy() if available else None
        sources['N'][str(owner)] = dict(available=available, scores=score,
            used_frames=[] if row is None else row['used_frames'])
    with np.load(folder / 'query_scores.npz', allow_pickle=False) as arrays:
        if not np.array_equal(arrays['valid_ids'], ids):
            raise ValueError('Real Q target class axis differs from F/N')
        positions = {int(owner): i for i, owner in enumerate(arrays['owner_ids'])}
        for obj in bundle['objects']:
            i = positions.get(obj['owner'])
            available = i is not None and bool(arrays['available'][i])
            sources['Q'][str(obj['owner'])] = dict(available=available,
                scores=arrays['scores'][i] if available else None)
    result = write(folder / 'sources.json', dict(status='DONE', bundle=bundle['identity'],
        source_model=model, actual_execution=child['identity'], source_receipt=file_identity(folder / 'native_query_receipt.json'),
        query_completion=receipt['query_completion'], sources=sources, valid_ids=ids,
        physical_image_encodings=receipt['physical_image_encodings'], actual_gpu_lock=job['gpu_lock'],
        new_training_updates=0, GT_labels_read=False))
    index.save()
    return result


def predict_scene(binding, benchmark, scene):
    from .native_inputs import build
    from .runtime import execute
    from .standard_export import VARIANTS, score_F, classify_scores, save_manifest
    root = Path(binding['output_root'])
    bundle = build(binding, benchmark, scene)
    bundle_path = root / 'benchmark/bundles' / benchmark / scene / 'bundle.json'
    folder = root / 'benchmark/predictions' / benchmark / scene
    protocol = verified(root / 'protocol_lock.json')
    scope = protocol['scopes'][benchmark]
    index = ConsumedInputs(root)
    f_path = root / 'benchmark/F' / benchmark / scene / 'receipt.json'
    execute(binding, f'benchmark/{benchmark}/{scene}/F_MV1000',
        [binding['inherited']['FC_python'], '-u', '-m', 'static_ovmap.benchmark_completion_v2.fc_worker',
         '--binding', root / 'binding.json', '--bundle', bundle_path], cwd=REPO, env=worker_env(binding),
        inputs=[dict(bundle=bundle['identity'], model_lock=bundle['model_lock']),
                index.identity(Path(__file__).with_name('fc_worker.py'))], outputs=[f_path], gpu=True)
    f = verified(f_path)
    index.identity(f['arrays']['path'], f['arrays'])
    with np.load(f['arrays']['path'], allow_pickle=False) as arrays:
        f_arrays = {name: arrays[name].copy() for name in arrays.files}
    with np.load(bundle['arrays']['geometry']['path'], allow_pickle=False) as arrays:
        geometry = {name: arrays[name].copy() for name in ('owners', 'indptr', 'indices', 'target_xyz')}
    if not np.array_equal(geometry['owners'], f_arrays['owners']):
        raise ValueError('GLOBAL F changed the immutable all-owner domain')
    base_mask = scope['mapping']['base_mask']
    variants = VARIANTS if base_mask is not None else ('U00', 'U10')
    # Construct the F-only labels before even requesting N/Q. This retains a real
    # IO boundary for source-off validation and supports independent descendants.
    fq = {}
    for si, seed in enumerate(SEEDS):
        fq[seed] = []
        for i, owner in enumerate(geometry['owners']):
            fq[seed].append((None, None) if not f_arrays['available'][i] else
                score_F(f_arrays['b'][i], f_arrays['z'][si, i], f_arrays['text'], base_mask,
                        calibrated=base_mask is not None))
    nq = None
    methods, traces = {}, {}
    temperatures = template_context(binding, benchmark)['temperatures']
    for variant in sorted(variants, key=lambda v: v in ('U10', 'U11')):
        if variant in ('U10', 'U11') and nq is None:
            nq = native_query(binding, bundle)
        for seed in SEEDS:
            accesses, labels, owner_audit = [], [], []
            for i, owner in enumerate(geometry['owners']):
                def source_loader(name):
                    accesses.append((int(owner), name))
                    row = nq['sources'][name][str(owner)]
                    return row['scores'] if row['available'] else None
                result = classify_scores(variant, f_arrays['valid_ids'], *fq[seed][i], source_loader, temperatures)
                labels.append(result['label'])
                owner_audit.append(dict(owner=int(owner), **result))
            if variant in ('U00', 'U01') and accesses:
                raise ValueError('MF-off illegally accessed N/Q')
            method = variant + '_seed' + str(seed)
            row = save_manifest(folder / method, scene=scene, benchmark=benchmark, method=variant, seed=seed,
                owners=geometry['owners'], indptr=geometry['indptr'], indices=geometry['indices'], labels=labels,
                target_points=len(geometry['target_xyz']), valid_instance_ids=scope['instance_ids'],
                provenance=dict(bundle=bundle['identity'], model_lock=bundle['model_lock'], protocol=protocol['identity'],
                    F=f['identity'], NQ=nq['identity'] if variant in ('U10', 'U11') else None,
                    MFoff_NQ_accesses=0 if variant in ('U00', 'U01') else None,
                    calibrator='CAP10_gamma.025' if variant in ('U01', 'U11') else 'RAW',
                    source_availability=owner_audit, exposure_role=scope.get('exposure_role', 'SEE_PROTOCOL_LOCK')))
            methods[method] = dict(identity=row['identity'], path=str(folder / method / 'manifest.json'))
            traces[method] = dict(NQ_access_count=len(accesses), availability=owner_audit)
    result = write(folder / 'receipt.json', dict(status='DONE' if len(variants) == 4 else 'BLOCKED_DEPENDENCY',
        scene=scene, benchmark=benchmark, bundle=bundle['identity'], methods=methods, traces=traces,
        expected_rows=8, completed_rows=len(methods), missing_variants=sorted(set(VARIANTS) - set(variants)),
        missing_reason=None if len(variants) == 4 else scope['mapping']['unresolved'],
        all_available_predictions_frozen_before_new_GT_scoring=True, GLOBAL=True,
        all_original_owners_retained=True, GT_labels_read=False))
    index.save()
    print('GLOBAL_PREDICTED', benchmark, scene, len(methods), flush=True)
    return result
