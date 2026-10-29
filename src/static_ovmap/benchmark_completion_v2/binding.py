"""Immutable public-A binding and finite STRUCT→FG/LR/ATTR asset resolution."""
from pathlib import Path
from .common import (REPO, BASE, TASK, read, verified, write, canonical_digest,
                     ConsumedInputs, Resolver, StageFailure)


def public_folder(root):
    root = Path(root)
    options = [root, root / 'artifacts/static_ovmap/semantic_update_evidence_v1']
    for folder in options:
        if (folder / 'A/source_vectors.npz').is_file() and (folder / 'A/source_scores.npz').is_file():
            return folder
    # Original OUTs keep import.npz; the same exact published bundle is in REPO.
    if (root / 'A/import.npz').is_file():
        meta = verified(root / 'A/import.json')
        folder = REPO / 'artifacts/static_ovmap/semantic_update_evidence_v1'
        if verified(folder / 'A/import.json')['identity'] == meta['identity']:
            return folder
    raise StageFailure('Exact public A scores/vectors bundle is absent', missing=[str(root)])


def structure_document(root, public, relative, index):
    root = Path(root) if root else None
    choices = ([root / relative] if root else []) + [public / 'records' / Path(relative).with_suffix('.json.gz')]
    for path in choices:
        if path.is_file():
            index.identity(path)
            return verified(path), str(path)
    return None, None


def bind(args):
    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    spec = read(args.spec)
    if spec['task_id'] != TASK or spec['base_commit'] != BASE:
        raise ValueError('Wrong frozen task configuration')
    if Path(args.spec).read_bytes() != (REPO / 'docs/paper/static_ovmap/benchmark_completion_v2/prompt/02_冻结实验配置.json').read_bytes():
        raise ValueError('Frozen configuration changed')
    prior = read(root / 'task_paths.json') if (root / 'task_paths.json').exists() else {}
    assets = read(args.asset_config) if args.asset_config else {}
    mapping = read(args.path_map) if args.path_map else assets.get('path_map', prior.get('path_map', {}))
    resolver = Resolver(mapping)
    semantic = args.semantic_root or prior.get('semantic_root') or assets.get('semantic_root') or str(REPO / spec['cache']['public_folder'])
    semantic = public_folder(resolver.resolve(semantic))
    structure = args.structure_root or prior.get('structure_root') or assets.get('structure_root')
    structure = resolver.resolve(structure) if structure else None
    public = REPO / spec['model']['export_folder']
    index = ConsumedInputs(root)
    index.identity(args.spec)
    required = ['A/import.json', 'A/source_scores.npz', 'A/source_vectors.npz',
                'A/candidates.json.gz', 'A/predictions.npz', 'A/crossfit.json.gz']
    descriptors = {name: index.identity(semantic / name) for name in required}
    metadata = verified(semantic / 'A/import.json')
    documents, document_paths = {}, {}
    for name in ['source_binding.json', 'model_selection_lock.json', 'predictor_lock.json',
                 'class_split.json', 'split_manifest.json', 'role_lock.json']:
        documents[name], document_paths[name] = structure_document(structure, public, name, index)
    inherited = resolver.rewrite(documents['source_binding.json']) if documents['source_binding.json'] else None
    if inherited and str(root) in [inherited.get(k) for k in ['output_root', 'fg_parent_root', 'lr_parent_root']]:
        raise ValueError('Parent scientific directories must remain read-only')
    actual_gpu = str(inherited['gpu']) if inherited else assets.get('gpu_device')
    if args.gpu is not None and str(args.gpu) != actual_gpu:
        raise ValueError('GPU must match the inherited actual GPU')
    if assets.get('gpu_device') is not None and str(assets['gpu_device']) != actual_gpu:
        raise ValueError('Asset-config GPU conflicts with inherited GPU')
    result = write(root / 'binding.json', dict(status='DONE', task_id=TASK, base_commit=BASE,
        specification=spec, semantic_root=str(semantic), structure_root=structure,
        structure_public=str(public), public_A=descriptors, public_A_identity=metadata['identity'],
        structure_documents=document_paths, inherited=inherited, path_map=mapping,
        gpu=actual_gpu, asset_config=assets, asset_identity=canonical_digest(assets),
        interpreters=dict(controller=assets.get('controller_python') or prior.get('controller_python'),
                          fc=assets.get('fc_python') or (inherited or {}).get('FC_python')),
        output_root=str(root), input_identity=canonical_digest(dict(public_A=descriptors,
            structure={k:v.get('identity') if v else None for k,v in documents.items()}, assets=assets))))
    index.save()
    return result


def restore_A(binding):
    import numpy as np
    from static_ovmap.semantic_update_evidence_v1.replay import restore_public_inputs
    root = Path(binding['output_root'])
    folder = Path(binding['semantic_root'])
    # E02 copies its original sealed metadata, scores and primary real vectors.
    restore_public_inputs(folder, root)
    meta = verified(root / 'A/import.json')
    objects = meta['objects']
    if (len(objects), len({o['family'] for o in objects}), sum(not o['supported'] for o in objects)) != (279, 12, 9):
        raise ValueError('Original object/family/unsupported identity differs')
    with np.load(root / 'A/import.npz', allow_pickle=False) as arrays:
        ids = arrays['valid_ids']
        text = arrays['text']
        from static_ovmap.semantic_update_evidence_v1.readers import column_index
        columns = [column_index(meta['columns'], 'B_FC', None)] + [column_index(meta['columns'], method, seed)
            for method in binding['specification']['calibration']['methods'] for seed in (17, 29)]
        b = arrays['embeddings'][:, columns, 0]
        scores = arrays['scores'][:, columns, 0]
        defined = arrays['defined'][:, columns, 0]
        if not np.isfinite(b[defined]).all() or not np.isfinite(scores[defined]).all():
            raise ValueError('Primary defined score/vector is nonfinite')
        error = float(np.max(np.abs((b[defined] @ text.T) - scores[defined])))
        if error > 1e-5 or not np.allclose(np.linalg.norm(b[defined], axis=-1), 1, atol=1e-5, rtol=0):
            raise ValueError('Real vector/text reconstruction failed')
    return write(root / 'A/restoration.json', dict(status='DONE', original_metadata=meta['identity'],
        primary_columns=columns, originals=279, families=12, unsupported=9,
        score_reconstruction_max_absolute_error=error, text_renormalized=False,
        source_vectors='PUBLIC_REAL_CLEAN8V; original_column_indices aligned', new_encodings=0))
