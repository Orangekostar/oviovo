"""Official vocabularies, label roles and finite asset inventory, before new GT."""
import ast
from collections import defaultdict
import csv
from pathlib import Path
import re

from .common import (read, verified, write, ConsumedInputs, Resolver, source_identity,
                     canonical_digest, StageFailure)


def normalize_name(name):
    return ' '.join(re.findall(r'\w+', name.casefold().replace('_', ' ')))


def map_training_roles(target_names, training_ids, training_names, base_ids, aliases=None):
    """No ID-size, embedding, GT-frequency or unmatched→novel heuristics."""
    if len(training_ids) != len(training_names) or len(set(training_ids)) != len(training_ids):
        raise ValueError('Training semantic ID/name axis is not unique')
    direct = defaultdict(list)
    for label, name in zip(training_ids, training_names, strict=True):
        direct[normalize_name(name)].append(dict(class_id=int(label), evidence='OFFICIAL_CANONICAL_NAME'))
    known_ids, base = set(map(int, training_ids)), set(map(int, base_ids))
    rows, unresolved = [], []
    for name in target_names:
        key = normalize_name(name)
        candidates = direct.get(key) or (aliases or {}).get(key, [])
        ids = sorted({int(row['class_id']) for row in candidates})
        if any(label not in known_ids for label in ids):
            raise ValueError('Alias points outside the actual training vocabulary')
        resolved = len(ids) == 1
        rows.append(dict(target_name=name, normalized_name=key, candidate_training_ids=ids,
            base=(ids[0] in base) if resolved else None, evidence=candidates,
            status='DONE' if resolved else 'BLOCKED_DEPENDENCY'))
        if not resolved:
            unresolved.append(name)
    return dict(status='DONE' if not unresolved else 'BLOCKED_DEPENDENCY', rows=rows,
        unresolved=unresolved, base_mask=[r['base'] for r in rows] if not unresolved else None,
        GT_labels_used=False, rule='official semantic IDs > canonical names > sourced aliases')


def _literal(path, symbol, *, size=None):
    """Read official constant declarations without importing model/CUDA modules."""
    values = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not isinstance(node, ast.Assign) or not any(isinstance(t, ast.Name) and t.id == symbol for t in node.targets):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute) and value.func.attr == 'array':
            value = value.args[0]
        try:
            value = ast.literal_eval(value)
        except (ValueError, TypeError):
            continue
        if size is None or len(value) == size:
            values.append(value)
    if len(values) != 1:
        raise ValueError('Official constant is absent or ambiguous: ' + str(path) + ':' + symbol)
    return list(values[0])


def _overlaps(path, benchmark):
    """Evaluate only the pinned evaluator's threshold declaration, exactly."""
    import numpy as np
    nodes = []
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not isinstance(node, ast.Assign):
            continue
        target = ast.unparse(node.targets[0])
        if target == ("opt['overlaps']" if benchmark == 'scannet200_val' else 'self.ious'):
            nodes.append(node.value)
    if len(nodes) != 1:
        raise ValueError('Official evaluator threshold declaration is ambiguous')
    # All code is from an already hashed pinned official source. Only this
    # literal NumPy declaration executes; no evaluator imports or GT access.
    return eval(compile(ast.Expression(nodes[0]), str(path), 'eval'), {'np': np, '__builtins__': {}}).tolist()


def _aliases(path, training_ids, training_names, index):
    """TSV identity and exact shared WordNet synset only; no broad NYU grouping."""
    if path is None or not Path(path).is_file():
        return {}, None
    descriptor = index.identity(path)
    with Path(path).open() as stream:
        records = list(csv.DictReader(stream, delimiter='\t'))
    canonical = {normalize_name(name): int(label) for label, name in zip(training_ids, training_names, strict=True)}
    synsets = defaultdict(set)
    for row in records:
        names = {normalize_name(row[field]) for field in ('raw_category', 'category')}
        ids = {canonical[name] for name in names if name in canonical}
        if row['wnsynsetid']:
            synsets[row['wnsynsetid']].update(ids)
    aliases = defaultdict(list)
    for row in records:
        names = {normalize_name(row[field]) for field in ('raw_category', 'category')}
        ids = {canonical[name] for name in names if name in canonical}
        if row['wnsynsetid']:
            ids |= synsets[row['wnsynsetid']]
        for name in names:
            for label in sorted(ids):
                evidence = dict(class_id=label, evidence='OFFICIAL_SCANNET_TSV_CANONICAL_OR_EXACT_WORDNET_SYNONYM',
                    official_raw_id=row['id'], official_raw_category=row['raw_category'],
                    official_category=row['category'], synset=row['wnsynsetid'], source_sha256=descriptor['sha256'])
                if evidence not in aliases[name]:
                    aliases[name].append(evidence)
    return dict(aliases), descriptor


def _scannet_inventory(scenes, roots):
    suffixes = ('.sens', '_vh_clean_2.ply', '.aggregation.json', '_vh_clean_2.0.010000.segs.json',
                '_vh_clean_2.labels.ply', '.txt')
    inventory = {}
    for scene in scenes:
        candidates = [Path(root) / prefix / scene for root in roots for prefix in ('scans', '')]
        files = {}
        for suffix in suffixes:
            found = next((folder / (scene + suffix) for folder in candidates if (folder / (scene + suffix)).is_file()), None)
            files[suffix] = dict(path=str(found), bytes=found.stat().st_size) if found else None
        inventory[scene] = dict(files=files, missing=[suffix for suffix, value in files.items() if value is None])
    return inventory


def _label_roles(binding, context, index):
    resolver = Resolver(binding['path_map'])
    roles_path = binding['structure_documents'].get('role_lock.json')
    if not roles_path:
        return dict(status='BLOCKED_DEPENDENCY', reason='STRUCT_ROLE_LOCK_ABSENT', components=[])
    roles = verified(roles_path)['roles']
    classes = verified(binding['structure_documents']['class_split.json'])
    components = [dict(component='MV_ATTN', gradient_scenes=roles['TRAIN'], selection_scenes=roles['DEV'],
        metric_only_scenes=roles['CONF'] + roles['REG'], training_class_ids=classes['base_ids'],
        evidence=[index.identity(roles_path), index.identity(binding['structure_documents']['split_manifest.json'])],
        label_role='BASE_ONLY_GRADIENT;CONF12_NOVEL_LABEL_CALIBRATION_EXPOSED')]
    if context:
        checkpoint = resolver.rewrite(context['checkpoint'])
        query_root = Path(checkpoint['path']).parent
        calibration_path = query_root / 'calibration_receipt.json'
        select_path = query_root / 'select_receipt.json'
        if calibration_path.is_file() and select_path.is_file():
            calibration = read(calibration_path)
            if calibration['checkpoint']['sha256'] != checkpoint['sha256']:
                raise ValueError('Q training receipt is not for the actual inference checkpoint')
            index.identity(checkpoint['path'], checkpoint)
            acquire = next((r for r in calibration['inputs'] if Path(r['path']).name == 'acquisition_lock.json'), None)
            if acquire is None:
                raise ValueError('Q training scenes have no bound acquisition lock')
            acquire = resolver.rewrite(acquire)
            index.identity(acquire['path'], acquire)
            selected = read(acquire['path'])['selected']
            split = {role: [r['scene_id'] for r in selected if r['role'] == role]
                     for role in ('fit', 'cal', 'select', 'confirm')}
            source = next(r for r in calibration['sources'] if Path(r['path']).name == 'query_pipeline.py')
            source = resolver.rewrite(source)
            index.identity(source['path'], source)
            components.append(dict(component='Q_GAIN', gradient_scenes=split['fit'],
                selection_scenes=split['cal'] + split['select'], metric_only_scenes=split['confirm'],
                training_class_ids='original_query_training_target_vocabulary', checkpoint=checkpoint,
                evidence=[index.identity(calibration_path), index.identity(select_path), acquire, source],
                label_role='FIT_GRADIENT;CAL_EARLY_STOP;SELECT_COMPARATOR;CONFIRM_METRIC_ONLY'))
        else:
            components.append(dict(component='Q_GAIN', gradient_scenes=None, selection_scenes=None,
                                   metric_only_scenes=None, label_role='UNKNOWN', missing=[str(calibration_path), str(select_path)]))
        runtime = context['runtime']
        config = Path(runtime['cropformer_config'])
        components.append(dict(component='CropFormer_hornet_3x_03823a', gradient_scenes=[],
            selection_scenes=[], metric_only_scenes=[], training_dataset_declaration='EntityV2 entity_train_01/02/03',
            dataset_membership_audit='DECLARED_EXTERNAL_DATASET;IMAGE_MEMBERSHIP_NOT_INDEPENDENTLY_AUDITED',
            evidence=[index.identity(config)], checkpoint_path=runtime['cropformer_weights'],
            label_role='EXTERNAL_PRETRAINED_PROPOSAL;NO_LOCAL_TRAINING'))
        for name in ('FrozenFCBase', 'native_SigLIP'):
            components.append(dict(component=name, gradient_scenes=[], selection_scenes=[], metric_only_scenes=[],
                label_role='FROZEN_FOUNDATION_PRETRAINING;NO_LOCAL_TRAINING',
                original_pretraining_sample_membership='UNKNOWN',
                evidence='bound parent FC/native physical model manifests; not a claim of internet pretraining decontamination'))
    return dict(status='DONE', components=components, roles=dict(
        gradient_training='labels affect optimizer updates', selection='labels select recipe/checkpoint/hyperparameter',
        metric_only='reported metrics have been observed', unknown='role provenance unavailable'),
        current_CONF12='ALL_EXPOSED;NOVEL_LABELS_USED_FOR_CALIBRATION_SELECTION',
        LOFO_independent_confirmation=False, original_pretraining_decontamination_proven=False)


def build(binding):
    root = Path(binding['output_root'])
    index = ConsumedInputs(root)
    resolver = Resolver(binding['path_map'])
    assets = binding['asset_config']
    external = root / 'assets/external'
    datasets = assets.get('datasets', {})
    scan_assets, replica_assets = datasets.get('scannet200_val', {}), datasets.get('replica48', {})
    paths = dict(
        split_scan=scan_assets.get('official_split_path') or external / 'Open3DIS/open3dis/dataset/scannetv2_val.txt',
        split_scan_check=external / 'OpenMask3D/openmask3d/evaluation/val_scenes_scannet200.txt',
        split_replica=replica_assets.get('official_eight_scene_list') or external / 'Open3DIS/open3dis/dataset/replica_8scenes.txt',
        classes_replica=replica_assets.get('class48_mapping_path') or external / 'Open3DIS/open3dis/dataset/replica.py',
        evaluator_scan=scan_assets.get('official_evaluator_path') or external / 'OpenMask3D/openmask3d/evaluation/eval_semantic_instance.py',
        evaluator_replica=replica_assets.get('official_evaluator_path') or external / 'Open3DIS/open3dis/evaluation/scannetv2_inst_eval.py')
    present = {key: Path(resolver.resolve(value)) for key, value in paths.items()}
    sources = {key: index.identity(path) for key, path in present.items() if path.is_file()}
    inherited = binding['inherited']
    context = None
    if inherited and inherited['scenes']:
        first = next(iter(inherited['scenes'].values()))
        context_path = resolver.resolve(first['context']['path'])
        context = resolver.rewrite(verified(context_path))
        index.identity(context_path)
    classes_path = binding['structure_documents'].get('class_split.json')
    classes = verified(classes_path) if classes_path else verified(Path(binding['semantic_root']) / 'A/import.json')['class_split']
    if classes_path:
        index.identity(classes_path)
    ids = classes.get('official_ids')
    names = classes.get('official_names')
    if ids is None or names is None:
        raise StageFailure('Official training semantic ID/name axis is absent', missing=[str(classes_path)])
    tsv = scan_assets.get('label_mapping_path')
    if not tsv and binding['structure_root']:
        tsv = Path(binding['structure_root']) / 'confirmation/raw/scannetv2-labels.combined.tsv'
    aliases, alias_source = _aliases(tsv, ids, names, index)
    roles = _label_roles(binding, context, index)
    scopes, overlap = {}, {}
    for benchmark in ('scannet200_val', 'replica48'):
        required = ['split_scan', 'evaluator_scan'] if benchmark == 'scannet200_val' else ['split_replica', 'classes_replica', 'evaluator_replica']
        missing = [str(present[name]) for name in required if name not in sources]
        if missing:
            scopes[benchmark] = dict(status='BLOCKED_DEPENDENCY', missing=missing, required_sources=required)
            continue
        scenes = present['split_scan' if benchmark == 'scannet200_val' else 'split_replica'].read_text().split()
        expected_count = binding['specification']['benchmarks'][benchmark]['expected_scenes']
        if len(scenes) != expected_count or len(set(scenes)) != expected_count:
            raise ValueError('Official split has unexpected count or duplicate scene: ' + benchmark)
        if benchmark == 'scannet200_val':
            if 'split_scan_check' in sources and set(present['split_scan_check'].read_text().split()) != set(scenes):
                raise ValueError('Two pinned official ScanNet val lists disagree')
            instance_ids = _literal(present['evaluator_scan'], 'VALID_CLASS_IDS', size=198)
            instance_names = _literal(present['evaluator_scan'], 'CLASS_LABELS', size=198)
            if {int(k): v for k, v in zip(ids, names, strict=True) if k in instance_ids} != dict(zip(instance_ids, instance_names, strict=True)):
                raise ValueError('Official instance vocabulary differs from the 200-class text mapping')
            target_ids, target_names = ids, names
            target_order = 'official *_vh_clean_2.ply vertex order'
        else:
            target_names = _literal(present['classes_replica'], 'INSTANCE_CAT_REPLICA', size=48)
            target_ids = list(range(1, 49))
            instance_ids, instance_names = target_ids, target_names
            target_order = 'author replica_3d/{scene}.pth original coords vertex order'
        mapping = map_training_roles(target_names, ids, names, classes['base_ids'], aliases)
        membership = set(scenes)
        overlap[benchmark] = []
        for component in roles['components']:
            overlap[benchmark].append(dict(component=component['component'], **{
                key: sorted(membership.intersection(component[key])) if component.get(key) is not None else None
                for key in ('gradient_scenes', 'selection_scenes', 'metric_only_scenes')}))
        trained = [r for r in overlap[benchmark] if r['gradient_scenes']]
        unknown = [r['component'] for r in overlap[benchmark] if r['gradient_scenes'] is None]
        scopes[benchmark] = dict(status='DONE', scenes=scenes, families=sorted({s.split('_')[0] if s.startswith('scene') else s for s in scenes}),
            expected_scenes=expected_count, text_ids=target_ids, text_names=target_names,
            instance_ids=instance_ids, instance_names=instance_names, instance_classes=len(instance_ids),
            excluded_text_classes=[dict(id=i, name=n) for i, n in zip(target_ids, target_names, strict=True) if i not in instance_ids],
            target_point_order=target_order, GT_format='semantic_id*1000+instance_id; full target point domain',
            minimum_region_vertices=100, IoU_thresholds=_overlaps(present['evaluator_scan' if benchmark == 'scannet200_val' else 'evaluator_replica'], benchmark),
            threshold_source=sources['evaluator_scan' if benchmark == 'scannet200_val' else 'evaluator_replica'],
            AP_definition='official mean over .50,.55,.60,.65,.70,.75,.80,.85,.90; AP25 separate',
            mapping=mapping, full_AC_available=mapping['base_mask'] is not None,
            clean_gradient_audit='BLOCKED_DEPENDENCY' if trained or unknown else 'DONE',
            gradient_overlap_components=trained, unknown_gradient_role_components=unknown,
            requires_clean_training=bool(trained), clean_training_not_run=True,
            historical_metric_exposure=[r for r in overlap[benchmark] if r['metric_only_scenes']],
            confidence_ours=1., baseline_confidence='PRESERVE_AUTHOR_NATIVE_SCORE',
            source_frame_rule=binding['specification']['standard']['frame_rule'], F_views_max=8,
            immutable_owner_domain='ALL_ORIGINAL_OVI_NATIVE_OWNERS;NO_G1_RECOVERY',
            missing_F_MF_off='ABSTAIN0;NQ_NEVER_ACCESSED', missing_F_MF_on='AVAILABLE_NQ_ONLY')
    legacy = {}
    if context:
        old_names = context['models']['native']['class_names']
        legacy = dict(replica51=map_training_roles(old_names, ids, names, classes['base_ids'], aliases),
                      old51_text_names=old_names, old51_text_ids=list(range(1, len(old_names) + 1)))
        if 'replica48' in scopes and 'text_names' in scopes['replica48']:
            new_names = scopes['replica48']['text_names']
            legacy['class_name_difference'] = dict(
                old51_only=sorted(set(map(normalize_name, old_names)) - set(map(normalize_name, new_names))),
                new48_only=sorted(set(map(normalize_name, new_names)) - set(map(normalize_name, old_names))),
                common=len(set(map(normalize_name, old_names)) & set(map(normalize_name, new_names))),
                reusing_old51_AP_as_new48_forbidden=True)
    roots = list((inherited or {}).get('data_roots', []))
    if scan_assets.get('raw_root'):
        roots.insert(0, scan_assets['raw_root'])
    inventory = _scannet_inventory(scopes.get('scannet200_val', {}).get('scenes', []), roots)
    inventory_record = write(root / 'protocol/scannet_asset_inventory.json', dict(status='DONE', scenes=inventory,
        checked_roots=roots, complete_raw_scenes=sum(not row['missing'] for row in inventory.values()),
        labels_opened=False, search='finite official scene names in exact authorized roots, no historical recursive search'))
    result = write(root / 'protocol_lock.json', dict(status='DONE', producer=source_identity(('protocol',)),
        official_sources=sources, semantic_alias_source=alias_source, scopes=scopes, legacy=legacy,
        component_roles=roles, overlap=overlap, asset_inventory=inventory_record['identity'],
        input_identity=canonical_digest(dict(binding=binding['input_identity'], sources=sources, alias_source=alias_source)),
        current_calibration_exposure='CONF12_ALREADY_EXPOSED;NOVEL_LABELS_USED;EXPLORATORY_LOFO',
        GT_semantic_labels_read=False, new_training_updates=0))
    index.save()
    return result
