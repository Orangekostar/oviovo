"""Bounded authorized discovery and pre-outcome family/category separation."""
import ast
import hashlib
import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor

from static_ovmap.module_validation.scannet_download import DownloaderLayout, remote_identity, transfer
from .common import ConsumptionIndex, read, verified, write

RAW_SUFFIXES = ('.sens', '.txt', '_vh_clean_2.ply', '_vh_clean_2.0.010000.segs.json', '.aggregation.json')


def family_id(scan):
    if not re.fullmatch(r'scene\d{4}_\d{2}', scan):
        raise ValueError(f'Invalid ScanNet ID: {scan}')
    return scan.split('_')[0]


def ordered_families(scans, denied):
    families = {}
    for scan in scans:
        family = family_id(scan)
        if family not in denied:
            families.setdefault(family, []).append(scan)
    order = sorted(families, key=lambda f: (hashlib.sha256(('LR1-family-20261009|' + f).encode()).hexdigest(), f))
    return [(f, sorted(families[f], key=lambda s: int(s.rsplit('_', 1)[1]))) for f in order]


def family_split(scans, denied):
    choices = [members[0] for _, members in ordered_families(scans, set(denied))]
    if len(choices) < 32:
        raise ValueError('BLOCKED_TRAINING_DATA: need32 independent complete families')
    return dict(train=choices[:24], dev=choices[24:28], holdout=choices[28:32])


def class_split(ids):
    ids = list(map(int, ids))
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError('Exact200 distinct ScanNet IDs required')
    ranked = sorted(ids, key=lambda i: (hashlib.sha256(('LR1-class-20261009|' + str(i)).encode()).hexdigest(), i))
    novel = set(ranked[:40])
    return [i for i in ids if i not in novel], [i for i in ids if i in novel]


def inventory_roots(roots):
    scans, inspected = {}, []
    for root in map(Path, roots):
        for directory in (root, root / 'scans', root / 'scans_train'):
            if not directory.is_dir():
                continue
            inspected.append(str(directory.resolve()))
            for path in sorted(directory.iterdir()):
                if not path.is_dir() or not re.fullmatch(r'scene\d{4}_\d{2}', path.name):
                    continue
                files = {suffix: str(path / (path.name + suffix)) for suffix in RAW_SUFFIXES}
                missing = [suffix for suffix, name in files.items() if not Path(name).is_file() or Path(name).stat().st_size == 0]
                item = dict(directory=str(path.resolve()), complete=not missing, files=files, missing=missing)
                if path.name not in scans or item['complete']:
                    scans[path.name] = item
    return dict(scans=scans, roots=list(map(str, roots)), inspected_immediate_directories=inspected,
                annotation_files_parsed=0, recursive_disk_search=False)


def official_classes(path):
    tree = ast.parse(Path(path).read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('VALID_CLASS_IDS_200', 'CLASS_LABELS_200'):
                    values[target.id] = ast.literal_eval(node.value)
    ids, names = values['VALID_CLASS_IDS_200'], values['CLASS_LABELS_200']
    if len(ids) != 200 or len(names) != 200:
        raise ValueError('Pinned official ScanNet200 constants incomplete')
    return list(map(int, ids)), list(names)


def materialize(binding):
    root = Path(binding['output_root'])
    spec = binding['specification']
    upstream = root / 'external/upstream_sources/ScanNet'
    train = upstream / 'Tasks/Benchmark/scannetv2_train.txt'
    constants = upstream / 'BenchmarkScripts/ScanNet200/scannet200_constants.py'
    index = ConsumptionIndex(root / 'input_verifications.json')
    index.identity(train); index.identity(constants)
    ids, names = official_classes(constants)
    base, novel = class_split(ids)
    write(root / 'class_split.json', dict(status='LOCKED', official_ids=ids, official_names=names,
          base_ids=base, heldout_ids=novel, positive_supervision='BASE_ONLY',
          constants=index.identity(constants), inherited_text_mapping='VERIFY_AT_FEATURE_BINDING'))
    prior = Path(binding['prior_data_root'])
    lock = read(prior / 'acquisition_lock.json')
    if lock.get('authorization') != 'User explicitly confirmed prior ScanNet authorization and terms agreement':
        raise PermissionError('Existing ScanNet authorization does not match reviewed authority')
    denied = set(spec['data']['deny_families'])
    denied.update(family_id(row['scene_id']) for row in lock['selected'])
    # Any additional physically named scan in inherited context/lineage is prior exposure.
    for scene in binding['scenes']:
        if scene.startswith('scene'):
            denied.add(family_id(scene))
    for item in binding['reference'].get('lineage_inputs', []):
        denied.update(re.findall(r'scene\d{4}(?=_\d{2})', item.get('path', '')))
    scans = [line.strip() for line in train.read_text().splitlines() if line.strip()]
    inventory = inventory_roots(binding['data_roots'])
    write(root / 'data_inventory.json', dict(inventory, status='INVENTORIED', exclusions=sorted(denied),
          official_train_file=index.identity(train), usable_new_families=len({family_id(s) for s, v in inventory['scans'].items() if v['complete'] and family_id(s) not in denied})))
    plan_path = root / 'data/acquisition_plan.json'
    candidates = ordered_families(scans, denied)
    if len(candidates) < 32:
        raise ValueError('BLOCKED_TRAINING_DATA: official TRAIN queue lacks32 new families')
    numerical_order = [dict(family=f, captures=members) for f, members in candidates]
    if plan_path.exists():
        plan = verified(plan_path)
        if plan['candidate_queue'] != numerical_order or plan['exclusions'] != sorted(denied):
            raise ValueError('Frozen acquisition queue changed')
    else:
        plan = write(plan_path, dict(status='PRE_OUTCOME_QUEUE_LOCKED', candidate_queue=numerical_order,
                     exclusions=sorted(denied), source_train=index.identity(train),
                     annotation_selection=False, target_family_count=32))
    split_path = root / 'split_manifest.json'
    if split_path.exists():
        previous = verified(split_path)
        if (previous['candidate_queue'] != plan['identity'] or previous['exclusions'] != sorted(denied)
                or previous['roles'] != family_split([r['scene'] for r in previous['selected']], denied)):
            raise ValueError('Completed fixed scene roles or acquisition queue changed')
        for row in previous['selected']:
            for suffix, path in row['files'].items():
                expected = row.get('download_receipts', {}).get(suffix)
                index.identity(path, {'sha256': expected['sha256']} if expected else None)
        index.write_memo(root / 'input_verifications.json')
        return previous
    acquisition = read(prior / 'acquisition_plan.json')
    script = Path(acquisition['layout']['script_path'])
    index.identity(script, {'sha256': acquisition['layout']['script_sha256']})
    layout = DownloaderLayout.from_script(script)
    target = root / 'data/scannet'
    label_source = prior / 'scannetv2-labels.combined.tsv'
    index.identity(label_source)
    target.mkdir(parents=True, exist_ok=True)
    label_target = target / label_source.name
    if label_target.exists() and label_target.read_bytes() != label_source.read_bytes():
        raise ValueError('Official label TSV changed')
    if not label_target.exists():
        label_target.write_bytes(label_source.read_bytes())
    selected, rejected = [], []
    for family, members in candidates:
        for scan in members:
            item = inventory['scans'].get(scan)
            if item and item['complete']:
                selected.append(dict(scene=scan, family=family, directory=item['directory'], files=item['files'], source='AUTHORIZED_LOCAL'))
                break
            files = {suffix: target / 'scans' / scan / (scan + suffix) for suffix in RAW_SUFFIXES}
            try:
                # Bounded four-way transfer; no GT annotations are parsed here.
                with ThreadPoolExecutor(max_workers=4) as pool:
                    futures = {suffix: pool.submit(transfer, layout.scene_url(scan, suffix), path) for suffix, path in files.items()}
                    results = {}
                    errors = []
                    for suffix, future in futures.items():
                        try:
                            results[suffix] = future.result()
                        except Exception as exc:
                            errors.append(exc)
                    if errors:
                        raise errors[0]
                selected.append(dict(scene=scan, family=family, directory=str(files['.sens'].parent),
                                     files={s: str(p) for s, p in files.items()}, source='AUTHORIZED_FIXED_QUEUE_DOWNLOAD', download_receipts=results))
                break
            except Exception as exc:
                import urllib.error
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                    rejected.append(dict(scene=scan, reason='REQUIRED_REMOTE_FILE_404'))
                    continue
                write(root / 'data/materialization_failure.json', dict(status='BLOCKED_TRAINING_DATA', scene=scan,
                      error=repr(exc), selected=selected, rejected=rejected, acquisition_identity=plan['identity']))
                raise RuntimeError('BLOCKED_TRAINING_DATA: ' + repr(exc)) from exc
        if len(selected) == 32:
            break
    if len(selected) < 32:
        raise ValueError('BLOCKED_TRAINING_DATA: fewer than32 complete families')
    roles = family_split([r['scene'] for r in selected], denied)
    result = write(root / 'split_manifest.json', dict(status='LOCKED_FILES_BEFORE_ANNOTATION_GENERATION',
                   roles=roles, selected=selected, exclusions=sorted(denied), rejected=rejected,
                   candidate_queue=plan['identity'], data_authority=binding['scan_authority'],
                   holdout_annotation_reads=0, supervision_scope='GT_DERIVED_PROPOSAL_RECOGNITION'))
    index.write_memo(root / 'input_verifications.json')
    print('DATA MATERIALIZED', {role: len(values) for role, values in roles.items()}, flush=True)
    return result
