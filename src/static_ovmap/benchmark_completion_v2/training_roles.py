"""Explicit semantic evidence for binary exposure roles, without GT or scores."""
from copy import deepcopy
import csv
from pathlib import Path

from .common import Resolver, source_identity, verified
from .protocol import normalize_name


def manifest_path(binding):
    value = binding['asset_config'].get('datasets', {}).get('replica48', {}).get('training_role_aliases_path')
    return Path(Resolver(binding.get('path_map')).resolve(value)) if value else None


def load(binding, training_ids, base_ids, taxonomy, index):
    path = manifest_path(binding)
    if path is None:
        return {}, None
    descriptor = index.identity(path)
    manifest = verified(path)
    if (manifest['policy'] != 'ANY_BASE_IN_CONFIRMED_SYNONYM_GROUP'
            or manifest['GT_labels_used'] is not False
            or manifest['training_axis'] != dict(ids=training_ids, base_ids=base_ids)):
        raise ValueError('Role manifest changed the frozen training axis or semantic policy')
    resolver = Resolver(binding.get('path_map'))
    inputs = {}
    for name in ('taxonomy', 'crosswalk'):
        declared = manifest[name]
        inputs[name] = index.identity(resolver.resolve(declared['path']), declared)
    if inputs['taxonomy']['sha256'] != taxonomy['sha256']:
        raise ValueError('Role evidence must use the frozen official global taxonomy')
    with Path(inputs['taxonomy']['path']).open() as stream:
        global_ids = {int(row['id']) for row in csv.DictReader(stream, delimiter='\t')}
    with Path(inputs['crosswalk']['path']).open() as stream:
        crosswalk = list(csv.DictReader(stream))
    rows = {}
    for row in manifest['rows']:
        key = normalize_name(row['target_name'])
        ids = row['semantic_ids']
        if (key in rows or not ids or any(type(i) is not int for i in ids)
                or ids != sorted(set(ids)) or not set(ids) <= global_ids):
            raise ValueError('Role evidence has duplicate names or invalid global semantic IDs')
        manual = row['user_confirmed_synonyms']
        named = [r for r in crosswalk if key in
                 {normalize_name(n) for n in r['replicaclass'].split(',') if n.strip()}]
        # These are raw Replica taxonomy IDs, not the evaluator's 1..48 IDs.
        replica_ids = set()
        for source in named:
            names = [normalize_name(n) for n in source['replicaclass'].split(',') if n.strip()]
            labels = [int(n.strip()) for n in source['replicaid'].split(',') if n.strip()]
            if len(names) != len(labels):
                raise ValueError('Explicit Replica name/ID crosswalk axis is ambiguous')
            replica_ids.update(label for name, label in zip(names, labels, strict=True) if name == key)
        sourced = {int(r['id']) for r in crosswalk if r['id'] and
            (r in named or replica_ids.intersection(
                int(n.strip()) for n in r['replicaid'].split(',') if n.strip()))}
        if type(manual) is not bool or (manual and not manifest.get('user_confirmation')):
            raise ValueError('Manual synonym roles require an explicit user confirmation')
        if (not manual or sourced) and set(ids) != sourced:
            raise ValueError('Role IDs differ from the complete explicit Replica crosswalk: ' + row['target_name'])
        roles = {i in base_ids for i in ids}
        if len(roles) > 1 and not manual:
            raise ValueError('Conflicting roles require a confirmed synonym group')
        rows[key] = dict(source_semantic_ids=ids,
            candidate_training_ids=[i for i in ids if i in training_ids], base=any(roles),
            role_evidence=dict(manual_synonym_group=manual, original_roles_conflict=len(roles) > 1,
                policy=manifest['policy'], manifest_identity=manifest['identity'],
                user_confirmation=manifest.get('user_confirmation') if manual else None))
    return rows, dict(descriptor, inputs=inputs, producer=source_identity(('training_roles',)))


def extend(mapping, roles):
    if not roles:
        return mapping
    result = deepcopy(mapping)
    for row in result['rows']:
        evidence = roles.get(row['normalized_name'])
        if evidence is None:
            continue
        if row['status'] == 'DONE':
            if row['base'] != evidence['base']:
                raise ValueError('Explicit role evidence contradicts a resolved training role')
            continue
        row.update(evidence, status='DONE')
    result['unresolved'] = [r['target_name'] for r in result['rows'] if r['status'] != 'DONE']
    result['status'] = 'BLOCKED_DEPENDENCY' if result['unresolved'] else 'DONE'
    result['base_mask'] = None if result['unresolved'] else [r['base'] for r in result['rows']]
    return result
