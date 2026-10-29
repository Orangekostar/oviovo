"""Finite, path-redacted derived export; raw RGB/GT/meshes and weights stay local."""
import gzip
import hashlib
import json
from pathlib import Path
import re

from .common import REPO, ARTIFACTS, write, verified, digest_file, canonical_digest


def redact(value, binding):
    roots = {str(binding.get(key, '')): token for key, token in (
        ('output_root', '$OUT'), ('semantic_root', '$SEMANTIC_ROOT'),
        ('structure_root', '$STRUCTURE_ROOT')) if binding.get(key)}
    roots[str(REPO)] = '$REPO'
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            safe_key = redact(key, binding)
            if safe_key != key and '$LOCAL_ASSET/' in safe_key:
                safe_key += '#' + hashlib.sha256(key.encode()).hexdigest()
            if safe_key in result:
                raise ValueError('Redacted dictionary keys collide')
            result[safe_key] = ('REDACTED' if re.search(r'(^|_)(token|password|secret|credential)($|_)', key, re.I)
                                else redact(item, binding))
        return result
    if isinstance(value, list):
        return [redact(item, binding) for item in value]
    if not isinstance(value, str):
        return value
    for source, token in sorted(roots.items(), key=lambda row: -len(row[0])):
        value = value.replace(source, token)
    return re.sub(r'(?:(?<!\w)|(?<=-I)|(?<=-L)|(?<=-B)|(?<=-isystem)|(?<=-iquote)|(?<=-idirafter)|(?<=-isysroot))/(?:home|mnt|tmp|goal|opt|usr)/[^:\s"\'<>`\)\],;]+',
                  lambda match: '$LOCAL_ASSET/' + Path(match[0]).name, value)


def decode_record(payload):
    return json.loads(gzip.decompress(payload))


def _execution_provenance_files(root):
    """Validate only explicitly declared task-local source, receipts and logs."""
    manifest = root/'publication/execution_provenance.json'
    if not manifest.exists() and not manifest.is_symlink():
        return {}

    def regular(relative):
        cursor = root
        for part in relative.parts:
            cursor = cursor/part
            if cursor.is_symlink():
                raise ValueError('Execution provenance cannot follow symlinks: '+str(relative))
        if not cursor.is_file():
            raise ValueError('Declared execution provenance file is missing: '+str(relative))
        return cursor

    value = verified(regular(Path('publication/execution_provenance.json')))
    if (value.get('status') != 'DONE' or value.get('schema') != 'OVMAP_EXECUTION_PROVENANCE_V1'
            or not isinstance(value.get('files'), list) or not value['files']):
        raise ValueError('Execution provenance requires a complete explicit manifest')
    suffixes = dict(PYTHON_SOURCE='.py', SEALED_RECEIPT='.json', EXECUTION_LOG='.log')
    selected = {}
    for row in value['files']:
        if not isinstance(row, dict) or not isinstance(row.get('source'), str):
            raise ValueError('Execution provenance entry requires a relative source path')
        relative = Path(row['source'])
        if (relative.is_absolute() or '..' in relative.parts or len(relative.parts) < 2
                or relative.parts[0] != 'bootstrap' or str(relative) != row['source']
                or row.get('kind') not in suffixes or relative.suffix != suffixes[row['kind']]
                or row['source'] in selected or type(row.get('bytes')) is not int or row['bytes'] < 0
                or not isinstance(row.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', row['sha256'])):
            raise ValueError('Execution provenance path, kind or descriptor is invalid: '+row['source'])
        path = regular(relative)
        if path.stat().st_size != row['bytes'] or digest_file(path) != row['sha256']:
            raise ValueError('Execution provenance bytes changed: '+row['source'])
        if row['kind'] == 'SEALED_RECEIPT' and verified(path)['identity'] != row.get('identity'):
            raise ValueError('Execution provenance receipt identity differs: '+row['source'])
        selected[row['source']] = row
    return selected


def candidates(root):
    """Only known task result layouts; never recurse into datasets or other tasks."""
    names = ['protocol_lock.json', 'model_lock.json', 'simple_recipe_lock.json',
        'result_store.json', 'tables.json', 'cell_provenance.jsonl', 'remaining_jobs.json',
        'resource_ledger.json', 'cost_estimate.json', 'job_manifest.json',
        'calibration/summary.json', 'calibration/candidates.json.gz',
        'calibration/crossfit.json.gz', 'calibration/predictions.npz',
        'A/import.json', 'A/import.npz', 'A/restoration.json',
        'publication/readout_replay.json', 'publication/class_scores.json', 'publication/legacy_class_scores.json',
        'publication/execution_provenance.json',
        'legacy/prediction_lock.json', 'legacy/score.json',
        'external/author_reported.json', 'external/recipes.json',
        'resources/hardware.json', 'review/coverage.json', 'review/primary_final.json']
    patterns = ['tables/*.md', 'tables/*.tex', 'tables/*_preview.pdf', 'tables/*_preview.png',
        'tables/*_compile.log', 'statistics/*.json', 'statistics/*.npz', 'supplement/*.json',
        'cases/*/*/data.json', 'cases/*/*/receipt.json', 'cases/*/*/caption.md',
        'cases/*/*/figure.pdf', 'cases/*/*/figure.png',
        'resources/capacity/*/*.json', 'resources/storage/*/*.json',
        'readout_cache/*.npz', 'readout_cache/*.json',
        'publication/readout_inputs/*.json', 'publication/readout_inputs/*.npz',
        'publication/class_scores/*/manifest.json', 'publication/class_scores/*/scores.npz',
        'publication/legacy_class_scores/*/manifest.json', 'publication/legacy_class_scores/*/scores.npz',
        'legacy/inputs/*/verifications.json', 'legacy/predictions/*/*/manifest.json',
        'legacy/predictions/*/*/*.npz', 'legacy/predictions/*/receipt.json',
        'legacy/evaluation/*/*_matches.json.gz', 'legacy/pools/*/*.json',
        'benchmark/F/*/*/vectors.npz', 'benchmark/F/*/*/receipt.json',
        'benchmark/NQ/*/*/query_scores.npz', 'benchmark/NQ/*/*/target_text.npz',
        'benchmark/NQ/*/*/sources.json', 'benchmark/NQ/*/*/native_query_receipt.json',
        'benchmark/predictions/*/*/receipt.json',
        'benchmark/predictions/*/*/*/manifest.json', 'benchmark/predictions/*/*/*/*.npz',
        'benchmark/evaluation/*/*/receipt.json', 'benchmark/evaluation/*/*/*_matches.json.gz',
        'benchmark/evaluation/*/*/G3_official_entry_agreement.json', 'benchmark/pools/*/*.json',
        'external/*/recipe.json', 'external/*/pools/*.json',
        'external/*/predictions/*/*/manifest.json', 'external/*/predictions/*/*/*.npz',
        'external/*/evaluation/*/*/receipt.json', 'external/*/evaluation/*/*/*_matches.json.gz',
        'external/*/evaluation/*/*/matches.json.gz',
        'external/MaskClustering/evaluation/replica48/*/G3_official_entry_agreement.json',
        'external/MaskClustering/pipeline/replica48/*/receipt.json',
        'external/MaskClustering/pipeline/replica48/*/availability.json',
        'external/MaskClustering/pipeline/replica48/*/blocked.json',
        'external/MaskClustering/execution_replica48.json',
        'external/MaskClustering/GT/replica48/*/receipt.json',
        'external/OpenMask3D/evaluation/replica48/*/G3_official_entry_agreement.json',
        'external/OpenMask3D/pipeline/replica48/*/receipt.json',
        'external/OpenMask3D/pipeline/replica48/*/availability.json',
        'external/OpenMask3D/pipeline/replica48/*/blocked.json',
        'external/OpenMask3D/execution_replica48.json',
        'external/OpenMask3D/GT/replica48/*/receipt.json',
        'external/OpenMask3D/geometry/replica48/*.json',
        'external/replica_rgbd/*/layout.json',
        'diagnostics/*/*/receipt.json', 'diagnostics/*/summary.json',
        'review/G*.json', 'review/visual*.json', 'review/G3_OVI_author_format/*/*/receipt.json',
        'review/score_recovery/*/*.json',
        'jobs/benchmark/*/*/*/receipt.json', 'jobs/external/MaskClustering/replica48/*/*/receipt.json',
        'jobs/external/OpenMask3D/replica48/*/*/receipt.json']
    selected = {root/name for name in names if (root/name).is_file()}
    for pattern in patterns:
        selected.update(p for p in root.glob(pattern) if p.is_file())
    selected.update(root/name for name in _execution_provenance_files(root))
    return sorted(selected)


def _is_remote_result_source(source):
    return source == 'result_store.json' or (
        source.startswith('benchmark/evaluation/') and source.endswith('/receipt.json'))


def export(binding, *, repo=REPO, blob_limit=40*2**20, git_limit=512*2**20,
           chunk_size=32*2**20):
    root, repo = Path(binding['output_root']), Path(repo)
    chunk_size = min(chunk_size, blob_limit)
    if min(chunk_size, git_limit) <= 0:
        raise ValueError('Publication size limits must be positive')
    declared = _execution_provenance_files(root)
    provenance_source = 'publication/execution_provenance.json'
    provenance_sha = digest_file(root/provenance_source) if declared else None
    provenance_identity = verified(root/provenance_source)['identity'] if declared else None
    from .readout_replay import prepare
    from .score_artifacts import prepare as prepare_scores
    from .legacy_score_artifacts import prepare as prepare_legacy_scores
    replay = prepare(binding)
    prepare_scores(binding, replay)
    prepare_legacy_scores(binding, replay)
    public, local = [], []
    used, stored = 0, {}
    for path in sorted(candidates(root), key=lambda path: (
            not _is_remote_result_source(str(path.relative_to(root))), str(path))):
        if path.is_symlink():
            raise ValueError('Derived publication cannot follow symlinks: ' + str(path))
        source = str(path.relative_to(root))
        original = path.read_bytes()
        original_sha = hashlib.sha256(original).hexdigest()
        if source in declared and (original_sha != declared[source]['sha256']
                or len(original) != declared[source]['bytes']):
            raise ValueError('Execution provenance changed during export: '+source)
        if source == provenance_source and original_sha != provenance_sha:
            raise ValueError('Execution provenance manifest changed during export')
        encoding = 'BINARY_UNCHANGED'
        if path.suffix == '.json' or path.name.endswith('.json.gz'):
            value = json.loads(gzip.decompress(original) if path.name.endswith('.gz') else original)
            original_identity = value.get('identity') if isinstance(value, dict) else None
            if original_identity and canonical_digest({k:v for k,v in value.items() if k!='identity'}) != original_identity:
                raise ValueError('Scientific export identity differs: '+source)
            envelope = dict(schema='PATH_REDACTED_RECORD_V1', original_identity=original_identity,
                            original_sha256=original_sha, record=redact(value, binding))
            payload = gzip.compress(json.dumps(envelope, ensure_ascii=False, sort_keys=True,
                                   allow_nan=False).encode(), mtime=0)
            encoding = 'GZIP_PATH_REDACTED_RECORD_V1'
        elif path.suffix in ('.jsonl','.md','.tex','.log','.py'):
            payload = gzip.compress(redact(original.decode(), binding).encode(), mtime=0)
            encoding = 'GZIP_PATH_REDACTED_TEXT'
        else:
            payload = original
        parts = []
        payload_sha = hashlib.sha256(payload).hexdigest()
        for start in range(0, len(payload), chunk_size):
            block = payload[start:start+chunk_size]
            block_sha = hashlib.sha256(block).hexdigest()
            if block_sha not in stored:
                in_git = used + len(block) <= git_limit
                name = block_sha + '.part'
                dest = repo/ARTIFACTS/'chunks'/name if in_git else root/'publication/release_assets'/name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    temp = dest.with_suffix('.tmp')
                    temp.write_bytes(block)
                    temp.replace(dest)
                if digest_file(dest) != block_sha:
                    raise ValueError('Immutable derived shard differs: ' + name)
                if in_git:
                    used += len(block)
                stored[block_sha] = dict(path=str(dest.relative_to(repo)) if in_git else name,
                    bytes=len(block), sha256=block_sha,
                    distribution='GIT' if in_git else 'LOCAL_RETAINED_RELEASE_PENDING')
            parts.append(dict(stored[block_sha], offset=start))
        row = dict(source=source, sha256=original_sha, original_bytes=len(original),
                   payload_sha256=payload_sha, encoding=encoding, parts=parts)
        public.append(row)
        local.append(dict(source=source, local_path=str(path), sha256=original_sha))
    record_data=dict(status='DONE', files=public,
        git_derived_bytes=used, git_derived_limit=git_limit, blob_limit=blob_limit,
        shard_size=chunk_size, original_record_identities_preserved=True,
        transformed_records_are_envelopes_not_original_sealed_records=True,
        raw_RGB_GT_mesh_weights_excluded=True,
        execution_provenance=None if not declared else dict(source=provenance_source,
            original_identity=provenance_identity, files=len(declared)),
        local_retention_instructions='Acquire original datasets/models using protocol source URLs and locks. '
          'The execution handoff resolves $OUT, $REPO, $SEMANTIC_ROOT and $STRUCTURE_ROOT. '
          'Concatenate parts by offset and verify payload_sha256; decode declared encoding. '
          'Use original_identity to resolve scientific record references across envelopes.')
    record=finalize_index(binding,record_data,repo=repo)
    write(root/'publication/local_artifacts.json', dict(status='DONE', files=local,
        public_index=record['identity'], release_parts_local_root=str(root/'publication/release_assets')))
    return record


def finalize_index(binding, record_data, *, repo=REPO):
    """Enforce the actual Git byte limit again after adding release URLs."""
    root,repo=Path(binding['output_root']),Path(repo)
    public=record_data['files']
    git_limit=record_data['git_derived_limit']
    stored={part['sha256']:part.copy() for row in public for part in row['parts']}
    used=sum(part['bytes'] for part in stored.values() if part['distribution']=='GIT')
    record_data['git_derived_bytes']=used
    index_path=repo/ARTIFACTS/'artifact_index.json'
    # Reuse a verified fixed point when exporting the same records again, so
    # intermediate size calculations do not rewrite an otherwise identical index.
    size_keys = ('git_manifest_bytes', 'git_total_derived_bytes')
    if index_path.is_file():
        previous = verified(index_path)
        comparable = lambda value: {key: item for key, item in value.items()
                                    if key not in ('identity', *size_keys)}
        if (comparable(previous) == comparable(record_data)
                and previous.get('git_manifest_bytes') == index_path.stat().st_size
                and previous.get('git_total_derived_bytes') == used + index_path.stat().st_size):
            record_data.update({key: previous[key] for key in size_keys})
    # Count the manifest itself, and prune only obsolete *export copies*. The
    # complete original scientific files in OUT are never removed or altered.
    while True:
        record=write(index_path,record_data)
        size=index_path.stat().st_size
        for _ in range(10):
            record_data.update(git_manifest_bytes=size,git_total_derived_bytes=used+size)
            record=write(index_path,record_data)
            current=index_path.stat().st_size
            if size==current:
                break
            size=current
        if used+size<=git_limit:
            break
        remaining=[row for row in stored.values() if row['distribution']=='GIT']
        if not remaining:
            raise ValueError('Derived manifest alone exceeds the configured Git export budget')
        part=remaining[-1]
        old=repo/part['path']
        dest=root/'publication/release_assets'/old.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(old.read_bytes())
        used-=part['bytes']
        for row in public:
            for item in row['parts']:
                if item['sha256']==part['sha256']:
                    item.update(path=dest.name,distribution='LOCAL_RETAINED_RELEASE_PENDING')
        part.update(path=dest.name,distribution='LOCAL_RETAINED_RELEASE_PENDING')
        record_data['git_derived_bytes']=used
    if index_path.stat().st_size>40*2**20:
        raise ValueError('Derived index needs a manifest shard before Git publication')
    retained={part['path'] for row in public for part in row['parts'] if part['distribution']=='GIT'}
    for path in (repo/ARTIFACTS/'chunks').glob('*.part'):
        if str(path.relative_to(repo)) not in retained:
            path.unlink()
    local_path=root/'publication/local_artifacts.json'
    if local_path.is_file():
        local=verified(local_path)
        local['public_index']=record['identity']
        write(local_path,local)
    return record
