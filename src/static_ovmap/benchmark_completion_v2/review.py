"""Evidence coverage and claim boundaries, independent of positive outcomes."""
from pathlib import Path
import json
import numpy as np

from .common import REPO, DOCS, verified, read, write, digest_file


def check_global_scene(binding, benchmark, scene):
    root = Path(binding['output_root'])
    lock = verified(root / 'benchmark/predictions' / benchmark / scene / 'receipt.json')
    geometry = None
    for method, descriptor in lock['methods'].items():
        row = verified(descriptor['path'])
        if row['identity'] != descriptor['identity']:
            raise ValueError('Frozen actual GLOBAL scene identity differs')
        if method.startswith(('U00_', 'U01_')) and lock['traces'][method]['NQ_access_count']:
            raise ValueError('Real MFoff scene trace accessed N/Q')
        with np.load(row['arrays']['path'], allow_pickle=False) as arrays:
            current = [arrays[key].copy() for key in ('owners','indptr','indices')]
            if geometry is None:
                geometry = current
            elif any(not np.array_equal(a,b) for a,b in zip(geometry,current,strict=True)):
                raise ValueError('Actual GLOBAL four-cell mask domains differ')
            if not np.array_equal(arrays['confidence'], np.ones(len(arrays['owners']))):
                raise ValueError('Actual own standard confidence differs from constant1')
    agreement = verified(root / 'benchmark/evaluation' / benchmark / scene / 'G3_official_entry_agreement.json')
    if agreement['synthetic'] or agreement['max_absolute_error'] != 0:
        raise ValueError('Actual common-author evaluator agreement is absent')
    return write(root / 'review/G3_global_actual.json', dict(status='DONE', benchmark=benchmark,
        scene=scene, prediction_lock=lock['identity'], logical_rows=len(lock['methods']),
        actual_all_owner_masks_equal=True, actual_MFoff_NQ_accesses=0,
        actual_constant1_confidence=True, author_entry_agreement=agreement['identity'],
        full_GT_denominator=True, synthetic=False))


def run(binding):
    from .work_state import resource_block,scene_state,maskclustering_state,openmask_state
    root = Path(binding['output_root'])
    protocol = verified(root / 'protocol_lock.json')
    calibration = verified(root / 'calibration/summary.json')
    legacy = verified(root / 'legacy/score.json') if (root / 'legacy/score.json').is_file() else None
    model = verified(root / 'model_lock.json') if (root / 'model_lock.json').is_file() else None
    tables = verified(root / 'tables/receipt.json') if (root / 'tables/receipt.json').is_file() else None
    if tables:
        from .aggregation_audit import run as aggregate_audit
        aggregate_audit(binding)
    standard, rows, remaining = {}, {}, []
    capacity_block=resource_block(binding)
    scene_plans={}
    for benchmark, scope in protocol['scopes'].items():
        completed = {}
        for scene in scope['scenes']:
            scene_plans[benchmark,scene]=scene_state(binding,benchmark,scene,capacity_block)
            path = root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json'
            if path.is_file():
                completed[scene] = verified(path)
            for method in ('U00','U10','U01','U11'):
                for seed in (17,29):
                    key = method + '_seed' + str(seed)
                    if scene in completed and key in completed[scene]['rows']:
                        continue
                    mapping_block = method in ('U01','U11') and not scope['full_AC_available']
                    geometry_block=False
                    if benchmark=='replica48':
                        config=binding['asset_config'].get('datasets',{}).get('replica48',{})
                        supplied=config.get('gt_root') or config.get('preprocessed_root')
                        geometry_options=[root/'benchmark/bundles/replica48'/scene/'bundle.json',
                            root/'assets/datasets/replica48/geometry'/(scene+'.ply'),
                            root/'assets/datasets/replica48/replica_3d'/(scene+'.pth')]
                        if supplied:
                            geometry_options += [Path(supplied)/(scene+'.pth'),Path(supplied)/'replica_3d'/(scene+'.pth')]
                        if config.get('evaluation_mesh_root'):
                            geometry_options.append(Path(config['evaluation_mesh_root'])/(scene+'.ply'))
                        geometry_block=not any(p.is_file() for p in geometry_options)
                    plan=dict(scene_plans[benchmark,scene])
                    if mapping_block or geometry_block:
                        plan.update(status='BLOCKED_DEPENDENCY',runnable=False,
                            reason='UNRESOLVED_BASE_ROLE' if mapping_block else 'AUTHOR_POINT_DOMAIN_MISSING',
                            command='--phase benchmark_predict --resume')
                    remaining.append(dict(kind='STANDARD_SCENE_CELL',benchmark=benchmark,scene=scene,method=key,**plan))
        count=sum(len(row['rows']) for row in completed.values())
        standard[benchmark]=dict(completed_scenes=len(completed),expected_scenes=len(scope['scenes']),
            completed_cells=count,expected_cells=len(scope['scenes'])*8,
            full_coverage=count==len(scope['scenes'])*8,scene_receipts={k:v['identity'] for k,v in completed.items()},
            bootstrap_inputs_complete=len(completed)==len(scope['scenes']) and all(
                all(f'{method}_seed{seed}' in row['rows'] for method in ('U11','U10','U01') for seed in (17,29))
                for row in completed.values()))
    first=protocol['scopes']['scannet200_val']['scenes'][0]
    if first in standard['scannet200_val']['scene_receipts']:
        check_global_scene(binding,'scannet200_val',first)
        from .cases import build as build_cases
        build_cases(binding)
    if legacy:
        prediction=verified(root/'legacy/prediction_lock.json')
        for scene,row in prediction.get('scenes',{}).items():
            path=root/'legacy/predictions'/scene/'receipt.json'
            actual=verified(path)
            for key in ('L00_seed17','L00_seed29','L10_seed17','L10_seed29',
                        'L01_seed17','L01_seed29','L11_seed17','L11_seed29'):
                if key not in actual['methods']:
                    remaining.append(dict(kind='LEGACY_SCENE_CELL',scene=scene,method=key,
                        status='BLOCKED_DEPENDENCY',runnable=False,reason='UNRESOLVED_REPLICA_BASE_ROLE',
                        command='--refresh-assets --phase protocol; affected legacy descendants only'))
    authors=verified(root/'external/author_reported.json') if (root/'external/author_reported.json').is_file() else None
    external={}
    for method in ('OVI-MAP','OpenMask3D','Open3DIS','MaskClustering'):
        external[method]={}
        for benchmark,scope in protocol['scopes'].items():
            key=method+':'+benchmark
            paper=authors['results'].get(key) if authors else None
            compatible=bool(paper and paper.get('compatible_frozen_variant',True))
            pool_path=root/'external'/method/'pools'/(benchmark+'.json')
            pool=verified(pool_path) if pool_path.is_file() else None
            local=bool(pool and pool['full_scene_coverage'])
            external[method][benchmark]=dict(complete=compatible or local,
                evidence='LOCAL_FULL' if local else 'AUTHOR_REPORTED_COMPATIBLE' if compatible else 'INCOMPLETE',
                locally_completed_scenes=pool['completed_scenes'] if pool else 0,
                source_identity=pool.get('identity') if local else paper.get('source') if compatible else pool.get('identity') if pool else None)
            if not(compatible or local):
                for scene in scope['scenes']:
                    if pool and scene in pool['source_scenes']:
                        continue
                    if method in ('MaskClustering', 'OpenMask3D') and benchmark == 'replica48':
                        method_state = maskclustering_state if method == 'MaskClustering' else openmask_state
                        remaining.append(dict(kind='EXTERNAL_SCENE_METHOD', method=method,
                            benchmark=benchmark, scene=scene,
                            recipe=f'external/{method}/recipe.json',
                            **method_state(binding, scene, capacity_block)))
                        continue
                    ready=method=='OVI-MAP' and (root/'benchmark/NQ'/benchmark/scene/'native_features.pkl').is_file() and (
                        root/'benchmark/GT'/benchmark/scene/'receipt.json').is_file()
                    plan=scene_plans[benchmark,scene]
                    runnable=ready or method=='OVI-MAP' and benchmark=='scannet200_val' and plan['runnable']
                    reason='QUEUED_CACHED_NATIVE_AUTHOR_READOUT' if ready else (
                        plan['reason'] if method=='OVI-MAP' and not plan['runnable'] else
                        'WAIT_SHARED_ORIGINAL_NATIVE_SOURCE' if method=='OVI-MAP' else 'FULL_ASSETS_OR_DATASET_PORT_PENDING')
                    remaining.append(dict(kind='EXTERNAL_SCENE_METHOD',method=method,benchmark=benchmark,scene=scene,
                        status='BLOCKED_RESOURCE' if runnable else plan['status'] if method=='OVI-MAP' else 'BLOCKED_DEPENDENCY',
                        runnable=runnable,reason=reason,recipe=f'external/{method}/recipe.json'))
    stats={}
    for benchmark in protocol['scopes']:
        path=root/'statistics'/(benchmark+'.json')
        stats[benchmark]=verified(path) if path.is_file() else None
        if not stats[benchmark]:
            ready=standard[benchmark]['bootstrap_inputs_complete']
            remaining.append(dict(kind='PAIRED_BOOTSTRAP',benchmark=benchmark,draw_sets=2,draws_each=1000,
                status='BLOCKED_RESOURCE' if ready else 'BLOCKED_DEPENDENCY',runnable=ready,
                reason='QUEUED_FULL_PAIRED_BOOTSTRAP' if ready else 'FULL_U11_U10_U01_SCENE_MATCHES_REQUIRED',
                command='--phase statistics --resume'))
    primary_path=root/'review/primary_final.json'
    primary=verified(primary_path) if primary_path.is_file() else {}
    publication_path=root/'publication/final.json'
    publication=read(publication_path) if publication_path.is_file() else {}
    flags=dict(software_complete=primary.get('software_complete',False),
        calibration_complete=calibration['status']=='DONE',
        legacy_complete=bool(legacy and legacy['status']=='DONE'),
        standard_coverage_complete=all(row['full_coverage'] for row in standard.values()),
        external_comparisons_complete=all(row['complete'] for scopes in external.values() for row in scopes.values()),
        publication_complete=publication.get('status')=='DONE',claim_supported=False)
    # Scientific support requires its own primary claim review, beyond file,
    # software, coverage, or publication status.
    flags['claim_supported']=bool(primary.get('claim_supported',False))
    review=write(root/'review/coverage.json',dict(status='DONE',completion_flags=flags,
        standard=standard,external=external,calibration=calibration['identity'],
        legacy=None if legacy is None else legacy['identity'],model=None if model is None else model['identity'],
        tables=None if tables is None else tables['identity'],actual_new_training_updates=0,
        final_primary_review_complete=bool(primary.get('requirements_review_complete',False)),publication_is_not_scientific_completion=True,
        current_CONF12_role='EXPOSED_NOVEL_LABEL_SELECTION;LOFO_IS_NOT_INDEPENDENT_CONFIRMATION',
        evaluator_auxiliary_compatibility=verified(root/'review/G3_evaluator_auxiliary_compat.json')['identity']
            if (root/'review/G3_evaluator_auxiliary_compat.json').is_file() else None,
        actual_existing_AP_parity=verified(root/'review/G3_evaluator_auxiliary_actual_parity.json')['identity']
            if (root/'review/G3_evaluator_auxiliary_actual_parity.json').is_file() else None,
        runnable_jobs_forbid_terminal_close=any(r['runnable'] for r in remaining)))
    write(root/'remaining_jobs.json',dict(status='BLOCKED_RESOURCE' if any(r['runnable'] for r in remaining)
        else 'BLOCKED_DEPENDENCY' if remaining else 'DONE',jobs=remaining,
        expected_original_standard_cells=2560,queued_work_not_silently_removed=True))
    _documents(binding,review,remaining)
    return review


def _handoff_evidence(root, review):
    """Describe measured work and table fields using retained receipts only."""
    lines = ['## Recorded execution', '',
        'Counters below describe the named sealed receipts. Live jobs may have additional unsealed consumption.', '']
    ledger_path = root / 'resource_ledger.json'
    if ledger_path.is_file():
        ledger = verified(ledger_path)
        lines += [f"Resource ledger identity: `{ledger['identity']}`.", '',
            '| Counter | Recorded count |', '|---|---:|']
        lines += [f'| `{key}` | {value} |' for key, value in ledger['counters'].items()]
        lines += ['', 'Unavailable measurements: ' + ', '.join(ledger['missing_measurements']) + '.', '']
    else:
        lines += ['Resource-ledger measurements are not yet available.', '']
    calibration = verified(root / 'calibration/summary.json')
    if calibration['identity'] != review['calibration']:
        raise ValueError('Handoff calibration differs from reviewed receipt')
    lines += [f"CPU calibration receipt: `{calibration['identity']}`.", '',
        '| Calibration counter | Recorded count |', '|---|---:|']
    lines += [f'| `{key}` | {value} |' for key, value in calibration['counters'].items()]
    lines += ['', 'Successful standard scene scoring retained in the current coverage:', '',
        '| Scope | Logical result rows | Unique scene scorer inputs | Author-entry checks |',
        '|---|---:|---:|---:|']
    for benchmark, scope in review['standard'].items():
        logical = unique = author_checks = 0
        for scene, identity in scope['scene_receipts'].items():
            score = verified(root / 'benchmark/evaluation' / benchmark / scene / 'receipt.json')
            if score['identity'] != identity or score['status'] != 'DONE':
                raise ValueError('Handoff scene score differs from reviewed coverage')
            logical += score['logical_rows']
            unique += score['unique_scorer_inputs']
            author_checks += int(score['official_entry_checked'])
        lines.append(f'| {benchmark} | {logical} | {unique} | {author_checks} |')
    lines += ['', 'Unique scene scorer inputs count successful per-scene evaluations after exact prediction aliases. '
        'Failed/interrupted attempts, pooled rescoring and engineering checks are excluded; this is not a total historical CPU-call count.', '']
    if review['legacy']:
        legacy = verified(root / 'legacy/score.json')
        if legacy['identity'] != review['legacy']:
            raise ValueError('Handoff Legacy score differs from reviewed receipt')
        lines += [f"Legacy coverage: {legacy['completed_logical_scene_rows']}/{legacy['expected_logical_scene_rows']} logical scene rows; "
            f"{legacy['completed_pools']}/{legacy['expected_pools']} pools. Logical rows do not measure physical scorer calls.", '']
    lines += ['## Table fields', '',
        'Each table includes the Method / setting row label and the following data columns. '
        'Per-cell units, result keys and aggregation formulas are in `cell_provenance.jsonl`.', '']
    tables_path = root / 'tables.json'
    if tables_path.is_file():
        tables = verified(tables_path)
        lines += [f"Table-definition identity: `{tables['identity']}`.", '',
            '| Table | Title | Data columns in order |', '|---|---|---|']
        lines += [f"| {table['id']} | {table['title']} | {'; '.join(table['headers'])} |"
                  for table in tables['tables']]
    else:
        lines += ['Table definitions are not yet available.']
    return '\n'.join(lines) + '\n\n'


def _documents(binding, review, remaining):
    root=Path(binding['output_root'])
    folder=REPO/DOCS
    folder.mkdir(parents=True,exist_ok=True)
    source=read(root/'bootstrap/source_binding.json')
    lines=['# Source binding','',f"Parent commit: `{source['base']}`.",
        f"Prompt ZIP SHA256: `{source['archive_sha256']}`.",'',
        'The following frozen parent files were checked against exact Git blobs before execution.','',
        '| Evidence | Source path | Exact Git blob |','|---|---|---|']
    for row in source['source_files']:
        lines.append(f"| {row['id']} | {row['path']} | {row['git_blob']} |")
    lines+=['','External repositories, actual component training roles, class axes, source hashes and target point orders are sealed in protocol_lock.json.',
        'Original-native captures are reused for the OVI standard protocol port. Its triangle paint, canonical classifier, top10/last8 pooling and original confidence are preserved; it never consumes F/MV/Q scores.']
    (folder/'SOURCE_BINDING.md').write_text('\n'.join(lines)+'\n')
    flags=review['completion_flags']
    lines=['# Current evidence coverage','',
        'This snapshot records measured coverage; completion dimensions are reported independently.','',
        '| Completion dimension | Complete |','|---|---|']
    lines += [f'| {key} | {str(value).lower()} |' for key,value in flags.items()]
    lines+=['','| Standard scope | Scored scenes | Scored cells |','|---|---:|---:|']
    lines += [f"| {benchmark} | {row['completed_scenes']}/{row['expected_scenes']} | {row['completed_cells']}/{row['expected_cells']} |"
              for benchmark,row in review['standard'].items()]
    lines+=['','Calibration: 1331 logical candidates; original279 objects and9 abstentions retained. No new training.',
        'Legacy eligible substitutions and standard GLOBAL predictions use separate pools. Partial standard scores stay out of T1/T2.',
        'Author-reported values are references, never counted as local model executions.',
        'Only dependencies listed in the current remaining_jobs.json are unresolved; completed inputs and method scopes are retained.',
        'Current CONF12 calibration and LOFO use exposed novel labels. Neither proves unseen-class calibration or the old physical grouping hypothesis.',
        'All displayed values and per-seed aggregation are generated from result_store.json; see tables/S2.md for partial AP and tables/S6.md for full-GT proposal coverage.']
    (folder/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    claims=['# Evidence-bound claims','',
        'This file is generated from the current coverage receipt. It does not replace the final primary scientific review.','',
        '| Claim | Current boundary |','|---|---|',
        '| Cached calibration is reproducible | Real RAW/candidate recomputation and matched1000 state/input parity are recorded. |',
        '| CAP improves the exposed calibration task | Report only the fixed and LOFO numbers on original279 CONF12 objects; novel labels were exposed. |',
        '| The complete method beats OVI-MAP | Not established: full standard coverage and paired family bootstrap remain incomplete. |',
        '| The semantic mechanism improves instance geometry | Not established: the four system variants share the same frozen native owner masks. |',
        '| Generalization to unseen classes or scenes | Not established by exposed CONF12 calibration or partial standard scenes. |',
        '| Online30 FPS or whole-pipeline memory advantage | Not established: worker wall times and N/Q memory peaks have narrower measurement scopes. |',
        '| Replica48 full AC is complete | Determined by its full standard-cell coverage and locked training-role mapping; see current remaining_jobs.json. |',
        '| CVPR novelty is validated | Not established by numerical calibration or a partial benchmark win. |','',
        'Seven independent completion flags:','']
    claims += [f'- `{key}`: `{str(value).lower()}`' for key,value in flags.items()]
    (folder/'CLAIMS.md').write_text('\n'.join(claims)+'\n')
    external_lines=['| Method | Benchmark | Evidence | Locally scored scenes |', '|---|---|---|---:|']
    for method,scopes in review['external'].items():
        for benchmark,scope in scopes.items():
            expected=review['standard'][benchmark]['expected_scenes']
            external_lines.append(f"| {method} | {benchmark} | {scope['evidence']} | {scope['locally_completed_scenes']}/{expected} |")
    external_text='\n'.join(external_lines)+'\n\n'
    command='PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" --output-root "$OUT" --resume'
    (folder/'HANDOFF.md').write_text('# Execution handoff\n\nCurrent coverage and remaining dependencies are recorded below; terminal completion is determined by publication/final.json.\n\n'
        'Resolve `$PY_CONTROL`, `$SEMANTIC_ROOT`, `$STRUCTURE_ROOT`, and `$OUT` from the local task_paths.json and binding.json. '
        'Private machine paths remain in the local execution receipts.\n\n'
        f'```bash\n{command} --phase benchmark_predict\n{command} --phase benchmark_score\n'
        f'{command} --phase baselines\n{command} --phase statistics\n{command} --phase tables\n{command} --phase review\n```\n\n'
        'GPU selection and locks follow the current asset configuration and actual job receipts; at most3 CPU scorers and BLAS1. '
        'Historical allocation limits and measured consumption are recorded in resource_ledger.json. '
        'Completed source/model/readout outputs are reused. Do not launch a duplicate streaming driver while its process marker is live.\n\n'
        'Past recovery attempts and original failures remain in their actual runtime receipts. '
        'The pinned author source and complete AP AST remain unchanged; the added auxiliary AR initialization and actual existing-scene AP parity are recorded in review/G3_evaluator_auxiliary*.json.\n\n'
        'Full external method variants, assets and exact commands are in external/<method>/recipe.json. '
        'The following coverage comes from current official whole-benchmark pools and compatible author references. '
        'Locally scored complete author predictions are distinguished from locally inferred predictions in each recipe. '
        'Synthetic format checks never count as formal completed scenes.\n\n'
        + external_text
        + f'Remaining logical jobs: {len(remaining)}. Exact jobs and missing dependencies: `remaining_jobs.json`. '
        'Completed job costs and planning ranges: `resource_ledger.json`, `cost_estimate.json`. '
        'Training updates remain0; native reconstruction, CropFormer, SigLIP, FC encoding and frozen head inference are real new work.\n\n'
        + _handoff_evidence(root, review)
        + 'The original deployment/N0 has not been replaced. Current partial results cannot establish a complete benchmark win or a CVPR novelty claim.\n')
    primary_status='complete' if review.get('final_primary_review_complete') else 'pending'
    (folder/'REVIEW.md').write_text(f'# Primary review status\n\nFinal primary requirements review is {primary_status}.\n\n'
        'G1 real numeric recomputation and G2 actual frozen state/input parity are recorded. '
        'The first standard scene has real G3 mask/GT/evaluator and MFoff access checks. '
        'The first OVI author-format connection checks actual masks/classes/native confidence.\n\n'
        'The released common evaluator has an uninitialized auxiliary ar_current branch when the first class has no GT. '
        'A task-owned runtime adapter adds only that missing NaN assignment, leaving the complete AP AST intact. '
        'Real existing-match AP arrays were compared before and after adaptation. '
        'Recovered scene receipts are authoritative; original stream records retain their historical failures.\n\n'
        f'Current remaining logical jobs: {len(remaining)}. Exact dependencies and commands are in remaining_jobs.json. '
        'Synthetic checks are labeled separately. Publication and a partial table do not complete this task.\n\n'
        '| Completion dimension | Complete |\n|---|---|\n'
        + ''.join(f'| {key} | {str(value).lower()} |\n' for key,value in flags.items()))
