"""Render measured paired-evidence tables; no invented benchmark values."""

import time
from collections import Counter
from pathlib import Path

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.module_validation.contracts import canonical_digest

from .binding import INDEX, ROOT
from .binding import read_json_or_gzip as read
from .diagnostics import METRICS
from .evaluation import RANKS
from .evidence import B


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def pp(value):
    return 'undefined' if value is None else f'{value * 100:.6f}'


def report(binding):
    start = time.monotonic()
    root = Path(binding['output_root'])
    spec, lock = read(binding['spec']), read(root / 'transfer_lock.json')
    diagnostics = read(root / 'diagnostics/all.json.gz')
    methods = diagnostics['methods']
    pools = {split: {rank: {m: read(root / 'pooled' / split / m / (rank + '.json')) for m in methods}
                     for rank in RANKS} for split in ('cal', 'replica')}
    nominee = lock['research_nominee']['method']
    delta = {metric: pools['replica'][RANKS[0]][nominee]['metrics'][metric] - pools['replica'][RANKS[0]][B]['metrics'][metric] for metric in METRICS}
    result = ['# Paired evidence: measured results', '',
              ('Offline frozen-evidence semantic refinement on two exposed CAL and eight historical Replica scenes. '
              'No new generalization confirmation, online mapping demonstration, or statistical significance claim.'), '',
              (f'CAL-frozen research nominee: **{nominee}**. Deployment: **N0_UNCHANGED**. '
              'Original wave-1 nomination remains untouched.'), '',
              ('APall/uAP averages the actual released 0.50–0.90 overlap vector; AP25 is separate. '
              'All values below are percentages, differences are percentage points. '
              'Official pooling calls the released evaluator on ordered scene files and sums semantic confusion matrices.'), '']
    for split in ('cal', 'replica'):
        for rank in RANKS:
            rows = []
            base, native = pools[split][rank][B]['metrics'], pools[split][rank]['N0']['metrics']
            for method in methods:
                values = pools[split][rank][method]['metrics']
                rows.append([method, *[pp(values[k]) for k in METRICS], pp(values['apall']-base['apall']),
                             pp(values['miou']-base['miou']), pp(values['apall']-native['apall'])])
            result += [f'## {split.upper()} / {rank} / RELEASED_DATASET_POOL', '',
                       table(['Method', *METRICS, 'ΔAPall vs B', 'ΔmIoU vs B', 'ΔAPall vs N0'], rows)]
    contrasts = [('PE_R3_LOCAL', m) for m in ('PE_R2_GLOBAL', 'PE_R0_BLEND', 'PE_R1_FOURWAY', 'PE_R3_LOCAL_CALDELTA')]
    contrasts += [('PE_D4_LINEAGE', m) for m in ('PE_D3_DIAGONAL', 'PE_D1_ANCHORED', 'PE_D2_GROUPED', 'PE_D4_SHUFFLED')]
    contrast_rows = []
    for split in ('cal', 'replica'):
        values = pools[split][RANKS[0]]
        for a, b in contrasts:
            contrast_rows.append([split, a + ' − ' + b, *[pp(values[a]['metrics'][k] - values[b]['metrics'][k]) for k in METRICS]])
    result += ['## Matched mechanism contrasts', '', table(['Split', 'Contrast', *METRICS], contrast_rows)]
    if lock['combination']['status'] == 'MEASURED':
        result += [('The one prespecified D4→R3 interaction was executed. It uses the best active CAL R3 eta '
                   f"{lock['combination']['parameter']}, the original B candidate set, and D4 mass within that set."), '']
        rows = []
        for split in ('cal', 'replica'):
            values = pools[split][RANKS[0]]
            # Active-R3 can differ from the selected R3; report interaction only on matching CAL active row.
            active = (read(root / 'pooled/cal' / lock['active_choices']['PE_R3_LOCAL']['id'] / 'OFFICIAL_CURRENT_CLASS.json')
                      if split == 'cal' else values['PE_R3_LOCAL']
                      if lock['choices']['PE_R3_LOCAL']['parameter'] == lock['combination']['parameter'] else None)
            if active is not None:
                rows.append([split, *[pp(values['PE_COMBO_D4_R3']['metrics'][k] - values['PE_D4_LINEAGE']['metrics'][k] - active['metrics'][k] + values[B]['metrics'][k]) for k in METRICS]])
        result += [table(['Split: combo − D4 − active R3 + B', *METRICS], rows),
                   ('Replica does not receive an extra active-R3 retuning run. The matched interaction is included when its selected eta agrees. If it differs, a matched '
                   'four-cell Replica interaction is not identifiable from canonical rows and is not fabricated.'), '']
    else:
        result += ['Composition: NOT_TRIGGERED_NO_ACTIVE_PAIR; no invented combination metrics.', '']
    coverage = []
    for scene, summary in diagnostics['summaries'].items():
        coverage.append([scene, summary['owners'], summary['identifiable'], summary['paired_coverage'],
                         summary['dependence_coverage'], summary['NQ_nonzero_covariance_owners'], summary['unknown_support_atoms'],
                         summary['D4_equals_D3_owners'], summary['text_comparison']['max_absolute_difference']])
    result += ['## Coverage and degeneracies', '', table(['Scene', 'Owners', 'Identifiable', 'Paired F/O', 'D inputs',
                'N/Q covariance active', 'Unknown atoms', 'D4=D3 structural', 'F/O text max difference'], coverage),
               ('Identifiable means unique geometry correspondence with IoU strictly greater than 0.5. '
               'All owners, including unidentifiable and fallback owners, remain in complete-map evaluation. '
               'Unknown mask support is isolated and labeled unknown; it is not evidence of independence. '
               'The proxy models only same-image/same-vision-space support. Cross-model and temporal correlations remain unmodeled.'), '']
    descriptor_rows, calibration_rows, matching_rows = [], [], []
    for split, scenes in [('cal', spec['datasets']['calibration']), ('replica', spec['datasets']['replica'])]:
        descriptors = [o for s in scenes for o in read(root / 'evidence' / s / 'frozen.json.gz')['objects'].values()]
        for method in ('PE_D3_DIAGONAL', 'PE_D4_LINEAGE', 'PE_D4_SHUFFLED'):
            entries = [o['dependence']['methods'][method] for o in descriptors if o['dependence']['status'] == 'COMPUTED']
            total_weights = Counter()
            active_sets = Counter()
            for entry in entries:
                total_weights.update(dict(zip(entry['sources'], entry['mean_source_weights'], strict=True)))
                for active in entry['active_sets']:
                    active_sets['+'.join(active['sources'])] += active['pairs']
            descriptor_rows.append([split, method, len(entries),
                '; '.join(f'{n}={total_weights[n]/len(entries):.6f}' for n in ('N', 'Q', 'F')),
                f"{sum(e['cycle_energy'] for e in entries)/len(entries):.6g}",
                f"{sum(e['graph_fit_squared_residual'] for e in entries)/len(entries):.6g}", dict(active_sets)])
        for method in methods:
            entries = [diagnostics['summaries'][s]['calibration'][method] for s in scenes]
            support = sum(e['support'] for e in entries)
            if support:
                calibration_rows.append([split, method, support,
                    f"{sum(e['sum_nll'] for e in entries)/support:.6f}",
                    f"{sum(e['sum_brier'] for e in entries)/support:.6f}"])
            records = [r for s in scenes for r in diagnostics['summaries'][s]['matcher'][f'{method}/{B}/OFFICIAL_CURRENT_CLASS']]
            for threshold in (.25, .5):
                matched = [r for r in records if r['overlap'] == threshold]
                matching_rows.append([split, method, threshold, *[sum(r[k] for r in matched) for k in ('added', 'lost', 'duplicates', 'ignored', 'unmatched_fp')]])
        shuffle_identity = sum(o['dependence'].get('shuffle_identity', False) for o in descriptors)
        result += [(f'{split.upper()}: shuffled correlation is an identity for {shuffle_identity}/{len(descriptors)} objects. '
                   'For the remaining objects it changes probabilities but the measured whole-map D4 and shuffled labels coincide. '
                   'This is not structural D4=D3 equality for every object.'), '']
    result += ['## Dependence weights and graph diagnostics', '',
               table(['Split', 'Method', 'Objects', 'Mean source weights (absent=0)', 'Mean cycle energy', 'Mean anchored fit residual', 'QP active sets / pair counts'], descriptor_rows),
               'QP pair counts describe numerical problems, not independent training examples.', '',
               '## Common-identifiable probability calibration', '',
               table(['Split', 'Method', 'Same support', 'NLL', 'Multiclass Brier'], calibration_rows),
               ('N0 canonical-relative confidence and old A7 are not assigned invented probability scales here. '
               'Calibration results use the fixed common identifiable support and cannot substitute for official map metrics.'), '',
               '## Actual released matching versus B', '',
               table(['Split', 'Method', 'Overlap', 'Added GT matches', 'Lost GT matches', 'Duplicates', 'Ignored events', 'Unmatched FP events'], matching_rows)]
    for split, scenes in [('cal', spec['datasets']['calibration']), ('replica', spec['datasets']['replica'])]:
        rows = []
        for method in methods:
            fields = ['changed_B', 'corrected_B', 'harmed_B', 'corrected_N0', 'harmed_N0', 'unused_correct_source', 'extra_O_correction_survived', 'F_correct_O_wrong_harm']
            rows.append([method, *[sum(diagnostics['summaries'][s]['methods'][method][k] for s in scenes) for k in fields]])
        result += [f'## {split.upper()} object outcomes', '', table(['Method', *fields], rows)]
        paired = Counter()
        blocking = []
        for s in scenes:
            paired.update(diagnostics['summaries'][s]['paired_outcomes'])
            blocking += diagnostics['summaries'][s]['blocking']
        result += [(f'Common-pair identifiable outcomes: `{dict(paired)}`. Probability-pool blocking margins >1: '
                   f"{sum(r['impossible_for_one_region_to_overcome'] for r in blocking)}/{len(blocking)} eligible wrong-base objects."), '',
                   ('Object corrections are not AP summands. Full all-threshold match-set changes, duplicates, ignored events, '
                   'false positives and per-class AP changes are saved in matcher_contrasts.json.gz. '
                   'The first two corrected and first two harmed owner IDs per scene/method are stored in summary.json without manual example selection.'), '']
    result += ['## Per-scene changes versus B (official)', '']
    rows = []
    for scene in spec['datasets']['replica']:
        base = read(root / 'rows' / scene / B / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
        for method in methods:
            actual = read(root / 'rows' / scene / method / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
            rows.append([scene, method, *[pp(actual[k] - base[k]) for k in METRICS]])
    result += [table(['Scene', 'Method', *['Δ' + k for k in METRICS]], rows)]
    sensitivity = []
    for omitted in spec['datasets']['replica']:
        base = root / 'sensitivity' / ('omit_' + omitted) / 'replica'
        a = read(base / nominee / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
        b = read(base / B / 'OFFICIAL_CURRENT_CLASS.json')['metrics']
        sensitivity.append([omitted, *[pp(a[k] - b[k]) for k in METRICS]])
    result += ['## Seven-scene sensitivity of frozen nominee', '', table(['Omitted', *['Δ'+k+' vs B' for k in METRICS]], sensitivity),
               'These are eight actual seven-scene released pools, not independent trials or bootstrap confidence intervals.', '']
    values = pools['replica'][RANKS[0]]
    best = max(methods, key=lambda m: values[m]['metrics']['apall'])
    frontier = [m for m in methods if not any(all(values[n]['metrics'][k] >= values[m]['metrics'][k] for k in METRICS)
                 and any(values[n]['metrics'][k] > values[m]['metrics'][k] for k in METRICS) for n in methods if n != m)]
    result += [f'Descriptive Replica best APall: **{best}**. Metric-only Pareto frontier across all five metrics: '
               + ', '.join(frontier) + '. These do not replace the CAL-frozen nominee.', '']
    result += [('The measured result favors the simple D2 grouped-probability control for descriptive Replica APall, '
                'while FC fusion retains higher mIoU. The CAL-frozen complex nominee does not transfer its CAL advantage: '
                f"Replica differences versus B are {pp(delta['apall'])} pp APall and {pp(delta['miou'])} pp mIoU. "
                'Keep the original FC research baseline and unchanged deployment; a future independent confirmation of the '
                'simple grouping tradeoff is better justified than advancing an unsupported dependence/acquisition mechanism.'), '']
    costs = []
    for method in methods:
        counts = Counter()
        for scene in spec['datasets']['replica']:
            counts.update(diagnostics['logical_costs'][scene]['methods'][method]['counts_by_kind'])
        costs.append([method, counts['image_crop'], counts['dense_image'], counts['region_pool_projection']])
    executions = [read(p) for p in sorted((root / 'execution').glob('*.json'))]
    timings = []
    for phase in ('recover', 'predict', 'evaluate'):
        first_jobs = {}
        for entry in executions:
            if entry['phase'] == phase:
                for row in entry['results']:
                    if 'job_seconds' in row:
                        first_jobs.setdefault(row['scene'], row)
        jobs = list(first_jobs.values())
        timings.append([phase, len(jobs), f"{sum(r['job_seconds'] for r in jobs):.3f}", max((r['process_peak_rss_kib'] for r in jobs), default=0)])
    recovery_timing = []
    for split, scenes in [('CAL', spec['datasets']['calibration']), ('Replica', spec['datasets']['replica'])]:
        receipts = [read(root / 'evidence' / scene / 'receipt.json') for scene in scenes]
        total = sum(r['elapsed_seconds'] for r in receipts)
        operator = sum(r['operator_seconds'] for r in receipts)
        recovery_timing.append([split, f'{total:.3f}', f'{operator:.3f}', f'{total-operator:.3f}'])
    result += ['## Physical and logical costs', '',
               ('This study invoked **0 image forwards, 0 image-model loads, 0 text-model forwards, 0 training jobs, '
               '0 new temperature fits, 0 geometry reconstructions and 0 downloaded bytes**. '
               'Necessary original model work is not free: the following Replica unions deduplicate only exact input/model operations.'), '',
               table(['Method', 'Crop inputs', 'Dense image encodes', 'Region pooling/projection'], costs),
               table(['CPU phase', 'Recorded jobs', 'Sum of job seconds (not wall latency)', 'Largest observed process RSS KiB'], timings),
               table(['Original recovery split', 'Sum of elapsed seconds', 'Dependence operator seconds', 'Remaining recovery/read/validation seconds'], recovery_timing),
               ('Timing includes I/O and cache validation. Concurrent process times are not summed into wall latency; '
               'process memory maxima are not summed into a system peak. Historical geometry/text inference costs and '
               'unrecorded failed-attempt durations are unknown, not zero. Original model/input manifests are retained as references. '
               'Scalar-grid evaluations share identical-output caches with explicit alias records.'), '']
    result += [('CAL sweep prediction/evaluation times are the two CAL scene records in execution/predict_*.json and '
                'execution/evaluate_*.json; Replica records are separate. Pool evaluation timers are in each pooled receipt. '
                'Report rendering timers are in report/render_*.json; push/publication time is in the external publication/final.json. '
                'Recovery totals above include the initial standalone CAL recovery; the process-job table can instead contain '
                'its later cache-validation invocation. These are distinct accounting views and must not be added.'), '']
    localmax = max((v['maximum_outside_drift'] for s in diagnostics['summaries'].values() for v in s['local'].values()), default=0)
    result += ['## Interpretation limits', '',
               (f'Largest measured local outside-probability drift: {localmax:.3g}. Mass conservation, PSD and a unique '
               'graph solution establish operator properties only; they do not establish accuracy or causal error correction.'), '',
               ('The four fixed CAL grids are the only new supervised scalar choices. Source temperatures and original Q policy '
               'already used supervision. Pair classes are not independent samples. Common-temperature residuals compare paired '
               'class scores, never latent vectors; any measured text difference prevents a visual-only attribution.'), '',
               ('Dependence is supported only if D4 exceeds D3, the same-anchor control, and shuffled dependence on relevant metrics. '
               'A simple-control win must not be relabeled as a proposed mechanism win. Conditional acquisition, E06, '
               'new models and geometry changes remain QUEUED_NOT_EXECUTED.'), '']
    # Machine-readable complete report inputs make every rendered number traceable.
    report_data = {'pools': pools, 'nominee': nominee, 'nominee_delta_Replica_vs_B': delta,
                   'descriptive_best_APall': best, 'metric_Pareto_frontier': frontier,
                   'contrasts': contrast_rows, 'seven_scene_sensitivity': sensitivity, 'timings': timings}
    report_data['identity'] = canonical_digest(report_data)
    write_once(root / 'report/data.json', report_data)
    selections = ['# Paired evidence CAL selection', '', f'Frozen research nominee: **{nominee}**; deployment N0_UNCHANGED.', '',
                  ('Sequential practical bands: APall 0.05 pp, mIoU 0.10 pp, AP50 0.10 pp; then fixed compute tier, '
                  'distance from identity, registry order. Bands express preference, not significance.'), '',
                  table(['Configuration', 'Parameter', 'Tier', *METRICS],
                        [[r['id'], r['parameter'], r['tier'], *[pp(r['metrics'][k]) for k in METRICS]] for r in lock['CAL_grid']]),
                  '## Frozen configurations', '',
                  table(['Method', 'Selected parameter', 'Selected grid ID', 'Best active parameter'],
                        [[m, r['parameter'], r['id'], lock['active_choices'].get(m, {}).get('parameter')] for m, r in lock['choices'].items()]),
                  f"Composition: `{lock['combination']}`", '',
                  (f"Transfer identity: `{lock['identity']}`. Full retained sets at each ranking stage are in transfer_lock.json. "
                  'Every choice was locked before Replica evaluation; DIRECT controls were eligible, shuffled dependence was not.')
                  ]
    handoff = ['# Paired evidence handoff', '', f"Branch: `{spec['branch']}`; base: `{spec['base_commit']}`.", '',
               f"Worktree: `{ROOT}`. Output: `{root}`. Parent binding: `{binding['wave1_binding']}`.", '',
               ('Runtime: `/home/ww/miniconda3/envs/ovimap-map/bin/python`. Existing NumPy/SciPy and original released evaluator; '
               'no environment reinstall. Scene workers use spawn, 4 BLAS threads, default 3 concurrent CPU processes. '
               'GPUs remain unused because the protocol prohibits new neural inference.'), '',
               ('Optional `--path-map mapping.json` accepts explicit old-root/new-root pairs. It creates recoverable directory '
                'symlinks only when the old root is absent and its parent exists, preserving embedded signed paths. '
                'It never replaces existing files/directories, and every consumed bound input still passes content-hash verification. '
                'No relocation was required for this measured run.'), '',
               ('```bash\n/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_paired_evidence.py '
               '--phase all --resume --threads 4 --workers 3\n```'), '',
               ('Phases: bind → recover → cal → freeze → replica → diagnose → report → publish. '
               'Independent scenes parallelize; prediction locks precede evaluation/GT; the transfer lock precedes Replica. '
               'Parent files and source-temperature fits remain unchanged. Exact evaluator-byte relocation is recorded separately '
               'and identity reuse has explicit aliases.'), '',
               ('The released pooler prints an outside-prediction-path notice when it consumes an unchanged parent mask manifest. '
                'The pinned evaluator still loads those masks; exact original-control scene and pooled metric parity was verified. '
                'These notices are retained in logs, not hidden by changing the evaluator.'), '',
               ('New modules: binding/evidence replace old branch-specific binding and recover caches; residuals/dependence/pair_graph '
               'implement fixed operators; evaluation wraps original SceneEvaluator/pool; selection freezes CAL; workflow schedules '
               'scenes; diagnostics joins outcomes; reporting/publication build measured deliverables. Old guards remain intact.'), '',
               ('Large prediction arrays, masks and released traces remain in shared storage. Compact sources, labels, probabilities, '
               'rows, pools, choices and diagnostics are published with a content manifest. Reproduction requires the bound original '
               'caches, not just GitHub. Missing external caches cannot be replaced with inference in this protocol.'), '',
               f"Binding identity: `{binding['identity']}`. Transfer identity: `{lock['identity']}`. Report input identity: `{report_data['identity']}`.", '',
               (f'External final publication receipt: `{root}/publication/final.json`. It records the containing commit SHA '
               'after a normal push and comparison; the receipt is not recursively committed.'), '',
               ('Not executed: conditional information acquisition, E06, new geometry/model/training work. '
               'All scenes are exposed; independent confirmation remains future work, not a completed claim.')]
    d4 = values['PE_D4_LINEAGE']['metrics']
    dep_supported = all(d4['apall'] > values[m]['metrics']['apall'] for m in ('PE_D3_DIAGONAL', 'PE_D1_ANCHORED', 'PE_D4_SHUFFLED'))
    residual_supported = all(values['PE_R3_LOCAL']['metrics']['apall'] > values[m]['metrics']['apall'] for m in ('PE_R2_GLOBAL', 'PE_R0_BLEND', 'PE_R1_FOURWAY'))
    claims = ['# Paired evidence claim ledger', '', table(['Claim', 'Status', 'Evidence/boundary'], [
        ['Local outside probabilities preserved', 'SUPPORTED_IN_THIS_STUDY' if localmax < 1e-12 else 'NOT_SUPPORTED', f'Measured maximum drift {localmax}; no implication of AP gain'],
        ['Local residual necessary for improved Replica APall over simple controls', 'SUPPORTED_IN_THIS_STUDY' if residual_supported else 'NOT_SUPPORTED', 'Matched R3−R2/R0/R1 table; exposed regression only'],
        ['Class-pair dependence adds APall beyond diagonal, anchor and shuffled controls', 'SUPPORTED_IN_THIS_STUDY' if dep_supported else 'NOT_SUPPORTED', 'Matched D4 contrasts and observed covariance coverage; not a true error-covariance estimate'],
        ['Fresh generalization', 'UNTESTED', 'All ten scenes exposed'],
        ['Online speedup', 'UNTESTED', 'Offline cached evidence, no new visual inference'],
        ['Conditional information acquisition', 'UNTESTED', 'QUEUED_NOT_EXECUTED'],
        ['KL ratio / minimum-variance QP / graph ranking is new mathematics', 'NOT_SUPPORTED', 'Inherited tools; see original SOURCES.md'],
        ['Mass conservation or PSD guarantees benchmark gain', 'NOT_SUPPORTED', 'Algebraic property does not imply AP improvement'],
    ])]
    docs = ROOT / 'docs/paper/static_ovmap'
    contents = {'PAIRED_EVIDENCE_RESULTS.md': result, 'PAIRED_EVIDENCE_SELECTION.md': selections,
                'PAIRED_EVIDENCE_HANDOFF.md': handoff, 'PAIRED_EVIDENCE_CLAIMS.md': claims}
    outputs = []
    for name, lines in contents.items():
        path = docs / name
        path.write_text('\n'.join(lines).rstrip() + '\n')
        outputs.append(INDEX.identity(path))
    receipt = {'status': 'MEASURED_REPORTS_RENDERED', 'report_identity': report_data['identity'], 'outputs': outputs,
               'elapsed_seconds': time.monotonic() - start}
    # Report contents are deterministic; timing records use separate invocation names.
    write_once(root / 'report' / f'render_{time.time_ns()}.json', receipt)
    return report_data
