"""One result store feeds all manuscript MD/TeX tables and cell provenance."""
import json
from pathlib import Path
import subprocess
import numpy as np

from .common import (REPO, BASE, SEEDS, DOCS, verified, read, write,
                     digest_file, canonical_digest, source_identity)


def _mean(values):
    return None if not values or any(v is None for v in values) else float(np.mean(values))


def _cell(results, keys, metric, *, unit='PERCENT', expected=None):
    present = [key for key in keys if key in results]
    values = [results[key]['metrics'].get(metric) for key in present]
    complete = len(present) == (len(keys) if expected is None else expected)
    return dict(value=_mean(values) if complete else None, results=present, metric=metric,
                aggregation='MEAN_OF_INDEPENDENT_PER_SEED_METRICS' if len(keys) > 1 else 'IDENTITY', unit=unit)


def _text(value, sources=()):
    return dict(value=value, results=list(sources), unit='TEXT', aggregation='IDENTITY')


def _fmt(cell):
    value = cell['value']
    if value is None:
        return '--'
    if cell['unit'] == 'PERCENT':
        result = f'{100 * value:.2f}'
    elif cell['unit'] == 'COUNT':
        result = str(int(value))
    elif cell['unit'] in ('SECONDS','GiB'):
        result = f'{value:.3f}'
    else:
        result = str(value)
    return result + cell.get('annotation', '')


def external_cell(results, method, benchmark, metric):
    """A partial local run never replaces a compatible full author result."""
    local = f'external:{method}:{benchmark}'
    reference = f'author:{method}:{benchmark}'
    if local in results and results[local]['full_scene_coverage']:
        return _cell(results, [local], metric)
    if reference in results and results[reference].get('compatible_frozen_variant', True):
        return dict(_cell(results, [reference], metric), annotation='†')
    return _text(None)


def add_open3dis_result(binding, results):
    from .open3dis_score import pool
    from .open3dis_replica_import import pool as replica_pool
    for benchmark, row in [('scannet200_val', pool(binding)), ('replica48', replica_pool(binding))]:
        if row is not None:
            results['external:Open3DIS:' + benchmark] = dict(row,
                source=dict(path='external/Open3DIS/pools/' + benchmark + '.json', identity=row['identity']))


def _tex(value):
    for before, after in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'),
                          ('_', r'\_'), ('#', r'\#'), ('>', r'\textgreater{}'),
                          ('<', r'\textless{}'), ('†', r'\textsuperscript{$\dagger$}')):
        value = value.replace(before, after)
    return value


def _emit(root, table):
    name = table['id']
    folder = root / 'tables'
    folder.mkdir(parents=True, exist_ok=True)
    header = ['Method / setting', *table['headers']]
    lines = ['# ' + name + ': ' + table['title'], '', table['caption'], '',
             '| ' + ' | '.join(header) + ' |', '| ' + ' | '.join(['---'] * len(header)) + ' |']
    panel = None
    for row in table['rows']:
        if row.get('panel') != panel:
            panel = row.get('panel')
            if panel:
                lines.append('| **' + panel + '** | ' + ' | '.join([''] * len(table['headers'])) + ' |')
        lines.append('| ' + ' | '.join([row['label'], *[_fmt(c) for c in row['cells']]]) + ' |')
    (folder / (name + '.md')).write_text('\n'.join(lines) + '\n')
    tex = [r'\begin{table*}[t]', r'\centering', r'\small',
           r'\caption{' + _tex(table['caption']) + '}', r'\label{tab:benchmark-' + name.lower() + '}',
           r'\begin{tabular}{l' + 'r' * len(table['headers']) + '}', r'\toprule',
           ' & '.join(_tex(value) for value in header) + r' \\', r'\midrule']
    panel = None
    for row in table['rows']:
        if row.get('panel') != panel:
            if panel:
                tex.append(r'\midrule')
            panel = row.get('panel')
            if panel:
                tex.append(r'\multicolumn{' + str(len(header)) + r'}{l}{\emph{' + _tex(panel) + r'}} \\')
        tex.append(' & '.join(_tex(value) for value in [row['label'], *[_fmt(c) for c in row['cells']]]) + r' \\')
    tex.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}'])
    (folder / (name + '.tex')).write_text('\n'.join(tex) + '\n')


def build(binding, *, compile_previews=True):
    from static_ovmap.semantic_update_evidence_v1.metrics import recognition
    from .author_results import build as author_results
    from .pooling import standard
    from .diagnostics import collect as conditional
    from .resources import build as resource_ledger
    from .external_pooling import ovi as ovi_pools
    from .supplement import historical, availability, job_manifest
    root = Path(binding['output_root'])
    previous_previews={row['table']:row for row in verified(root/'tables/receipt.json')['previews']} \
        if (root/'tables/receipt.json').is_file() else {}
    results = {}
    summary = verified(root / 'calibration/summary.json')
    recipes = verified(root / 'simple_recipe_lock.json')
    protocol = verified(root / 'protocol_lock.json')
    model = verified(root / 'model_lock.json') if (root / 'model_lock.json').is_file() else None
    public_meta = verified(Path(binding['semantic_root']) / 'A/import.json')
    classes = public_meta['class_split']
    common = dict(benchmark='EXPOSED_CONF12', split_id='ORIGINAL279', metric_definition='POSITIVE_CLASS_MACRO_FULL_ORIGINAL_DENOMINATOR',
        exposure_role=summary['exposure_role'], checkpoint_profile='matched1000',
        prediction=dict(path='calibration/predictions.npz', sha256=digest_file(root / 'calibration/predictions.npz')),
        inputs=public_meta['identity'], model_lock=None if model is None else model['identity'],
        evaluator_revision=BASE, originals=279, unsupported=9)
    for row in summary['fixed_MV']:
        key = f"calibration:fixed:{row['recipe']['candidate_id']}:{row['seed']}"
        results[key] = dict(common, method_id='MV_ATTN', seed=row['seed'], metrics=row['metrics'],
            calibrator=row['recipe'], source=dict(path='calibration/summary.json', identity=summary['identity'],
            key=f"fixed_MV:seed{row['seed']}:{row['recipe']['candidate_id']}"))
    for row in summary['raw_and_post']:
        key = f"calibration:LOFO:{row['method']}:{row['panel']}:{row['seed']}"
        results[key] = dict(common, method_id=row['method'], seed=row['seed'], metrics=row['metrics'],
            calibrator=row['panel'], source=dict(path='calibration/summary.json', identity=summary['identity'],
            key=f"raw_and_post:{row['method']}:{row['panel']}:{row['seed']}"))
    with np.load(root / 'calibration/predictions.npz', allow_pickle=False) as arrays:
        for kind in ('CAP', 'MIX', 'NMIX'):
            for si, seed in enumerate(SEEDS):
                metrics = recognition(arrays['targets'], arrays['MV_ATTN_global_' + kind][si], classes['base_ids'], classes['heldout_ids'])
                key = f'calibration:global55:{kind}:{seed}'
                results[key] = dict(common, method_id='MV_ATTN', seed=seed, metrics=metrics,
                    calibrator=recipes['benchmark_simple_controls'][kind]['recipe'],
                    selection_role='EXPOSED_ALL_CONF12_SEED17_BEST_OF55;SAME_RECIPE_TO_BOTH_SEEDS',
                    source=dict(path='calibration/predictions.npz', sha256=common['prediction']['sha256'],
                                key='MV_ATTN_global_' + kind + ':' + str(si)), recipe_lock=recipes['identity'])
    legacy = verified(root / 'legacy/score.json') if (root / 'legacy/score.json').is_file() else None
    if legacy:
        for cohort, methods in legacy['pools'].items():
            for method, row in methods.items():
                results['legacy:' + cohort + ':' + method] = dict(method_id=method, benchmark=cohort,
                    metrics=row['metrics'], scope=row['scope'], full_scene_coverage=True,
                    ordered_scenes=row['ordered_scenes'], source=dict(path=f'legacy/pools/{cohort}/{method}.json',
                    identity=row['identity']), source_pool=row['identity'],
                    prediction_lock=verified(root / 'legacy/prediction_lock.json')['identity'],
                    protocol='OLD51_OR_CF18_FIXED_ELIGIBLE;NOT_STANDARD_GLOBAL', model_lock=model['identity'])
    standard_pools = {}
    for benchmark in ('scannet200_val', 'replica48'):
        pool = standard(binding, benchmark)
        standard_pools[benchmark] = pool
        for method, row in pool['rows'].items():
            results['standard:' + benchmark + ':' + method] = dict(row,
                source=dict(path=f'benchmark/pools/{benchmark}/{method}.json', identity=row['identity']),
                protocol=protocol['identity'], model_lock=model['identity'])
    authors = author_results(binding)
    for key, value in authors['results'].items():
        results['author:' + key] = dict(value, source_receipt=authors['identity'])
    native_external = ovi_pools(binding)
    add_open3dis_result(binding, results)
    for benchmark, row in native_external['benchmarks'].items():
        if row['metrics'] is not None:
            results['external:OVI-MAP:' + benchmark] = dict(row,
                source=dict(path=f'external/OVI-MAP/pools/{benchmark}.json',identity=row['identity']))
    from .mask_clustering_score import pool as maskclustering_pool
    from .openmask_score import pool as openmask_pool
    replica_ports = dict(MaskClustering=maskclustering_pool(binding), OpenMask3D=openmask_pool(binding))
    for method, port in replica_ports.items():
        results[f'external:{method}:replica48'] = dict(port,
            metrics=port['metrics'] or {},
            source=dict(path=f'external/{method}/pools/replica48.json', identity=port['identity']))
    if 'external:Open3DIS:replica48' in results:
        replica_ports['Open3DIS'] = results['external:Open3DIS:replica48']
    diagnostics = {benchmark: conditional(binding, benchmark) for benchmark in ('scannet200_val', 'replica48')}
    for benchmark, report in diagnostics.items():
        for method, row in report['rows'].items():
            results[f'conditional:{benchmark}:{method}'] = dict(row, benchmark=benchmark,
                scope=report['scope'], full_scene_coverage=report['full_scene_coverage'],
                condition='REAL_NATIVE_PROPOSAL_IOU_GT_.5;SHARED_MATCHES;UNSUPPORTED_RETAINED',
                source=dict(path=f'diagnostics/{benchmark}/summary.json', identity=report['identity'], key='rows:' + method),
                scene_receipts=report['scene_receipts'], protocol=protocol['identity'], model_lock=model['identity'])
        for scene_name, receipt_id in report['scene_receipts'].items():
            row = verified(root / 'diagnostics' / benchmark / scene_name / 'receipt.json')
            key = f'coverage:{benchmark}:{scene_name}'
            results[key] = dict(metrics={k: row['coverage'][k] for k in ('valid_GT_instances', 'matched_GT_instances', 'proposal_recall')},
                source=dict(path=f'diagnostics/{benchmark}/{scene_name}/receipt.json', identity=receipt_id, key='coverage'),
                protocol=protocol['identity'], prediction_lock=row.get('source', {}).get('prediction_lock', row.get('prediction_lock')))
            for method, value in row.get('interventions', {}).items():
                results[f'intervention:{benchmark}:{scene_name}:{method}'] = dict(metrics=value,
                    source=dict(path=f'diagnostics/{benchmark}/{scene_name}/receipt.json', identity=receipt_id,
                                key='interventions:' + method), protocol=protocol['identity'],
                    prediction_lock=row['source']['prediction_lock'])
    costs = resource_ledger(binding)
    historical_results, source_availability = historical(binding), availability(binding)
    job_manifest(binding)
    for key, value in costs['counters'].items():
        results['cost:' + key] = dict(metrics=dict(count=value), source=dict(path='resource_ledger.json',
            identity=costs['identity'], key='counters:' + key), source_receipts=costs['source_receipts'])
    for row in historical_results['rows']:
        results['historical:'+row['method']]=dict(row, source=historical_results['source'],
            source_receipt=historical_results['identity'],data_role='HISTORICAL_EXPOSED;NOT_NEW_STANDARD')
    for benchmark,report in source_availability['benchmarks'].items():
        for pattern,count in report['counts'].items():
            results[f'availability:{benchmark}:{pattern}']=dict(metrics=dict(count=count),
                source=dict(path='supplement/availability.json',identity=source_availability['identity'],
                            key=f'benchmarks:{benchmark}:counts:{pattern}'),
                completed_scenes=report['completed_scenes'],expected_scenes=report['expected_scenes'],
                scene_receipts=report['scene_receipts'],GT_labels_read=False)
    timing_rows=[]
    for stage in ('cropformer','native_map','F_MV1000','native_query'):
        measured={scene:row['stages'][stage] for scene,row in costs['per_scene'].items() if stage in row['stages']}
        if not measured:
            continue
        key='timing:'+stage
        results[key]=dict(metrics=dict(seconds_median=float(np.median([r['worker_wall_seconds'] for r in measured.values()]))),
            source=dict(path='resource_ledger.json',identity=costs['identity'],key='per_scene:stages:'+stage),
            actual_scene_jobs=measured,includes_CPU_and_model_load=True,not_isolated_module_latency=True)
        timing_rows.append(dict(label=stage+' worker wall (median)',cells=[_cell(results,[key],'seconds_median',unit='SECONDS'),
            _text('seconds'),_text(str(len(measured))+' completed scene jobs')]))
    peaks=[r['stages']['native_query']['peak_GPU_allocated_bytes'] for r in costs['per_scene'].values()
           if r['stages'].get('native_query',{}).get('peak_GPU_allocated_bytes') is not None]
    if peaks:
        key='memory:native_query'
        results[key]=dict(metrics=dict(GiB=max(peaks)/2**30),
            source=dict(path='resource_ledger.json',identity=costs['identity'],key='per_scene:stages:native_query:peak_GPU_allocated_bytes'),
            scope='NQ_WORKER_ALLOCATED_ONLY;WHOLE_PIPELINE_PEAK_UNKNOWN')
        timing_rows.append(dict(label='NQ peak allocated (max)',cells=[_cell(results,[key],'GiB',unit='GiB'),
            _text('GiB'),_text('NQ worker only')]))

    tables = []
    def add(name, title, headers, caption, rows):
        table = dict(id=name, title=title, headers=headers, caption=caption, rows=rows)
        if any(len(row['cells']) != len(headers) for row in rows):
            raise ValueError('Table row/column dimensions differ: ' + name)
        tables.append(table)
    rows = []
    for method in ('OVI-MAP', 'OpenMask3D', 'Open3DIS', 'MaskClustering', 'Ours'):
        cells = []
        for benchmark in ('scannet200_val', 'replica48'):
            if method == 'Ours':
                keys = [f'standard:{benchmark}:U11_seed{seed}' for seed in SEEDS]
                eligible = all(key in results and results[key]['full_scene_coverage'] for key in keys)
                eligible &= protocol['scopes'][benchmark]['clean_gradient_audit'] == 'DONE'
            else:
                cells.extend(external_cell(results, method, benchmark, metric)
                             for metric in ('apall', 'ap50', 'ap25'))
                continue
            cells.extend([_cell(results, keys, metric) if eligible else _text(None)
                          for metric in ('apall', 'ap50', 'ap25')])
        rows.append(dict(label='OVI-MAP (protocol port)' if method == 'OVI-MAP' else method, cells=cells))
    add('T1', 'Full external methods on official benchmarks',
        ['ScanNet AP', 'AP50', 'AP25', 'Replica AP', 'AP50', 'AP25'],
        'Official ScanNet200 validation (312 scenes, 198 instance classes) and Replica48 (8 scenes), percentages. '
        '† Marks author-reported values; unmarked values are local. '
        'Local MaskClustering Replica results use the dataset port. '
        'Open3DIS Replica uses the full ISBNet + Grounded SAM-HQ + refinement port with author SPP. '
        'OVI uses original native semantics and confidence on the common protocol. '
        'Ours requires complete coverage and reuses U11. Missing full results remain blank.', rows)
    rows = []
    for method, mf, ac in (('U00','No','No'),('U10','Yes','No'),('U01','No','Yes'),('U11','Yes','Yes')):
        cells = [_text(mf), _text(ac)]
        for benchmark in ('scannet200_val','replica48'):
            keys = [f'standard:{benchmark}:{method}_seed{seed}' for seed in SEEDS]
            full = all(key in results and results[key]['full_scene_coverage'] for key in keys)
            full &= protocol['scopes'][benchmark]['clean_gradient_audit'] == 'DONE'
            cells.extend([_cell(results, keys, metric) if full else _text(None) for metric in ('apall','ap50')])
        rows.append(dict(label=dict(U00='F only', U10='Multi-source', U01='Calibrated F', U11='Full system')[method], cells=cells))
    add('T2','Global all-instance source and calibration ablation',
        ['MF','AC','ScanNet AP','AP50','Replica AP','AP50'],
        'GLOBAL ablation over identical complete original-native instance masks. MFoff uses only F and leaves missing F unknown. '
        'AC is fixed CAP10 plus gamma 0.025 before F temperature. Two seeds are evaluated independently then averaged. '
        'Partial subsets and legacy eligible updates do not enter this table.', rows)
    rows = []
    for key, label in (('RAW','Raw MV'),('OFFSET','Offset'),('CAP10','Angle cap'),('CAP10_OFFSET','Angle cap + offset')):
        ids = [f'calibration:fixed:{key}:{seed}' for seed in SEEDS]
        rows.append(dict(label=label, panel='Fixed interventions', cells=[_cell(results, ids, m) for m in ('B_macro','N_macro','H_macro')]))
    for kind in ('CAP','MIX','NMIX'):
        ids = [f'calibration:global55:{kind}:{seed}' for seed in SEEDS]
        rows.append(dict(label=dict(CAP='Angle cap', MIX='Linear mix', NMIX='Normalized mix')[kind],
            panel='Equal search budget: 55 candidates each', cells=[_cell(results, ids, m) for m in ('B_macro','N_macro','H_macro')]))
    for benchmark, report in diagnostics.items():
        if not report['rows']:
            continue
        prefix = f"{benchmark}: geometric matches, {report['completed_scenes']}/{report['expected_scenes']} scenes"
        for method, label in (('RAW','Raw MV'),('OFFSET','Offset'),('CAP10','Angle cap'),
                              ('CAP10_OFFSET','Angle cap + offset'),('SEARCH_CAP','Locked cap search'),
                              ('SEARCH_MIX','Locked linear mix'),('SEARCH_NMIX','Locked normalized mix')):
            ids = [f'conditional:{benchmark}:{method}_seed{seed}' for seed in SEEDS]
            rows.append(dict(label=label, panel=prefix, cells=[_cell(results, ids, metric)
                        for metric in ('B_macro','N_macro','H_macro')]))
    add('T3','Calibration operators on the same frozen MV model',['Base','Novel','H'],
        'Exposed CONF12, all 279 original objects including 9 abstentions. Positive-class macro accuracy (%). '
        'Fixed cap 10 degrees and offset 0.025. Each searched operator receives 55 candidates, selected only on exposed seed 17 '
        'and applied to both seeds. Each seed H is computed before averaging. Novel labels participated in selection; '
        'New-scope panels condition on the same class-independent native proposal IoU>0.5 matches and retain unsupported F. '
        'They use the locked recipes without benchmark selection and do not replace full-denominator AP.',rows)
    rows=[]
    for method in ('OVI-MAP','OpenMask3D','Open3DIS','MaskClustering','Ours'):
        values = {
            'OVI-MAP':('Origin implementation','CropFormer / SigLIP','Native mapping; voxel0.01m'),
            'OpenMask3D':('Mask3D + SAM','CLIP-L14@336',
                'Supervised proposal; Replica port: '+str(replica_ports['OpenMask3D']['completed_scenes'])+'/8 formal scenes'),
            'Open3DIS':('ISBNet + Grounded-SAM','Refined CLIP-L14',
                'Supervised proposal; two3D passes; Replica port: '+
                str(replica_ports.get('Open3DIS', {}).get('completed_scenes', 0))+'/8 formal scenes'),
            'MaskClustering':('Full mask graph','CropFormer + CLIP-H14',
                'Replica port: '+str(replica_ports['MaskClustering']['completed_scenes'])+'/8 formal scenes'),
            'Ours':('Original-native masks','N / Q_GAIN / FC-MV1000','CONF12 exposed; gradient audit in protocol')}[method]
        cells = [_text(v) for v in values]
        if method in replica_ports:
            cells[2] = _text(values[2], [f'external:{method}:replica48'])
        rows.append(dict(label=method, cells=cells))
    add('S1','Protocol and complete method identities',['Proposal','Semantic source','Information condition'],
        'Published variants and local execution coverage are recorded separately. Foundation pretraining membership remains unknown.',rows)
    rows=[]
    for cohort in ('replica8','scannet_cf18'):
        for method in ('L00','L10','L01','L11'):
            ids=[f'legacy:{cohort}:{method}_seed{seed}' for seed in SEEDS]
            rows.append(dict(label=cohort+' '+method,cells=[_cell(results,ids,m) for m in ('apall','ap50','ap25','miou','macc')]))
    for benchmark,pool in standard_pools.items():
        for method in ('U00','U10','U01','U11'):
            ids=[f'standard:{benchmark}:{method}_seed{seed}' for seed in SEEDS]
            if not any(key in results for key in ids):continue
            full = all(key in results and results[key]['full_scene_coverage'] for key in ids)
            rows.append(dict(label=benchmark+(' FULL_STANDARD ' if full else ' PARTIAL_SUBSET ')+method,cells=[_cell(results,ids,m) for m in ('apall','ap50','ap25','miou','macc')]))
    for benchmark,row in native_external['benchmarks'].items():
        if row['metrics'] is not None:
            rows.append(dict(label=benchmark+' '+row['scope']+' OVI-MAP protocol port',
                cells=[_cell(results,['external:OVI-MAP:'+benchmark],m) for m in ('apall','ap50','ap25','miou','macc')]))
    for method, port in replica_ports.items():
        if port['metrics'] is not None:
            rows.append(dict(label='replica48 '+port['scope']+' '+method+' dataset port',
                cells=[_cell(results,[f'external:{method}:replica48'],m)
                       for m in ('apall','ap50','ap25','miou','macc')]))
    add('S2','Legacy fixed-eligible and standard results',['AP','AP50','AP25','mIoU','mAcc'],
        'Legacy Replica51/CF18 only modify the fixed eligible owner domain. They are excluded from T1/T2. '
        'Standard partial pools are explicitly PARTIAL_SUBSET and do not represent full benchmark results.',rows)
    rows=[]
    for benchmark,report in source_availability['benchmarks'].items():
        for pattern in ('NONE','F','N','Q','NQ','NF','QF','NQF'):
            present=[] if pattern=='NONE' else list(pattern)
            groups=int(any(n in present for n in ('N','Q')))+int('F' in present)
            weights={name:0. for name in ('N','Q','F')}
            for name in present:
                weights[name]=1/groups/(sum(n in present for n in ('N','Q')) if name!='F' else 1)
            key=f'availability:{benchmark}:{pattern}'
            rows.append(dict(label=pattern,panel=f"{benchmark}: {report['completed_scenes']}/{report['expected_scenes']} scenes",
                cells=[_cell(results,[key],'count',unit='COUNT'),*[_text(weights[n]) for n in ('N','Q','F')],
                       _text('F only' if 'F' in present else 'unknown')]))
    add('S3','Actual available-source fusion',['Owners','N weight','Q weight','F weight','MFoff'],
        'GT-free availability over all native owners in completed scenes, counted once across seeds and variants. '
        'N/Q form one group and F another; available groups are normalized and weighted equally. '
        'Unknown has no prediction. Source class scores retain their own temperatures.',rows)
    add('S4','Historical structural controls and two-level ablation',
        ['GT Base','GT Novel','GT All','REAL AP','REAL AP50'],
        'Previously published structural experiment, cited without new inference. '
        'GT recognition uses historical matched1000 per-class balanced accuracy over its original conditions; '
        'REAL uses historical selected checkpoints and pooled proposal AP, with per-seed means. '
        'The LL/UL/LU/UU rows retain both levels of the original structural ablation. '
        'These exposed diagnostic protocols differ from the new standard GLOBAL evaluation and do not establish physical-correspondence gains.',
        [dict(label=row['method'],cells=[_cell(results,['historical:'+row['method']],m)
            for m in ('GT_Base','GT_Novel','GT_All','Real_AP','Real_AP50')]) for row in historical_results['rows']])
    rows=[]
    for row in summary['raw_and_post']:
        key=f"calibration:LOFO:{row['method']}:{row['panel']}:{row['seed']}"
        rows.append(dict(label=f"{row['method']} {row['panel']} seed{row['seed']}",cells=[_cell(results,[key],m) for m in ('B_macro','N_macro','H_macro','B_micro','N_micro','H_micro')]))
    add('S5','Per-seed RAW and LOFO calibration',['Base','Novel','H','Base micro','Novel micro','H micro'],
        'Same165 candidates per learned method/seed; seed17 selects each fold on eleven other physical families. '
        'OOF predictions are stitched into original order before recomputing macro metrics. Static FC is measured once.',rows)
    rows=[]
    for benchmark,scope in protocol['scopes'].items():
        for scene in scope['scenes']:
            path=root/'benchmark/bundles'/benchmark/scene/'bundle.json'
            if not path.is_file():continue
            bundle=verified(path)
            key=f'bundle:{benchmark}:{scene}'
            results[key]=dict(metrics={k:bundle[k] for k in ('native_owner_count','owners_with_F_view','zero_F_view_owners','one_F_view_owners')},
                source=dict(path=str(path.relative_to(root)),identity=bundle['identity']),protocol=protocol['identity'])
            coverage_key=f'coverage:{benchmark}:{scene}'
            cells=[_cell(results,[key],k,unit='COUNT') for k in ('native_owner_count','owners_with_F_view','zero_F_view_owners','one_F_view_owners')]
            cells += [_cell(results,[coverage_key],k,unit='COUNT') for k in ('valid_GT_instances','matched_GT_instances')]
            cells += [_cell(results,[coverage_key],'proposal_recall')]
            rows.append(dict(label=scene,cells=cells))
    add('S6','Class-independent proposal support',['All owners','F supported','Zero F','One F','Valid GT','Matched GT','Recall'],
        'All original native owners remain in the immutable registry. Support depends on geometry and image masks, never GT labels. '
        'A single qualified region view is valid. Full GT coverage/conditional matches are score-only diagnostics.',rows)
    rows=[]
    for key, value in results.items():
        if key.startswith('intervention:'):
            rows.append(dict(label=key.removeprefix('intervention:'),cells=[_cell(results,[key],metric,unit='COUNT')
                for metric in ('originals','changed','corrected','harmed','correct_before','correct_after')]))
    add('S7','Calibration corrections and damage on common geometric matches',
        ['Matched','Changed','Corrected','Harmed','Correct before','Correct after'],
        'Score-only diagnostics on shared native proposal IoU>0.5 matches, including unsupported F. '
        'Counts are per seed and subset; complete official AP retains every valid GT instance. '
        'Full target score/rank arrays accompany each scene receipt.',rows)
    counters=summary['counters']
    add('S8','Measured execution and cache costs',['Value','Unit','Scope'],
        'Logical candidates, new CPU work, GPU inference and training are distinct. '
        'Worker wall times include loading and CPU overhead; they are not isolated online module latencies or measured 30 FPS. '
        'NQ allocated memory is not the whole-pipeline peak. Standard/new baseline image encoding is counted.',
        [dict(label=k,cells=[_text(v),_text('actual counter'),_text('CPU calibration')]) for k,v in counters.items() if isinstance(v,(int,float))]
        +[dict(label=k,cells=[_cell(results,['cost:'+k],'count',unit='COUNT'),_text('physical calls'),_text('Completed real jobs')])
          for k in costs['counters']]+timing_rows)
    # Narrative cells are also source-bound. Numeric experimental cells above
    # retain their more specific score/bundle/cost records.
    for table in tables:
        metadata='presentation:'+table['id']
        results[metadata]=dict(metrics={}, source=dict(path='src/static_ovmap/benchmark_completion_v2/tables.py',
            parent_revision=BASE, producer=source_identity(('tables',)), key='table:'+table['id']),
            specification=dict(path='configs/static_ovmap/benchmark_completion_v2.json',
                sha256=digest_file(REPO/'configs/static_ovmap/benchmark_completion_v2.json')),
            protocol=protocol['identity'], calibration_summary=summary['identity'],
            resource_ledger=costs['identity'], data_role='PRESENTATION_METADATA;SEE_SCIENTIFIC_CELL_RECORDS')
        for row in table['rows']:
            for cell in row['cells']:
                if cell['value'] is not None and not cell['results']:
                    cell['results']=[metadata]
    store = write(root / 'result_store.json', dict(status='DONE', results=results, metric_unit='FRACTION',
        summary=summary['identity'], simple_recipe_lock=recipes['identity'], model_lock=None if model is None else model['identity'],
        protocol_lock=protocol['identity'], generated_from_frozen_real_records=True,
        author_values_never_count_as_local_executions=True, producer=source_identity(('tables', 'author_results'))))
    table_record=write(root/'tables.json',dict(status='DONE',result_store=store['identity'],tables=tables,
        T1_U11_and_T2_U11_share_identical_result_keys=True,missing_values_are_null=True,per_seed_H_before_mean=True))
    provenance=[]
    for table in tables:
        _emit(root,table)
        for ri,row in enumerate(table['rows']):
            for ci,cell in enumerate(row['cells']):
                if cell['value'] is not None:
                    provenance.append(dict(table=table['id'],row=ri,column=ci,label=row['label'],
                        raw_value=cell['value'],display_unit=cell['unit'],result_keys=cell['results'],
                        annotation=cell.get('annotation', ''),
                        metric=cell.get('metric'),aggregation=cell['aggregation'],result_store=store['identity']))
    (root/'cell_provenance.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in provenance))
    previews=[]
    if compile_previews:
        for name in ('T1','T2','T3'):
            path=root/'tables'/(name+'_preview.tex')
            path.write_text('\\documentclass[10pt]{article}\n\\usepackage[margin=12mm,paperwidth=210mm,paperheight='+('220' if name=='T3' else '150')+'mm]{geometry}\n'
                '\\usepackage{booktabs}\n\\begin{document}\n\\pagestyle{empty}\n\\setcounter{table}{'+str(int(name[1:])-1)+'}\n\\input{'+name+'.tex}\n\\end{document}\n')
            tex_sha=digest_file(root/'tables'/(name+'.tex'))
            wrapper_sha=digest_file(path)
            previous=previous_previews.get(name)
            if (previous and previous.get('source_tex_sha256')==tex_sha
                    and previous.get('preview_source_sha256')==wrapper_sha
                    and path.with_suffix('.pdf').is_file() and path.with_suffix('.png').is_file()
                    and digest_file(path.with_suffix('.pdf'))==previous.get('pdf_sha256')
                    and digest_file(path.with_suffix('.png'))==previous.get('png_sha256')):
                previews.append(previous)
                continue
            command=['pdflatex','-interaction=nonstopmode','-halt-on-error',path.name]
            completed=subprocess.run(command,cwd=path.parent,capture_output=True,text=True)
            (path.parent/(name+'_compile.log')).write_text(completed.stdout+completed.stderr)
            if completed.returncode:
                raise RuntimeError('Actual TeX compilation failed: '+name)
            subprocess.run(['pdftoppm','-singlefile','-scale-to','1800','-png',str(path.with_suffix('.pdf')),
                            str(path.with_suffix(''))],check=True,capture_output=True)
            previews.append(dict(table=name,pdf=str(path.with_suffix('.pdf')),png=str(path.with_suffix('.png')),
                actual_compile_exit=completed.returncode,source_tex_sha256=tex_sha,preview_source_sha256=wrapper_sha,
                pdf_sha256=digest_file(path.with_suffix('.pdf')),png_sha256=digest_file(path.with_suffix('.png'))))
    return write(root/'tables/receipt.json',dict(status='DONE',tables=table_record['identity'],
        result_store=store['identity'],cell_provenance_sha256=digest_file(root/'cell_provenance.jsonl'),previews=previews,
        rendered_visual_review='PENDING_PRIMARY_INSPECTION',complete_standard_evidence=all(
            row['completed_full_pools']==8 for row in standard_pools.values())))
