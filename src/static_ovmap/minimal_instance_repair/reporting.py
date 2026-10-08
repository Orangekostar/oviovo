"""Three evidence-bearing tables, compact provenance and normal publication."""

from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess

from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read

from .binding import REPO, seal
from .output_diagnostics import PAIRS, analyze_outputs


ARTIFACT = Path('artifacts/static_ovmap/minimal_instance_repair_v1')
REPORTS = ('MINIMAL_REPAIR_RESULTS.md','MINIMAL_REPAIR_HANDOFF.md',
           'MINIMAL_REPAIR_SELECTION.md','MINIMAL_REPAIR_CLAIMS.md')
SHORT = {'IR00_D2':'D2','IR01_G1':'G1','IR02_NEAREST_ATTACH':'Nearest attach',
    'IR03_EVIDENCE_ATTACH':'Evidence attach','IR04_DIRECT_GROUP':'Direct group',
    'IR05_VERIFIED_REPAIR':'Verified repair','IR06_ANYUP_REREAD':'AnyUp reread',
    'IR07_BOUNDARY_STABLE':'Boundary stable','IR08_COMBINATION':'Combination'}


def cost_accounting(binding):
    root = Path(binding['output_root'])
    counts,frames,failed = Counter(),[],[]
    for scene in binding['scenes']:
        plan = read(root/'recognition'/scene/'plan.json')
        for fid in plan['frames']:
            p = root/'recognition'/scene/'frames'/(fid+'.json')
            row = read(p)
            if row['status']!='COMPLETE':
                raise ValueError('scientific cost requires actual complete region frames')
            counts.update(row['counts'])
            frames.append({'scene':scene,'frame_id':int(fid),'identity':row['identity'],
                           'elapsed_seconds':row['elapsed_seconds'],'counts':row['counts']})
    for p in root.rglob('*.json'):
        if not ('failed_' in p.name or '/invocations/' in str(p) or '/history/' in str(p)):
            continue
        row = read(p)
        if row.get('status')=='FAILED':
            failed.append({'receipt':str(p),'identity':row.get('identity'),'reason':row.get('reason'),
                'counts':row.get('counts'),'seconds':row.get('elapsed_seconds',row.get('seconds'))})
    # These logged pilot controller failures preceded invocation timing receipts.
    # Their actual duration is unrecorded and therefore explicitly null.
    pilot_log = (root/'pilot_execution.log').read_text()
    if 'AttributeError' in pilot_log and 'in1d' in pilot_log:
        failed.append({'receipt':str(root/'pilot_execution.log'),'reason':'RELEASED_EVALUATOR_IN_NUMPY_2_ENV',
                       'counts':None,'seconds':None,'corrected_by':'CONTROLLER_ENV_EVALUATION'})
    observer = [read(root/'observations'/s/'receipt.json') for s in binding['scenes']]
    parity = read(root/'pilots/ordinary_anyup_parity.json')
    if counts['new_FC_frame_inputs']>832 or counts['AnyUp_QK_computations']>832:
        raise ValueError('scientific inference cap exceeded')
    timing = read(root/'timing/summary.json')
    cold = Counter()
    for row in timing['summary']:
        cold.update(row['counts'])
    return seal({'science_successful_unique_frames':len(frames),'science_counts':dict(counts),
        'science_frame_receipts':frames,'successful_observer_rasterizations':sum(len(r['frames']) for r in observer),
        'extra_pilot_FC_inputs':parity['extra_successful_pilot_FC_inputs'],
        'extra_pilot_AnyUp_QK':parity['extra_successful_pilot_AnyUp_QK'],
        'cold_actual_call_count':timing['call_count'],'cold_counts':dict(cold),
        'observed_failed_work':failed,'unrecorded_failure_duration_is_null':True,
        'science_and_cold_counters_separate':True,'new_maps':0,'new_segmentation_inference':0,'new_NQ_inference':0})


def table_data(store):
    metrics = store['pooled_metrics']
    table1 = []
    for method in SHORT:
        row = {'method':method,'name':SHORT[method]}
        for cohort in metrics:
            row[cohort] = {'metrics':metrics[cohort][method]['metrics'],
                'pool_identity':metrics[cohort][method]['identity'],
                'delta_IR01_fraction':{m:v-metrics[cohort]['IR01_G1']['metrics'][m] for m,v in metrics[cohort][method]['metrics'].items()},
                'delta_D2_fraction':{m:v-metrics[cohort]['IR00_D2']['metrics'][m] for m,v in metrics[cohort][method]['metrics'].items()}}
        table1.append(row)
    table2 = [r for r in store['output_diagnostics']['comparisons'] if (r['candidate'],r['reference']) in PAIRS]
    timed = {r['method']:r for r in store['timing']['summary']}
    table3 = []
    for method in SHORT:
        row = {'method':method,'name':SHORT[method],
            'replica8':metrics['replica8'][method]['metrics'],'scannet_cf18':metrics['scannet_cf18'][method]['metrics'],
            'seconds_per_scene':None,'allocated_GiB':None,'reserved_GiB':None,'call_count':0,
            'FC_inputs_per_call':None,'AnyUp_QK_per_call':None,'AnyUp_regions_per_call':None,'observer_frames_per_call':None}
        if method in timed:
            t = timed[method]
            baseline = timed['IR01_G1']
            row.update(seconds_per_scene=t['mean_seconds_per_scene'],allocated_GiB=t['peak_cuda_allocated_bytes']/2**30,
                reserved_GiB=t['peak_cuda_reserved_bytes']/2**30,call_count=t['calls'],
                time_ratio_vs_IR01=t['mean_seconds_per_scene']/baseline['mean_seconds_per_scene'],
                allocated_ratio_vs_IR01=t['peak_cuda_allocated_bytes']/baseline['peak_cuda_allocated_bytes'],
                incremental_allocated_GiB=t['incremental_peak_cuda_allocated_bytes']/2**30,
                incremental_reserved_GiB=t['incremental_peak_cuda_reserved_bytes']/2**30,
                exclusive_host_seconds=t['mean_exclusive_host_seconds'])
            for key,count in [('FC_inputs_per_call','FC_frame_inputs'),('AnyUp_QK_per_call','AnyUp_QK_computations'),
                              ('AnyUp_regions_per_call','AnyUp_region_pools'),('observer_frames_per_call','observer_frames')]:
                row[key] = t['counts'].get(count,0)/t['calls']
        table3.append(row)
    return [table1,table2,table3]


def displayed_tables(tables):
    pct = lambda x:f'{100*x:.2f}'
    number = lambda x:'—' if x is None else f'{x:.2f}'
    t1,t2,t3 = tables
    head1 = ['Method','R APall','R AP50','R mIoU','CF APall','CF AP50','CF mIoU','CF ΔAP G1','CF ΔAP D2']
    rows1 = [[r['method'][:4]+' '+r['name'],*[pct(r[c]['metrics'][m]) for c in ('replica8','scannet_cf18') for m in ('apall','ap50','miou')],
              pct(r['scannet_cf18']['delta_IR01_fraction']['apall']),pct(r['scannet_cf18']['delta_D2_fraction']['apall'])] for r in t1]
    head2 = ['Pair','Cohort','ΔAPall','ΔAP50','ΔmIoU','Ops A/B','Geom50 +/−','Geom75 +/−','GT50 +/−','GT75 +/−','W→R','R→W']
    rows2 = []
    for r in t2:
        matched = lambda t:next(x for x in r['thresholds'] if abs(x['threshold']-t)<1e-12)['released_class_aware']['counts']
        geometry = lambda t:next(x for x in r['thresholds'] if abs(x['threshold']-t)<1e-12)['class_agnostic']['counts']
        pair = r['candidate'][:4]+'−'+r['reference'][:4]
        op,s = r['operations'],r['original_P_semantic_changes']
        rows2.append([pair,'Replica' if r['cohort']=='replica8' else 'CF18',
            *[pct(r['metric_deltas_fraction'][m]) for m in ('apall','ap50','miou')],
            str(op['candidate_applied_operations'])+'/'+str(op['reference_applied_operations']),
            *[str(geometry(t).get('new_unique_GT_matches',0))+'/'+str(geometry(t).get('lost_unique_GT_matches',0)) for t in (.5,.75)],
            *[str(matched(t).get('new_unique_GT_matches',0))+'/'+str(matched(t).get('lost_unique_GT_matches',0)) for t in (.5,.75)],
            str(s.get('matchable_wrong_to_right',0)),str(s.get('matchable_right_to_wrong',0))])
    head3 = ['Method','R APall','R mIoU','CF APall','CF mIoU','s/scene','Alloc GiB','Reserv GiB','FC/call','QK/call','Regions/call','Obs/call']
    rows3 = [[r['method'][:4]+' '+r['name'],*[pct(r[c][m]) for c in ('replica8','scannet_cf18') for m in ('apall','miou')],
        *[number(r[k]) for k in ('seconds_per_scene','allocated_GiB','reserved_GiB','FC_inputs_per_call',
                                 'AnyUp_QK_per_call','AnyUp_regions_per_call','observer_frames_per_call')]] for r in t3]
    notes = [
        'Official dataset pools (%), fixed surface, one exclusive partition. Deltas are percentage points; all five metrics in source data.',
        'Prespecified pairs. Geom: class-agnostic matching; GT: released class-aware matching. W/R: geometrically matchable original P supports. Full thresholds and label changes in supplement.',
        'Fresh paired cold calls, 8 Replica scenes × 2 rounds per measured arm. Required models resident; peaks are maxima, time is the mean. Unmeasured: —.']
    return [(head1,rows1,notes[0]),(head2,rows2,notes[1]),(head3,rows3,notes[2])]


def write_tables(destination, store):
    data = table_data(store)
    displays = displayed_tables(data)
    for i,(rows,(headers,display,note)) in enumerate(zip(data,displays,strict=True),1):
        name = destination/('table_'+str(i))
        atomic_write_json(name.with_suffix('.json'),seal({'table':i,'source_result_store_identity':store['identity'],
            'metric_unit':'FRACTION','rows':rows,'printed_metrics_unit':'PERCENT','caption':note}))
        with name.with_suffix('.csv').open('w',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(display)
        md = '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
        md += '\n'.join('| '+' | '.join(map(str,row))+' |' for row in display)+'\n\n'+note+'\n'
        name.with_suffix('.md').write_text(md)
        escape = lambda s:str(s).replace('_',r'\_').replace('%',r'\%').replace('−','-').replace('—','--').replace('Δ',r'$\Delta$').replace('→',r'$\to$')
        tex = '\\begin{table*}[t]\n\\centering\n\\small\n\\begin{tabular}{l'+('r'*(len(headers)-1))+'}\n\\toprule\n'
        tex += ' & '.join(map(escape,headers))+r' \\'+'\n\\midrule\n'
        tex += '\n'.join(' & '.join(map(escape,row))+r' \\' for row in display)
        tex += '\n\\bottomrule\n\\end{tabular}\n\\caption{'+escape(note)+'}\n\\label{tab:minimal-repair-'+str(i)+'}\n\\end{table*}\n'
        name.with_suffix('.tex').write_text(tex)
    return displays


def render_preview(destination, displays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    target = destination/'table_layout_preview.pdf'
    if target.exists():
        return
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':9})
    with PdfPages(target) as pdf:
        for i,(headers,rows,note) in enumerate(displays,1):
            fig,ax = plt.subplots(figsize=(14,8.3))
            ax.axis('off')
            ax.set_title('Table '+str(i)+' · Minimal instance repair',loc='left',pad=18,fontsize=16)
            tab = ax.table(cellText=rows,colLabels=headers,cellLoc='right',loc='upper center',bbox=[0,.16,1,.72])
            tab.auto_set_font_size(False)
            tab.set_fontsize(9 if i==1 else 8)
            tab.auto_set_column_width(range(len(headers)))
            for (r,c),cell in tab.get_celld().items():
                cell.set_edgecolor('#D9D9D9')
                cell.set_linewidth(.4)
                if r==0:
                    cell.set_facecolor('#F2F2F2')
                    cell.get_text().set_fontweight('bold')
                if c==0:
                    cell.get_text().set_ha('left')
            import textwrap
            ax.text(0,.08,textwrap.fill(note,145),transform=ax.transAxes,ha='left',va='top',fontsize=9)
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(destination/('table_'+str(i)+'_preview.png'),dpi=120)
            plt.close(fig)


def compact_sources(binding, store, destination):
    root = Path(binding['output_root'])
    for source in ('source_binding.json','storage_manifest.json','implementation_freeze.json','selection.json','assets.json',
                   'pool_cache_restoration.json',
                   'observation_storage_preflight.json','diagnostics/summary.json','diagnostics/output_summary.json',
                   'timing/binding.json','timing/summary.json','pilots/summary.json','pilots/ordinary_anyup_parity.json'):
        path = root/source
        if path.is_file():
            dest = destination/'provenance'/source
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    for folder in ('history/freeze','tests'):
        for path in (root/folder).glob('*.json'):
            dest = destination/'provenance'/folder/path.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    for scene in binding['scenes']:
        dest = destination/'scenes'/scene
        dest.mkdir(parents=True,exist_ok=True)
        for kind,source in [('pose_roles',root/'observations'/scene/'pose_roles.json'),
                            ('supports',root/'observations'/scene/'supports.json'),
                            ('proposals',root/'proposals'/scene/'receipt.json'),
                            ('semantic_decisions',root/'recognition'/scene/'decisions.json'),
                            ('prediction_lock',root/'predictions'/scene/'receipt.json')]:
            shutil.copy2(source,dest/(kind+'.json'))
        diagnostic = read(root/'diagnostics'/scene/'opportunity.json')
        atomic_write_json(dest/'opportunity_summary.json',{k:diagnostic[k] for k in ('identity','summary','matching')})
        with gzip.open(dest/'output_comparisons.json.gz','wt') as stream:
            json.dump(read(root/'diagnostics'/scene/'output_comparisons.json'),stream,separators=(',',':'))
        plan = read(root/'recognition'/scene/'plan.json')
        # Source identities and reconstructible references, no masks or imagery.
        atomic_write_json(dest/'recognition_plan.json',plan)
    for cohort,methods in store['pooled_metrics'].items():
        for method,pool in methods.items():
            path = Path(pool['per_class_receipt'])
            dest = destination/'per_class'/cohort/(method+'.json')
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dest)
    calls = [read(root/'timing/calls'/(str(c['repeat'])+'_'+c['scene']+'_'+c['method'])/'receipt.json') for c in store['timing']['calls']]
    atomic_write_json(destination/'timing_calls.json',calls)


def reports(binding, store, destination):
    docs = REPO/'docs/paper/static_ovmap'
    selected,costs = store['selection'],store['costs']
    lines = ['# Minimal instance repair results\n',
        f"Actual coverage: {store['scene_method_coverage']}/234 scene-method records, {store['full_cohort_pool_coverage']}/18 released ordered pools. Fixed surface; one exclusive instance/semantic output. All cohorts were previously exposed.\n",
        f"Research recommendation: `{selected['selected']}`. TARGET_MET={selected['TARGET_MET']}; MATERIAL_TARGET_MET={selected['MATERIAL_TARGET_MET']}. Deployment: N0_UNCHANGED.\n"]
    for i in range(1,4):
        lines.append('## Table '+str(i)+'\n\n'+(destination/('table_'+str(i)+'.md')).read_text())
    lines.append('\nHardware and model load: see [timing provenance](../../../'+str(ARTIFACT)+'/provenance/timing/summary.json). '
        '[OviMAP Table 8](https://arxiv.org/html/2603.26541v1#S8) reports ms per processed keyframe on RTX3090 + i7-12700K; this experiment measures seconds per scene on A40 + Xeon Silver 4314. '
        'Tables 7/8 do not report peak VRAM. Different hardware and pipeline boundaries prevent an external speedup ratio. Scene latency cannot be relabeled as per-frame skipped frames or validated online 30 FPS. '
        'The earlier module comparison remains available in [RESOURCE_COMPARISON.md](RESOURCE_COMPARISON.md).\n')
    timed = {r['method']:r for r in store['timing']['summary']}
    baseline = timed['IR01_G1']
    lines.append('\nWithin this new series (16 calls per arm), '+ '; '.join(
        SHORT[m]+f": wall time {r['mean_seconds_per_scene']/baseline['mean_seconds_per_scene']:.3f}× G1, allocated peak {r['peak_cuda_allocated_bytes']/baseline['peak_cuda_allocated_bytes']:.3f}× G1"
        for m,r in timed.items() if m!='IR01_G1')+'. Reserved peaks describe allocator reservations, not model-only memory. Common Native/SigLIP/segmentation construction is outside this scene-repair boundary.\n')
    (docs/REPORTS[0]).write_text('\n'.join(lines))
    command = '/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_minimal_instance_repair.py --spec configs/static_ovmap/minimal_instance_repair_v1.json --parent-root /mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-minimal-instance-repair-v1/attempt_001 --phase all --resume'
    (docs/REPORTS[1]).write_text('# Reproduction handoff\n\n'+f"Freeze commit: `{store['freeze']['commit']}`. Binding: `{binding['identity']}`. Canonical store: `{store['identity']}`.\n\n"+
        '```bash\n'+command+'\n```\n\n'+f"Logical output: `{binding['logical_root']}`; physical storage: `{binding['output_root']}`. Large raster/mask tensors, licensed scans and model weights remain outside Git; manifests bind their hashes. No old runs were deleted.\n\n"+
        f"Scientific costs: `{json.dumps(costs['science_counts'],sort_keys=True)}`; observer rasters={costs['successful_observer_rasterizations']}; extra pilot AnyUp QK={costs['extra_pilot_AnyUp_QK']}. Cold calls={costs['cold_actual_call_count']}, counted separately. Recorded failures are preserved; unrecorded durations are null.\n\n"+
        'The initial complete implementation freeze preceded main acquisition. Corrective freezes subsequently record diagnostic tuple handling, scoped resume orchestration, protected-raw-zero class-change abstention, exact ordered pool aliasing and completed-timing resume. Their receipts are retained in the provenance directory; the supplied numerical protocol was unchanged. A uniform incumbent relabel that would change an original raw-zero row keeps the original class, including on an enlarged host support.\n\n'+
        'Evaluation uses the original controller environment; FC/AnyUp uses the inherited FC environment. This avoids invoking the released NumPy evaluator in NumPy 2.4. No environment upgrades or GT input to prediction were used.\n')
    (docs/REPORTS[2]).write_text('# Exact exposed-cohort selection\n\n'+f"Selected `{selected['selected']}`; TARGET_MET={selected['TARGET_MET']}; MATERIAL_TARGET_MET={selected['MATERIAL_TARGET_MET']}.\n\n"+
        'All five metrics must not decrease versus G1 in both cohorts; CF18 APall must strictly exceed D2 and CF18 AP50 must be at least D2. Epsilon=1e-10; the separate material flag is 0.10 percentage points. No gates were relaxed.\n\n'+
        '\n'.join('- `'+n+'`: '+('PASS' if d['TARGET_MET'] else 'FAIL')+'; failed checks: '+', '.join(d.get('failed_checks',[])) for n,d in selected['method_outcomes'].items())+'\n\nDeployment remains N0_UNCHANGED.\n')
    claims = ['# Claims and evidence boundaries\n',
        'Implementation, scientific coverage, target achievement, resource measurement and publication are separate statuses. Actual geometry remains fixed; structural arms change partitions. Semantic arms preserve masks.\n',
        'The relaxed candidate-set matching ceiling uses incompatible/overlapping hypotheses. It is neither an AP bound nor a deployable oracle. R overlaps may expose forbidden incumbent transfers; they do not authorize such edits.\n',
        'Proposal and verification views occupy disjoint pose bins in already-known sequences. They are not independent datasets. CF18 is eighteen captures of seven physical scene families, not full ScanNet200 validation. No statistical significance or untouched-test claim is made.\n']
    for r in store['output_diagnostics']['comparisons']:
        if (r['candidate'],r['reference']) in PAIRS:
            delta = r['metric_deltas_fraction']
            claims.append(f"- {r['cohort']} {r['candidate']} − {r['reference']}: ΔAPall={100*delta['apall']:.4f} pp, ΔAP50={100*delta['ap50']:.4f} pp, ΔmIoU={100*delta['miou']:.4f} pp; original-support semantic changes={r['original_P_semantic_changes']}; rank changes={r['rank_changes']}.\n")
    claims.append('\nUnique GT matches, duplicate FP entries and ambiguous score ties are separate ledgers. Label/support edits can change current-class ranks across scenes; fewer owners or FP entries alone do not establish recall improvement. Timed outputs match their own locked scientific outputs, not G1 for all arms.\n')
    (docs/REPORTS[3]).write_text('\n'.join(claims))


def tables(binding):
    root = Path(binding['output_root'])
    store = read(root/'result_store.json')
    if store['status']!='SCIENCE_COMPLETE':
        raise ValueError('three complete research tables require all 234/18 records')
    from .recognition import require_freeze
    freeze = require_freeze(binding)
    analysis = analyze_outputs(binding,store)
    selection,timing = read(root/'selection.json'),read(root/'timing/summary.json')
    if timing['status']!='COLD_TIMING_COMPLETE' or timing['call_count']!=64:
        raise ValueError('complete same-boundary table requires all 64 timed outputs')
    science = store.get('science_identity',store['identity'])
    store = seal({**store,'science_identity':science,'selection':selection,'timing':timing,
        'output_diagnostics':analysis,'costs':cost_accounting(binding),'freeze':freeze,
        'resource_comparison_provenance':{'own_boundary':binding['specification']['timing']['boundary'],
            'own_unit':'SECONDS_PER_SCENE','external_reference':'https://arxiv.org/html/2603.26541v1#S8',
            'external_unit':'MS_PER_PROCESSED_KEYFRAME','external_hardware':'RTX3090 + i7-12700K',
            'external_peak_VRAM_in_tables_7_8':'NOT_REPORTED','cross_hardware_speedup_ratio_valid':False,
            'skipped_frames_from_scene_latency_valid':False},
        'source_binding':binding['identity'],'all_three_tables_from_one_store':True})
    atomic_write_json(root/'result_store.json',store)
    destination = REPO/ARTIFACT
    destination.mkdir(parents=True,exist_ok=True)
    atomic_write_json(destination/'result_store.json',store)
    displays = write_tables(destination,store)
    compact_sources(binding,store,destination)
    reports(binding,store,destination)
    (destination/'visual_contract.md').write_text('Three main tables: complete official accuracy; five prespecified mechanism comparisons; same-boundary cold time/memory/work. Source: sealed result_store.json. Metrics: fractions in JSON, percent once in print; differences: pp. No confidence intervals or significance claimed. Three landscape preview pages, readable vector fonts, grayscale header grouping. Per-class/scene/threshold ledgers are supplemental.\n')
    render_preview(destination,displays)
    (destination/'ATTRIBUTION.md').write_text('Reuses the repository released OviMAP evaluator/capture/runtime operators, frozen FC and original AnyUp. AnyUp: https://github.com/wimmerth/anyup, commit 351807a9c4287368732cc247f26c7c81c9139af4, checkpoint SHA256 9d035c0f27114a6f32bdd3d8ed93b6cd39dad4b8f8bf94e69fddbfbf022901b2, CC-BY-4.0. Model weights and licensed scans are not published. Existing repository licenses remain in force.\n')
    return store


def publish(binding):
    root,destination = Path(binding['output_root']),REPO/ARTIFACT
    store = read(destination/'result_store.json')
    qa = read(root/'visual_qa.json')
    if qa['status']!='RENDER_INSPECTED' or qa['result_store_identity']!=store['identity']:
        raise ValueError('inspect the three rendered preview pages before publication')
    from .recognition import require_freeze
    freeze = require_freeze(binding)
    size = sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())
    if size>=binding['specification']['resources']['publish_target_MiB']*2**20:
        raise ValueError('compact Git artifact budget exceeded')
    locks = [read(root/'predictions'/s/'receipt.json') for s in binding['scenes']]
    roles = [read(root/'observations'/s/'pose_roles.json') for s in binding['scenes']]
    bank_integrity = all(len(r['selected'])<=32 and
        not ({x['pose_bin'] for x in r['selected'] if x['bank']=='proposal'} &
             {x['pose_bin'] for x in r['selected'] if x['bank']=='verification'}) and
        all(x['bank']==('proposal' if i%2==0 else 'verification') for i,x in enumerate(r['selected'])) for r in roles)
    manifests = [read(r['manifest']) for lock in locks for r in lock['methods'].values()]
    pilot = read(root/'pilots/summary.json')
    validation = read(root/'tests/final_validation.json')
    diagnostics = [read(root/'diagnostics'/s/'output_comparisons.json') for s in binding['scenes']]
    checks = {'coverage_234':store['scene_method_coverage']==234,'pools_18':store['full_cohort_pool_coverage']==18,
        'frozen_spec':freeze['spec_identity']==binding['spec'],'cold_64':store['timing']['call_count']==64,
        'postlock_26':store['output_diagnostics']['scene_count']==26,'observer_cap':store['costs']['successful_observer_rasterizations']<=832,
        'science_FC_cap':store['costs']['science_counts'].get('new_FC_frame_inputs',0)<=832,
        'science_QK_cap':store['costs']['science_counts'].get('AnyUp_QK_computations',0)<=832,
        'three_main_tables':all((destination/('table_'+str(i)+suffix)).is_file() for i in (1,2,3) for suffix in ('.md','.csv','.json','.tex')),
        'all_five_gate':store['selection']['all_five_gate_unchanged'],'deployment_unchanged':store['deployment']=='N0_UNCHANGED',
        'compact_budget':size<50*2**20,'render_inspected':qa['status']=='RENDER_INSPECTED',
        'zero_maps_and_segmentation':store['new_maps']==store['new_segmentation_inference']==0,
        'zero_NQ_inference':store['costs']['new_NQ_inference']==0,
        'extra_pilot_caps':store['costs']['extra_pilot_FC_inputs']<=2 and store['costs']['extra_pilot_AnyUp_QK']<=3,
        'both_real_pilots_18':pilot['status']=='INTEGRATED_PILOTS_VERIFIED' and pilot['real_partitions_scored']==18,
        'production_boundary_tests':validation['returncode']==0,
        'all_234_predictions_locked_GT_free':len(manifests)==234 and all(m['locked'] and m['GT_input'] is False for m in manifests),
        'pose_banks_disjoint_and_capped':bank_integrity,
        'edits_capped_no_incumbent_transfer':all(len(r.get('audit',{}).get('applied',[]))<=8 and
            r.get('audit',{}).get('moved_incumbent_rows',0)==0 for lock in locks for r in lock['methods'].values()),
        'all_loaded_threshold_diagnostics':all(len(d['thresholds'])==10 and d['labels_used_after_all_predictions_locked'] and
            d['duplicate_score_multiplicity_preserved'] for d in diagnostics),
        'model_environment_unchanged':read(root/'assets.json')['environment_modified'] is False,
        'preserved_other_research':not subprocess.check_output(['git','diff',binding['specification']['base_commit'],'--',
            'src/static_ovmap/cvpr_compact','src/static_ovmap/evidence_exploration','src/static_ovmap/runtime_parity'],cwd=REPO)}
    review = seal({'status':'REQUIREMENTS_VERIFIED' if all(checks.values()) else 'REQUIREMENTS_INCOMPLETE',
        'checks':checks,'artifact_bytes':size,'canonical_result_store_identity':store['identity'],
        'freeze_commit':freeze['commit'],'review_count':1,'visual_QA':qa,
        'evidence':{'implementation':'src/static_ovmap/minimal_instance_repair',
            'numeric_protocol':'configs/static_ovmap/minimal_instance_repair_v1.json',
            'predictions':'scenes/*/prediction_lock.json','pose_roles':'scenes/*/pose_roles.json',
            'postlock_diagnostics':'scenes/*/output_comparisons.json.gz',
            'actual_ordinary_AnyUp_parity':'provenance/pilots/ordinary_anyup_parity.json',
            'final_boundary_validation':'provenance/tests/final_validation.json',
            'initial_and_corrective_freezes':'provenance/history/freeze',
            'cold_call_parity_and_memory':'timing_calls.json','complete_metrics_and_costs':'result_store.json'},
        'specification_files':sorted(p.name for p in (REPO/'docs/paper/static_ovmap/minimal_instance_repair_v1/spec').iterdir())})
    atomic_write_json(destination/'requirement_review.json',review)
    if not all(checks.values()):
        raise ValueError('requirement review failed: '+str([k for k,v in checks.items() if not v]))
    subprocess.run(['git','add','--',str(ARTIFACT),*[str(Path('docs/paper/static_ovmap')/n) for n in REPORTS]],cwd=REPO,check=True)
    if subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO):
        subprocess.run(['git','commit','-m','Publish minimal repair accuracy, mechanism and paired resource evidence'],cwd=REPO,check=True)
    branch = binding['specification']['branch']
    subprocess.run(['git','push','origin',branch],cwd=REPO,check=True)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote = subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]
    if head!=remote:
        raise ValueError('publication remote full SHA differs from local HEAD')
    result = seal({'status':'PUSH_VERIFIED','local_HEAD':head,'remote_HEAD':remote,'branch':branch,
        'result_store_identity':store['identity'],'requirements_identity':review['identity'],
        'science_coverage':234,'pool_coverage':18,'report_paths':list(REPORTS)})
    atomic_write_json(root/'publication/final.json',result)
    return result


def run_reporting_phase(binding, phase):
    return tables(binding) if phase=='tables' else publish(binding)
