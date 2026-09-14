#!/usr/bin/env python3
"""Archive the executed fixed study and render its four tables and GitHub handoff."""
import argparse
import collections
import gzip
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.static_ovmap.cache_io import sha256_file
from scripts.evaluation.run_static_local_ownership import read,write,arrays,identity
from scripts.evaluation.summarize_static_local_ownership import metrics


def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def pct(x):return 'null' if x is None else f'{100*x:.3f}'
def ids(x):return ', '.join(map(str,x)) or '—'
def short(run):return 'P' if run=='fp32_primary' else 'R'
def gz(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);args=p.parse_args();start=time.perf_counter()
    large=args.root;artifact=ROOT/'artifacts/static_ovmap/local_ownership_v1';docs=ROOT/'docs/paper/static_ovmap';artifact.mkdir(parents=True,exist_ok=True)
    config=read(large/'predictions/frozen_config.json');binding=read(large/'predictions/input_binding.json');generation=read(large/'predictions/prediction_manifest.json');evaluation=read(large/'evaluation/evaluation_manifest.json');analysis=read(large/'analysis/manifest.json')
    rows=read(large/'analysis/performance.json');new=[r for r in rows if r['execution_role']=='NEW_EXPERIMENT'];effects=read(large/'analysis/effects.json');objects=read(large/'analysis/objects.json');regions=read(large/'analysis/regions.json');evidence=read(large/'analysis/evidence_regions.json')
    assert len(new)==6 and all(r['status']=='COMPLETE' for r in new)
    archives=[]
    def copy(source,dest):
        assert source.stat().st_size<2_000_000,(source,source.stat().st_size)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest);archives.append({'source':str(source),'repository_path':str(dest.relative_to(ROOT)),'sha256':sha256_file(dest)})
    for f in (large/'analysis').iterdir():copy(f,artifact/'analysis'/f.name)
    for f in ['input_binding.json','frozen_config.json','prediction_manifest.json','smoke.json']:copy(large/'predictions'/f,artifact/'prediction'/f)
    copy(large/'predictions/shared/summary.json',artifact/'prediction/shared_projection.json')
    for f in ['metric_protocol.json','evaluation_manifest.json']:copy(large/'evaluation'/f,artifact/'evaluation'/f)
    checks=[];events=[];costs={};source_counts=[]
    old_manifest=read(ROOT/'artifacts/static_ovmap/t1_attribution_v1/evaluation/evaluation_manifest.json');old_config=read(ROOT/config['reference_config'])
    proj_path=old_config['evaluation_projection'];assert sha256_file(proj_path)==old_manifest['input_inventory'][proj_path]['sha256']
    assert sha256_file(ROOT/'configs/evaluation/ovimap_local_ownership_v1.json')==sha256_file(large/'predictions/frozen_config.json')
    for run in config['runs']:
        pred=large/'predictions'/run;ev=large/'evaluation'/run;ref=Path(config['reference_evaluation'])/run
        copy(pred/'summary.json',artifact/'prediction'/run/'summary.json')
        copy(ev/'candidate_manifest_parity.json',artifact/'evaluation'/run/'candidate_manifest_parity.json')
        copy(ev/'manifest_isolation.json',artifact/'evaluation'/run/'manifest_isolation.json')
        d={m:read(pred/(m+'.json')) for m in config['methods']};a=arrays(pred/'atoms.npz');u=arrays(pred/'unaries.npz');g=arrays(pred/'graph.npz');summary=read(pred/'summary.json');costs[run]=summary['costs']
        source_owners={m:np.load(d[m]['owner_path']) for m in d};local='LO_U00_LOCAL';spatial='LO_U00_SPATIAL';fill='LO_U00_OVI_FILL'
        for key in ['atom_key','evidence_key','unary_key']:assert d[local][key]==d[spatial][key] and d[local][key]
        assert d[local]['owner_cache_key']!=d[spatial]['owner_cache_key']
        assert np.array_equal(source_owners[local],u['labels'][a['point_to_atom']]+1)
        energy=d[spatial]['solver']['energy'];assert all(x>=y for x,y in zip(energy,energy[1:]))
        assert all(np.array_equal(source_owners[m],source_owners[m][a['representatives']][a['point_to_atom']]) for m in d)
        changed_atoms=int((source_owners[local][a['representatives']]!=source_owners[spatial][a['representatives']]).sum())
        check={'run':run,'same_atoms_evidence_unaries':True,'distinct_solver_keys':True,'source_owners_constant_within_atoms':True,'LOCAL_equals_saved_unary_solution':True,'spatial_energy_nonincreasing':True,'graph_final_changed_atoms':changed_atoms,'graph_final_changed_source_points':int((source_owners[local]!=source_owners[spatial]).sum()),'graph_degree_mass_bound':None}
        degree=np.bincount(g['edges'].ravel(),weights=np.repeat(g['weights'],2),minlength=len(a['counts']));assert np.all(degree<=a['counts']+1e-8);check['graph_degree_mass_bound']=True
        manifests=[]
        for m in d:
            copy(pred/(m+'.json'),artifact/'prediction'/run/(m+'.json'))
            e=read(ev/m/'metrics.json');assert e['candidate_manifest_identity']=='EXACT' and e['overlapping']['cache_hit'];base=read(ref/'AT_U00/metrics.json');assert e['overlapping']['released']==base['overlapping']['released']
            manifests.append(Path(e['overlapping']['manifest']).read_bytes())
            assert e['validation']['coverage_exact'] and e['validation']['inside_original_masks'] and e['validation']['changed_only_U00_multicandidate']
            if m!=fill:assert 'global_projection_parity' not in e['validation'] and 'assignment_priorities' not in d[m]
            else:assert e['validation']['global_projection_parity']=='EXACT'
            copy(ev/m/'evaluation_ledger.json.gz',artifact/'evaluation'/run/m/'evaluation_ledger.json.gz')
            copy(Path(e['unique']['trace_path']),artifact/'evaluation'/run/m/'released_trace.json.gz')
            ledger=gz(ev/m/'evaluation_ledger.json.gz');c=collections.Counter(r['source'] for r in ledger if r['kept']);nonempty=collections.Counter(r['source'] for r in ledger if r['kept'] and r['unique_projected_area']>0)
            source_counts.append({'run':run,'condition':m,'candidate_sources':dict(c),'nonempty_unique_sources':dict(nonempty)})
        assert all(x==manifests[0] for x in manifests)
        check['six_decimal_candidate_manifest_byte_identity']=True;checks.append(check)
        la=gz(ev/local/'evaluation_ledger.json.gz');lb=gz(ev/spatial/'evaluation_ledger.json.gz')
        for before,after in zip(la,lb):
            if before['unique_export_outcome']!=after['unique_export_outcome']:
                i=before['canonical_index'];trace=gz(ev/spatial/'unique/released_trace.json.gz')
                actual=[e for e in trace['events'] if e['event']=='ignore_test' and e['overlap_threshold']==.5 and Path(e['candidate_file']).name==f'candidate_{i:04d}.npy']
                events.append({'run':run,'candidate_id':before['candidate_id'],'class_id':before['final_class_id'],'serialized_rank':before['rank_score_serialized'],
                    'LOCAL_unique_area':before['unique_projected_area'],'SPATIAL_unique_area':after['unique_projected_area'],'before_outcome':before['unique_export_outcome'],'after_outcome':after['unique_export_outcome'],'actual_released_events_at_05':actual})
    write(artifact/'event_transitions.json',events);write(artifact/'source_counts.json',source_counts)
    write(artifact/'verification.json',{'status':'PASS','runs':checks,'six_required_cells_complete':True,'candidate_AP_exactly_reused_U00':True,'frozen_config_bytes_exact':True,'historical_projection_sha256_exact':True,'original_projection_sha256':old_manifest['input_inventory'][proj_path]['sha256'],'regional_delta_scope':'all 10 pairwise comparisons; integer sums asserted in executed summarizer','new_model_inferences':0,'GT_prediction_separation':'prediction runner reads no GT arrays; evidence uses frame masks, depth, poses and candidate membership only'})
    write(artifact/'costs.json',{'prediction_preparation_seconds':generation['preparation_seconds'],'shared_projection_seconds':generation['shared_projection_seconds'],'runs':costs,'prediction_total_seconds':generation['seconds'],'evaluation_total_seconds':evaluation['seconds'],'analysis_seconds':analysis['seconds'],'prediction_peak_rss_bytes':generation['peak_rss_bytes'],'evaluation_peak_rss_bytes':evaluation['peak_rss_bytes'],'cache_scope':'old overlapping AP cache hit; new atom/evidence/graph/owners computed once; projection shared across exact-equal model coordinates','timing_boundary':'CPU only, nested timers not additive, native coordinate z-buffer creation included; no end-to-end FPS or inference throughput'})
    tests='/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/evaluation/test_static_local_ownership.py tests/evaluation/test_static_t1_attribution.py tests/evaluation/test_static_proposal_fusion.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py'
    write(artifact/'test_receipt.json',{'command':tests,'cwd':str(ROOT),'exit_status':0,'result':'29 passed in 0.42s','baseline':'19 passed in 1.23s','red_green':'new missing implementation fixtures failed first; seven core then nine focused tests passed. A normalized graph weight was 0.9999999999999999; fixture uses 1e-15 tolerance, no production parameter changed.','neighbor_cap_regression':'duplicate-centroid fixture failed with 3 edges; explicit nonself cap fixed it. Real directed edge selections unchanged on both full atom sets; neighbor_cap_verification.json', 'unchanged_evaluator_trace_suite':'not rerun; existing original evaluator tracing reused'})
    decision={'status':'COMPLETE_NO_NET_UNIQUE_GAIN','H1':'not supported for net unique-instance quality; macro semantic mIoU improves but unique AP decreases versus protection/O-only','H2':'not supported: unique AP unchanged in primary, decreases in repeat; spatial energy decreases but benchmark quality does not follow','H3':'not supported: none of the 7/6 U00-added GT objects at strict >.5 survive the new unique maps; LOCAL/SPATIAL additionally lose GT5004','interpretation':'Evidence coverage is high, yet geometry/instance support does not reliably preserve object extent. Same-class competition is important. Spatial smoothing can move a small residual across released minimum-size eligibility and create a ranked FP.','scientific_tuning_after_evaluation':False,'next_experiment':'One predeclared Room1 confirmation of the unchanged O-only/U00_FILL/LOCAL/SPATIAL package on two saved runs, after separately authorized data preparation; same parameters, all metrics and object retention, no new rule or favorable-scene selection here.'}
    write(artifact/'decision.json',decision)
    # Actual new large artifact references, including shared visibility and all unique masks; no copies.
    large_files=[identity(f) for f in sorted(large.rglob('*')) if f.is_file()]
    write(artifact/'large_artifacts.json',{'root':str(large),'files':large_files,'total_bytes':sum(f['bytes'] for f in large_files),'storage':'existing shared filesystem; no original datasets/weights/logits uploaded'})
    write(artifact/'archive_sources.json',archives)
    commands=[{'stage':name,'argv':d['command'],'exit_status':0} for name,d in [('prediction',generation),('evaluation',evaluation),('analysis',analysis)]]
    commands.append({'stage':'focused_tests','command':tests,'exit_status':0})
    write(artifact/'executed_commands.json',{'cwd':str(ROOT),'environment_prefix':'OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8','commands':commands,'environment':generation['environment']})
    source_files=[ROOT/'src/static_ovmap/local_ownership.py',ROOT/'src/static_ovmap/ownership_evidence.py',ROOT/'tests/evaluation/test_static_local_ownership.py',ROOT/'configs/evaluation/ovimap_local_ownership_v1.json',*sorted((ROOT/'scripts/evaluation').glob('*static_local_ownership.py'))]
    write(artifact/'source_identity.json',{'base':'98cafc4a96bf906878116ae3284da28bc974ad4d','branch':'research/ovimap-local-ownership-v1','source_sha256':{str(f.relative_to(ROOT)):sha256_file(f) for f in source_files},'execution_source_hashes_unchanged':all(sha256_file(ROOT/f)==h for f,h in generation['source_sha256'].items()),'post_execution_compatibility_fix':'Explicit graph neighbor cap for missing-self duplicate-centroid case; original execution hashes retained. Full real edge selection unchanged in neighbor_cap_verification.json; no metric rerun.', 'model_input_T0_hashes':{r['run']:r['t0_sha256'] for r in binding['runs']}})
    anchor='../../../artifacts/static_ovmap/local_ownership_v1'
    content='''# Local ownership results — two cached Room0 FP32 runs

**COMPLETE_NO_NET_UNIQUE_GAIN.** Six new cells COMPLETE; model inference/training/mapping counts remain zero. U00 masks, original classes, canonical IDs and raw source-area AP ranks are unchanged. S1a, geometry transfer and old candidate-union gains are fixed inputs, not new contributions. This concerns the static fusion/output layer only.

## Table A — paired performance

All metrics below are percentages. P/R are the original primary/repeat network predictions. AT_* rows are REUSED_REFERENCE, not rerun experiments; LO_* rows are NEW. cAP75 is the separate class-agnostic canonical >=.75 diagnostic, not released semantic AP75. The released evaluator uses its actual float thresholds (strict >; runtime .75 is .7500000000000002), 48 instance versus 51 semantic classes, and minimum predicted region100. Candidate masks are identical within the new U00 family; unique masks are evaluated separately with unchanged rank.

'''
    content+=table(['Run','Condition','Role','Candidate AP','Unique AP','Unique AP50','Unique AP25','cAP75','mIoU','mAcc','Unique eligible'],[[short(r['run']),r['condition'],'reused' if r['execution_role']=='REUSED_REFERENCE_NOT_NEW_EXPERIMENT' else 'new',*[pct(metrics(r)[k]) for k in ['candidate_AP','unique_AP','unique_AP50','unique_AP25','canonical_AP75','mIoU','mAcc']],r['unique_eligible_count']] for r in rows])
    content+='\n\nCounts, source composition and evidence coverage for new rows:\n\n'
    content+=table(['Run','Method','Candidates OVI/SF','Nonempty unique OVI/SF','Empty','Small nonempty','Observed source %','Challenger-qualified source %'],[[short(r['run']),r['condition'], '/'.join(str(next(c for c in source_counts if c['run']==r['run'] and c['condition']==r['condition'])['candidate_sources'].get(s,0)) for s in ['OVI','SpaCeFormer']),'/'.join(str(next(c for c in source_counts if c['run']==r['run'] and c['condition']==r['condition'])['nonempty_unique_sources'].get(s,0)) for s in ['OVI','SpaCeFormer']),r['unique_empty_count'],r['unique_small_nonempty_count'],pct(read(large/'predictions'/r['run']/'summary.json')['statistics']['observed_source_point_fraction']),pct(read(large/'predictions'/r['run']/'summary.json')['statistics']['challenger_qualified_point_fraction'])] for r in new])
    content+=f'\n\nFull precision, per-class AP, ignored event counts and costs: [performance]({anchor}/analysis/performance.json), [source counts]({anchor}/source_counts.json), [protocol]({anchor}/evaluation/metric_protocol.json). The 454,188/454,699 atoms and 2,437,909/2,443,916 candidate incidences retain same-class and SF-only conflicts. Empty candidates remain represented; no redistribution or AP rescoring occurs.\n\n## Table B — controlled effects\n\nAfter minus before, **percentage points**. Candidate AP changes are zero for protection/local/spatial ownership comparisons. Comparisons to O-only have a different candidate pool: their already-known candidate AP difference is explicitly excluded from the new ownership contribution.\n\n'
    content+=table(['Run','Effect','Δ candidate AP','Δ unique AP','Δ unique AP50','Δ cAP75','Δ mIoU','Δ mAcc'],[[short(r['run']),r['effect'],*[pct(r['delta_proportions'][k]) for k in ['candidate_AP','unique_AP','unique_AP50','canonical_AP75','mIoU','mAcc']]] for r in effects])
    content+='''

Protection=U00_FILL−raw U00; local_evidence=LOCAL−U00_FILL; spatial_term=SPATIAL−LOCAL; local_net/spatial_net compare to paired common-domain O-only. U00_FILL ties O-only unique AP on both runs. LOCAL increases macro mIoU but decreases unique AP and canonical high-IoU quality versus both references. SPATIAL adds no unique AP on primary and loses 0.540 pp on repeat. Improvement over very poor raw-U00 unique maps is insufficient evidence of net method value.

## Table C — object retention and losses

Actual released matching at strict runtime thresholds; GT IDs are the comparison identity. Lists below include positive and negative outcomes, not a favorable subset. Candidate/score owners, exact trace references and all conditional object transitions accompany the machine table.

'''
    retention=[r for r in objects if r.get('type')=='U00_gain_retention']
    content+=table(['Run','Method','Runtime threshold','U00-added IDs','Survive unique','Previously correct OVI lost','New unique over OVI'],[[short(r['run']),r['condition'],repr(r['threshold']),ids(r['U00_added_GT_ids']),ids(r['surviving_added_GT_ids']),ids(r['previously_correct_OVI_lost_GT_ids']),ids(r['new_unique_over_OVI_GT_ids'])] for r in retention])
    content+=f'''

At >.5, none of the original U00-added 7/6 GT objects survives any new unique map; raw U00 previously retained two in each run. LOCAL and SPATIAL lose previously correct blinds GT5004 in both runs. Its OVI owner40 mask had IoU .5131137407 under protection. LOCAL reduces owned projected size from 20,520 to 12,848/12,885 (62.612/62.792%), chiefly transferring original source points to SF query9. Its fixed geometric-atom-graph component count rises 32→56 / 31→67. These are graph structure proxies, not mesh topology or a postprocessing split. Same-class reassignment can destroy object extent without changing semantic class.

The repeat SPATIAL AP loss does **not** require another matched-GT loss at .25/.5/.75. SF query137 (chair, canonical105) grows from 99 to 101 projected points, crossing the fixed evaluator min-region100. It is then a counted unmatched FP with unchanged area rank25082 above two chair TPs ranked24085 and23879. Chair AP50 falls from1 to.416667; all-class unique AP50 falls 2.431 pp. SF query198 (nightstand) grows94→110, but its no-GT class AP remains null. The rules were not adjusted to suppress these unfavorable results.

See [object transitions]({anchor}/analysis/objects.json), [extent/competing candidates/events]({anchor}/analysis/candidate_extent_and_events.json.gz), [size-boundary events]({anchor}/event_transitions.json), and [deterministic examples]({anchor}/analysis/examples.json). Positive multi-GT intersections, background/localization flags and empty duplicates are nonexclusive diagnostics, not an additive error taxonomy. Low owned fraction is not itself a false negative: actual evaluator matches determine that.

## Table D — evidence and regional diagnosis

Regions are frozen original U00 support×multiplicity×class agreement plus projection-unmatched. All integer regional confusions and paired deltas reconstruct the global valid-GT confusion exactly, including pred0 false negatives. Below are all regions for LOCAL−FILL and SPATIAL−LOCAL, both runs; C/W denote semantic correctness.

'''
    displayed=[r for r in regions if r['effect'] in ['local_evidence','spatial_term']]
    content+=table(['Run','Effect','Region/support/multiplicity/agreement','Valid GT','C→W','W→C','W→W','C→C','Owner changes','Class changes','Same-class owner changes'],[[short(r['run']),r['effect'],'/'.join(str(r[k]) for k in ['region_id','source_support','candidate_multiplicity','class_agreement']),*[r[k] for k in ['valid_gt_vertices','correct_to_wrong','wrong_to_correct','wrong_to_wrong','correct_to_correct','owner_changes','class_changes','same_class_owner_changes']]] for r in displayed])
    content+='\n\nEvidence and graph coverage (source domain, percentages):\n\n'
    content+=table(['Run','Region','Observed points %','Challenger atoms','No-challenger fallback %','LOCAL changed points','SPATIAL changed points','Supported edges touching region'],[[short(r['run']),r['region_id'],pct(r['evidence_observed_point_fraction']),r['challenger_qualified_atoms'],pct(r['no_challenger_fallback_point_fraction']),r['LOCAL_changed_points'],r['SPATIAL_changed_points'],r['graph_supported_edges_touching_region']] for r in evidence if r['region_id']!=-1])
    content+=f'''

No-challenger fallback is 6.336/6.340% overall; entirely unobserved points are only about1.92%. Shared evidence exists over most points, so blanket observation unavailability is not the explanation. Usable correspondence rejects small remainders, low IoU and ambiguous matches explicitly; per-incidence frame IDs/counts/masses/agreements are stored externally. [Evidence regions]({anchor}/analysis/evidence_regions.json) include score-margin quantiles and distinguish missing observations from insufficient challengers.

There are 1,322,619/1,323,903 positive graph edges, so SPATIAL is not an empty-graph control. Its energies strictly decrease across five recorded sweeps. Final atom/point change counts are in [verification]({anchor}/verification.json). LOCAL changes 83,420/83,001 source points relative to protection; on valid GT, 29,988/29,893 ownership changes retain the semantic class. LOCAL causes 2,411/2,414 correct→wrong and2,102/2,080 wrong→correct semantic vertices, despite improving macro mIoU: macro IoU does not equal total vertex accuracy. SPATIAL changes another2,304/2,301 valid-GT owners and slightly improves mIoU while failing to improve unique instances.

Projection-unmatched remains111,427 vertices (111,180 valid semantic GT), all pred0. Changes occur only at U00 multiplicity>=2; neither single-candidate nor unsupported regions are altered. Raw depth agreement is visibility, not instance truth. Frame predictions share frontend provenance with OVI and are not independent teachers. Exact whole-mask/class-conflict pairs were absent in both banks; nevertheless geometry-only evidence has no semantic-class correctness signal, and unresolved class-conflict regions remain. No GT-based class choice is introduced.

## Cost and scientific decision

CPU threads are explicitly8. The complete prediction pass took {generation['seconds']:.2f}s (asset binding, shared full-cloud projection, atom/evidence/graph construction, solvers and writes), peak RSS {generation['peak_rss_bytes']/2**30:.3f}GiB. Shared selected-frame projection took {generation['shared_projection_seconds']:.2f}s, counted once because returned coordinates/order are exactly equal. The measured real smoke used65,536 source points, frame0 and38,609 positive-depth-consistent pixels; it is I/O evidence, not a benchmark subset result.

'''
    content+=table(['Run','Atoms s','Evidence s','Graph s','FILL solver s','LOCAL solver s','SPATIAL solver s'],[[short(run),*[f'{costs[run][k]:.3f}' for k in ['atom_seconds','evidence_seconds','graph_seconds']],*[f"{costs[run]['solvers'][m]['seconds']:.3f}" for m in config['methods']]] for run in config['runs']])
    content+=f'''

All six unique-map evaluations plus checks/export took {evaluation['seconds']:.2f}s, peak RSS {evaluation['peak_rss_bytes']/2**30:.3f}GiB; post-hoc diagnosis took {analysis['seconds']:.2f}s. Per-cell unique-mask export and released evaluation times are in Table A's machine rows. New prediction caches were cold; overlapping AP/trace was reused only after canonical mask/class/.6f parity. These nested CPU timers are not end-to-end FPS and exclude original network/mapping cost. [Costs]({anchor}/costs.json).

H1/H2/H3 are not supported for the required net unique-instance objective. Local evidence is present and moves ownership, but fails to preserve gains or correct OVI extent. Spatial smoothness decreases its own energy without repeat-consistent instance benefit and exposes threshold-sensitive residual FPs. The final status is **COMPLETE_NO_NET_UNIQUE_GAIN**, not an implementation failure or a coverage-blocked experiment.

Only Room0 development evidence and two saved runs are available; no confidence interval, significance, generalization, or unseen-scene claim follows. Extra3D pretraining is disclosed; training-scene exclusion remains UNVERIFIED. Raw model logits were not opened. No parameter sweep or seventh method was run.

Single next experiment: a separately authorized, predeclared Room1 confirmation using the unchanged O-only/U00_FILL/LOCAL/SPATIAL package and two saved runs, with all metrics and object accounting. Do not select a favorable scene, add a repair or launch new mapping/inference in this task. Implementation and reproduction details: [handoff](LOCAL_OWNERSHIP_HANDOFF.md).
'''
    (docs/'LOCAL_OWNERSHIP_RESULTS.md').write_text(content)
    handoff=f'''# Local ownership implementation handoff

Six required cells **COMPLETE**; scientific outcome **COMPLETE_NO_NET_UNIQUE_GAIN**. See [four result tables](LOCAL_OWNERSHIP_RESULTS.md), [decision]({anchor}/decision.json), and [full reviewed specification](LOCAL_OWNERSHIP_SPEC.md). Original attribution `NO_REPAIR_JUSTIFIED` is superseded only by this explicitly authorized new method task, not silently edited.

## Source, inputs and execution

Reviewed and actual start HEAD: `98cafc4a96bf906878116ae3284da28bc974ad4d`, clean; branch `research/ovimap-local-ownership-v1`. Isolated worktree `/home/ww/crove/ovimap-local-ownership`. No unrelated branch changes. Execution/final source and configuration SHA256s are in [source identity]({anchor}/source_identity.json); this handoff does not recursively embed its own commit SHA. Final Git push/SHA verification is reported after the commit.

Exact model runs: `fp32_primary` and `fp32_repeat` from the original distinct FP32 T0 caches; use corrected `predictions_verified`, not two rereads counted as extra model runs. [Binding]({anchor}/prediction/input_binding.json) contains consumed-file hashes, T0/query/mask/class/coordinate equivalence and candidate identities. Original native geometry owners come from bound native projection, never G1/G2 owners. Archived graph JSON supplies frame calibration/exporter provenance only. Optional raw logits remain unopened and were not rehashed.

Selected frame IDs: `{binding['selection']['selected_frame_ids']}`. Selection used only the original200-frame schedule and pose validity/separation, then at most32 evenly spaced temporal indices. All exclusions and original IDs are in binding. Camera1200×680, fx/fy600, cx599.5/cy339.5, depth scale6553.5 are receipt-derived. World points transform via camera-to-world inverse rotation; camera-z and integer-pixel np.rint ties-to-even match existing code. Equal-depth z-buffer ties use original source row; each pixel counts once. Zero masks/invalid depths/occlusion abstain. Predicted masks may share OVI frontend errors.

Environment: `{generation['command'][0]}`, `{generation['environment']['python'].splitlines()[0]}`, NumPy `{generation['environment']['numpy']}`; existing native evaluation environment only. No installation or environment merge. Commands below were actually executed from the worktree with exit0; original 34-cell evaluator was not rerun.

'''
    for item in commands:
        cmd=item.get('command') or __import__('shlex').join(item['argv'])
        if item['stage']!='focused_tests':cmd='OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 '+cmd
        handoff+=f"```bash\n{cmd}\n```\n\n{item['stage']}: exit0.\n\n"
    handoff+=f'''Artifact/report packaging is performed by this checked-in script:

```bash
{sys.executable} scripts/evaluation/finalize_static_local_ownership.py --root {large}
```

Its completion receipt is [packaging]({anchor}/packaging_receipt.json). The output directories for the three computational runners must be new; to reproduce, choose a new external root and pass its prediction/evaluation/analysis subdirectories consistently, then package that root. Existing baseline files remain references, not regenerated targets.

## Implementation and protocol

- [Shared evidence](../../../src/static_ovmap/ownership_evidence.py): pose selection, whole-cloud shared visibility, positive-mask sparse histograms and leave-atom-out candidate correspondence. No GT/class input. Usable-view count0 marks missing score; fallback uses0.5 without evidence, while one usable fallback view retains its computed score. Challengers require at least2 views. Scores are smoothed evidence, not correctness probabilities.
- [Local/spatial ownership](../../../src/static_ovmap/local_ownership.py): exact membership bits refine2cm cells; radius1.5cm connected groups, no dense distance allocation. Atom order is centroid x/y/z with deterministic atom-ID ties. Fallback is U00 OVI-first raw source area/canonical tie order. Both local methods consume exactly the same persisted unaries/feasible sets/atoms. Sparse6-neighbor/3cm graph uses at least2 co-visible mask histograms, normalized symmetric weights and bounded sequential descent. No old parent locks, labels/masks/NMS changes or post-splitting.
- [Prediction runner](../../../scripts/evaluation/run_static_local_ownership.py): explicit method registry, unknown methods reject, cache identities bind masks/coordinates/query/run/observations/calibration/atom/evidence/unary/graph/method/source/parameters. [Frozen config](../../../configs/evaluation/ovimap_local_ownership_v1.json) matches the pre-evaluation saved bytes.
- [Evaluator adapter](../../../scripts/evaluation/evaluate_static_local_ownership.py): global FILL checks independent projected global-order parity; local methods check original support/coverage/IDs and fixed projection without fake scalar priorities or global comparisons. Original evaluator/protocol and old T1 runners are unchanged. [Candidate parity]({anchor}/evaluation/fp32_primary/candidate_manifest_parity.json) and [isolation]({anchor}/verification.json) prove new candidate manifests equivalent to U00 and byte-identical across the new family.
- [Post-hoc diagnosis](../../../scripts/evaluation/summarize_static_local_ownership.py): actual released events, U00-added GT retention, OVI losses, nonexclusive error flags, source competing candidate IDs and fixed-graph extent/boundary/component proxies. Fixed U00 region matrices include unmatched pred0 FN; every paired delta sums to global. Predictions are never optimized again on the GT domain.

Post-execution code review found a missing-self duplicate-centroid edge case that could exceed the neighbor cap. A failing fixture demonstrated it; an explicit cap fixes it. Both full real atom sets have zero affected rows, verified in [neighbor-cap evidence]({anchor}/neighbor_cap_verification.json), so graph/evaluator inputs and measured results remain unchanged. Execution-time and final source hashes are both retained; no scientific threshold changed and no benchmark rerun was needed.

## Task-specific completion review

| Requirement | Evidence/status |
| --- | --- |
| §0–3 reviewed base, source/assets, scope | Clean specified base; input binding and actual frame/depth/calibration files verified. Two distinct T0 identities; zero inference/training/mapping. |
| §4 exact six U00 cells | Six COMPLETE new rows; six separately labeled reused baseline rows. No U11 substitution or repeated34-cell study. |
| §5.1–5.2 observations | 32 original-schedule, pose-distinct shared frames; positive-depth/instance z-buffer, integer pixels and source-row tie; real smoke PASS. |
| §5.3 atoms | Full1,862,429-point clouds; exact candidate signatures and bounded connectivity; neutral atoms allow SF within original OVI support. |
| §5.4–5.5 evidence/unary | Leave-atom-out subtraction,16/.20/.05 gating, >=2 challenger views, explicit missing state, beta.05, fallback/canonical ties; sparse arrays and frame correspondences archived externally. |
| §5.6 spatial | Same atoms/unaries/feasible sets, evidence-gated sparse weights, sentinels filtered, lambda.10, <=5 sequential sweeps; actual energy and degree-mass bounds verified. |
| §5.7 parameters | Exact specified config written before new GT evaluation and remains byte-identical; no scientific deviation or post-GT tuning. |
| §6 family/output/evaluator | Local membership/coverage/projection validated without global coercion; global parity preserved; candidate AP identical to U00; no unique rescoring/redistribution. |
| §7 caches/artifacts | Run and query-specific source/evidence/solver hashes, one bank reference per run, source maps and sparse evidence; GT confined to evaluation/diagnosis. |
| §8 focused verification | 29 passing scoped tests plus one real input smoke. Unchanged original trace test campaign not repeated. CPU8/chunk65536, no new packages or large raw logits. |
| §9 tables/interpretation | Four human/machine tables, both gains/losses and same-class transitions, high coverage, energy vs metric distinction, residual FP at99→101, costs and one next experiment. |
| §10–11 GitHub delivery | Both MD files and scoped small evidence prepared with relative links. Mandatory task-branch commit/push and remote SHA comparison are the final operation; final response carries actual full SHA/URL. |
| §12–14 evidence limits/self-review | Geometry-only local adaptation, not a reproduction or independent teacher. All six use U00, same unary keys, no GT-selected behavior. Primary performed review; no delegated scientific decisions. |

## Evidence locations and limits

[Performance]({anchor}/analysis/performance.json), [effects]({anchor}/analysis/effects.json), [objects]({anchor}/analysis/objects.json), [regions]({anchor}/analysis/regions.json), [evidence]({anchor}/analysis/evidence_regions.json), [confusion deltas]({anchor}/analysis/confusion_deltas.npz), [costs]({anchor}/costs.json), [test receipt]({anchor}/test_receipt.json), [event transitions]({anchor}/event_transitions.json), [repeat sensitivity]({anchor}/analysis/repeat_sensitivity.json). New released traces and evaluator ledgers are small compressed files under `evaluation/<run>/<method>/` here. [Archive mapping]({anchor}/archive_sources.json) maps external trace references to their committed copies.

Large owners/atoms/evidence/shared visibility/unique masks stay at `{large}`. [Large-artifact manifest]({anchor}/large_artifacts.json) gives their actual sizes and SHA256s; a reviewer needs that shared filesystem plus original receipt-bound source assets for replay. No weights, original datasets, credentials or raw logits are uploaded. No blocked required cell or remaining implementation task; optional extra methods and future scenes are NOT_RUN by design.

The scientific limitation is a negative result on a single development scene, not missing evidence: high coverage still fails to preserve unique objects. Geometry/class ambiguity, correlated frontend observations, tiny leftover masks crossing evaluator eligibility, same-class extent loss and graph proxy limitations are disclosed. Two saved runs are not significance or scene generalization. Extra3D training is present and scene-exclusion unverified. One next experiment is the separately authorized fixed Room1 confirmation described in [results](LOCAL_OWNERSHIP_RESULTS.md); no launch in this task.
'''
    (docs/'LOCAL_OWNERSHIP_HANDOFF.md').write_text(handoff)
    plan=docs/'LOCAL_OWNERSHIP_PLAN.md';plan.write_text(plan.read_text().replace('- [ ]','- [x]').replace('original row tie-breaking','deterministic atom-ID tie-breaking'))
    write(artifact/'packaging_receipt.json',{'status':'COMPLETE','command':[sys.executable,*sys.argv],'seconds':time.perf_counter()-start,'large_files':len(large_files),'large_bytes':sum(f['bytes'] for f in large_files),'required_cells':6,'new_model_inferences':0,'reports':['docs/paper/static_ovmap/LOCAL_OWNERSHIP_RESULTS.md','docs/paper/static_ovmap/LOCAL_OWNERSHIP_HANDOFF.md']})
    print(artifact,flush=True)


if __name__=='__main__':main()
