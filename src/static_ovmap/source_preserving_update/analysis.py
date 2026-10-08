"""Post-lock geometry, released traces and source-injection diagnostics."""

from collections import Counter
from pathlib import Path

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.evidence_exploration.analysis import trace_entries,zipped
from static_ovmap.evidence_exploration.diagnostic_details import compare_entries
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,read

from .binding import seal
from .decisions import MATCHED_METHODS,NEW_METHODS


def semantic_outcomes(a,b,opportunities,threshold):
    counts=Counter();ledger=[]
    for item in opportunities:
        owner=int(item['owner']);best=item['best'];old,new=b[owner],a[owner]
        matchable=bool(best and best['iou']>threshold and item['instance_size_eligible'])
        if best is None:category='NO_ELIGIBLE_GT'
        elif not matchable:category='GEOMETRY_INSUFFICIENT'
        elif new==old:category='UNCHANGED_RIGHT' if new==best['class'] else 'UNCHANGED_WRONG'
        elif old!=best['class'] and new==best['class']:category='WRONG_TO_RIGHT'
        elif old==best['class'] and new!=best['class']:category='RIGHT_TO_WRONG'
        else:category='WRONG_TO_WRONG'
        counts[category]+=1
        if new!=old:
            ledger.append({'owner':owner,'old_class':old,'new_class':new,'threshold':threshold,
                'geometrically_matchable':matchable,'category':category,'closest_eligible_GT':best})
    return {'counts':dict(counts),'changed_ledger':ledger,'original_incumbent_denominator':len(opportunities),
            'rule':'strict IoU > threshold; released eligible GT; fixed original P support'}


def diagnose_scene(binding,scene,rows):
    root=Path(binding['output_root']);lock=read(root/'predictions'/scene/'receipt.json')
    if lock['status']!='PREDICTIONS_LOCKED' or len(rows)!=9:raise ValueError('complete predictions/evaluation required before diagnosis')
    index=ConsumptionIndex(root/'diagnostics'/scene/'verifications.json')
    key=canonical_digest({'lock':lock['identity'],'evaluations':{r['method']:r['identity'] for r in rows},
        'producer':index.identity(__file__)})
    dest=root/'diagnostics'/scene/'receipt.json'
    if dest.exists():
        previous=read(dest);_verified_identity(previous)
        if previous['input_identity']==key:
            for item in previous['dependencies']:index.identity(item['path'],item)
            return previous
        from .resume import invalidate_descendants
        invalidate_descendants(binding,scene,'diagnose','changed locked inputs or diagnostic producer')
    opportunity_path=Path(binding['parent_root'])/'diagnostics'/scene/'opportunity.json'
    opportunity=read(opportunity_path);_verified_identity(opportunity);dependencies=[index.identity(opportunity_path)]
    manifest=read(root/'evidence'/scene/'manifest.json');domain=read(root/'eligibility'/(scene+'.json'))
    pmap={int(r['owner']):r for r in opportunity['incumbents']}
    for owner,item in manifest['incumbents'].items():
        if pmap[int(owner)]['support_hash']!=item['selection']['support_hash']:raise ValueError('parent geometric support lineage changed')
    labels={};ranks={};trace={};traces={};partitions={};decisions={}
    for row in rows:
        method=row['method'];payload=load_prediction(lock['methods'][method]['manifest'])
        labels[method]=owner_labels(payload);ranks[method]=dict(payload.instance_ranks)
        partitions[method]=row['owner_partition_digest'];score=read(row['evaluation_receipt'])
        path=Path(score['manifest']).with_name('trace.json.gz')
        if str(path) not in traces:
            dependencies.append(index.identity(path));raw=zipped(path)
            if not all(raw['parity'][k] for k in ('ap_exact','pr_and_fn_exact')):raise ValueError('released diagnostic trace parity failed')
            traces[str(path)]=trace_entries(raw)
        trace[method]=traces[str(path)]
        decisions[method]=read(root/'decisions'/scene/(method+'.json'))['incumbents']
    if any(partitions[m]!=partitions['SU01_G1'] for m in (*MATCHED_METHODS,*NEW_METHODS)):
        raise ValueError('fixed G1 owner partition differs in post-lock diagnosis')
    pairs=[tuple(r) for r in binding['specification']['comparisons']]+[(m,'SU01_G1') for m in (*MATCHED_METHODS,*NEW_METHODS)]
    comparisons=[]
    for a,b in pairs:
        outcomes={str(t):semantic_outcomes(labels[a],labels[b],opportunity['incumbents'],t) for t in (.5,.75)}
        matches={}
        for t in (.5,.75):
            entries=[[r for r in trace[m] if abs(r['overlap_threshold']-t)<1e-12] for m in (a,b)]
            # Preserve all duplicate score entries; legacy helper suffix is independent of threshold.
            matches[str(t)]={k.removesuffix('50'):v for k,v in compare_entries(*entries).items()}
        changed=[]
        for owner,item in manifest['incumbents'].items():
            oid=int(owner)
            if labels[a][oid]==labels[b][oid]:continue
            decision=decisions[a][owner];coarse=[]
            for v in item['full_views']:
                path=root/'coarse'/scene/str(v['frame_id'])/'receipt.json'
                if path.exists():
                    rec=read(path)['records'].get(v['region_id'])
                    if rec and rec['available']:coarse.append(rec['scores'])
            mean_a=np.mean([v['scores'] for v in item['full_views']],axis=0) if item['full_views'] else None
            mean_c=np.mean(coarse,axis=0) if len(coarse)==len(item['full_views']) and coarse else None
            old_ix=manifest['valid_ids'].index(labels[b][oid]);new_ix=manifest['valid_ids'].index(labels[a][oid])
            source={}
            for name,scores in {**item['scores'],'A':mean_a,'C':mean_c}.items():
                source[name]=None if scores is None else {'old_class_cosine':float(scores[old_ix]),
                    'new_class_cosine':float(scores[new_ix]),'new_minus_old':float(scores[new_ix]-scores[old_ix])}
            changed.append({'owner':oid,'reference_class':labels[b][oid],'candidate_class':labels[a][oid],
                'source_availability':{n:s is not None for n,s in item['scores'].items()},'source_margins':source,
                'same_view_A_minus_C':None if mean_a is None or mean_c is None else {
                    'old_class':float(mean_a[old_ix]-mean_c[old_ix]),'new_class':float(mean_a[new_ix]-mean_c[new_ix])},
                'decision_reference':{'method':a,'owner':owner},'third_class_winner':decision.get('third_class_winner',False),
                'same_encoder_NQ_are_distinct_streams_not_independent_evidence':True})
        rank_changes=[{'owner':o,'old_rank':ranks[b][o],'new_rank':ranks[a][o],
                       'recovered_owner':o not in pmap} for o in sorted(ranks[a]) if ranks[a][o]!=ranks[b][o]]
        comparisons.append({'scene':scene,'cohort':binding['scenes'][scene]['cohort'],'candidate':a,'reference':b,
            'output_identity_tie':lock['methods'][a]['prediction_key']==lock['methods'][b]['prediction_key'],
            'semantic_outcomes':outcomes,'released_matches':matches,'changed_source_ledger':changed,'rank_changes':rank_changes})
    coverage={}
    hard_effects={}
    for m in (*MATCHED_METHODS,*NEW_METHODS):
        changed=sum(r['changed'] for r in decisions[m].values())
        coverage[m]={'selected':len(manifest['incumbents']),'common':domain['eligible_count'],'applied_changes':changed,
            'proposed_changes':sum(r['proposed_class']!=r['old_class'] for r in decisions[m].values()),
            'posterior_changes':sum(r.get('posterior_changed',False) for r in decisions[m].values()),
            'protected':sum(bool(r['protected_raw_zero_count']) for r in manifest['incumbents'].values()),
            'new_proposals_cancelled_by_raw_zero':sum(r['proposed_class']!=r['applied_class'] for r in decisions[m].values()),
            'unavailable':sum(any('UNAVAILABLE' in s or 'NO_PARENT' in s for s in d['exclusions']) for d in domain['incumbents'].values()),
            'successful_FULL_A_views':sum(len(r['full_views']) for r in manifest['incumbents'].values()),
            'common_FULL_pairs':sum(len(manifest['incumbents'][o]['full_views']) for o,d in domain['incumbents'].items() if d['eligible']),
            'source_available_counts':{n:sum(r['scores'].get(n) is not None for r in manifest['incumbents'].values()) for n in ('N','Q','F')}}
        hard_effects[m]={}
        for t in (.5,.75):
            effect=Counter()
            for o,item in manifest['incumbents'].items():
                owner=int(o);p=pmap[owner];best=p['best']
                if not best or not p['instance_size_eligible'] or best['iou']<=t:continue
                old=labels['SU01_G1'][owner];hard=labels['SU02_HARD_MATCHED'][owner];new=labels[m][owner];gt=best['class']
                if old==gt and hard!=gt:
                    effect['hard_damage']+=1;effect['damage_prevented' if new==gt else 'damage_remains']+=1
                if old!=gt and hard==gt:
                    effect['hard_useful_correction']+=1;effect['correction_retained' if new==gt else 'correction_lost']+=1
            hard_effects[m][str(t)]=dict(effect)
    history=[]
    parent_store=read(binding['parent_result_store']['path'])
    for m,old,field in [('SU02_HARD_MATCHED','IR06_ANYUP_REREAD','IR06_class'),('SU03_STABLE_MATCHED','IR07_BOUNDARY_STABLE','IR07_class')]:
        parent_payload=load_prediction(binding['scenes'][scene]['parent_predictions'][old]['manifest'])
        original_labels=owner_labels(parent_payload)
        old_row=next(r for r in parent_store['scene_metrics'] if r['scene']==scene and r['method']==old)
        new_row=next(r for r in rows if r['method']==m)
        history.append({'method':m,'historical_method':old,'historical_prediction_key':parent_payload.prediction_key,
            'matched_prediction_key':lock['methods'][m]['prediction_key'],
            'historical_actual_changed':sum(original_labels[int(o)]!=r['old_class'] for o,r in manifest['incumbents'].items()),
            'historical_proposed_changed':sum(r['historical'][field]!=r['old_class'] for r in manifest['incumbents'].values()),
            'matched_changed':coverage[m]['applied_changes'],
            'coverage_removed_original_edits':sum(original_labels[int(o)]!=labels[m][int(o)] for o in manifest['incumbents']),
            'output_identical':parent_payload.prediction_key==lock['methods'][m]['prediction_key'],
            'metric_delta_from_historical':{k:new_row['metrics'][k]-old_row['metrics'][k] for k in ('apall','ap50','ap25','miou','macc')},
            'historical_evaluation_identity':old_row['evaluation_identity']})
    result=seal({'status':'POSTLOCK_DIAGNOSTICS_COMPLETE','scene':scene,'input_identity':key,
        'comparisons':comparisons,'coverage':coverage,'hard_control_effects':hard_effects,
        'historical_to_matched':history,'dependencies':dependencies,
        'raw_zero_effect':'protected owners excluded before C acquisition; no unmeasured new-update counterfactual claimed',
        'prediction_lock':lock['identity'],'recovered_mask_and_class_parity':'EXACT',
        'parent_geometric_opportunity_identity':opportunity['identity'],'GT_used_only_after_prediction_lock':True})
    atomic_write_json(dest,result);index.write_memo(root/'diagnostics'/scene/'verifications.json');return result


def analyze(binding,store):
    from .orchestration import require_freeze
    require_freeze(binding);root=Path(binding['output_root']);results=[];blocked=[]
    for cohort,names in binding['cohorts'].items():
        for scene in names:
            rows=[r for r in store['scene_metrics'] if r['scene']==scene]
            try:results.append(diagnose_scene(binding,scene,rows))
            except (OSError,ValueError,KeyError) as exc:blocked.append({'scene':scene,'reason':type(exc).__name__+': '+str(exc)})
    aggregate=[];coverage={};history=[];class_deltas=[];hard_effects={}
    pairs=[tuple(r) for r in binding['specification']['comparisons']]+[(m,'SU01_G1') for m in (*MATCHED_METHODS,*NEW_METHODS)]
    for cohort,names in binding['cohorts'].items():
        available=[r for r in results if r['scene'] in names];coverage[cohort]={}
        hard_effects[cohort]={}
        for m in (*MATCHED_METHODS,*NEW_METHODS):
            total=Counter();sources=Counter()
            for r in available:
                c=r['coverage'][m];total.update({k:v for k,v in c.items() if isinstance(v,int)});sources.update(c['source_available_counts'])
            coverage[cohort][m]={**dict(total),'source_available_counts':dict(sources),'scenes':len(available),'full_cohort':len(available)==len(names)}
            hard_effects[cohort][m]={}
            for t in ('0.5','0.75'):
                effect=Counter()
                for r in available:effect.update(r['hard_control_effects'][m][t])
                hard_effects[cohort][m][t]=dict(effect)
        for a,b in pairs:
            details=[d for r in available for d in r['comparisons'] if (d['candidate'],d['reference'])==(a,b)]
            if not details:continue
            counts={};matches={}
            for t in ('0.5','0.75'):
                c=Counter();mc=Counter();gains=[];losses=[]
                for d in details:
                    c.update(d['semantic_outcomes'][t]['counts'])
                    match=d['released_matches'][t];mc.update({k:v for k,v in match.items() if isinstance(v,int)})
                    gains.extend({'scene':d['scene'],'class_and_GT_id':g} for g in match['new_unique_GT_ids'])
                    losses.extend({'scene':d['scene'],'class_and_GT_id':g} for g in match['lost_unique_GT_ids'])
                counts[t]=dict(c);matches[t]={'counts':dict(mc),'new_unique_GT':gains,'lost_unique_GT':losses}
            metrics=None
            if a in store['pooled_metrics'].get(cohort,{}) and b in store['pooled_metrics'][cohort]:
                metrics={k:store['pooled_metrics'][cohort][a]['metrics'][k]-store['pooled_metrics'][cohort][b]['metrics'][k]
                         for k in ('apall','ap50','ap25','miou','macc')}
                ap=read(store['pooled_metrics'][cohort][a]['per_class_receipt'])['classes']
                bp=read(store['pooled_metrics'][cohort][b]['per_class_receipt'])['classes'];bmap={r['class_id']:r for r in bp}
                for row in ap:
                    old=bmap[row['class_id']];diff={k:None if row[k] is None or old[k] is None else row[k]-old[k]
                        for k in ('apall','ap50','ap25','iou','accuracy')}
                    class_deltas.append({'cohort':cohort,'candidate':a,'reference':b,'class_id':row['class_id'],
                        'class_name':row['class_name'],'deltas':diff})
            aggregate.append({'cohort':cohort,'candidate':a,'reference':b,'full_cohort':len(details)==len(names),
                'metric_deltas_fraction':metrics,'semantic_outcomes':counts,'released_matches':matches,
                'output_identity_tie_all_scenes':all(d['output_identity_tie'] for d in details),
                'rank_changes':sum(len(d['rank_changes']) for d in details),
                'recovered_rank_changes':sum(x['recovered_owner'] for d in details for x in d['rank_changes'])})
        for r in available:history.extend({'scene':r['scene'],'cohort':cohort,**x} for x in r['historical_to_matched'])
    result=seal({'status':'DIAGNOSTICS_COMPLETE' if len(results)==26 else 'PARTIAL_DEPENDENCY_BLOCK',
        'scene_receipts':{r['scene']:r['identity'] for r in results},'blocked':blocked,'comparisons':aggregate,
        'coverage':coverage,'hard_control_effects':hard_effects,'historical_to_matched':history,'per_class_pooled_deltas':class_deltas,
        'source_agreement_independence_claim':False})
    atomic_write_json(root/'diagnostics/summary.json',result)
    contrast=lambda a,b:[r for r in aggregate if (r['candidate'],r['reference'])==(a,b)]
    from .selection import select
    selection=select(store['pooled_metrics'],binding['specification']['selection'])
    interpretations=[]
    for a,b in binding['specification']['comparisons']:
        observed=contrast(a,b)
        complete=len(observed)==2 and all(x['full_cohort'] and x['metric_deltas_fraction'] is not None for x in observed)
        nondecrease=complete and all(v>=-1e-10 for x in observed for v in x['metric_deltas_fraction'].values())
        nonincrease=complete and all(v<=1e-10 for x in observed for v in x['metric_deltas_fraction'].values())
        interpretations.append({'candidate':a,'reference':b,'complete':complete,
            'all_five_metrics_both_cohorts_nondecrease':nondecrease,
            'metric_relation':('OUTPUT_IDENTITY_TIE' if complete and all(x['output_identity_tie_all_scenes'] for x in observed)
                else 'ALL5_NONDECREASE' if nondecrease else 'ALL5_NONINCREASE' if nonincrease else 'TRADEOFF' if complete else 'INCOMPLETE'),
            'measured_contrasts':observed})
    assessment=seal({'status':result['status'],'source_preservation_vs_hard':contrast('SU04_F_REPLACE','SU02_HARD_MATCHED'),
        'retain_old_F_vs_replace':contrast('SU05_F_BLEND','SU04_F_REPLACE'),
        'same_view_AnyUp_vs_coarse':contrast('SU05_F_BLEND','SU06_F_COARSE'),
        'F_group_vs_equal_A_mass_global':contrast('SU05_F_BLEND','SU07_GLOBAL_BLEND'),
        'paired_change_vs_absolute_A':contrast('SU08_PAIRED_DELTA','SU05_F_BLEND'),
        'stability_control':contrast('SU03_STABLE_MATCHED','SU02_HARD_MATCHED'),
        'remaining_changes_vs_G1':[r for r in aggregate if r['reference']=='SU01_G1'],
        'exclusion_coverage':coverage,'hard_control_effects':hard_effects,'interpretations':interpretations,
        'fixed_gate_selection':selection,
        'recommendation':('retain '+selection['selected']+' as an exposed-cohort candidate; independently specify confirmation and standalone costs'
            if selection['target_met'] else 'stop these five fixed transformations under this target; no weight sweep is authorized'),
        'future_update_risk_study':'only separately specified with training/physical-family separation; not executed',
        'availability_limitation':{c:{'selected':next(iter(ms.values())).get('selected') if ms else None,
            'common':next(iter(ms.values())).get('common') if ms else None,
            'protected':next(iter(ms.values())).get('protected') if ms else None} for c,ms in coverage.items()},
        'next_stage_executed':False,'independent_confirmation':False})
    atomic_write_json(root/'next_stage_assessment.json',assessment);return result
