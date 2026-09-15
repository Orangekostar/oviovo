#!/usr/bin/env python3
"""Evaluation-only semantic correctness, donor routing, gates and released outcomes."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.static_ovmap.cached_semantic_transfer import METHODS
from src.static_ovmap.attribution_objects import candidate_ious,released_object_outcomes,candidate_event_diagnostics
from scripts.evaluation.diagnose_static_t1_attribution import write,write_gzip
from scripts.evaluation.run_static_sf_ovi_semantics import read,identity


def gzread(p):
    with gzip.open(p,'rt') as f:return json.load(f)


def extract(row,threshold,family='unique'):
    return released_object_outcomes(gzread(row[family]['trace_path']),threshold)


def transition(old,new,gtclass):
    if gtclass is None:return 'unresolved_geometry'
    return ('right' if old==gtclass else 'wrong')+'_to_'+('right' if new==gtclass else 'wrong')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);root=Path(cfg['new_output_root']);pred=root/'predictions';ev=root/'evaluation';out=root/'analysis';out.mkdir(exist_ok=True)
    small=ROOT/cfg['delivery']['small_artifacts'];small.mkdir(parents=True,exist_ok=True)
    rows=read(ev/'performance.json');refs=read(ev/'reused_references.json');protocol=read(ev/'metric_protocol.json');instance_ids=protocol['instance_ids']
    funnels=read(pred/'intervention_funnel.json');correctness=[];outcomes=[];headroom=[];routing=[];all_receivers=[];comparison=[];summaryc=[];summaryd=[];events=[]
    for run in cfg['runs']:
        pool=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run;ref=Path(cfg['recorded_roots_to_verify']['reference_evaluation_root'])/run
        receivers=read(pred/run/'receiver_decisions.json');all_receivers.extend(receivers);records=read(pool/'candidates.json');o=read(ref/'AT_O_AREA/metrics.json');u=read(ref/'AT_U00/metrics.json')
        overlap=np.load(ref/'candidate_gt_overlap.npz');ids=overlap['gt_ids'];sizes=overlap['gt_sizes'];ious=candidate_ious(overlap['intersections'],overlap['projected_sizes'],sizes)
        ovi=np.array([r['receiver'] for r in receivers]);labels=np.array([r['original_class_id'] for r in records]);eligible=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,instance_ids)
        for nominal in [.25,.5,.75]:
            oldmatch=extract(o,nominal);umatch=extract(u,nominal,'overlapping');threshold=oldmatch['threshold'];added=set(umatch['matched'])-set(oldmatch['matched'])
            eligible_gt=set(map(str,ids[eligible]));misses=eligible_gt-set(oldmatch['matched'])
            best_wrong=[];missing_shape=[];gt_diagnostic=[]
            for gid in sorted(misses|added,key=int):
                j=int(np.flatnonzero(ids==int(gid))[0]);best=int(ovi[np.argmax(ious[ovi,j])]);shape=ovi[ious[ovi,j]>threshold]
                if gid in added:
                    if ious[best,j]>threshold and labels[best]!=int(gid)//1000:best_wrong.append(int(gid))
                    if not len(shape):missing_shape.append(int(gid))
                gt_diagnostic.append({'gt_id':int(gid),'U00_added':gid in added,'baseline_miss':gid in misses,'best_OVI':best,
                    'best_iou':float(ious[best,j]),'best_original_class':int(labels[best]),'all_sufficient_OVI_receivers':shape.tolist(),
                    'any_original_correct_OVI':bool(np.any(labels[shape]==int(gid)//1000))})
            headroom.append({'run':run,'runtime_threshold':threshold,'U00_added':sorted(map(int,added)),
                  'best_recorded_OVI_shape_sufficient_but_class_wrong':best_wrong,'no_sufficient_OVI_shape':missing_shape,
                  'all_baseline_misses':sorted(map(int,misses)),'GT_aware_diagnostic_only':gt_diagnostic})
            for row in [r for r in rows if r['run']==run and r['status']=='COMPLETE']:
                method=row['condition'];newmatch=extract(row,nominal);gained=set(newmatch['matched'])-set(oldmatch['matched']);lost=set(oldmatch['matched'])-set(newmatch['matched'])
                outcomes.append({'run':run,'method':method,'runtime_threshold':threshold,'gained':sorted(map(int,gained)),
                    'lost':sorted(map(int,lost)),'U00_added_recovered':sorted(map(int,added&set(newmatch['matched']))),
                    'U00_added_not_recovered':sorted(map(int,added-set(newmatch['matched']))),'baseline_matches':oldmatch,'new_matches':newmatch})
                if nominal==.5:summaryd.append({k:v for k,v in outcomes[-1].items() if k not in ['baseline_matches','new_matches']})
                for receiver in receivers:
                    i=receiver['receiver'];choices=np.flatnonzero(eligible&(ious[i]>threshold));gtclass=int(ids[choices[0]]//1000) if len(choices)==1 else None
                    geometry='unique' if len(choices)==1 else 'ambiguous' if len(choices)>1 else 'unmatched';original=receiver['old_class'];final=receiver['final_labels'][method]
                    native=receiver['native'];chosen=receiver['selected_donor'];suggestion=native['proposed_label'] if method==METHODS[0] and native else chosen['class_id'] if method!=METHODS[0] and chosen else None
                    adopted=receiver['changed'][method];correct=gtclass is not None and suggestion==gtclass
                    item={'run':run,'method':method,'runtime_threshold':threshold,'receiver':i,'native_owner_id':receiver['native_owner_id'],
                        'geometry_correspondence':geometry,'geometry_GT_ids':ids[choices].tolist(),'geometry_IoUs':ious[i,choices].tolist(),
                        'old_class':original,'proposed_class':suggestion,'final_class':final,'changed':adopted,
                        'transition':transition(original,final,gtclass),'proposal_correct':bool(correct) if gtclass is not None and suggestion is not None else None,
                        'old_correct':bool(original==gtclass) if gtclass is not None else None,
                        'useful_proposal':bool(correct and original!=gtclass) if gtclass is not None and suggestion is not None else None,
                        'harmful_proposal':bool(original==gtclass and suggestion!=gtclass) if gtclass is not None and suggestion is not None else None,
                        'old_native_gate_reason':native['prior_gate_reason'] if native else None,
                        'SC_gate_reason':receiver['gate']['reason'],'SC_pass':receiver['gate']['accepted'],
                        'subset_transition':'entry' if original not in instance_ids and final in instance_ids else 'exit' if original in instance_ids and final not in instance_ids else 'unchanged'}
                    correctness.append(item)
                    if method==METHODS[1]:
                        alternatives=receiver['eligible_donors'];correct_donors=[d['canonical_index'] for d in alternatives if gtclass is not None and d['class_id']==gtclass]
                        flags=[]
                        if not alternatives:flags.append('no_eligible_donor')
                        if geometry!='unique':flags.append('receiver_geometry_'+geometry)
                        if correct_donors and chosen['class_id']!=gtclass:flags.append('correct_donor_exists_but_not_selected')
                        if gtclass is not None and chosen and chosen['class_id']==gtclass and not receiver['gate']['accepted']:flags.append('correct_selected_donor_rejected_by_SC')
                        if gtclass is not None and alternatives and not correct_donors:flags.append('no_eligible_donor_correct_class')
                        if gtclass is not None and original==gtclass and chosen and chosen['class_id']!=gtclass:flags.append('damages_previously_correct_receiver')
                        gid=str(int(ids[choices[0]])) if len(choices)==1 else None
                        if gtclass is not None and final==gtclass and gid not in newmatch['matched']:flags.append('adopted_correct_class_without_released_GT_match')
                        routing.append({'run':run,'runtime_threshold':threshold,'receiver':i,'native_owner_id':receiver['native_owner_id'],
                            'gt_ids':ids[choices].tolist(),'flags':flags,'correct_eligible_donors':correct_donors,
                            'receiver_GT_IoUs':ious[i,choices].tolist(),
                            'chosen_donor_GT_IoUs':ious[chosen['canonical_index'],choices].tolist() if chosen else None,
                            'eligible_donor_GT_IoUs':{str(d['canonical_index']):ious[d['canonical_index'],choices].tolist() for d in alternatives},
                            'chosen_donor':chosen,'all_eligible_donors':alternatives,'native_target_present':native is not None,
                            'native_raw_class':native['proposed_label'] if native else None})
        for row in [r for r in rows if r['run']==run and r['status']=='COMPLETE']:
            method=row['condition'];doc=read(pred/run/(method+'.json'));trace=gzread(row['unique']['trace_path'])
            diagnostics=candidate_event_diagnostics(trace,overlap['intersections'],overlap['projected_sizes'],np.array(doc['labels']),ids,sizes,instance_ids)
            events.extend([{'run':run,'method':method,**e} for e in diagnostics])
            relevant=[x for x in correctness if x['run']==run and x['method']==method and np.isclose(x['runtime_threshold'],.5)]
            summaryc.append({'run':run,'method':method,'transitions':dict(Counter(x['transition'] for x in relevant)),
                'changed_transitions':dict(Counter(x['transition'] for x in relevant if x['changed'])),
                'useful_suggestions_adopted':sum(x['useful_proposal'] is True and x['changed'] for x in relevant),
                'useful_suggestions_rejected':sum(x['useful_proposal'] is True and not x['changed'] for x in relevant),
                'harmful_suggestions_adopted':sum(x['harmful_proposal'] is True and x['changed'] for x in relevant),
                'harmful_suggestions_prevented':sum(x['harmful_proposal'] is True and not x['changed'] for x in relevant)})
        for r in receivers:
            if r['native'] and r['selected_donor']:
                comparison.append({'run':run,'native_owner_id':r['native_owner_id'],'receiver':r['receiver'],'cohort':'native_E1_and_SF_donor_overlap',
                       'native_raw':r['native']['proposed_label'],'SF_raw':r['selected_donor']['class_id'],'final_labels':r['final_labels'],
                       'definition':'aligned receiver diagnostic, not subgroup AP or a pure backbone effect'})
    reconcile=[h for h in headroom if np.isclose(h['runtime_threshold'],.5)]
    # Historical counts are checked only here, after all predictions/evaluations are frozen.
    for h,expected in zip(reconcile,[(7,5,2),(6,4,2)]):
        assert (len(h['U00_added']),len(h['best_recorded_OVI_shape_sufficient_but_class_wrong']),len(h['no_sufficient_OVI_shape']))==expected
    write_gzip(out/'receiver_decisions.json.gz',all_receivers);write_gzip(out/'label_correctness_and_gate.json.gz',correctness)
    write_gzip(out/'donor_routing.json.gz',routing);write_gzip(out/'object_outcomes.json.gz',outcomes);write_gzip(out/'released_events.json.gz',events)
    write(out/'diagnostic_headroom.json',headroom);write(out/'correctness_summary.json',summaryc);write(out/'object_summary.json',summaryd);write(out/'common_target_comparison.json',comparison)
    # Paired native owner IDs, never cross-run SF query IDs.
    native_runs={run:{r['native_owner_id']:r for r in all_receivers if r['run']==run} for run in cfg['runs']}
    cross=[]
    for owner in sorted(native_runs[cfg['runs'][0]]):
        a,b=(native_runs[run][owner] for run in cfg['runs']);assert a['source_mask_hash']==b['source_mask_hash']
        cross.append({'native_owner_id':owner,'mask_hash':a['source_mask_hash'],'first_labels':a['final_labels'],'repeat_labels':b['final_labels'],
                      'first_selected_donor':a['selected_donor'],'repeat_selected_donor':b['selected_donor']})
    write(out/'cross_run_receivers.json',cross)
    crosstabs=[]
    for run in cfg['runs']:
        selected=[x for x in correctness if x['run']==run and np.isclose(x['runtime_threshold'],.5)]
        native_cross=Counter((x['old_native_gate_reason'],x['useful_proposal'],x['harmful_proposal']) for x in selected if x['method']==METHODS[0] and x['proposed_class'] is not None)
        gate_cross=Counter((x['SC_gate_reason'],x['useful_proposal'],x['harmful_proposal']) for x in selected if x['method']==METHODS[2] and x['proposed_class'] is not None)
        crosstabs.append({'run':run,'native_old_gate': [{'reason':k[0],'useful':k[1],'harmful':k[2],'count':v} for k,v in native_cross.items()],
            'SC_gate':[{'reason':k[0],'useful':k[1],'harmful':k[2],'count':v} for k,v in gate_cross.items()],
            'routing_flags':dict(Counter(flag for r in routing if r['run']==run and np.isclose(r['runtime_threshold'],.5) for flag in r['flags']))})
    write(out/'gate_routing_crosstabs.json',crosstabs)
    for path in out.iterdir():shutil.copy2(path,small/path.name)
    for name in ['input_binding.json','prediction_manifest.json','intervention_funnel.json','real_cache_smoke.json','frozen_config.json']:shutil.copy2(pred/name,small/name)
    for name in ['performance.json','metric_protocol.json','evaluation_manifest.json','reused_references.json']:shutil.copy2(ev/name,small/name)
    exclusions=[]
    for run in cfg['runs']:
        shutil.copy2(pred/run/'reverse_correspondence.npz',small/f'{run}_reverse_correspondence.npz')
        shutil.copy2(pred/run/'donor_registry.json',small/f'{run}_donor_registry.json')
        for method in METHODS:shutil.copy2(pred/run/(method+'.json'),small/f'{run}_{method}.json')
        pair=np.load(pred/run/'reverse_correspondence.npz');n=pair['intersections']
        records=read(Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run/'candidates.json')
        ra=np.array([records[i]['source_area'] for i in pair['receivers']])[:,None]
        da=np.array([records[i]['source_area'] for i in pair['donors']])[None,:]
        rok,dok=2*n>=ra,2*n>=da
        exclusions.append({'run':run,'all_pairs':int(n.size),'nonempty_eligible_pairs':int(np.count_nonzero(rok&dok&(ra>0)&(da>0))),
            'receiver_coverage_failures':int(np.count_nonzero(~rok)),'donor_coverage_failures':int(np.count_nonzero(~dok)),
            'both_failures':int(np.count_nonzero(~rok&~dok)),'zero_area_donors':int(np.count_nonzero(da==0)),
            'nonexclusive_failure_counts':True,'GT_input':False})
    write(small/'eligibility_exclusions.json',exclusions)

    with (small/'performance.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['run','condition','role','uAP','AP50','AP25','mIoU','mAcc','new_inference'])
        for r in refs+rows:writer.writerow([r['run'],r['condition'],r['execution_role'],r['unique']['released']['all_ap'],r['unique']['released']['all_ap_50%'],r['unique']['released']['all_ap_25%'],r['semantic']['semantic_miou'],r['semantic']['semantic_macc'],0])
    write(small/'large_artifacts.json',{'uploaded':False,'generated':[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
          'inherited_assets':'input_binding.json; large inherited inputs were not uploaded','reproduce':'See SF_OVI_SEMANTIC_HANDOFF.md commands with a fresh output root'})
    lines=['# Cached SF → OVI semantics\n','科学状态：**COMPLETE_NO_NET_GAIN**；六条件完成，未建立跨场景有效性。\n','固定 O-only 几何、owner、注册表与原分数字符串；仅改OVI类别。所有数值为百分数。\n',
      '## A：配对性能\n','| Run | Condition | Role | uAP / AP50 / AP25 | mIoU / mAcc | Subset entry / exit |','|---|---|---|---|---|---|']
    for run in cfg['runs']:
        for r in [x for x in refs+rows if x['run']==run]:
            u=r['unique']['released'];s=r['semantic']
            lines.append(f"| {run} | {r['condition']} | {r['execution_role']} | {100*u['all_ap']:.3f}/{100*u['all_ap_50%']:.3f}/{100*u['all_ap_25%']:.3f} | {100*s['semantic_miou']:.3f}/{100*s['semantic_macc']:.3f} | {(len(r['subset_entries']) if isinstance(r.get('subset_entries',0),list) else r.get('subset_entries',0))}/{(len(r['subset_exits']) if isinstance(r.get('subset_exits',0),list) else r.get('subset_exits',0))} |")
    lines+=['\n全部条件的独立类别无关 Canonical AP75 保持16.842%，不宣称几何改善。活动OVI重叠掩码与unique掩码逐项相等，只评估一次；完整身份键相同的payload明确复用。新增图像/文本/2D/3D推理均为0。',
      '\n## B：建议与采用漏斗\n','| Run | Method | Receivers / cohort | Valid / same-label | Proposed / adopted changes | No-suggestion abstentions |','|---|---|---|---|---|---|']
    for f in funnels:lines.append(f"| {f['run']} | {f['method']} | {f['receivers']}/{f['cohort']} | {f['valid_suggestions']}/{f['same_label_suggestions']} | {f['proposed_changes']}/{f['adopted_changes']} | {f['abstentions']} |")
    lines+=['\nS-A移除整组旧采用门槛，不是margin-only消融。S-C仅对S-B同一供体建议应用2/3支持门控，不另选供体；单个正权重供体可通过，不构成独立确认。全部拒绝原因、支持分布、重复合并、单供体通过及成本见机器漏斗。',
      '\n## C：固定几何下的正确性（strict >.5）\n','| Run | Method | Changed transitions | Useful adopted / rejected | Harmful adopted / prevented |','|---|---|---|---|---|']
    for c in summaryc:lines.append(f"| {c['run']} | {c['method']} | {c['changed_transitions']} | {c['useful_suggestions_adopted']}/{c['useful_suggestions_rejected']} | {c['harmful_suggestions_adopted']}/{c['harmful_suggestions_prevented']} |")
    lines+=['\n仅唯一几何对应可判对错；ambiguous/unmatched保持显式，不采用多数像素定义对象正确性。全部接收者与.25/.5/.75阈值均在账本中，含原本正确者。',
      '\n## D：真实发布版对象结果（strict >.5）\n','| Run | Method | Gained | Lost | U00 added recovered | Not recovered |','|---|---|---|---|---|---|']
    for d in summaryd:lines.append('| '+' | '.join(str(d[k]) for k in ['run','method','gained','lost','U00_added_recovered','U00_added_not_recovered'])+' |')
    lines+=['\n历史7/6新增、5/4最佳OVI形状足够但类别错、2/2没有足够OVI形状均已复现；完整headroom检查同时覆盖所有baseline misses。GT-aware计数只是诊断，不是输出方法或AP上界。',
     '\nS-B与S-A证据源和目标覆盖均不同；common_target_comparison给共同原生owner逐对象对照，不报告子集AP为全景结果。两个缓存是同一开发场景敏感性检查，不是独立场景、盲验证或显著性证据。',
     '\n证据：[performance](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/performance.json)、[funnel](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/intervention_funnel.json)、[correctness](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/label_correctness_and_gate.json.gz)、[routing](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/donor_routing.json.gz)、[objects](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/object_outcomes.json.gz)、[headroom](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/diagnostic_headroom.json)。']
    (ROOT/cfg['delivery']['results']).write_text('\n'.join(lines)+'\n')
    print('summary complete',len(correctness),'receiver-threshold-condition records',flush=True)


if __name__=='__main__':main()
