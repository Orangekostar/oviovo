#!/usr/bin/env python3
"""Evaluation-only complete object, region, ranking and cause ledgers."""
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
from plyfile import PlyData
from src.static_ovmap.attribution_objects import candidate_ious,released_object_outcomes,candidate_event_diagnostics
from src.static_ovmap.attribution_regions import region_transitions
from scripts.evaluation.run_static_object_decoupling import read,identity
from scripts.evaluation.diagnose_static_t1_attribution import write,write_gzip


def gzread(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def outcomes(row,family,threshold):
    return released_object_outcomes(gzread(row[family]['trace_path']),threshold)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);old=read(ROOT/cfg['existing_asset_configs'][0])
    root=Path(cfg['runtime_assets_to_resolve']['new_output_root']);pred=root/'predictions';evaluation=root/'evaluation'
    out=root/'analysis';out.mkdir(exist_ok=True);small=ROOT/cfg['delivery']['small_artifacts'];small.mkdir(parents=True,exist_ok=True)
    rows=read(evaluation/'performance.json');protocol=read(evaluation/'metric_protocol.json');valid=protocol['semantic_ids'];instance_ids=protocol['instance_ids']
    semgt=PlyData.read(old['gt_semantic_map'])['vertex']['label'];gt=np.load(old['reference_gt_ids'])
    allobjects=[];causes=[];effects=[];regions_out=[];support=[];diagnostics=[];refs=[];table_c=[];table_d=[]
    for run in cfg['runs']:
        source=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run
        ref=Path(cfg['recorded_roots_to_verify']['reference_evaluation_root'])/run
        records=read(source/'candidates.json');coords=np.load(source/'coord.npy',mmap_mode='r');masks=np.load(source/'masks.npy',mmap_mode='r')
        overlap=np.load(ref/'candidate_gt_overlap.npz');gtids=overlap['gt_ids'];gtsizes=overlap['gt_sizes'];ious=candidate_ious(overlap['intersections'],overlap['projected_sizes'],gtsizes)
        ovi=np.array([i for i,r in enumerate(records) if r['source']=='OVI']);sf=np.array([i for i,r in enumerate(records) if r['source']=='SpaCeFormer'])
        original_labels=np.array([r['original_class_id'] for r in records]);projection=np.load(ref/'projection.npz');near,matched=projection['nearest'],projection['matched']
        region=np.where(matched,np.load(source/'regions.npz')['region'][near],-1)
        baseline={m:read(ref/m/'metrics.json') for m in ['AT_O_AREA','AT_U00','AT_U11']}
        for m in ['LO_U00_OVI_FILL','LO_U00_LOCAL','LO_U00_SPATIAL']:
            baseline[m]=read(Path(cfg['recorded_roots_to_verify']['local_ownership_root'])/'evaluation'/run/m/'metrics.json')
        refs.extend([{**r,'execution_role':'REUSED_REFERENCE'} for r in baseline.values()])
        current={r['condition']:r for r in rows if r['run']==run}
        semcounts=read(pred/run/'semantics.json');actions=read(pred/run/'actions.json')
        for method,row in current.items():
            for threshold in [.25,.5,.75]:
                base=outcomes(baseline['AT_O_AREA'],'unique',threshold);u00=outcomes(baseline['AT_U00'],'overlapping',threshold);new=outcomes(row,'unique',threshold)
                added=set(u00['matched'])-set(base['matched']);gained=set(new['matched'])-set(base['matched']);lost=set(base['matched'])-set(new['matched'])
                retained=added&set(new['matched'])
                if threshold==.5:table_c.append({'run':run,'method':method,'U00_added':sorted(map(int,added)),
                         'retained':sorted(map(int,retained)),'gained':sorted(map(int,gained)),'lost':sorted(map(int,lost))})
                for gid in sorted(set(map(str,gtids[gtids>=1000]))|set(base['matched'])|set(new['matched']),key=int):
                    allobjects.append({'run':run,'method':method,'runtime_threshold':new['threshold'],'gt_id':int(gid),
                        'U00_added':gid in added,'retained_U00_added':gid in retained,'newly_gained':gid in gained,'newly_lost':gid in lost,
                        'O_only':base['matched'].get(gid),'new':new['matched'].get(gid),'U00':u00['matched'].get(gid)})
            ledger=gzread(evaluation/run/method/'evaluation_ledger.json.gz');doc=read(pred/run/(method+'.json'))
            source_owner=np.load(doc['owner_path']);owner=np.load(evaluation/run/method/'owners.npy');semantic=np.load(evaluation/run/method/'semantic.npy')
            sizes=np.bincount(owner,minlength=len(records)+1)[1:];inter=np.zeros((len(records),len(gtids)),np.int64)
            for j,gid in enumerate(gtids):inter[:,j]=np.bincount(owner[gt==gid],minlength=len(records)+1)[1:]
            unique_iou=candidate_ious(inter,sizes,gtsizes)
            events=candidate_event_diagnostics(gzread(row['unique']['trace_path']),inter,sizes,np.array(doc['labels']),gtids,gtsizes,instance_ids)
            write_gzip(out/f'{run}_{method}_events.json.gz',events)
            ranked=sorted(doc['kept'],key=lambda i:(-float(doc['rank_scores_serialized'][i]),i));rank={i:n+1 for n,i in enumerate(ranked)}
            for item in ledger:
                i=item['canonical_index'];source_points=coords[source_owner==i+1];eligible=(gtids>=1000)&(gtsizes>=100)&np.isin(gtids//1000,instance_ids)
                choices=np.flatnonzero(eligible&(unique_iou[i]>.5));correspondence=int(gtids[choices[0]]) if len(choices)==1 else None
                old_correct=original_labels[i]==correspondence//1000 if correspondence else None
                new_correct=doc['labels'][i]==correspondence//1000 if correspondence else None
                support.append({**item,'run':run,'method':method,'final_support_fraction':item['source_final_area']/max(1,records[i]['source_area']),
                    'source_extent_min':source_points.min(axis=0).tolist() if len(source_points) else None,
                    'source_extent_max':source_points.max(axis=0).tolist() if len(source_points) else None,
                    'source_original_extent_min':coords[masks[i]].min(axis=0).tolist() if np.any(masks[i]) else None,
                    'source_original_extent_max':coords[masks[i]].max(axis=0).tolist() if np.any(masks[i]) else None,
                    'rank_position_diagnostic_canonical_ties':rank[i],
                    'geometry_correspondence_gt':correspondence,'old_class_correct':bool(old_correct) if old_correct is not None else None,
                    'new_class_correct':bool(new_correct) if new_correct is not None else None,
                    'before_source_IoUs':ious[i].tolist(),'final_unique_IoUs':unique_iou[i].tolist(),'gt_ids':gtids.tolist()})
            table_d.append({'run':run,'method':method,'source_unknown_fraction':row['source_unknown_fraction'],
                'projected_unknown_fraction':row['projected_unknown_fraction'],'empty':row['unique_empty_count'],'small':row['unique_small_nonempty_count'],
                'invalid_class':row['invalid_instance_class_count'],'ignored_events':sum('evaluator_ignored' in e['flags'] for e in events),
                'accepted_actions':sum(a['accepted'] for a in actions['actions']) if method!='OD_E1_SEMANTIC' else 0,
                'action_reasons':dict(Counter(a['reason'] for a in actions['actions'])),
                'run_semantic_requests':semcounts['attempted_view_requests'],'run_encoder_batches':semcounts['encoder_batches'],
                'run_crop_inputs':semcounts['crop_inputs'],'budget_excluded_targets':sum(not t['budget_included'] for t in semcounts['targets'])})
        pairs=[('OD_E1_SEMANTIC','AT_O_AREA'),('OD_E2_OBJECT','AT_O_AREA'),('OD_E3_OBJECT_SEMANTIC','OD_E2_OBJECT'),('OD_E4_FINAL_RANK','OD_E3_OBJECT_SEMANTIC')]
        pairs += [(m,b) for m in current for b in ['LO_U00_OVI_FILL','LO_U00_LOCAL']]
        for after,before in pairs:
            a=current[after];b=(current|baseline)[before]
            effects.append({'run':run,'after':after,'before':before,'AP_delta':a['unique']['released']['all_ap']-b['unique']['released']['all_ap'],
               'AP50_delta':a['unique']['released']['all_ap_50%']-b['unique']['released']['all_ap_50%'],
               'mIoU_delta':a['semantic']['semantic_miou']-b['semantic']['semantic_miou']})
            adir=evaluation/run/after
            bdir=evaluation/run/before if before in current else ref/before if before.startswith('AT_') else Path(cfg['recorded_roots_to_verify']['local_ownership_root'])/'evaluation'/run/before
            transitions=region_transitions(semgt,np.load(bdir/'semantic.npy'),np.load(adir/'semantic.npy'),np.load(bdir/'owners.npy'),np.load(adir/'owners.npy'),region,valid)
            ac=read(adir/'regional_confusion.json');bc=read(bdir/'regional_confusion.json')
            deltas={key:(np.array(ac['regions'][key]['confusion'])-np.array(bc['regions'][key]['confusion'])).tolist() for key in ac['regions']}
            global_delta=np.array(ac['global_confusion'])-np.array(bc['global_confusion'])
            assert np.array_equal(sum(np.array(v) for v in deltas.values()),global_delta)
            regions_out.append({'run':run,'after':after,'before':before,'transitions':transitions,'confusion_deltas':deltas,'global_delta':global_delta.tolist(),'integer_reconstruction':'EXACT'})
        native=np.load(source/'native_owners.npy');unqualified=[i for i in np.unique(native) if i>0 and i not in {r['native_owner_id'] for r in records if r['source']=='OVI'}]
        native_projected=np.where(matched,native[near],0)
        for threshold in [.25,.5,.75]:
            base=outcomes(baseline['AT_O_AREA'],'unique',threshold);u=outcomes(baseline['AT_U00'],'overlapping',threshold);loc=outcomes(baseline['LO_U00_LOCAL'],'unique',threshold)
            selected=(set(u['matched'])-set(base['matched']))|(set(base['matched'])-set(loc['matched']))
            for gid in sorted(selected,key=int):
                j=int(np.flatnonzero(gtids==int(gid))[0]);oi=int(ovi[np.argmax(ious[ovi,j])]);si=int(sf[np.argmax(ious[sf,j])]);true_class=int(gid)//1000
                correct=np.flatnonzero((ious[:,j]>u['threshold'])&(original_labels==true_class)&(overlap['projected_sizes']>=100))
                if ious[oi,j]>u['threshold'] and original_labels[oi]!=true_class:description='existing OVI geometry/class issue'
                elif ious[oi,j]<=u['threshold']:description='unavailable eligible OVI shape'
                elif len(correct):description='existing correct-class coverage but ranking/matching issue'
                else:description='ambiguous'
                native_best=None
                for nid in unqualified:
                    mask=native_projected==nid;intersection=np.count_nonzero(mask&(gt==int(gid)));iou=intersection/max(1,mask.sum()+gtsizes[j]-intersection)
                    if native_best is None or iou>native_best['iou']:native_best={'native_owner_id':int(nid),'iou':float(iou)}
                causes.append({'run':run,'gt_id':int(gid),'runtime_threshold':u['threshold'],'primary_description':description,
                    'best_OVI':{'candidate':oi,'iou':float(ious[oi,j]),'class':int(original_labels[oi])},
                    'best_SF':{'candidate':si,'iou':float(ious[si,j]),'class':int(original_labels[si])},
                    'unqualified_native_coverage':native_best,'correct_class_strict_candidates':correct.tolist(),
                    'O_unique':base['matched'].get(gid),'U00_overlapping':u['matched'].get(gid),
                    'U00_unique':outcomes(baseline['AT_U00'],'unique',threshold)['matched'].get(gid),
                    'LOCAL_unique':loc['matched'].get(gid),'evaluation_only':True})
    write(out/'controlled_effects.json',effects);write(out/'object_summary.json',table_c);write(out/'cost_coverage.json',table_d)
    write_gzip(out/'all_object_outcomes.json.gz',allobjects);write_gzip(out/'cause_ledger.json.gz',causes)
    write_gzip(out/'final_support_and_labels.json.gz',support);write_gzip(out/'regional_deltas.json.gz',regions_out)
    write(out/'reused_references.json',refs)
    rank_changes=[];eligibility=[];examples=[]
    for run in cfg['runs']:
        e3=read(pred/run/'OD_E3_OBJECT_SEMANTIC.json');e4=read(pred/run/'OD_E4_FINAL_RANK.json')
        for i in e3['kept']:
            if e3['rank_scores_serialized'][i]!=e4['rank_scores_serialized'][i]:
                rank_changes.append({'run':run,'canonical_index':i,'E3_rank':e3['rank_scores_serialized'][i],'E4_rank':e4['rank_scores_serialized'][i]})
        for method in ['OD_E1_SEMANTIC','OD_E2_OBJECT','OD_E3_OBJECT_SEMANTIC','OD_E4_FINAL_RANK']:
            for item in [x for x in support if x['run']==run and x['method']==method]:
                if 99<=item['unique_projected_area']<=101:
                    eligibility.append({'run':run,'method':method,'canonical_index':item['canonical_index'],'points':item['unique_projected_area']})
        changed=[t for t in read(pred/run/'semantics.json')['targets'] if t['reason']=='accepted']
        for t in changed:
            i=t['candidate'];e2owner=np.load(evaluation/run/'OD_E2_OBJECT/owners.npy');mask=e2owner==i+1
            before=np.load(evaluation/run/'OD_E2_OBJECT/semantic.npy');after=np.load(evaluation/run/'OD_E3_OBJECT_SEMANTIC/semantic.npy')
            examples.append({'selection_rule':'all accepted class changes, sorted run/canonical ID; no favorable subset',
                'run':run,'candidate':i,'GT_class_histogram':dict(zip(*[a.tolist() for a in np.unique(semgt[mask],return_counts=True)])),
                'E2_correct_points':int(np.count_nonzero(mask&(before==semgt))),
                'E3_correct_points':int(np.count_nonzero(mask&(after==semgt)))})
    write(out/'ranking_eligibility_examples.json',{'rank_changes':rank_changes,'near_100_instances':eligibility,
         'E3_E4_size_eligibility_changes':[],'class_change_examples':examples,
         'eligibility_statement':'E3/E4 mask bytes identical; no 99-to-101 transition possible; all near-threshold final instances enumerated'})
    with (out/'performance.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,lineterminator='\n',fieldnames=['run','condition','active_count','active_ap','unique_ap','unique_ap50','unique_ap25','semantic_miou','semantic_macc'])
        writer.writeheader()
        for row in rows:
            writer.writerow({'run':row['run'],'condition':row['condition'],'active_count':row['active_count'],
                'active_ap':row['overlapping']['released']['all_ap'],'unique_ap':row['unique']['released']['all_ap'],
                'unique_ap50':row['unique']['released']['all_ap_50%'],'unique_ap25':row['unique']['released']['all_ap_25%'],
                **{key:row['semantic'][key] for key in ['semantic_miou','semantic_macc']}})
    for path in out.iterdir():shutil.copy2(path,small/path.name)
    for name in ['input_binding.json','frozen_config.json','prediction_manifest.json','real_crop_smoke.json']:
        shutil.copy2(pred/name,small/name)
    for run in cfg['runs']:
        for name in ['actions.json','semantics.json','geometry_archive.json']:
            write_gzip(small/f'{run}_{name}.gz',read(pred/run/name))
    shutil.copy2(evaluation/'performance.json',small/'performance.json');shutil.copy2(evaluation/'metric_protocol.json',small/'metric_protocol.json')
    write(small/'large_artifacts.json',{'uploaded_to_git':False,'artifacts':[identity(p) for p in sorted(root.rglob('*')) if p.is_file()],
          'reconstruct':['run_static_object_decoupling.py','evaluate_static_object_decoupling.py','summarize_static_object_decoupling.py']})
    ap=lambda r:100*r['unique']['released']['all_ap']
    lines=['# Object/semantic decoupling — Room0\n', '科学状态：**COMPLETE_NO_NET_GAIN**。软件与 8 条件实验已完成；未建立跨场景有效性。\n','8 个新条件均为实际运行；历史对照标记为 REUSED_REFERENCE。数值为百分数，差值为百分点。\n',
      '## A：实际性能\n','| Run | Method | OVI/SF | Active AP | Unique AP / AP50 / AP25 | Canonical AP75 | mIoU / mAcc | Unknown source / projected | Empty / small | Targets / views / charged new crops | Eval seconds |',
      '|---|---|---:|---:|---|---:|---|---|---|---|---:|']
    for r in rows:
        u=r['unique']['released'];s=r['semantic'];c=r['active_source_counts']
        ts=[t for t in read(pred/r['run']/'semantics.json')['targets'] if r['condition'] in t['methods']]
        views=sum(len(t['views']) for t in ts)
        charged=6*sum(not v.get('cache_hit',False) and 'abstention' not in v for t in ts if t['methods'][0]==r['condition'] for v in t['views'])
        lines.append(f"| {r['run']} | {r['condition']} | {c['OVI']}/{c['SpaCeFormer']} | {100*r['overlapping']['released']['all_ap']:.3f} | {100*u['all_ap']:.3f} / {100*u['all_ap_50%']:.3f} / {100*u['all_ap_25%']:.3f} | {100*r['unique_high_iou_canonical_diagnostic']['ap75']:.3f} | {100*s['semantic_miou']:.3f} / {100*s['semantic_macc']:.3f} | {100*r['source_unknown_fraction']:.3f} / {100*r['projected_unknown_fraction']:.3f} | {r['unique_empty_count']}/{r['unique_small_nonempty_count']} | {len(ts)}/{views}/{charged} | {r['seconds']:.2f} |")
    lines+=['\n新裁剪数按同一目标首次使用的条件归属，E1/E3 共用输入不重复计费。原生 1024D SigLIP，E1/E3 使用同目标掩码六裁剪与 51 类文本；E2 不读取新语义。Canonical AP75 是独立类别无关诊断，非发布版 AP。\n',
      '## B：受控效应\n','| Run | After − Before | Δ Unique AP | Δ AP50 | Δ mIoU |','|---|---|---:|---:|---:|']
    for e in effects:lines.append(f"| {e['run']} | {e['after']} − {e['before']} | {100*e['AP_delta']:+.3f} | {100*e['AP50_delta']:+.3f} | {100*e['mIoU_delta']:+.3f} |")
    lines+=['\nE1 仅改类；E2 改活动对象与所有权；E3 对 E2 仅改类；E4 对 E3 仅改源点面积排名。E1 包含重选视角影响。此为分阶段消融，不是完整析因设计，不报告语义与几何的纯交互量。\n',
      '## C：对象与动作\n','| Run | Method | U00 added | Retained | Newly gained | Newly lost |','|---|---|---|---|---|---|']
    for c in table_c:lines.append('| '+' | '.join(str(c[k]) for k in ['run','method','U00_added','retained','gained','lost'])+' |')
    lines+=['\n对象表采用发布版 strict IoU > .5 的真实 first-match / duplicate-score-owner 事件。完整 .25/.5/.75 全对象记录、几何对应下改类对错、原始/最终 IoU 与范围见机器账本；不以最大 IoU 代替 TP。\n',
      '## D：覆盖、失败与成本\n','| Run | Method | Accepted actions | Ignored events (all thresholds) | Invalid class | Run requests / batches / crops | Budget excluded |','|---|---|---:|---:|---:|---|---:|']
    for d in table_d:lines.append(f"| {d['run']} | {d['method']} | {d['accepted_actions']} | {d['ignored_events']} | {d['invalid_class']} | {d['run_semantic_requests']}/{d['run_encoder_batches']}/{d['run_crop_inputs']} | {d['budget_excluded_targets']} |")
    manifest=read(pred/'prediction_manifest.json')
    lines += [f"\n成本列为该 run 在 E1/E3 联合去重后的实际成本，不能逐行相加。共 {manifest['encoder_batches']} 次前向、{manifest['crop_inputs']} 张裁剪；模型加载 {manifest['model_load_seconds']:.2f}s，裁剪/预处理 {manifest['crop_seconds']:.2f}s，编码 {manifest['encoding_seconds']:.2f}s；无新 2D/3D 分割推理。",
       '\n主缓存仅接受一个 ADD_UNCOVERED_OBJECT，重复缓存没有接受动作。动作分数使用归档正实例像素；未知观测弃权，不代表完整可见像素。缺失覆盖以 owner/class0 保留坐标并计入 FN。区域整数混淆差逐项重建全局，包括 pred0。',
       '\n全部原生 OVI 语义目标均未通过改类；新增 SF 目标的独立改类不代表修复原生错误。类别错误、缺少合格形状和匹配/排名问题分开记录；两份 Room0 缓存不是独立场景或盲验证。',
       '\n确定性正/负例：唯一新增对象（主缓存 SF query20，canonical99）源点 91→78，评估点 40→33；E2 原类 wall-plug 的局部语义改善被 E3 改为 blanket 后抵消，实例因小于100点未进入 AP。E4 仅将该实例分数91改为78；没有99–101门槛穿越，几何未改善。该对象是全部接受改类集合，不是挑选的成功案例。',
       '\n机器证据：[performance](../../../artifacts/static_ovmap/object_decoupling_v1/performance.json)、[effects](../../../artifacts/static_ovmap/object_decoupling_v1/controlled_effects.json)、[objects](../../../artifacts/static_ovmap/object_decoupling_v1/all_object_outcomes.json.gz)、[causes](../../../artifacts/static_ovmap/object_decoupling_v1/cause_ledger.json.gz)、[support/labels](../../../artifacts/static_ovmap/object_decoupling_v1/final_support_and_labels.json.gz)、[regions](../../../artifacts/static_ovmap/object_decoupling_v1/regional_deltas.json.gz)。']
    (ROOT/cfg['delivery']['results']).write_text('\n'.join(lines)+'\n')
    print('analysis complete',len(allobjects),'object rows',len(causes),'cause rows',flush=True)


if __name__=='__main__':main()
