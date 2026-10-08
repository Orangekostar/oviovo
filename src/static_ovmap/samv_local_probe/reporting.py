"""Three measured table families, mask-only contact sheets and compact evidence."""

from pathlib import Path
import csv
import gzip
import json
import shutil

import numpy as np

from .binding import load_scene
from .common import METHODS,REPO,ConsumptionIndex,read,verified,write

LABELS={'REF_D2':'D2 reference','SV00_G1':'G1','SV01_SAM2_GEOM':'SAM2 geometry',
    'SV02_SAMV_GEOM':'SAM-V geometry','SV03_OLDMASK_FC':'OLD-mask FC',
    'SV04_SAMV_FC':'SAM-V-mask FC','SV05_COMBINED':'SAM-V + fixed FC'}
METRICS=('apall','ap50','ap25','miou','macc')
HEADERS={'method':'Method','cohort':'Cohort','model':'Model','targets':'Targets','anchor_success':'Anchor OK',
    'raw_positive_rows':'Raw rows','admitted_positive_rows':'Admitted rows','suppressed_rows':'Blocked rows',
    'mean_best_iou':'Best IoU','mean_fixed_iou':'Fixed-GT IoU','joint_eligible':'Joint eligible',
    'wrong_to_right':'Wrong to right','right_to_wrong':'Right to wrong','undefined_fixed_reference':'No fixed GT',
    'scene':'Scene','owner':'Owner','calls':'Calls','window_frames':'Frames','dtype':'Dtype',
    'mean_seconds':'Mean (s)','median_seconds':'Median (s)','peak_allocated_GiB':'Allocated (GiB)',
    'peak_reserved_GiB':'Reserved (GiB)','observation':'Observer (s)',
    'query_planning_and_canonical_preparation':'Planning/JPEG (s)','SAM2_lifting':'SAM2 lifting (s)',
    'SAMV_lifting':'SAM-V lifting (s)','FC_worker_wall_including_first_load':'FC wall (s)',
    'output_rank_construction':'Output/rank (s)','released_evaluation_registry_and_export':'Evaluation (s)',
    'diagnostics':'Diagnosis (s)','new_observer_frames':'New ray frames','new_observer_rays':'New rays'}
for _cohort,_label in [('replica_probe2','Replica'),('cf_probe2','CF')]:
    for _metric,_title in [('apall','APall'),('ap50','AP50'),('ap25','AP25'),('miou','mIoU'),('macc','mAcc')]:
        HEADERS[_cohort+'_'+_metric+'_percent']=_label+' '+_title
        HEADERS[_metric+'_delta_pp']=_title+' delta (pp)'


def table(output,name,rows,columns):
    output.mkdir(parents=True,exist_ok=True)
    write(output/(name+'.json'),{'columns':columns,'rows':rows,'source_unit':'EXPLICIT_PER_COLUMN',
        'undefined_cells_are_null':True})
    with (output/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    def display(value):
        if value is None:return '—'
        if isinstance(value,float):return f'{value:.3f}'
        return str(value)
    headings=[HEADERS.get(c,c) for c in columns]
    body=['| '+' | '.join(headings)+' |','| '+' | '.join(['---']*len(columns))+' |']
    body+=['| '+' | '.join(display(row.get(c)) for c in columns)+' |' for row in rows]
    (output/(name+'.md')).write_text('\n'.join(body)+'\n')
    def tex(value):
        return display(value).replace('—',r'\textemdash{}').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
    text=['\\begin{tabular}{'+('l'*len(columns))+'}',r'\toprule',' & '.join(tex(c) for c in headings)+r' \\',r'\midrule']
    text+=[' & '.join(tex(row.get(c)) for c in columns)+r' \\' for row in rows]
    text +=[r'\bottomrule',r'\end{tabular}']
    (output/(name+'.tex')).write_text('\n'.join(text)+'\n')
    return '\n'.join(body)


def contact_sheets(binding,output,diagnostics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from .query_plan import load_observation
    output.mkdir(parents=True,exist_ok=True);cases=[];receipts=[]
    for scene in binding['scenes']:
        inputs=load_scene(binding,scene);plan=verified(Path(binding['output_root'])/'query_plan'/scene/'receipt.json')
        images=[]
        for query in plan['queries']:
            slot=query['anchor_slot'];frame=query['frames'][slot];source,valid=load_observation(frame,len(inputs.xyz))
            old=np.zeros(valid.shape,bool);old[valid]=inputs.g1.owner_ids[source[valid]]==query['owner']
            old=np.asarray(Image.fromarray(old).resize((1024,1024),Image.Resampling.NEAREST))
            masks=[old];sources=[]
            for model in ('SAM2','SAMV'):
                row=verified(Path(binding['output_root'])/'segmentation'/scene/model/str(query['owner'])/'receipt.json')
                with np.load(row['raw_arrays']['path'],allow_pickle=False) as arrays:masks.append(arrays[f'canonical_mask_{slot}'])
                sources.append(row['identity'])
            target=next(t for t in diagnostics[scene]['targets'] if t['owner']==query['owner'])
            images.append((query,masks,target));delta=target['models']['SAMV']['best_iou_delta']
            cases.append({'scene':scene,'owner':query['owner'],'delta':delta,'anchor':target['models']['SAMV']['active'],
                'images':(query,masks,target)})
            receipts.append({'scene':scene,'owner':query['owner'],'query_identity':query['identity'],'segmentation_identities':sources})
        n=max(len(images),1);fig,axes=plt.subplots(n,3,figsize=(6.6,max(2.,n*1.35)),squeeze=False)
        if not images:
            for ax in axes.flat:ax.axis('off')
            axes[0,0].text(.05,.5,'NO QUERIABLE TARGET',transform=axes[0,0].transAxes)
        for pos,(query,masks,target) in enumerate(images):
            for j,(name,mask) in enumerate(zip(('G1 visible mask','SAM2 raw mask','SAM-V raw mask'),masks)):
                ax=axes[pos,j];ax.imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
                pts=np.asarray(query['points_xy_canonical']);ax.scatter(pts[:,0],pts[:,1],s=8,c='#E69F00',marker='+')
                title=f"{name} | owner {query['owner']}"
                if j==0:title+=f"\nanchor {query['anchor_frame_id']} | {query['pool']}"
                ax.set_title(title,fontsize=7,pad=7)
                if j==2:ax.text(0,-.06,f"anchor={target['models']['SAMV']['active']} | 3D ΔIoU={target['models']['SAMV']['best_iou_delta']:+.6f}",transform=ax.transAxes,fontsize=6)
        fig.suptitle(scene+' — all locked targets; raw anchor masks, no RGB or 2D GT',fontsize=9)
        fig.tight_layout(rect=(0,0,1,.98));fig.savefig(output/(scene+'_all_targets.png'),dpi=160);fig.savefig(output/(scene+'_all_targets.pdf'));plt.close(fig)
    success=next((c for c in cases if c['anchor'] and c['delta']>0),None)
    failure=next((c for c in cases if not c['anchor'] or c['delta']<0),None)
    pair={'status':'LOCAL_SUPPORT_GAIN_AND_FAILURE_PAIR' if success and failure else 'NO_BOTH_GAIN_AND_FAILURE_CASES',
        'selection_rule':'FIRST_IN_FIXED_SCENE_OWNER_ORDER_WITH_FINAL_3D_BEST_IOU_GAIN_OR_LOSS/ANCHOR_FAILURE',
        'success':{k:v for k,v in success.items() if k!='images'} if success else None,
        'failure':{k:v for k,v in failure.items() if k!='images'} if failure else None,
        'success_is_local_support_gain_not_pilot_promotion':True}
    if success and failure:
        fig,axes=plt.subplots(2,3,figsize=(6.6,3.8))
        for i,(kind,case) in enumerate([('LOCAL 3D SUPPORT GAIN',success),('LOSS / ANCHOR FAILURE',failure)]):
            query,masks,_=case['images']
            for j,mask in enumerate(masks):
                axes[i,j].imshow(mask,cmap='gray',vmin=0,vmax=1);axes[i,j].axis('off')
                title=('G1','SAM2','SAM-V')[j]
                if j==0:title=kind+f"\n{case['scene']}/{case['owner']} | ΔIoU={case['delta']:+.6f} | G1"
                axes[i,j].set_title(title,fontsize=7,pad=7)
        fig.tight_layout();fig.savefig(output/'labeled_gain_failure_pair.png',dpi=180);plt.close(fig)
    write(output/'mask_sources.json',{'selected_cases':receipts,'case_pair':pair,'RGB_published':False,
        'rendered_from_actual_binary_masks':True,'GT_images_published':False})
    return pair


def compact(binding,output,store):
    """Actual small results plus content references to heavy read-only inputs."""
    root=Path(binding['output_root']);index=ConsumptionIndex(output/'verification_memo.json')
    write(output/'result_store.json',store)
    for name in ['model_assets.json','resource_profile.json','freeze.json','selection.json','costs.json','diagnostics/summary.json','pilots/summary.json','no_evidence_builder_check.json']:
        value=verified(root/name);write(output/'receipts'/name,value)
    write(output/'source_references.json',{'binding_identity':binding['identity'],'base_commit':binding['base_commit'],
        'parent_store':binding['parent_result_store'],'scenes':{s:{'source_update_prediction_receipt':r['source_update_prediction_receipt']} for s,r in binding['scenes'].items()},
        'large_assets_external':True,'licensed_raw_RGB_depth_not_published':True,'binding_external':index.identity(root/'source_binding.json')})
    diagnostics={};trace_refs={}
    for scene in binding['scenes']:
        plan=verified(root/'query_plan'/scene/'receipt.json');write(output/'queries'/scene/'receipt.json',plan)
        decision=verified(root/'readout'/scene/'decisions.json');write(output/'decisions'/(scene+'.json'),decision)
        lock=verified(root/'predictions'/scene/'receipt.json');write(output/'predictions'/(scene+'.json'),lock)
        diag=verified(root/'diagnostics'/scene/'receipt.json');diagnostics[scene]=diag
        # Preserve complete real post-lock diagnostics compressed, not server-only links.
        p=output/'diagnostics'/(scene+'.json.gz');p.parent.mkdir(parents=True,exist_ok=True)
        with gzip.GzipFile(filename=str(p),mode='wb',mtime=0) as f:f.write(json.dumps(diag,sort_keys=True,allow_nan=False).encode())
        for model in ('SAM2','SAMV'):
            lift=verified(root/'lifted'/scene/model/'receipt.json');write(output/'lifting'/scene/(model+'.json'),lift)
            # Changed rows are enough to reconstruct the actual full partition from G1.
            with np.load(lift['partition']['path'],allow_pickle=False) as arrays:
                values={k:arrays[k] for k in ['moved_source_rows','old_owners','new_owners']}
            dest=output/'edits'/scene/(model+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**values)
            for query in plan['queries']:
                row=verified(root/'segmentation'/scene/model/str(query['owner'])/'receipt.json')
                write(output/'segmentation'/scene/model/(str(query['owner'])+'.json'),row)
                with np.load(row['raw_arrays']['path'],allow_pickle=False) as arrays:
                    masks={k:arrays[k] for k in arrays.files if k.startswith(('mask_','canonical_mask_'))}
                dest=output/'masks'/scene/model/(str(query['owner'])+'.npz');dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**masks)
        for row in [r for r in store['scene_metrics'] if r['scene']==scene]:
            score=read(row['evaluation_receipt']);write(output/'scoring'/scene/(row['method']+'.json'),
                {'evaluation_identity':score['identity'],'released_receipt':score})
            trace=Path(score['manifest']).with_name('trace.json.gz');identity=index.identity(trace)
            dest=output/'traces'/(identity['sha256']+'.json.gz');dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copyfile(trace,dest)
            trace_refs[scene+':'+row['method']]={'source':identity,'published':str(dest.relative_to(output)),'evaluation_identity':row['evaluation_identity']}
    for cohort,methods in store['pooled_metrics'].items():
        for method,pool in methods.items():write(output/'per_class'/cohort/(method+'.json'),verified(pool['per_class_receipt']))
    write(output/'traces/index.json',trace_refs)
    pair=contact_sheets(binding,output/'visuals',diagnostics)
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if size>30*2**20:raise RuntimeError(f'compact publication exceeds30 MiB: {size/2**20:.3f}')
    return {'bytes':size,'MiB':size/2**20,'case_pair':pair}


def audit(binding,store,costs,diagnostics):
    root=Path(binding['output_root']);freeze=verified(root/'freeze.json');pilots=verified(root/'pilots/summary.json')
    plans={s:verified(root/'query_plan'/s/'receipt.json') for s in binding['scenes']}
    checks={'four_exact_scenes':set(plans)=={'office1','room0','scene0011_00','scene0050_00'},
        'verbatim_protocol':Path(binding['spec']['path']).read_bytes()==(REPO/'docs/paper/static_ovmap/samv_local_probe_v1/spec/PROTOCOL_SPEC.json').read_bytes(),
        '28_complete_scene_rows':len(store['scene_metrics'])==28 and all(r['status']=='COMPLETE' for r in store['scene_metrics']),
        '14_exact_two_scene_pools':sum(map(len,store['pooled_metrics'].values()))==14 and all(p['scene_order']==binding['cohorts'][c] for c,methods in store['pooled_metrics'].items() for p in methods.values()),
        'two_fresh_G1_parity_checks':len(store['baseline_parity'])==2 and all(all(r['checks'].values()) for r in store['baseline_parity']),
        'real_two_cohort_pilots_before_AP':pilots['status']=='REAL_ENGINEERING_PILOTS_VERIFIED' and not pilots['new_AP_scored'],
        'max32_locked_targets':sum(len(p['queries']) for p in plans.values())<=32 and all(len(p['queries'])<=8 for p in plans.values()),
        'GT_free_target_selection':all(not p['GT_used'] and not p['models_used_by_selection'] for p in plans.values()),
        'one_global_frozen_profile':all(p['effective_window_length']==freeze['resource_profile']['window_length'] for p in plans.values()),
        '8_extra_timing_calls':costs['timing']['actual_extra_calls']==8 or bool(costs['timing']['absent_cohorts']),
        'all_timed_raw_masks_exact':all(r['parity']['raw_binary_masks_exact'] for r in costs['timing']['rows']),
        'bounded_model_and_FC_work':costs['budgets_verified'],'no_new_maps_AnyUp_NQ_training':all(costs[k]==0 for k in ['new_maps','new_AnyUp','new_NQ','new_training']),
        'GT_diagnostics_after_global_lock':all(verified(root/'diagnostics'/s/'receipt.json')['GT_read_only_after_all_four_predictions_locked'] for s in plans),
        'target_projection_surface_unchanged':all(verified(root/'predictions'/s/'receipt.json')['one_exclusive_partition_for_AP_and_mIoU'] for s in plans),
        'exact_combination_factorization':all(verified(root/'predictions'/s/'receipt.json')['SV05_owners_exactly_SV02'] and verified(root/'predictions'/s/'receipt.json')['SV05_class_map_exactly_SV04'] for s in plans),
        'deployment_N0_unchanged':verified(root/'selection.json')['deployment']=='N0_UNCHANGED',
        'focused_tests_pass':freeze['focused_tests']['exit_code']==0 and '12 passed' in Path(freeze['focused_tests']['log']['path']).read_text(),
        'complete_implementation_committed_before_science':bool(freeze['commit']) and freeze['all_implementation_present']}
    no_evidence=verified(root/'no_evidence_builder_check.json')
    checks['real_four_map_no_evidence_builder_parity']=len(no_evidence['rows'])==4 and all(all(r['checks'].values()) for r in no_evidence['rows'])
    if not all(checks.values()):raise ValueError('completion evidence audit failed: '+str({k:v for k,v in checks.items() if not v}))
    return write(root/'requirement_audit.json',{'status':'MEASURED_REQUIREMENTS_VERIFIED','checks':checks,
        'source_freeze_identity':freeze['identity'],'result_store_identity':store['identity'],
        'publication_and_full_CLI_terminal_are_separate_final_gates':True,'final_primary_review_required':True})


def report(binding):
    root=Path(binding['output_root']);store=verified(root/'result_store.json');diagnostics=verified(root/'diagnostics/summary.json')
    costs=verified(root/'costs.json')
    from .selection import select
    selection=write(root/'selection.json',select(store['pooled_metrics'],binding['specification']['selection']))
    # Report the controls separately, using the already fixed lexicographic order.
    from functools import cmp_to_key
    controls=['SV01_SAM2_GEOM','SV03_OLDMASK_FC']
    def control_compare(a,b):
        for cell in binding['specification']['selection']['lexicographic']:
            cohort,metric=cell.split('.');delta=store['pooled_metrics'][cohort][a]['metrics'][metric]-store['pooled_metrics'][cohort][b]['metrics'][metric]
            if abs(delta)>1e-10:return -1 if delta>0 else 1
        return controls.index(a)-controls.index(b)
    best_control=sorted(controls,key=cmp_to_key(control_compare))[0]
    control_report={'best_simple_control':best_control,'candidates':controls,
        'fixed_lexicographic_order':binding['specification']['selection']['lexicographic'],
        'control_is_not_a_proposed_method':True,'metrics':{c:{m:store['pooled_metrics'][c][m]['metrics'] for m in controls} for c in binding['cohorts']}}
    store=write(root/'result_store.json',{**store,'diagnostics_identity':diagnostics['identity'],'costs_identity':costs['identity'],
        'selection':selection,'freeze_identity':verified(root/'freeze.json')['identity']})
    output=REPO/'artifacts/static_ovmap/samv_local_probe_v1';tables=output/'tables'
    write(output/'best_simple_control.json',control_report)
    main=[];full=[]
    for method in ('REF_D2',*METHODS):
        row={'method':LABELS[method]};complete={'method':method,'label':LABELS[method],'cell_sources':{}}
        for cohort in ('replica_probe2','cf_probe2'):
            pool=store['pooled_metrics'][cohort][method]
            for k in METRICS:complete[cohort+'_'+k+'_percent']=pool['metrics'][k]*100 if pool['metrics'][k] is not None else None
            for k in ('apall','ap50','miou'):row[cohort+'_'+k+'_percent']=complete[cohort+'_'+k+'_percent']
            complete['cell_sources'][cohort]=pool['identity']
        main.append(row);full.append(complete)
    columns=['method',*[c+'_'+k+'_percent' for c in ('replica_probe2','cf_probe2') for k in ('apall','ap50','miou')]]
    t1=table(tables,'table1_probe_metrics',main,columns)
    table(tables,'table1_all_five_metrics',full,['method',*[c+'_'+k+'_percent' for c in ('replica_probe2','cf_probe2') for k in METRICS]])
    write(tables/'table1_cell_sources.json',{'rows':full})
    mechanism=[]
    for cohort,data in diagnostics['cohorts'].items():
        for model in ('SAM2','SAMV'):
            r=data[model];mechanism.append({'cohort':cohort,'model':model,'targets':r['targets'],'anchor_success':r['anchor_success'],
                'raw_positive_rows':r['raw_positive_rows'],'admitted_positive_rows':r['admitted_positive_rows'],
                'suppressed_rows':r['suppressed_positive_rows'],'GT50':r['class_agnostic_counts']['0.5'],'GT75':r['class_agnostic_counts']['0.75'],
                'mean_best_iou':r['mean_best_iou'],'mean_fixed_iou':r['mean_fixed_reference_iou'],'cell_source':diagnostics['identity']})
    t2=table(tables,'table2_mechanism',mechanism,['cohort','model','targets','anchor_success','raw_positive_rows','admitted_positive_rows','suppressed_rows','GT50','GT75','mean_best_iou','mean_fixed_iou'])
    sem=[]
    for cohort,data in diagnostics['cohorts'].items():
        a,b=(store['pooled_metrics'][cohort][m]['metrics'] for m in ('SV04_SAMV_FC','SV03_OLDMASK_FC'))
        sem.append({'cohort':cohort,**data['semantic'],**{k+'_delta_pp':(a[k]-b[k])*100 for k in METRICS},
            'diagnostic_source':diagnostics['identity'],'metric_sources':{m:store['pooled_metrics'][cohort][m]['identity'] for m in ('SV04_SAMV_FC','SV03_OLDMASK_FC')}})
    t2s=table(tables,'table2_semantic',sem,['cohort','joint_eligible','wrong_to_right','right_to_wrong','undefined_fixed_reference',*[k+'_delta_pp' for k in METRICS]])
    t3=table(tables,'table3_window_timing',costs['timing']['summaries'],['scene','owner','model','calls','window_frames','dtype','mean_seconds','median_seconds','peak_allocated_GiB','peak_reserved_GiB'])
    write(tables/'cell_sources.json',{'table1':full,'table2_structure':mechanism,'table2_semantic':sem,
        'table3_window':[{'scene':r['scene'],'model':r['model'],'owner':r['owner'],'aggregate_source':costs['timing']['identity'],
            'measurement_sources':[v['identity'] for v in costs['timing']['rows'] if v['scene']==r['scene'] and v['model']==r['model']]} for r in costs['timing']['summaries']],
        'table3_stages':costs['identity']})
    stages=[{'scene':s,**{k:v for k,v in row.items() if isinstance(v,(int,float)) or v is None},'cell_source':costs['identity']} for s,row in costs['stage_seconds'].items()]
    stage_columns=['scene','observation','query_planning_and_canonical_preparation','SAM2_lifting','SAMV_lifting','FC_worker_wall_including_first_load','output_rank_construction','released_evaluation_registry_and_export','diagnostics','new_observer_frames','new_observer_rays']
    table(tables,'table3_actual_stage_costs',stages,stage_columns)
    write(tables/'table2_complete_mechanism.json',diagnostics);write(tables/'table3_complete_actual_costs.json',costs)
    docs=REPO/'docs/paper/static_ovmap';artifact='../../../artifacts/static_ovmap/samv_local_probe_v1'
    captions='固定完整场景 office1、room0、scene0011_00、scene0050_00；前两者为 Replica-probe2，后两者为 CF-probe2。每组由两个场景按固定顺序调用 released evaluator 池化，非逐场景 AP 平均。APall 使用 .50:.05:.90，最小实例100点。全部场景此前已曝光。'
    text=['# SAM-V 固定局部探针结果','',captions,'',t1,'',
        f"研究选择：{selection['selected']}；状态 {selection['status']}；MATERIAL={selection['material_met']}。部署 N0_UNCHANGED。",'',
        f"固定词典序下的最佳简单对照：{best_control}。该对照单独报告，不重命名为候选新方法。",'',
        '原始、允许修改与最终独占支持分别保留；class-agnostic GT50/75 不等于官方 AP，也不以查询数作为 precision 分母。','',t2,'',t2s,'',
        '窗口计时：相同预锁定 JPEG/提示、BF16、模型驻留、无特征/结果缓存；读图到原尺寸二值 mask，加载/lifting/FC/评分/保存在计时外。第二轮反序；OS 页缓存未控制。每个输出与科学结果逐像素一致。','',t3,'',
        f"逻辑记录28、两场景池14；新场景 scorer 调用 {store['physical_new_scene_scorer_calls']}。独立的工程检查、失败、模型加载和缓存成本见机器数据。",'',
        f"[完整五项指标]({artifact}/tables/table1_all_five_metrics.md) · [阶段成本]({artifact}/tables/table3_actual_stage_costs.md) · [真实 mask 全目标展示]({artifact}/visuals) · [规范结果库]({artifact}/result_store.json)",'',
        '完整地图独立延迟为 NOT_MEASURED_THIS_PROBE；不声称在线30 FPS。未自动扩大到 Replica8/CF18。']
    (docs/'SAMV_PROBE_RESULTS.md').write_text('\n'.join(text)+'\n')
    (docs/'SAMV_PROBE_SELECTION.md').write_text('# 固定选模规则与结论\n\n'+json.dumps(selection,ensure_ascii=False,indent=2)+'\n\n所有判定使用未舍入 fraction：每个候选五项指标在两个池均不低于G1−1e−10；CF APall>D2+1e−10且AP50≥D2−1e−10；额外 MATERIAL 需 APall−D2≥.001。按 CF APall、CF AP50、Replica APall、CF mIoU 排序，1e−10内视作相等，再优先SV02、SV04、SV05。\n\n'+f'最佳简单对照为 {best_control}，按同一固定词典序比较，仅作对照报告。\n\n'+('建议先分析本轮局部正负案例，再独立审批全量确认实验。' if selection['target_met'] else '本轮未达到固定晋升条件；保持 G1 研究参照，不开展自动全量扩展。')+'\n')
    (docs/'SAMV_PROBE_CLAIMS.md').write_text('# 证据与主张边界\n\n'+json.dumps(diagnostics['mechanism_flags'],ensure_ascii=False,indent=2)+'\n\nSAM-V 是外部预训练分割器。本实验只检验固定提示、实测深度观察、共同 lifter 下的模块集成，不构成新几何感知解码器，也不证明对所有多视角方法更优。\n\n结构与语义主张独立于选模目标。SV04 对照 SV03 使用同一双源双视角成功域；SV05 固定复制 SV02 分区和 SV04 标签，等同某组件时不声称协同增益。未获取新2D GT，不把前插入 panoptic 预测作为GT。\n\n四个已曝光场景只提供开发证据，不是独立确认。两场景池类别覆盖有限，MATERIAL并非统计显著性。仅窗口计时可比较，完整地图延迟及在线FPS未测。\n')
    command='/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_samv_local_probe.py --spec configs/static_ovmap/samv_local_probe_v1.json --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001 --phase all --resume'
    (docs/'SAMV_PROBE_HANDOFF.md').write_text('# 交接\n\n'+f"完整代码/资源配置以外部 freeze.json 为准，基座 {binding['base_commit']}，研究分支 {binding['branch']}。全局窗口 {verified(root/'resource_profile.json')['window_length']}，SAM-V/SAM2独立环境；原始FC保持FP32。\n\n"+'```bash\n'+command+'\n```\n\n'+f"持久存储：{root}。输入、稠密地图、RGB/depth与模型权重留在共享盘，只发布实际紧凑 mask、改动行、类决策、指标与 trace。公开引用见 artifacts/static_ovmap/samv_local_probe_v1/source_references.json。\n\n"+'全流程实际终态/退出码见 execution/all_terminal.json；普通 push 的本地/远程完整SHA在外部 publication/final.json。上述最后两项需在最终发布时实测核验，当前报告不将其预先标记成功。\n\n'+f"模型失败调用数：{len(costs['failed_attempts'])}，详细真实日志/已知耗时/未知null见 costs.json。两次导入问题已纠正，成功的同身份分割叶子在resume时复用；新增前向不被计作缓存命中。部署仍为N0。\n")
    visual_contract={'artifact':'MASK_ONLY_FIXED_TARGET_CONTACT_SHEETS','question':'What masks did both actual models return for every locked target?',
        'evidence_layer':'QUALITATIVE_AND_LIMITATION','source':'locked segmentation receipts and measured old projected masks',
        'statistics':'No uncertainty or quantitative2D claim; final3D best-IoU deltas are post-lock diagnostics',
        'panel_map':'fixed owner rows; G1/SAM2/SAMV anchor columns; all targets included',
        'palette':'gray binary masks; Okabe-Ito orange positive point markers','formats':['PNG','PDF'],
        'traceability':'visuals/mask_sources.json; query/segmentation identities'}
    write(output/'visual_contract.json',visual_contract)
    compact_result=compact(binding,output,store);evidence=audit(binding,store,costs,diagnostics)
    write(output/'requirement_audit.json',evidence)
    return write(root/'report_receipt.json',{'status':'REPORT_COMPLETE','result_store_identity':store['identity'],
        'artifacts':str(output),'compact':compact_result,'tables':3,'all_five_metrics_preserved':True,
        'four_reports':[str(docs/('SAMV_PROBE_'+n+'.md')) for n in ['RESULTS','HANDOFF','SELECTION','CLAIMS']],
        'source_cell_identities_preserved':True,'visual_render_inspection_pending':True,'requirement_audit_identity':evidence['identity']})


def partial(binding,failure):
    root=Path(binding['output_root']);docs=REPO/'docs/paper/static_ovmap';output=REPO/'artifacts/static_ovmap/samv_local_probe_v1'
    status=write(root/'partial.json',{'status':'PARTIAL_DEPENDENCY_BLOCK','failure':failure,'deployment':'N0_UNCHANGED'})
    write(output/'partial.json',status)
    for name in ['RESULTS','HANDOFF','SELECTION','CLAIMS']:
        (docs/('SAMV_PROBE_'+name+'.md')).write_text('# SAM-V 局部探针：未完成\n\n'+json.dumps(status,ensure_ascii=False,indent=2)+'\n\n未测单元保持缺失；不作为零mask、零耗时或完整两场景池。成功叶子保留以便内容核验后resume。\n')
    return status
