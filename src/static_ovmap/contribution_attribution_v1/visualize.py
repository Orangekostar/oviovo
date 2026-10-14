"""Prelocked median cases, actual saved geometry, and JSON-derived statistics."""
from pathlib import Path
import time
import numpy as np
from .common import (ConsumptionIndex,verified,read_document,write,immutable_record,unit_identity,cached_record)
from .binding import load_binding
from .records import Records
from .nomination import prelock

def choose_cases(rows,module):
    strata=dict(success=[r for r in rows if r.get('path')=='CORRECTED' or module=='R' and r.get('domain')=='NEW_RECOVERY' and r.get('matched_IoU50') and r['new_label']==r.get('GT_class')],
                failure=[r for r in rows if r.get('path')=='HARMED' or module=='R' and r.get('domain')=='NEW_RECOVERY' and not (r.get('matched_IoU50') and r['new_label']==r.get('GT_class'))],
                unchanged=[r for r in rows if r.get('path')=='REMAINED_CORRECT' or module=='R' and r.get('domain')=='INCUMBENT' and r['old_label']==r['new_label']])
    result=[];shortfalls=[]
    for stratum,bank in strata.items():
        comparable=[r for r in bank if r.get('margin_change') is not None or r.get('IoU') is not None]
        measure=lambda r:abs(r['margin_change']) if r.get('margin_change') is not None else r['IoU']
        median=float(np.median([measure(r) for r in comparable])) if comparable else None
        comparable.sort(key=lambda r:(abs(measure(r)-median),('replica8','scannet_cf18').index(r['cohort']),r['family'],r['scene'],r['owner']))
        selected=[];families=set()
        for r in comparable:
            if r['family'] in families:continue
            selected.append(r);families.add(r['family'])
            if len(selected)==2:break
        if len(selected)<2:
            for r in comparable:
                if r in selected:continue
                selected.append(r)
                if len(selected)==2:break
        result += [dict(r,module=module,stratum=stratum,stratum_median=median,selection_measure='ABS_GT_MARGIN_CHANGE' if r.get('margin_change') is not None else 'DESCRIPTIVE_MAX_IOU') for r in selected]
        shortfalls.append(dict(module=module,stratum=stratum,available=len(comparable),selected=len(selected),shortfall=2-len(selected)))
    return result,shortfalls

def statistics(root,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    root=Path(root);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    history=verified(root/'history.json');attribute=verified(root/'attribute.json');surface=verified(root/'surface.json');rank=verified(root/'rank.json');main=verified(root/'main_score.json')
    times={}
    for row in verified(root/'stage_exits.json')['records']:
        if row['exit_code']==0 and not row.get('exact_completed_output_reused',False):times.setdefault(row['phase'],row['wall_seconds'])
    source=write(folder/'figure_sources.json',dict(history=history['ER_effects'],source_errors=attribute['source_summaries'],matched_step=surface['history']['matched_step'],shuffle={k:dict(metrics=v['metrics'],shuffle_mean=v['shuffle_mean']) for k,v in surface['intervention'].items()},
                 budgets=dict(head=[surface['workload']['new_head_calls_this_invocation'],4000],main_rows=[main['logical_scene_rows'],312],rank_rows=[rank['physical_scene_assign_calls'],104]),phase_wall_seconds=times,phase_wall_scope='FIRST_OBSERVED_SUCCESS_PER_PHASE;FAILURES_AND_REUSE_SEPARATE_IN_STAGE_EXITS',new_counts=dict(training=0,RGB=0,SAM=0,map=0)))
    def save(fig,name):
        for ext in ('png','svg','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax,cohort in zip(axes,('replica8','scannet_cf18'),strict=True):
        row=next(r for r in source['history'] if r['cohort']==cohort and r['metric']=='apall');keys=['E_add','R_cond','interaction'];values=[100*row['effects'][k] for k in keys]
        ax.bar(keys,values,color=['#0072B2','#D55E00','#999999'],hatch=['','//','..']);ax.axhline(0,color='black',lw=.8);ax.set(title=cohort,ylabel='APall change (percentage points)')
    save(fig,'ER_effects')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax,cohort in zip(axes,('replica8','scannet_cf18'),strict=True):
        rows=[r for r in source['source_errors'] if r['cohort']==cohort and r['stratum']!='ALL'];x=np.arange(len(rows));width=.25
        for shift,field,color in [(-1,'NQ_common_errors','#999999'),(0,'F_corrects_NQ','#0072B2'),(1,'fusion_harmed','#D55E00')]:ax.bar(x+shift*width,[r[field] for r in rows],width,color=color,label=field)
        ax.set_xticks(x,['All agree','NQ agree / F differs','Other disagreement'],rotation=15);ax.set(title=cohort,ylabel='Objects (all three available, IoU > 0.5)');ax.legend(fontsize=8)
    save(fig,'source_error_paths')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained')
    for ax,seed,name in zip(axes,(17,29),('FG2_SURFACE','S29_FG2_SURFACE'),strict=True):
        matched=[r for r in source['matched_step'] if r['seed']==seed];m={r['method']:r['metrics']['A']*100 for r in matched if r['status']=='COMPLETE'}
        shuffled=source['shuffle'][name];values=[m.get('FG1_VIEW',np.nan),m.get('FG2_SURFACE',np.nan),100*shuffled['metrics']['ORIGINAL']['A'],100*shuffled['shuffle_mean']['A']]
        ax.bar(range(4),values,color=['#999999','#0072B2','#56B4E9','#D55E00']);ax.scatter([3]*5,[100*v['A'] for k,v in shuffled['metrics'].items() if k!='ORIGINAL'],color='black',marker='x',zorder=4)
        ax.set_xticks(range(4),['FG1 step1000','FG2 step1000','FG2 selected','5-shuffle mean'],rotation=15);ax.set(title=f'Seed {seed}',ylabel='DEV A (%)',ylim=(0,100))
    save(fig,'S_matched_and_shuffle')
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),layout='constrained');items=list(source['budgets'].items())
    axes[0].barh([k for k,v in items],[v[0]/v[1] for k,v in items],color='#0072B2');axes[0].set(xlim=(0,1.23),xticks=[0,.5,1.],xlabel='Actual use / fixed limit')
    for i,(k,v) in enumerate(items):axes[0].text(v[0]/v[1],i,f' {v[0]}/{v[1]}',va='center',fontsize=9)
    times=source['phase_wall_seconds'];axes[1].barh(list(times),list(times.values()),color='#999999');axes[1].set(xlabel='Phase wall seconds (nested / overlapping; do not sum)')
    save(fig,'budget_and_cost')
    return source

def _render_case(root,records,case,folder):
    import matplotlib.pyplot as plt
    from .recovery import official_entries,saved_surface
    from .sources import source_probabilities
    from .attribute import probability_measures
    from scipy.special import softmax
    scene=case['scene'];owner=case['owner'];module=case['module'];pair=dict(E_SOURCE=('A0','A1'),E_GROUP=('E6','E7'),R=('A1','A3'),S=('FG0_FC8','FG2_SURFACE'))[module]
    context=read_document(Path(records.binding['compact_root'])/'contexts'/f'{scene}.json',records.index);surface=saved_surface(records,context)
    with np.load(surface['path'],allow_pickle=False) as arrays:xyz=arrays['surface_xyz'].copy()
    a,b=[records.payload(scene,m) for m in pair];mask=b.owner_ids==owner
    if not mask.any():return dict(status='VISUAL_BLOCKED_GEOMETRY',reason='NO_SAVED_OWNER_SUPPORT',case=case)
    points=xyz[mask];low=points.min(0);high=points.max(0);pad=np.maximum((high-low)*.3,.1);keep=np.all((xyz>=low-pad)&(xyz<=high+pad),axis=1);selected=np.flatnonzero(keep)
    if len(selected)>30000:selected=selected[np.linspace(0,len(selected)-1,30000,dtype=np.int64)]
    capture=read_document(records.resolver.resolve(context['capture_manifest']),records.index);chosen_camera=None;skipped=[]
    readout=verified(Path(root)/'main_predict'/scene/'readouts.json');valid_ids=readout['valid_ids'];truth=case.get('GT_class')
    sources={n:read_document(d['path'],records.index) for n,d in readout['source_descriptors'].items()}
    # Recovery owners can lie outside the incumbent-source request registry.
    # Missing requests are explicitly unavailable in this display-only card.
    for source in sources.values():
        if str(owner) not in source['objects']:source['objects'][str(owner)]=dict(available=False,scores=None,used_frames=[],reason='OUTSIDE_INCUMBENT_SOURCE_REQUEST_DOMAIN')
    preferred=set(sources['N']['objects'].get(str(owner),{}).get('used_frames',[]))
    cameras=sorted(capture['frames'],key=lambda f:(f['frame_id'] not in preferred,f['frame_id']))
    for camera in cameras:
        pose=np.asarray(camera['pose_c2w'],np.float64);K=np.asarray(camera['intrinsics'],np.float64)
        local=(xyz[selected]-pose[:3,3])@pose[:3,:3];positive=local[:,2]>1e-6
        pixels=local@K.T;pixels=np.divide(pixels[:,:2],local[:,2,None],out=np.full((len(selected),2),np.nan),where=positive[:,None])
        h,w=camera['image_size_hw'];visible=positive&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
        owner_visible=visible&(b.owner_ids[selected]==owner)
        if not owner_visible.any():skipped.append(dict(frame_id=camera['frame_id'],reason='SELECTED_OWNER_FULLY_OUTSIDE_SAVED_CAMERA_FRUSTUM'));continue
        chosen_camera={k:camera[k] for k in ('frame_id','pose_c2w','intrinsics','image_size_hw')}
        crop=pixels[owner_visible];pixel_low=crop.min(0);pixel_high=crop.max(0);pixel_pad=np.maximum((pixel_high-pixel_low)*.3,8.)
        use=visible&np.all((pixels>=pixel_low-pixel_pad)&(pixels<=pixel_high+pixel_pad),axis=1);coords=pixels[use];selected=selected[use];break
    if chosen_camera is None:return dict(status='VISUAL_BLOCKED_GEOMETRY',reason='OWNER_FULLY_INVISIBLE_IN_ALL_SAVED_CAMERAS',case=case,skipped_views=skipped)
    semantics=[a.semantic_labels[selected],b.semantic_labels[selected]]
    ids=context['models']['native']['valid_ids'];names=context['models']['native']['class_names'];class_names=dict(zip(map(int,ids),names,strict=True));class_names[0]='unknown'
    focus=sorted({int(i) for i in (case['old_label'],case['new_label'],truth) if i is not None and i!=0})
    colors={int(i):'#dddddd' for i in ids};colors[0]='#999999'
    colors.update(dict(zip(focus,('#0072B2','#D55E00','#009E73'),strict=False)))
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.0),layout='constrained')
    for ax,labels,title,payload in zip(axes[:2],semantics,pair,(a,b),strict=True):
        ax.scatter(coords[:,0],coords[:,1],c=[colors[int(i)] for i in labels],s=1.4,rasterized=True)
        highlight=payload.owner_ids[selected]==owner
        if highlight.any():ax.scatter(coords[highlight,0],coords[highlight,1],facecolors='none',edgecolors='black',s=5,lw=.15)
        ax.set(title=title,aspect='equal',xlabel='Saved camera u (pixel)',ylabel='Saved camera v (pixel)',xlim=(pixel_low[0]-pixel_pad[0],pixel_high[0]+pixel_pad[0]),ylim=(pixel_high[1]+pixel_pad[1],pixel_low[1]-pixel_pad[1]))
    official=[]
    for method in pair:
        score=records.score(scene,method);trace=read_document(Path(score['manifest']).with_name('trace.json.gz'),records.index,sealed=False);entries=official_entries(trace,.5)
        official.append(dict(method=method,definite_TP=owner in entries['definite_TP_owners'],definite_FP_score_entries=entries['definite_FP_score_entry_owners'].count(owner),rank=dict(records.payload(scene,method).instance_ranks).get(owner)))
    text=[f'{module} / {case["stratum"]}',f'{scene} / owner {owner}',f'Old: {class_names.get(case["old_label"],str(case["old_label"]))}',f'New: {class_names.get(case["new_label"],str(case["new_label"]))}',f'GT reference: {class_names.get(case.get("GT_class"),str(case.get("GT_class")))}',f'Max IoU: {case["IoU"]:.3f}',f'Path: {case.get("path",case.get("domain"))}']
    text += [f'{r["method"]}: rank={r["rank"]}, TP={r["definite_TP"]}, FP entries={r["definite_FP_score_entries"]}' for r in official]
    probabilities=source_probabilities(sources,readout['temperatures'],valid_ids,owner);source_evidence={}
    for name,probability in probabilities.items():
        top=None if probability is None else valid_ids[int(np.argmax(probability))];measures=probability_measures(probability,valid_ids,truth)
        source_evidence[name]=dict(available=probability is not None,top1=top,**measures)
        margin='n/a' if measures['margin'] is None else f'{measures["margin"]:+.3f}'
        text.append(f'{name}: {class_names.get(top,"unavailable")}; GT margin {margin}')
    if module=='S':
        for name in ('J0_FC8_FONLY','J1_FG2_S17_FONLY'):
            row=readout['methods'][name].get(str(owner));vector=None if row is None else row.get('cosines')
            probability=None if vector is None else softmax(np.asarray(vector,np.float64)/readout['temperatures']['F'])
            measures=probability_measures(probability,valid_ids,truth)
            source_evidence[name]=dict(available=vector is not None,cosines=vector,**measures)
            margin='n/a' if measures['margin'] is None else f'{measures["margin"]:+.3f}'
            text.append(f'{name}: GT margin {margin}')
    axes[2].axis('off');axes[2].text(0,1,'\n'.join(text),va='top',fontsize=8,wrap=True);fig.suptitle(f'POINT_PROJECTION: shared saved camera frame {chosen_camera["frame_id"]}; geometry, no RGB/depth occlusion rendering',fontsize=10)
    from matplotlib.patches import Patch
    legend=[Patch(color=colors[i],label=class_names.get(i,str(i))) for i in focus]
    legend += [Patch(color='#dddddd',label='Other classes'),Patch(color=colors[0],label='Unknown')]
    fig.legend(handles=legend,loc='lower center',ncol=len(legend),bbox_to_anchor=(.5,-.12),fontsize=8)
    path=folder/(module+'_'+scene+f'_owner{owner}_{case["stratum"]}.png');fig.savefig(path,dpi=160,bbox_inches='tight');plt.close(fig)
    return dict(status='COMPLETE',case=case,figure=str(path),projection='POINT_PROJECTION_SAVED_CAMERA_PERSPECTIVE',saved_geometry=surface,saved_camera=chosen_camera,skipped_views=skipped,official_status=official,source_evidence=source_evidence,raw_RGB_published=False,raw_GT_image_published=False)

def run(root):
    import matplotlib
    matplotlib.use('Agg')
    root=Path(root);binding=load_binding(root);index=ConsumptionIndex();records=Records(root,binding,index);locked=prelock(root);attribute=verified(root/'attribute.json');recovery=verified(root/'recovery.json')
    key=unit_identity('qualitative',dict(shortlist=locked['identity'],attribute=attribute['identity'],recovery=recovery['identity'],rank=verified(root/'rank.json')['identity']),producers=('common','records','recovery','nomination','visualize'))
    previous=cached_record(root/'qualitative.json',key)
    if previous:return previous
    folder=root/'qualitative';folder.mkdir(parents=True,exist_ok=True);cases=[];shortfalls=[]
    contract=dict(target='CVPR_evidence_package',claim='Quantitative attribution and inspectable examples on exposed data',source_data=['history.json','attribute.json','surface.json','rank.json','stability.json'],
                  statistics='ORIGINAL_METRICS_FAMILY_LFO_BOOTSTRAP',case_policy='MEDIAN_EFFECT_FIXED_TIES_FAMILY_DIVERSITY',palette='OKABE_ITO_WITH_HATCH_AND_DIRECT_LABELS',formats=['PNG','PDF','SVG'],raw_RGB_or_GT_images=False)
    write(folder/'visual_contract.json',contract)
    for selected in locked['selection']:
        module=selected['module'];rows=[r for r in attribute['S_owner_rows'] if r['seed']==17 and r['eligible']] if module=='S' else recovery['object_rows'] if module=='R' else [r for r in attribute['E_owner_rows'] if r['module']==module]
        chosen,missing=choose_cases(rows,module);cases+=chosen;shortfalls+=missing
    if len(cases)>12:raise ValueError('Qualitative case budget exceeded')
    immutable_record(folder/'case_lock.json',dict(status='COMPLETE',cases=cases,shortfalls=shortfalls,shortlist=locked['identity'],before_rendering=True))
    start=time.perf_counter();figures=statistics(root,folder/'statistics');rendered=[]
    for case in cases:
        try:rendered.append(_render_case(root,records,case,folder))
        except FileNotFoundError as exc:rendered.append(dict(status='VISUAL_BLOCKED_GEOMETRY',case=case,reason=str(exc)))
        records.cache.clear()
    index.write_memo(folder/'verifications.json')
    return write(root/'qualitative.json',dict(status='COMPLETE',input_identity=key,shortlist=locked['identity'],cases=rendered,shortfalls=shortfalls,statistical_figure_source=figures['identity'],statistics_figures=4,
                 case_PNG_count=sum(r['status']=='COMPLETE' for r in rendered),wall_seconds=time.perf_counter()-start,raw_RGB_or_GT_published=False,render_QA='PENDING_PRIMARY_RENDERED_REVIEW'))
