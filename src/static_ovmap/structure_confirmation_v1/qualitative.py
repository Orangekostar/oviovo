"""Evidence-bound figures and deterministic success/failure/unchanged cases."""
from pathlib import Path
import numpy as np
from .common import verified,write,immutable_record,unit_key,existing,ConsumptionIndex

COLORS={'S_LL':'#0072B2','V_LL':'#E69F00','G_LL':'#009E73','MV_ATTN':'#CC79A7','S_UL':'#56B4E9','S_LU':'#D55E00','S_UU':'#666666','S_PRE':'#222222'}

def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    return plt

def save(fig,path,index):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);out={}
    for suffix in ('.pdf','.svg','.png'):
        dest=path.with_suffix(suffix);fig.savefig(dest,bbox_inches='tight',dpi=220)
        if suffix=='.svg':dest.write_text('\n'.join(line.rstrip() for line in dest.read_text().splitlines())+'\n')
        out[suffix[1:]]=index.identity(dest)
    return out

def run(binding,*,phase='qualitative'):
    root=Path(binding['output_root']);real=verified(root/'confirmation/real_score.json');gt=verified(root/'confirmation/GT/score.json');selection=verified(root/'model_selection_lock.json');stats=verified(root/'statistics/receipt.json')
    identity=unit_key('qualitative',dict(real=real['identity'],GT=gt['identity'],selection=selection['identity'],statistics=stats['identity']),producers=('qualitative',));path=root/'figures/receipt.json';old=existing(path,identity)
    if old:return old
    plt=setup();index=ConsumptionIndex(root/'figures/verifications.json');sources={};figures={};methods=['S_LL','V_LL','G_LL','MV_ATTN'];markers={17:'o',29:'s'}
    from .statistics import column
    real_cols=verified(root/'confirmation/real_prediction_lock.json')['columns']
    contract=dict(status='COMPLETE',target='CVPR research evidence package; full width figures',claim='Display measured structure contrasts, factorial effects, fixed budget and coverage without presuming gains',reviewer_questions=['Does true correspondence help?','Are both learned stages needed?','How does evidence vary with updates/views?','How much real/novel support exists?'],evidence_layers=['comparison','mechanism','robustness','limitation','qualitative'],sources={'real_score':real['identity'],'GT_score':gt['identity'],'selection':selection['identity'],'statistics':stats['identity']},statistics='Seed points shown separately; family intervals only from fixed CONF bootstrap',palette='Okabe-Ito with markers and direct labels',formats=['PDF','SVG','PNG'],case_rule='Each S/G and S/MV: success2 failure2 unchanged2, closest to stratum median margin delta then family and key',caption_role='Metric, profile, support and unit explicit; provided-reconstruction proposal scope',placement='Main structure comparison and appendix mechanism/limitations',raw_RGB_or_mesh_published=False)
    immutable_record(root/'figures/visual_contract.json',contract)
    fig,ax=plt.subplots(figsize=(7.1,2.8));data=[]
    display_methods=methods+['B_FC','B_SURFACE_MEAN','B_PFE_ADAPTED']
    for j,method in enumerate(display_methods):
        static=method.startswith('B_')
        for seed in ((None,) if static else (17,29)):
            ci=column(real_cols,method,seed,'selected');row=next(r for r in real['pools'] if r['column']==ci);value=row['metrics']['apall'];data.append(dict(method=method,seed=seed,APall=value,column=ci))
            if value is not None:ax.scatter(j+(0 if static else -.07 if seed==17 else .07),100*value,color='#666666' if static else COLORS[method],marker='D' if static else markers[seed],s=38,label='static' if method=='B_FC' else 'seed'+str(seed) if j==0 else None)
    ax.set_xticks(range(len(display_methods)),['S_LL','V_LL','G_LL','MV_ATTN','B_FC','SURFACE','PFE']);ax.tick_params(axis='x',labelsize=8);ax.set_ylabel('Pooled sparse-proposal APall (%)');ax.set_title('Fixed CONF12 · selected learned models and static controls');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.tight_layout();figures['grouping']=save(fig,root/'figures/grouping',index);plt.close(fig);sources['grouping']=data
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),constrained_layout=True);cells=[]
    for ax,seed in zip(axes,(17,29)):
        matrix=np.asarray([[selection['arms'][m+':seed'+str(seed)]['matched1000']['A'] for m in row] for row in [('S_UU','S_UL'),('S_LU','S_LL')]])*100;cells.append(dict(seed=seed,metric='DEV A (%)',step=1000,matrix=matrix.tolist()))
        factorial_values=[selection['arms'][m+':seed'+str(s)]['matched1000']['A']*100 for m in ('S_LL','S_UL','S_LU','S_UU') for s in (17,29)]
        ax.imshow(matrix,cmap='cividis',vmin=min(factorial_values),vmax=max(factorial_values))
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white',bbox=dict(facecolor='black',alpha=.4,edgecolor='none',pad=2))
        ax.set_xticks([0,1],['outer uniform','outer learned']);ax.set_yticks([0,1],['inner uniform','inner learned']);ax.set_title('seed'+str(seed)+' · matched step1000')
    figures['factorial']=save(fig,root/'figures/factorial',index);plt.close(fig);sources['factorial']=cells
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2));curves=[]
    for method,color in COLORS.items():
        for seed in (17,29):
            values=selection['curves'][method+':seed'+str(seed)];axes[0].plot([r['step'] for r in values],[100*r['A'] for r in values],color=color,ls='-' if seed==17 else '--',lw=1,alpha=.85,label=method if seed==17 else None);curves.append(dict(method=method,seed=seed,steps=[r['step'] for r in values],A=[r['A'] for r in values]))
    prefix_data=[]
    for method in methods:
        for seed in (17,29):
            ys=[]
            for prefix in (2,4,8):
                ci=next(i for i,c in enumerate(gt['columns']) if c['method']==method and c['seed']==seed and c['profile']=='selected' and c['prefix']==prefix);ys.append(gt['metrics'][str(ci)]['full_originals']['all']['ordinary_top1'])
            axes[1].plot([2,4,8],[np.nan if y is None else 100*y for y in ys],color=COLORS[method],marker=markers[seed],ls='-' if seed==17 else '--',lw=1);prefix_data.append(dict(method=method,seed=seed,prefixes=[2,4,8],ordinary_top1=ys))
    axes[0].set(xlabel='Completed optimizer updates',ylabel='DEV A (%)',title='Original DEV · all eight methods');axes[1].set(xlabel='Readout view prefix',ylabel='GT diagnostic clean Top1 (%)',title='Fixed CONF originals · selected weights');axes[1].set_xticks([2,4,8]);fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=(0,.15,1,1));figures['budget']=save(fig,root/'figures/budget',index);plt.close(fig);sources['budget']=dict(training=curves,prefix=prefix_data,seed_styles='solid17 dashed29; no confidence interval inferred from two seeds')
    inputs=verified(root/'confirmation/real_input_lock.json');GTinputs=verified(root/'confirmation/GT/input_lock.json');sam=verified(root/'confirmation/SAM/generator_lock.json');matched=[r for r in real['proposal_matches'] if r['class_id'] is not None and r['IoU']>.5];classes=verified(root/'class_split.json')
    coverage=dict(generated=sum(r['generated'] for r in sam['frames'].values()),retained=inputs['retained_proposals'],multiview=inputs['supported'],IoU50_proposals=len(matched),IoU50_unique_GT=len({(r['scene'],r['GT_owner']) for r in matched}),IoU50_novel_unique_GT=len({(r['scene'],r['GT_owner']) for r in matched if r['class_id'] in classes['heldout_ids']}))
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2));names=list(coverage);axes[0].barh(np.arange(len(names)),list(coverage.values()),color='#0072B2');axes[0].set_yticks(np.arange(len(names)),['AMG generated','Retained masks','Multiview support','IoU > .5 proposals','IoU > .5 unique GT','Novel unique GT']);axes[0].set_xlabel('Count (different units labelled)');axes[0].invert_yaxis()
    GTcoverage=[]
    for j,(name,base) in enumerate([('base',True),('novel',False)]):
        selected=[o for o in GTinputs['objects'] if o['base']==base];n=sum(o['supported'] for o in selected);m=len(selected)-n;axes[1].bar(j,n,color='#0072B2',label='supported' if j==0 else None);axes[1].bar(j,m,bottom=n,color='#A6A6A6',hatch='//',label='abstention' if j==0 else None);GTcoverage.append(dict(partition=name,supported=n,abstention=m,all_originals=len(selected)))
    axes[1].set_xticks([0,1],['base','novel']);axes[1].set_ylabel('Fixed GT originals');axes[1].legend(frameon=False);fig.tight_layout();figures['coverage']=save(fig,root/'figures/coverage',index);plt.close(fig);sources['coverage']=dict(real=coverage,GT=GTcoverage)
    cases=case_figures(binding,index,plt);sources['cases']=cases
    source_record=immutable_record(root/'figures/source_data.json',dict(status='COMPLETE',data=sources,contract=contract))
    index.write_memo(root/'figures/verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,statistical_figures=figures,cases=cases,case_figures_n=sum(len(v['cases']) for v in cases),source_data=source_record['identity'],render_QA='PENDING_PRIMARY_VISUAL_INSPECTION',scientific_data_not_generated=True))

def case_figures(binding,index,plt):
    from .statistics import column
    root=Path(binding['output_root']);real=verified(root/'confirmation/real_score.json');inputs=verified(root/'confirmation/real_input_lock.json');pred=verified(root/'confirmation/real_prediction_lock.json');sources={o['key']:o for o in inputs['objects']};pointers={r['key']:r for r in pred['records']};classes=verified(root/'class_split.json');names=dict(zip(classes['official_ids'],classes['official_names']));cases=[]
    left=column(pred['columns'],'S_LL',17,'selected')
    for other in ('G_LL','MV_ATTN'):
        right=column(pred['columns'],other,17,'selected');buckets={k:[] for k in ('success','failure','unchanged')}
        for row in real['proposal_matches']:
            if row['class_id'] is None or row['IoU']<=.5:continue
            a=row['predictions'][left];b=row['predictions'][right];t=row['class_id'];kind='success' if a==t and b!=t else 'failure' if a!=t and b==t else 'unchanged' if a==b else None
            if kind is None:continue
            record=verified(pointers[row['key']]['path']);index.identity(record['arrays']['path'],record['arrays'])
            with np.load(record['arrays']['path'],allow_pickle=False) as arr:
                ids=arr['valid_ids'].tolist();scores=arr['full200_cosines'][[left,right],0].copy()
            target=ids.index(t);margins=[float(s[target]-np.max(np.delete(s,target))) for s in scores]
            buckets[kind].append(dict(row,left_prediction=a,right_prediction=b,left_margin=margins[0],right_margin=margins[1],margin_delta=margins[0]-margins[1],prediction_source=record['arrays']))
        for kind,rows in buckets.items():
            undefined_margins=sum(not np.isfinite(r['margin_delta']) for r in rows)
            rows=[r for r in rows if np.isfinite(r['margin_delta'])]
            median=float(np.median([r['margin_delta'] for r in rows])) if rows else None;chosen=sorted(rows,key=lambda r:(abs(r['margin_delta']-median),r['family'],r['key']))[:2];rendered=[]
            for j,row in enumerate(chosen):
                obj=sources[row['key']];geometry=verified(root/'confirmation/geometry'/obj['scene']/'manifest.json');frame=next(f for f in geometry['frames'] if f['frame_id']==obj['seed_frame'])
                index.identity(obj['support']['path'],obj['support'])
                with np.load(obj['support']['path'],allow_pickle=False) as a:mask=a['original_sam_mask'].copy()
                truth=verified(root/'confirmation/GT'/obj['scene']/'manifest.json')['GT_targets'];index.identity(truth['path'],truth)
                with np.load(truth['path'],allow_pickle=False) as a:owners=a['owners'].copy()
                index.identity(frame['arrays']['path'],frame['arrays'])
                with np.load(frame['arrays']['path'],allow_pickle=False) as a:sr=a['source_rows'].copy();valid=a['source_valid'].copy();depth=a['depth_m'].copy()
                good=valid&np.isfinite(depth)&(depth>0);shade=np.full(mask.shape,.95,np.float32);lo=hi=None
                if bool(good.any()):
                    lo,hi=map(float,np.quantile(depth[good],[.02,.98]));shade[good]=.85-.65*np.clip((depth[good]-lo)/max(hi-lo,1e-6),0,1)
                image=np.repeat(shade[...,None],3,axis=-1)
                gtmask=np.zeros(mask.shape,bool);gtmask[valid]=owners[sr[valid]]==row['GT_owner'];fig,axes=plt.subplots(1,3,figsize=(7.1,2.5))
                titles=['GT: '+names[row['class_id']],'S_LL: '+names.get(row['left_prediction'],'unknown'),other+': '+names.get(row['right_prediction'],'unknown')]
                for ax,title,overlay in zip(axes,titles,(gtmask,mask,mask)):
                    ax.imshow(image);ax.imshow(np.dstack([np.full(mask.shape,.0),np.full(mask.shape,.45),np.full(mask.shape,.7),overlay.astype(float)*.45]));ax.set_title(title,fontsize=8,wrap=True);ax.axis('off')
                fig.suptitle(obj['scene']+' · frame '+str(obj['seed_frame'])+' · '+kind+' · depth/mesh projection',fontsize=9);fig.tight_layout();stem=root/'figures/cases'/('S_vs_'+other+'_'+kind+'_'+str(j+1));files=save(fig,stem,index);plt.close(fig)
                rendered.append(dict(row,files=files,seed_frame=frame,proposal_source=obj['support'],GT_source=truth,background='REGISTERED_DEPTH_MESH_PROJECTION_NO_RGB',background_source=frame['arrays'],shared_depth_scale_metres=[lo,hi],shared_camera_and_scale=True,selection_median=median))
            cases.append(dict(pair=['S_LL',other],stratum=kind,available_candidates=len(rows),excluded_undefined_margin=undefined_margins,selection_requires_defined_margin_for_both_models=True,requested=2,cases=rendered,unavailable_slots=2-len(rendered)))
    return cases
