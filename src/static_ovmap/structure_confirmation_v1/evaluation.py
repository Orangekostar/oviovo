"""Sparse overlapping-mask views, independent from exclusive map payloads."""
import numpy as np

def proposal_manifest_entries(supports,labels,valid_ids,minimum=100):
    if len(supports)!=len(labels):raise ValueError('Proposal/label count differs')
    allowed=set(map(int,valid_ids));rows=[];size=None
    for index,(mask,label) in enumerate(zip(supports,labels)):
        mask=np.asarray(mask)
        if mask.ndim!=1 or mask.dtype!=np.bool_:raise ValueError('Independent boolean proposal mask required')
        if size is None:size=len(mask)
        if len(mask)!=size:raise ValueError('Proposal mesh index domains differ')
        if label!=0 and int(label) not in allowed:raise ValueError('Predicted class outside complete vocabulary')
        if int(label)>0 and int(mask.sum())>=minimum:rows.append(dict(proposal_index=index,label=int(label),rank='1.000000',area=int(mask.sum())))
    return rows

def released(binding,index):
    from pathlib import Path
    from .common import read_document
    from static_ovmap.released_loader import load_released_module
    from static_ovmap.preservation_routing.scorer_compat import adapt
    from static_ovmap.cvpr_compact.protocol import validate_overlaps
    scene=binding['cohorts']['scannet_cf18'][0];source=read_document(binding['baseline_rows']['REF_G1'][scene]['evaluation_receipt'],index,sealed=False)
    desc=source['context']['evaluator'];index.identity(desc['path'],desc)
    namespace=load_released_module(desc['path']);namespace['init']('Scannet200');adapt(namespace);validate_overlaps(namespace['overlaps'])
    if namespace['min_region_sizes'].tolist()!=[100]:raise ValueError('Released minimum differs')
    return namespace,desc

def ap_metrics(namespace,ap):
    from .common import plain
    averages=plain(namespace['compute_averages'](ap))
    return dict(apall=averages['all_ap'],ap50=averages['all_ap_50%'],ap25=averages['all_ap_25%']),averages

def gzip_record(path,value):
    import gzip,json
    from pathlib import Path
    from .common import plain
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with gzip.GzipFile(filename=str(temp),mode='wb',mtime=0) as stream:stream.write(json.dumps(plain(value),ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    temp.replace(path)

def GT_recognition(binding):
    from pathlib import Path
    from .common import verified,write,unit_key,existing,immutable_record,ConsumptionIndex
    from .metrics import partitions,score_losses
    root=Path(binding['output_root']);inputs=verified(root/'confirmation/GT/input_lock.json');predictions=verified(root/'confirmation/GT/prediction_lock.json');classes=verified(root/'class_split.json')
    identity=unit_key('GT_recognition',dict(inputs=inputs['identity'],predictions=predictions['identity']),producers=('evaluation','metrics'))
    path=root/'confirmation/GT/score.json';old=existing(path,identity)
    if old:return old
    source={r['key']:r for r in inputs['objects']};columns=predictions['columns'];records={str(i):[] for i in range(len(columns))};index=ConsumptionIndex(root/'confirmation/GT/score_verifications.json')
    for pointer in predictions['records']:
        record=verified(pointer['path']);obj=source[record['key']];index.identity(record['arrays']['path'],record['arrays'])
        with np.load(record['arrays']['path'],allow_pickle=False) as a:
            score=a['full200_cosines'].copy();defined=a['defined'].copy();pred=a['predictions'].copy();ids=a['valid_ids'].tolist()
        for ci,col in enumerate(columns):
            losses=[score_losses(score[ci,c] if defined[ci,c] else None,obj['class_id'],obj['base'],ids,classes['base_ids']) for c in range(4)]
            records[str(ci)].append(dict(key=obj['key'],scene=obj['scene'],family=obj['family'],class_id=obj['class_id'],base=obj['base'],predictions=pred[ci].tolist(),defined=defined[ci].tolist(),NLL=[v[0] for v in losses],CE=[v[1] for v in losses],supported=obj['supported']))
    metrics={ci:dict(full_originals=partitions(rows),common_supported=partitions([r for r in rows if all(r['defined'])]),per_family={f:partitions([r for r in rows if r['family']==f]) for f in sorted({r['family'] for r in rows})}) for ci,rows in records.items()}
    value=immutable_record(path,dict(status='COMPLETE',input_identity=identity,columns=columns,records=records,metrics=metrics,full_original_denominator=True,competition_classes=200,heldout_CE160=None,scope='GT_DERIVED_DIAGNOSTIC'))
    index.write_memo(root/'confirmation/GT/score_verifications.json');return value

def real_scene(job):
    import os
    from pathlib import Path
    from .common import (verified,write,ConsumptionIndex,unit_key,existing,immutable_record,canonical_digest,read_document,plain)
    from static_ovmap.released_trace import trace_released_matches
    binding,scene=job;root=Path(binding['output_root']);predictions=verified(root/'confirmation/real_prediction_lock.json');inputs=verified(root/'confirmation/real_input_lock.json');gt=verified(root/'confirmation/GT'/scene/'manifest.json')
    identity=unit_key('real_scene_score',dict(predictions=predictions['identity'],inputs=inputs['identity'],GT=gt['identity']),producers=('evaluation',),scene=scene)
    folder=root/'confirmation/evaluation'/scene;path=folder/'receipt.json';old=existing(path,identity)
    if old:return old
    index=ConsumptionIndex(folder/'verifications.json');namespace,descriptor=released(binding,index);objs=[o for o in inputs['objects'] if o['scene']==scene];bykey={r['key']:verified(r['path']) for r in predictions['records'] if r['key'].split(':')[0]==scene}
    masks=[];mask_paths=[];GTdesc=gt['GT_targets'];index.identity(GTdesc['path'],GTdesc)
    with np.load(GTdesc['path'],allow_pickle=False) as a:official=a['official_instance_ids'].copy();owners=a['owners'].copy()
    gt_path=folder/'GT.npy';folder.mkdir(parents=True,exist_ok=True)
    if gt_path.exists():
        if not np.array_equal(np.load(gt_path,allow_pickle=False),official):raise ValueError('Full GT changed')
    else:np.save(gt_path,official)
    for oi,obj in enumerate(objs):
        index.identity(obj['support']['path'],obj['support'])
        with np.load(obj['support']['path'],allow_pickle=False) as a:rows=a['source_rows'].copy()
        mask=np.zeros(len(official),bool);mask[rows]=True;target=folder/'masks'/f'{oi:04d}.npy';target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            if not np.array_equal(np.load(target,allow_pickle=False),mask):raise ValueError('Raw proposal support changed')
        else:np.save(target,mask)
        masks.append(mask);mask_paths.append(target)
    all_predictions=[]
    for obj in objs:
        desc=bykey[obj['key']]['arrays'];index.identity(desc['path'],desc)
        with np.load(desc['path'],allow_pickle=False) as a:all_predictions.append(a['predictions'][:,0].copy())
    predictions_matrix=np.asarray(all_predictions,np.int64).reshape(len(objs),len(predictions['columns']))
    scoring_columns=[(i,c) for i,c in enumerate(predictions['columns']) if c['prefix']==8]
    scored={};results=[]
    for ci,col in scoring_columns:
        labels=predictions_matrix[:,ci].tolist();entries=proposal_manifest_entries(masks,labels,namespace['VALID_CLASS_IDS'])
        sig=canonical_digest(dict(labels=labels,geometry=inputs['identity'],GT=gt['identity'],evaluator=descriptor))
        if sig in scored:
            results.append(dict(scored[sig],column=ci,column_spec=col,exact_scorer_alias=scored[sig]['column']));continue
        dest=folder/'scores'/str(ci);dest.mkdir(parents=True,exist_ok=True);manifest=dest/'pred_inst_sem_mapping.txt'
        manifest.write_text(''.join(os.path.relpath(mask_paths[r['proposal_index']],dest)+' '+str(r['label'])+' 1.000000\n' for r in entries))
        gt2pred,pred2gt=namespace['assign_instances_for_scan'](str(folder),str(manifest),str(gt_path));matches={scene:dict(gt=gt2pred,pred=pred2gt)}
        ap=namespace['evaluate_matches'](matches);metrics,averages=ap_metrics(namespace,ap)
        match_path=dest/'matches.json.gz';gzip_record(match_path,matches)
        result=write(dest/'receipt.json',dict(status='COMPLETE',column=ci,column_spec=col,scene=scene,input_identity=sig,metrics=metrics,per_class=averages['classes'],matches=index.identity(match_path),manifest=index.identity(manifest),GT=index.identity(gt_path),evaluator=descriptor,runtime_overlaps=namespace['overlaps'].tolist(),retained_proposals=len(objs),scored_proposals=len(entries),unknown_or_small_proposals=len(objs)-len(entries),constant_ranks=True,independent_overlapping_masks=True,projection='IDENTITY_ORIGINAL_MESH_VERTEX_ROWS',GT_original_instances=sum(len(v) for v in gt2pred.values()),valid_GT_instance_classes=int(np.isfinite(ap[0,:,0]).sum())))
        results.append(result);scored[sig]=result
    registry={int(k):v for k,v in gt['GT_registry'].items()};counts={o:int((owners==o).sum()) for o in registry};matches=[]
    for oi,obj in enumerate(objs):
        seen,n=np.unique(owners[masks[oi]],return_counts=True);ious=[(int(o),int(count)/(int(masks[oi].sum())+counts[int(o)]-int(count))) for o,count in zip(seen,n) if int(o) in registry and counts[int(o)]>=100]
        best=sorted(ious,key=lambda x:(-x[1],x[0]))[0] if ious else (None,0.)
        target=registry[best[0]]['class_id'] if best[0] is not None and best[1]>.25 else None
        matches.append(dict(key=obj['key'],scene=scene,family=obj['family'],GT_owner=best[0] if target is not None else None,IoU=best[1],class_id=target,predictions=predictions_matrix[oi].tolist(),supported=obj['supported'],retained_support_vertices=int(masks[oi].sum())))
    index.write_memo(folder/'verifications.json')
    print('CONF_REAL_SCORED',scene,len(results),flush=True)
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,scene=scene,rows=results,proposal_matches=matches,all_retained_proposals=len(objs),scoring_prefix=8,semantic_map_mIoU=None,full_GT_preserved=True))

def real_score(binding):
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path
    import warnings
    from .common import verified,write,unit_key,existing,ConsumptionIndex,immutable_record,read_document,arrays_record
    from static_ovmap.released_trace import trace_released_matches
    from static_ovmap.contribution_attribution_v1.stability import namespace_matches
    root=Path(binding['output_root']);prediction_lock=verified(root/'confirmation/real_prediction_lock.json');scenes=verified(root/'role_lock.json')['roles']['CONF'];verified(root/'confirmation/GT/input_lock.json')
    identity=unit_key('real_score',prediction_lock['identity'],producers=('evaluation',),scenes=scenes);path=root/'confirmation/real_score.json';old=existing(path,identity)
    if old:return old
    with ProcessPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(real_scene,[(binding,s) for s in scenes]))
    index=ConsumptionIndex(root/'confirmation/pool_verifications.json');namespace,descriptor=released(binding,index);pools=[];scored={}
    for ci,col in enumerate(prediction_lock['columns']):
        if col['prefix']!=8:continue
        rows=[next(r for r in receipt['rows'] if r['column']==ci) for receipt in receipts];source=tuple(r['matches']['sha256'] for r in rows)
        if source in scored:pools.append(dict(scored[source],column=ci,column_spec=col,exact_scorer_alias=scored[source]['column']));continue
        matches={}
        for si,row in enumerate(rows):matches.update(namespace_matches(read_document(row['matches']['path'],index,sealed=False,expected=row['matches']),str(si)))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning);ap,trace=trace_released_matches(namespace,matches);metrics,averages=ap_metrics(namespace,ap)
        if not trace['parity']['ap_exact'] or not trace['parity']['pr_and_fn_exact']:raise ValueError('Actual released pool trace parity failed')
        folder=root/'confirmation/pools'/str(ci);gzip_record(folder/'matches.json.gz',matches);gzip_record(folder/'trace.json.gz',trace)
        array=arrays_record(folder/'AP.npz',dict(AP=ap,valid_ids=np.asarray(namespace['VALID_CLASS_IDS']),overlaps=np.asarray(namespace['overlaps'])),index)
        result=write(folder/'receipt.json',dict(status='COMPLETE',column=ci,column_spec=col,metrics=metrics,per_class=averages['classes'],arrays=array,trace=index.identity(folder/'trace.json.gz'),matches=index.identity(folder/'matches.json.gz'),trace_parity=trace['parity'],ordered_scenes=scenes,ordered_inputs=[r['input_identity'] for r in rows],valid_GT_instance_classes=int(np.isfinite(ap[0,:,0]).sum()),aggregation='ORIGINAL_RELEASED_EVALUATE_MATCHES_ALL_ORDERED_SCENES',evaluator=descriptor,semantic_mIoU=None))
        scored[source]=result;pools.append(result)
    if len(pools)!=35 or sum(len(r['rows']) for r in receipts)!=420:raise ValueError('Required35 AP pools/420 scene rows incomplete')
    proposal_matches=[r for receipt in receipts for r in receipt['proposal_matches']];classes=verified(root/'class_split.json')
    coverage=[]
    for ci,col in enumerate(prediction_lock['columns']):
        rows=[]
        for threshold in (.25,.5):
            selected=[r for r in proposal_matches if r['class_id'] is not None and r['IoU']>threshold];per_gt={}
            for r in selected:per_gt.setdefault((r['scene'],r['GT_owner']),[]).append(int(r['predictions'][ci]==r['class_id']))
            one_to_one={}
            for r in sorted(selected,key=lambda r:(-r['IoU'],r['key'])):one_to_one.setdefault((r['scene'],r['GT_owner']),r)
            rows.append(dict(IoU_strictly_greater=threshold,matched_proposals=len(selected),unique_GT=len(per_gt),families=len({r['family'] for r in selected}),proposal_top1=float(np.mean([r['predictions'][ci]==r['class_id'] for r in selected])) if selected else None,unique_GT_balanced_top1=float(np.mean([np.mean(v) for v in per_gt.values()])) if per_gt else None,one_to_one_top1=float(np.mean([r['predictions'][ci]==r['class_id'] for r in one_to_one.values()])) if one_to_one else None,heldout_unique_GT=len({(r['scene'],r['GT_owner']) for r in selected if r['class_id'] in classes['heldout_ids']}),heldout_families=len({r['family'] for r in selected if r['class_id'] in classes['heldout_ids']})))
        coverage.append(dict(column=ci,column_spec=col,rows=rows))
    index.write_memo(root/'confirmation/pool_verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,scene_receipts={r['scene']:r['identity'] for r in receipts},scene_rows=[r for receipt in receipts for r in receipt['rows']],pools=pools,proposal_matches=proposal_matches,recognition_coverage=coverage,scene_rows_n=420,pools_n=35,all_original_GT_retained=True,retained_proposals=len(proposal_matches),CPU_workers=3,scope='CONF12_SPARSE_PROPOSAL_AP_ON_PROVIDED_RECONSTRUCTION',semantic_mIoU=None))

def run(binding,*,phase='confirm_score'):
    from pathlib import Path
    from .common import write
    real=real_score(binding);gt=GT_recognition(binding)
    return write(Path(binding['output_root'])/'confirmation/score.json',dict(status='COMPLETE',real=real['identity'],GT=gt['identity']))
