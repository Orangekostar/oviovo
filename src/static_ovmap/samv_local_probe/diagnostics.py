"""Post-lock full-map matching and separately scoped raw/support/class evidence."""

from collections import Counter
from pathlib import Path
import time

import numpy as np

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.evidence_exploration.analysis import trace_entries,zipped
from static_ovmap.evidence_exploration.diagnostic_details import compare_entries
from static_ovmap.minimal_instance_repair.diagnostics import maximum_matches,support_overlap
from static_ovmap.minimal_instance_repair.output_diagnostics import describe_partition
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.released_loader import load_released_module

from .binding import load_scene
from .common import METHODS,_array_digest,canonical_digest,read,verified,write


def physical_weights(xyz,faces):
    weights=np.zeros(len(xyz),np.float64)
    for begin in range(0,len(faces),262144):
        tri=faces[begin:begin+262144];p=xyz[tri].astype(np.float64)
        area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)/6
        for j in range(3):np.add.at(weights,tri[:,j],area)
    return weights


def mesh_components(owners,faces,incidence):
    """Actual indexed topology only; disjoint face vertices are never welded."""
    ids,n=np.unique(owners[owners>0],return_counts=True)
    counts=Counter(dict(zip(map(int,ids),map(int,n))));shared=int((incidence>1).sum())
    if shared==0:
        values=np.sort(owners[faces],axis=1)
        repeated=np.concatenate([values[:,i][(values[:,i]>0)&(values[:,i]==values[:,i+1])] for i in (0,1)])
        ids,n=np.unique(repeated,return_counts=True)
        counts.subtract(dict(zip(map(int,ids),map(int,n))))
        return {str(o):int(n) for o,n in counts.items()},'DISJOINT_FACE_ROWS_NO_WELDING'
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    a=np.concatenate([faces[:,0],faces[:,1],faces[:,2]]);b=np.concatenate([faces[:,1],faces[:,2],faces[:,0]])
    keep=(owners[a]>0)&(owners[a]==owners[b]);a,b=a[keep],b[keep]
    graph=coo_matrix((np.ones(len(a),np.uint8),(a,b)),shape=(len(owners),len(owners))).tocsr()
    _,labels=connected_components(graph,directed=False,return_labels=True)
    pairs=np.unique(np.stack((owners[owners>0],labels[owners>0]),axis=1),axis=0)
    ids,n=np.unique(pairs[:,0],return_counts=True)
    return {str(int(o)):int(c) for o,c in zip(ids,n)},'ORIGINAL_INDEXED_FACE_EDGES'


def eligible_ground_truth(binding,inputs):
    row=binding['baseline_rows']['SU01_G1'][inputs.scene];score=read(row['evaluation_receipt'])
    inputs.index.identity(score['gt_path'],score['context']['gt'])
    path=Path(score['manifest']).with_name('matches.json.gz');inputs.index.identity(path)
    matches=next(iter(zipped(path).values()));namespace=load_released_module(score['context']['evaluator']['path'])
    namespace['init']('Replica' if inputs.data['dataset']=='Replica' else 'Scannet200')
    minimum=int(namespace['min_region_sizes'][0]);distance=namespace['dist_threshes'][0];confidence=namespace['dist_confs'][0]
    eligible={int(g['instance_id']):g for values in matches['gt'].values() for g in values
        if g['instance_id']>=1000 and g['vert_count']>=minimum and g['med_dist']<=distance and g['dist_conf']>=confidence}
    return np.load(score['gt_path'],allow_pickle=False).reshape(-1),eligible,minimum


def diagnose_scene(binding,scene,store):
    root=Path(binding['output_root']);inputs=load_scene(binding,scene)
    lock=verified(root/'predictions'/scene/'receipt.json');plan=verified(root/'query_plan'/scene/'receipt.json')
    evaluation=verified(root/'evaluation'/scene/'receipt.json');dest=root/'diagnostics'/scene/'receipt.json'
    key=canonical_digest({'lock':lock['identity'],'evaluation':evaluation['identity'],'plan':plan['identity'],
        'operator':inputs.index.identity(__file__)})
    if dest.exists():
        prior=verified(dest)
        if prior['input_identity']!=key:raise ValueError('diagnostic inputs changed')
        return prior
    if verified(root/'predictions/summary.json')['status']!='ALL_PREDICTIONS_LOCKED':raise ValueError('diagnostics require global prediction lock')
    begin=time.perf_counter();gt,eligible,minimum=eligible_ground_truth(binding,inputs)
    weights=physical_weights(inputs.xyz,inputs.faces);incidence=np.bincount(inputs.faces.reshape(-1),minlength=len(inputs.xyz))
    partitions={};methods={};traces={};payloads={}
    for method in (*METHODS,'REF_D2'):
        payload=load_prediction(lock['methods'][method]['manifest']);payloads[method]=payload
        digest=_array_digest(payload.owner_ids)
        if digest not in partitions:
            supports=describe_partition(payload,inputs,gt,eligible,minimum)
            components,topology=mesh_components(payload.owner_ids,inputs.faces,incidence)
            partitions[digest]={'supports':supports,'matching':{str(t):maximum_matches(supports,eligible,t) for t in (.5,.75)},
                'mesh_components':components,'topology_convention':topology,'rows_shared_by_multiple_faces':int((incidence>1).sum())}
        methods[method]=partitions[digest]
        row=next(r for r in evaluation['rows'] if r['method']==method);score=read(row['evaluation_receipt'])
        trace_path=Path(score['manifest']).with_name('trace.json.gz');inputs.index.identity(trace_path)
        if str(trace_path) not in traces:traces[str(trace_path)]=trace_entries(zipped(trace_path))
        methods[method]={**methods[method],'trace_entries':traces[str(trace_path)],'scoring_identity':score['identity']}
    old_by={r['owner']:r for r in methods['SV00_G1']['supports']};target_rows=[]
    for query in plan['queries']:
        owner=query['owner'];old=old_by.get(owner,{'overlaps':[],'best':None,'evaluation_points':0})
        # Freeze one reference from maximum old intersection, not per-view IoU.
        fixed=min(old['overlaps'],key=lambda r:(-r['intersection'],r['gt_id'])) if old['overlaps'] else None
        target={'owner':owner,'pool':query['pool'],'fixed_reference_GT_id':fixed['gt_id'] if fixed else None,
            'fixed_reference_reason':'OLD_MAXIMUM_OVERLAP_TIE_MIN_GT' if fixed else 'NO_VALID_OLD_ASSOCIATION',
            'old_best_iou':old['best']['iou'] if old['best'] else 0.,'old_support_rows':int((inputs.g1.owner_ids==owner).sum()),
            'old_physical_area_m2':float(weights[inputs.g1.owner_ids==owner].sum()),'models':{}}
        for model,method in [('SAM2','SV01_SAM2_GEOM'),('SAMV','SV02_SAMV_GEOM')]:
            lift=verified(root/'lifted'/scene/model/'receipt.json');evidence=lift['targets'][str(owner)]
            current=next((r for r in methods[method]['supports'] if r['owner']==owner),{'overlaps':[],'best':None})
            fixed_match=next((r for r in current['overlaps'] if fixed and r['gt_id']==fixed['gt_id']),None)
            areas={};support_diagnostics={}
            with np.load(evidence['evidence_arrays']['path'],allow_pickle=False) as arrays:
                for name in ['raw_foreground_rows','raw_positive_proposal_rows','admitted_positive_rows','suppressed_positive_rows','raw_negative_proposal_rows']:
                    areas[name.removesuffix('_rows')+'_area_m2']=float(weights[arrays[name]].sum())
                for name in ('raw_positive_proposal_rows','admitted_positive_rows'):
                    support_diagnostics[name]=support_overlap(arrays[name],inputs.nearest,inputs.matched,gt,eligible,minimum=minimum)
                sparse_rows=arrays['source_rows'];n,k=arrays['n'],arrays['k']
                areas['foreground_n_below_2_area_m2']=float(weights[sparse_rows[(k>0)&(n<2)]].sum())
            final=payloads[method].owner_ids==owner;before=inputs.g1.owner_ids==owner
            raw_count=evidence['raw_qualified_positive_rows']
            target['models'][model]={**evidence,**areas,'support_diagnostics':support_diagnostics,'raw_qualified_support_is_final_mask':False,
                'final_physical_area_m2':float(weights[final].sum()),'added_area_m2':float(weights[final&~before].sum()),
                'lost_area_m2':float(weights[before&~final].sum()),
                'best_iou':current['best']['iou'] if current['best'] else 0.,
                'fixed_reference_iou':(fixed_match['iou'] if fixed_match else 0.) if fixed else None,
                'best_iou_delta':(current['best']['iou'] if current['best'] else 0.)-target['old_best_iou'],
                'overmerge_indicator':len(current['overlaps'])>len(old['overlaps']) and (current['best']['iou'] if current['best'] else 0.)<target['old_best_iou'],
                'old_indexed_components':methods['SV00_G1']['mesh_components'].get(str(owner),0),
                'new_indexed_components':methods[method]['mesh_components'].get(str(owner),0),
                'raw_positive_rows':raw_count}
        target_rows.append(target)
    decisions=verified(root/'readout'/scene/'decisions.json');semantic=[]
    for target in target_rows:
        owner=target['owner'];row=decisions['targets'][str(owner)];fixed=target['fixed_reference_GT_id']
        truth=int(eligible[fixed]['label_id']) if fixed else None
        old_correct=row['OLD_class']==truth if truth is not None else None;new_correct=row['NEW_class']==truth if truth is not None else None
        semantic.append({'owner':owner,'original_support_fixed_GT':fixed,'GT_class':truth,
            'OLD_correct':old_correct,'NEW_correct':new_correct,'wrong_to_right':old_correct is False and new_correct is True,
            'right_to_wrong':old_correct is True and new_correct is False,**row})
    pairs=[('SV01_SAM2_GEOM','SV00_G1'),('SV02_SAMV_GEOM','SV00_G1'),('SV02_SAMV_GEOM','SV01_SAM2_GEOM'),
           ('SV04_SAMV_FC','SV03_OLDMASK_FC'),('SV05_COMBINED','SV02_SAMV_GEOM'),('SV05_COMBINED','SV04_SAMV_FC')]
    comparisons=[]
    for candidate,reference in pairs:
        a,b=methods[candidate],methods[reference];thresholds=[]
        for t in (.5,.75):
            ga,gb=set(a['matching'][str(t)]['gt_ids']),set(b['matching'][str(t)]['gt_ids'])
            ea,eb=([r for r in methods[m]['trace_entries'] if abs(r['overlap_threshold']-t)<1e-12] for m in (candidate,reference))
            thresholds.append({'threshold':t,'class_agnostic':{'gained_GT_ids':sorted(ga-gb),'lost_GT_ids':sorted(gb-ga),
                'net_unique_matches':len(ga)-len(gb)},'released':{k.removesuffix('50'):v for k,v in compare_entries(ea,eb).items()}})
        ranks_a,ranks_b=dict(payloads[candidate].instance_ranks),dict(payloads[reference].instance_ranks)
        comparisons.append({'candidate':candidate,'reference':reference,'thresholds':thresholds,
            'rank_changes':[{'owner':o,'before':ranks_b.get(o),'after':ranks_a.get(o)} for o in sorted(set(ranks_a)|set(ranks_b)) if ranks_a.get(o)!=ranks_b.get(o)]})
    donor_rows={}
    for model,method in [('SAM2','SV01_SAM2_GEOM'),('SAMV','SV02_SAMV_GEOM')]:
        lift=verified(root/'lifted'/scene/model/'receipt.json');by={r['owner']:r for r in methods[method]['supports']};donors=[]
        for owner,value in lift['donors'].items():
            o=int(owner);old=old_by.get(o,{}).get('best');new=by.get(o,{}).get('best')
            donors.append({'owner':o,**value,'old_best_iou':old['iou'] if old else 0.,'new_best_iou':new['iou'] if new else 0.,
                'donor_harmed_best_iou':(new['iou'] if new else 0.)<(old['iou'] if old else 0.),'vanished':value['remaining_rows']==0})
        donor_rows[model]={'audit':lift['audit'],'protected_core_preserved':lift['protected_core_preserved'],'donors':donors}
    result=write(dest,{'status':'DIAGNOSTICS_COMPLETE','scene':scene,'cohort':binding['scenes'][scene]['cohort'],
        'input_identity':key,'targets':target_rows,'semantic':semantic,'comparisons':comparisons,'donor_ledger':donor_rows,
        'partitions':{m:{k:v for k,v in value.items() if k!='trace_entries'} for m,value in methods.items()},
        'released_score_entries':{m:methods[m]['trace_entries'] for m in methods},
        'eligible_GT_ids':sorted(eligible),'GT_read_only_after_all_four_predictions_locked':True,
        'two_D_GT_metrics':None,'two_D_GT_reason':'NO_NEW_ALIGNED_ANNOTATIONS; PREINSERTION_PANOPTIC_IS_PREDICTION',
        'indexed_topology_not_welded':True,'elapsed_seconds':time.perf_counter()-begin})
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json');print('DIAGNOSED',scene,len(target_rows),'targets',flush=True);return result


def analyze(binding):
    root=Path(binding['output_root']);store=verified(root/'result_store.json')
    scenes={s:diagnose_scene(binding,s,store) for s in binding['scenes']};cohorts={}
    for cohort,names in binding['cohorts'].items():
        summary={};all_targets=[t for s in names for t in scenes[s]['targets']]
        for model,method in [('SAM2','SV01_SAM2_GEOM'),('SAMV','SV02_SAMV_GEOM')]:
            values=[t['models'][model] for t in all_targets]
            summary[model]={'targets':len(values),'anchor_success':sum(r['active'] for r in values),
                'raw_positive_rows':sum(r['raw_qualified_positive_rows'] for r in values),
                'admitted_positive_rows':sum(r['admitted_positive_rows'] for r in values),
                'suppressed_positive_rows':sum(r['suppressed_positive_rows'] for r in values),
                'mean_best_iou':float(np.mean([r['best_iou'] for r in values])) if values else None,
                'mean_fixed_reference_iou':float(np.mean([r['fixed_reference_iou'] for r in values if r['fixed_reference_iou'] is not None])) if any(r['fixed_reference_iou'] is not None for r in values) else None,
                'class_agnostic_counts':{str(t):sum(scenes[s]['partitions'][method]['matching'][str(t)]['count'] for s in names) for t in (.5,.75)}}
        sem=[r for s in names for r in scenes[s]['semantic']]
        summary['semantic']={'joint_eligible':sum(r['common_update_eligible'] for r in sem),'wrong_to_right':sum(r['wrong_to_right'] for r in sem),
            'right_to_wrong':sum(r['right_to_wrong'] for r in sem),'undefined_fixed_reference':sum(r['GT_class'] is None for r in sem)}
        cohorts[cohort]=summary
    signals={}
    signals['SEGMENTOR_SUPPORT_SIGNAL']=all(any(cohorts[c]['SAMV']['class_agnostic_counts'][str(t)]>cohorts[c]['SAM2']['class_agnostic_counts'][str(t)] for t in (.5,.75))
        and cohorts[c]['SAMV']['mean_best_iou'] is not None and cohorts[c]['SAMV']['mean_best_iou']>=cohorts[c]['SAM2']['mean_best_iou'] for c in cohorts)
    signals['SEMANTIC_MASK_SIGNAL']=all(all(store['pooled_metrics'][c]['SV04_SAMV_FC']['metrics'][k]>=store['pooled_metrics'][c]['SV03_OLDMASK_FC']['metrics'][k]-1e-10 for k in ('apall','ap50','ap25','miou','macc'))
        and any(store['pooled_metrics'][c]['SV04_SAMV_FC']['metrics'][k]>store['pooled_metrics'][c]['SV03_OLDMASK_FC']['metrics'][k]+1e-10 for k in ('apall','ap50','ap25','miou','macc')) for c in cohorts)
    signals['COMBINATION_EQUALS_COMPONENT']=any(all(verified(root/'predictions'/s/'receipt.json')['methods']['SV05_COMBINED']['prediction_key']==verified(root/'predictions'/s/'receipt.json')['methods'][m]['prediction_key'] for s in binding['scenes']) for m in ('SV02_SAMV_GEOM','SV04_SAMV_FC'))
    return write(root/'diagnostics/summary.json',{'status':'DIAGNOSTICS_COMPLETE','scene_receipts':{s:r['identity'] for s,r in scenes.items()},
        'cohorts':cohorts,'mechanism_flags':signals,'GT_ids_qualified_by_scene':True,'class_agnostic_counts_are_not_AP':True})
