#!/usr/bin/env python3
"""Post-prediction object/region/extent accounting for the six frozen local cells."""
import argparse
import collections
import gzip
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from plyfile import PlyData
from src.static_ovmap.attribution_objects import released_object_outcomes,candidate_ious,candidate_event_diagnostics
from src.static_ovmap.attribution_regions import region_transitions
from scripts.evaluation.run_static_local_ownership import read,write,arrays
from scripts.evaluation.diagnose_static_t1_attribution import write_gzip


def load_gzip(path):
    with gzip.open(path,'rt') as f:return __import__('json').load(f)


def metrics(row):
    return {'candidate_AP':row['overlapping']['released']['all_ap'],'unique_AP':row['unique']['released']['all_ap'],
        'unique_AP50':row['unique']['released']['all_ap_50%'],'unique_AP25':row['unique']['released']['all_ap_25%'],
        'canonical_AP75':row['unique_high_iou_canonical_diagnostic']['ap75'],
        'mIoU':row['semantic']['semantic_miou'],'mAcc':row['semantic']['semantic_macc']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','predictions','evaluation','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();config=read(args.config);old=read(config['reference_config'])
    performance=read(args.evaluation/'reused_baselines.json')+read(args.evaluation/'performance.json');protocol=read(args.evaluation/'metric_protocol.json')
    gt=np.load(old['reference_gt_ids']);semgt=PlyData.read(old['gt_semantic_map'])['vertex']['label'];valid=protocol['semantic_ids'];instance_ids=protocol['instance_ids']
    thresholds=[next(x for x in protocol['overlaps_runtime'] if np.isclose(x,t)) for t in [.25,.5,.75]]
    effects=[];object_rows=[];transitions=[];confusions=[];evidence_rows=[];extent={};examples=[];repeat=[]
    pairs=[('protection','AT_U00','LO_U00_OVI_FILL'),('local_evidence','LO_U00_OVI_FILL','LO_U00_LOCAL'),
        ('spatial_term','LO_U00_LOCAL','LO_U00_SPATIAL'),('local_net','AT_O_AREA','LO_U00_LOCAL'),('spatial_net','AT_O_AREA','LO_U00_SPATIAL')]
    outcome_registry={}
    for run in config['runs']:
        pred=args.predictions/run;source=Path(config['predictions'])/run;ev=args.evaluation/run;ref=Path(config['reference_evaluation'])/run
        local={r['condition']:r for r in performance if r['run']==run and r['status']=='COMPLETE'}
        docs={c:read(source/(c+'.json')) if c.startswith('AT_') else read(pred/(c+'.json')) for c in local}
        roots={c:ref/c if c.startswith('AT_') else ev/c for c in local};owners={c:np.load(path/'owners.npy',mmap_mode='r') for c,path in roots.items()};semantic={c:np.load(path/'semantic.npy',mmap_mode='r') for c,path in roots.items()}
        regions=np.load(ev/'regions.npy');registry=read(source/'region_registry.json');registry['-1']={'source_support':'PROJECTION_UNMATCHED','candidate_multiplicity':None,'class_agreement':None}
        outcomes={};traces={}
        for c,row in local.items():
            traces[c]={rep:load_gzip(row[rep]['trace_path']) for rep in ['overlapping','unique']}
            outcomes[c]={rep:{str(t):released_object_outcomes(traces[c][rep],t) for t in thresholds} for rep in traces[c]}
            row['ignored_outcomes']={rep:{str(t):{'ignored_unmatched':sum(e['event']=='ignore_test' and not e['counted_fp'] and e['overlap_threshold']==t for e in trace['events']),
                'duplicate_events':sum(e['event']=='duplicate' and e['overlap_threshold']==t for e in trace['events'])} for t in thresholds} for rep,trace in traces[c].items()}
        outcome_registry[run]=outcomes
        for effect,before,after in pairs:
            if before not in local or after not in local:effects.append({'run':run,'effect':effect,'status':'BLOCKED'});continue
            a,b=metrics(local[before]),metrics(local[after]);effects.append({'run':run,'effect':effect,'before':before,'after':after,'status':'COMPLETE','delta_proportions':{k:b[k]-a[k] for k in a},'candidate_pool_same':not before=='AT_O_AREA'})
            tr=region_transitions(semgt,semantic[before],semantic[after],owners[before],owners[after],regions,valid)
            bc=read(roots[before]/'regional_confusion.json');ac=read(roots[after]/'regional_confusion.json');total=np.zeros((52,52),np.int64)
            for region,v in tr.items():
                delta=np.array(ac['regions'][region]['confusion'])-np.array(bc['regions'][region]['confusion']);total+=delta
                transitions.append({'run':run,'effect':effect,'before':before,'after':after,'region_id':int(region),**registry[region],**v,'confusion_delta_index':len(confusions)});confusions.append(delta)
            if not np.array_equal(total,np.array(ac['global_confusion'])-np.array(bc['global_confusion'])):raise ValueError('regional deltas do not reconstruct global')
            for t in thresholds:
                am=outcomes[before]['unique'][str(t)]['matched'];bm=outcomes[after]['unique'][str(t)]['matched']
                for change,ids in [('gained',set(bm)-set(am)),('lost',set(am)-set(bm))]:
                    for gid in sorted(ids,key=int):
                        object_rows.append({'run':run,'effect':effect,'before':before,'after':after,'threshold':t,'GT_id':int(gid),'transition':change,'old_match':am.get(gid),'new_match':bm.get(gid),'trace_references':[local[before]['unique']['trace_path'],local[after]['unique']['trace_path']]})
        for c in config['methods']:
            if c not in local:continue
            for t in thresholds:
                om=outcomes['AT_O_AREA']['overlapping'][str(t)]['matched'];um=outcomes['AT_U00']['overlapping'][str(t)]['matched'];new=outcomes[c]['unique'][str(t)]['matched'];ovi_unique=outcomes['AT_O_AREA']['unique'][str(t)]['matched']
                added=set(um)-set(om)
                object_rows.append({'run':run,'condition':c,'type':'U00_gain_retention','threshold':t,'U00_added_GT_ids':sorted(map(int,added)),
                    'surviving_added_GT_ids':sorted(map(int,added&set(new))),'lost_added_GT_ids':sorted(map(int,added-set(new))),
                    'previously_correct_OVI_lost_GT_ids':sorted(map(int,set(ovi_unique)-set(new))),
                    'new_unique_over_OVI_GT_ids':sorted(map(int,set(new)-set(ovi_unique))),
                    'surviving_matches':{gid:new[gid] for gid in sorted(added&set(new),key=int)}})
        if 'LO_U00_LOCAL' not in local:continue
        atoms=arrays(pred/'atoms.npz');evidence=arrays(pred/'evidence.npz');unary=arrays(pred/'unaries.npz');graph=arrays(pred/'graph.npz');diagnostic=arrays(pred/'evidence_diagnostics.npz')
        atom_count=len(atoms['counts']);aa=np.repeat(np.arange(atom_count),np.diff(atoms['indptr']));fallback=atoms['fallback'];cc=atoms['candidates']
        fb_cost=np.full(atom_count,np.nan);is_fallback=cc==fallback[aa];fb_cost[aa[is_fallback]]=unary['costs'][is_fallback]
        challenger=np.full(atom_count,np.inf);choose=unary['feasible']&~is_fallback;np.minimum.at(challenger,aa[choose],unary['costs'][choose]);margin=fb_cost-challenger
        projection=np.load(ref/'projection.npz');near=projection['nearest'];matched=projection['matched'];point_atoms=atoms['point_to_atom']
        source_regions=np.load(source/'regions.npz')['region'];run_summary=read(pred/'summary.json');local_source=np.load(docs['LO_U00_LOCAL']['owner_path']);fill_source=np.load(docs['LO_U00_OVI_FILL']['owner_path']);spatial_source=np.load(docs['LO_U00_SPATIAL']['owner_path'])
        for region in np.unique(source_regions):
            selected=source_regions[atoms['representatives']]==region;point_n=int(atoms['counts'][selected].sum());qualified=diagnostic['challenger_qualified']&selected;observed=(evidence['observed_views']>0)&selected;finite=selected&np.isfinite(margin)
            source_mask=source_regions==region
            evidence_rows.append({'run':run,'region_id':int(region),**registry[str(region)],'source_points':point_n,'atoms':int(selected.sum()),
                'evidence_observed_point_fraction':float(atoms['counts'][observed].sum()/max(point_n,1)), 'challenger_qualified_atoms':int(qualified.sum()),
                'no_challenger_fallback_point_fraction':float(atoms['counts'][selected&~diagnostic['challenger_qualified']].sum()/max(point_n,1)),
                'no_observed_evidence_point_fraction':float(atoms['counts'][selected&(evidence['observed_views']==0)].sum()/max(point_n,1)),
                'fallback_minus_best_challenger_unary_margin_quantiles':np.quantile(margin[finite],[0,.25,.5,.75,1]).tolist() if finite.any() else None,
                'LOCAL_changed_points':int((source_mask&(local_source!=fill_source)).sum()),'SPATIAL_changed_points':int((source_mask&(spatial_source!=local_source)).sum()),
                'graph_supported_edges_touching_region':int(np.count_nonzero(selected[graph['edges'][:,0]]|selected[graph['edges'][:,1]]))})
        evidence_rows.append({'run':run,'region_id':-1,'source_support':'PROJECTION_UNMATCHED','evaluation_vertices':int((~matched).sum()),'valid_GT_vertices':int(((~matched)&np.isin(semgt,valid)).sum()),'evidence_state':'no source correspondence; prediction0 FN remains'})
        # Evaluation-only structural diagnostics on one fixed radius-capped centroid graph.
        d,nb=cKDTree(atoms['xyz']).query(atoms['xyz'],k=7,distance_upper_bound=.03,workers=8)
        a=np.broadcast_to(np.arange(atom_count)[:,None],nb.shape);keep=np.isfinite(d)&(nb<atom_count)&(nb>=0)&(nb!=a)
        edges=np.unique(np.sort(np.column_stack([a[keep],nb[keep]]),axis=1),axis=0)
        adjacency=csr_matrix((np.ones(2*len(edges),np.int8),(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(atom_count,atom_count))
        incidence=csr_matrix((np.ones(len(cc),bool),(aa,cc)),shape=(atom_count,len(docs['LO_U00_LOCAL']['labels']))).tocsc()
        gtdata=np.load(ref/'candidate_gt_overlap.npz');gt_ids=gtdata['gt_ids'];gt_sizes=gtdata['gt_sizes'];original_inter=gtdata['intersections'];sizes=gtdata['projected_sizes'];point_gt=np.searchsorted(gt_ids,gt);evaluable=(gt_ids>=1000)&(gt_sizes>=100)&np.isin(gt_ids//1000,instance_ids)
        before_iou=candidate_ious(original_inter,sizes,gt_sizes);method_details={}
        for c in config['methods']:
            source_owners=np.load(docs[c]['owner_path']);atom_owners=source_owners[atoms['representatives']];predowners=owners[c]
            inter=np.bincount(predowners*len(gt_ids)+point_gt,minlength=(len(sizes)+1)*len(gt_ids)).reshape(len(sizes)+1,len(gt_ids))[1:];owned_sizes=inter.sum(axis=1);after_iou=candidate_ious(inter,owned_sizes,gt_sizes)
            details=[]
            for i in docs[c]['kept']:
                candidate_atoms=incidence.indices[incidence.indptr[i]:incidence.indptr[i+1]];owned_atoms=candidate_atoms[atom_owners[candidate_atoms]==i+1]
                components_before=connected_components(adjacency[candidate_atoms][:,candidate_atoms],directed=False,return_labels=False) if len(candidate_atoms) else 0
                components_after=connected_components(adjacency[owned_atoms][:,owned_atoms],directed=False,return_labels=False) if len(owned_atoms) else 0
                on=np.zeros(atom_count,bool);on[owned_atoms]=True;boundary=int(np.count_nonzero(on[edges[:,0]]!=on[edges[:,1]]))
                old_on=np.zeros(atom_count,bool);old_on[candidate_atoms]=True;old_boundary=int(np.count_nonzero(old_on[edges[:,0]]!=old_on[edges[:,1]]))
                stolen=np.bincount(atom_owners[candidate_atoms],weights=atoms['counts'][candidate_atoms],minlength=len(sizes)+1).astype(np.int64)
                competitors=[{'candidate_id':docs[c]['ledger'][j-1]['candidate_id'],'points':int(stolen[j])} for j in np.flatnonzero(stolen) if j>0 and j!=i+1]
                competitors.sort(key=lambda x:(-x['points'],x['candidate_id']))
                row={'candidate_id':docs[c]['ledger'][i]['candidate_id'],'canonical_index':i,'source':docs[c]['ledger'][i]['source'],'class_id':docs[c]['labels'][i],
                    'projected_size':int(sizes[i]),'owned_projected_size':int(owned_sizes[i]),'owned_fraction':float(owned_sizes[i]/sizes[i]) if sizes[i] else None,
                    'unique_empty':bool(owned_sizes[i]==0),'unique_small':bool(0<owned_sizes[i]<100),
                    'geometric_atom_graph_components_before':int(components_before),'geometric_atom_graph_components_after':int(components_after),
                    'geometric_atom_graph_boundary_edges_before':old_boundary,'geometric_atom_graph_boundary_edges_after':boundary,
                    'GT_positive_intersections_before':gt_ids[evaluable&(original_inter[i]>0)].tolist(),'GT_positive_intersections_after':gt_ids[evaluable&(inter[i]>0)].tolist(),
                    'class_correct_threshold_coverage':{str(t):{'before':gt_ids[evaluable&(gt_ids//1000==docs[c]['labels'][i])&(before_iou[i]>t)].tolist(),'after':gt_ids[evaluable&(gt_ids//1000==docs[c]['labels'][i])&(after_iou[i]>t)].tolist()} for t in thresholds},
                    'conflicting_candidates_taking_original_source_points':competitors,'structure_definition':'fixed 6-neighbor 3cm centroid graph proxy, evaluation only; not mesh topology or post-splitting; multi-GT overlap and empty duplicates are not automatically errors'}
                details.append(row)
            method_details[c]={'candidates':details,'released_event_flags':candidate_event_diagnostics(traces[c]['unique'],inter,owned_sizes,np.array(docs[c]['labels']),gt_ids,gt_sizes,instance_ids)}
        extent[run]=method_details
        # Deterministic representative evidence cards; no best-looking/GT-based policy selection.
        for effect in ['local_evidence','spatial_term']:
            for change in ['gained','lost']:
                found=next((x for x in object_rows if x.get('run')==run and x.get('effect')==effect and x.get('transition')==change),None)
                examples.append({'run':run,'effect':effect,'category':change,'selection':'first nominal threshold in .25/.5/.75 then smallest GT ID','evidence':found,'status':'PRESENT' if found else 'NOT_PRESENT'})
        run_summary['evidence_geometry_class_ambiguity']={'limitation':'identical geometric evidence cannot identify which of conflicting fixed classes is correct; no class rescue is performed'}
        write(args.output/(run+'_evidence_summary.json'),run_summary)
    if len(outcome_registry)==2:
        a,b=config['runs']
        for c in config['methods']:
            if c not in outcome_registry[a] or c not in outcome_registry[b]:continue
            for t in thresholds:
                first=outcome_registry[a][c]['unique'][str(t)]['matched'];second=outcome_registry[b][c]['unique'][str(t)]['matched']
                repeat.append({'condition':c,'threshold':t,'common_GT_ids':sorted(map(int,set(first)&set(second))),'primary_only_GT_ids':sorted(map(int,set(first)-set(second))),'repeat_only_GT_ids':sorted(map(int,set(second)-set(first))),'comparison':'GT identity, never equal model query indices'})
    write(args.output/'performance.json',performance);write(args.output/'effects.json',effects);write(args.output/'objects.json',object_rows)
    write(args.output/'regions.json',transitions);write(args.output/'evidence_regions.json',evidence_rows);np.savez_compressed(args.output/'confusion_deltas.npz',deltas=np.stack(confusions))
    write_gzip(args.output/'candidate_extent_and_events.json.gz',extent);write(args.output/'examples.json',examples);write(args.output/'repeat_sensitivity.json',repeat)
    write(args.output/'manifest.json',{'status':'COMPLETE','command':[sys.executable,*sys.argv],'seconds':time.perf_counter()-start,'region_deltas_sum_global':'EXACT','new_method_policy_changes':0})
    print(args.output,flush=True)


if __name__=='__main__':main()
