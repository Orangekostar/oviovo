#!/usr/bin/env python3
"""Evaluate frozen attribution predictions; GT and exact matcher traces stay here."""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.released_loader import load_released_module
from src.static_ovmap.released_trace import trace_released_matches, _plain
from src.static_ovmap.attribution_regions import regional_confusions
from src.static_ovmap.fusion_attribution import resolve_native_owners
from src.evaluation.static_projected_instances import projected_instance_metrics
from scripts.evaluation.evaluate_static_ovmap_instances import load_exports
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics


def write(path,data):
    path.write_text(json.dumps(_plain(data),indent=2,allow_nan=False)+'\n')


def write_gzip(path,data):
    with gzip.open(path,'wt',encoding='utf-8') as f:
        json.dump(_plain(data),f,allow_nan=False)


def check(path,expected=None):
    p=Path(path);actual=sha256_file(p)
    if expected is not None and actual!=expected:
        raise ValueError('bound evaluation input changed: '+str(p))
    return {'sha256':actual,'bytes':p.stat().st_size}


def bind_protocol(config,output):
    historical=json.loads(Path(config['historical_evaluation']).read_text())
    inventory={}
    for name in ('gt_instance_map','gt_semantic_map','reference_gt_ids','text_cache'):
        path=config[name];inventory[path]=check(path,historical['input_sha256'][path])
    original=json.loads(Path(config['historical_native_instances_receipt']).read_text())
    source=Path(config['evaluator_root'])/'scripts'
    for name in ('eval_utils.py','eval_sem_seg.py','eval_inst_seg.py','utils/semantic_const.py'):
        inventory[str(source/name)]=check(source/name,original['source_sha256'][name])
    evaluator=load_released_module(source/'eval_utils.py');evaluator['init']('Replica')
    constants=load_released_module(source/'utils/semantic_const.py')
    valid=np.load(config['text_cache'],allow_pickle=False)['valid_ids']
    if valid.tolist()!=list(range(1,len(constants['REPLICA_51'])+1)):
        raise ValueError('text IDs differ from actual Replica semantic vocabulary')
    overlaps=evaluator['overlaps']
    protocol={'overlaps_runtime':overlaps.tolist(), 'overlaps_float_hex':[float(x).hex() for x in overlaps],
        'released_match_operator':'>', 'all_ap_overlap_indices':np.flatnonzero(~np.isclose(overlaps,.25)).tolist(),
        'AP_integration':'np.dot(precision, np.convolve(padded_recall, [-0.5,0,0.5], valid))',
        'confidence_ties':'np.argsort ascending (default NumPy kind), np.unique score thresholds; no epsilon',
        'confidence_export':'.6f; assignment retains full precision; parsed AP values stored per candidate',
        'duplicate_handling':'GT iteration order; first match marks prediction visited; later duplicates append min-score FP and update max-score TP without setting visited; observe unchanged source',
        'min_region_sizes':evaluator['min_region_sizes'].tolist(),
        'distance_thresholds':evaluator['dist_threshes'].tolist(),'distance_confidences':evaluator['dist_confs'].tolist(),
        'infinity_policy':'JSON null in runtime distance arrays means signed infinity; distance=+inf confidence=-inf',
        'GT_filter':'instance_id >=1000; vert_count >=100; med_dist <=distance_thresh; dist_conf >=distance_conf',
        'prediction_filter':'valid instance class; projected mask size >=100 before evaluate_matches',
        'void_group_rule':'invalid GT class is void; group IDs<1000/small/distance-ineligible GT intersections contribute ignore; unmatched prediction is FP iff ignore_fraction <= overlap threshold',
        'absent_class_AP':'NaN -> null with no-GT reason; all-AP uses np.nanmean; GT-present no-pred AP=0',
        'semantic_ids':valid.tolist(),'semantic_labels':constants['REPLICA_51'],
        'instance_ids':evaluator['VALID_CLASS_IDS'],'instance_labels':evaluator['CLASS_LABELS'],
        'semantic_rule':'ignore GT-invalid rows only; prediction0 keeps valid-GT FN; mean over GT-supported classes',
        'native_to_common':'world metres; native mesh vertices to returned T0 coordinate rows; 1NN squared_distance < 0.05**2',
        'common_to_evaluation':'returned T0 coordinate rows to full GT evaluation vertices; 1NN squared_distance < 0.05**2',
        'canonical_diagnostic':'full raw GT instance domain; min_region100; greedy confidence-ranked one-to-one IoU>=.75; precision-envelope AP; not released AP',
        'source_inventory':inventory,'numpy_version':np.__version__}
    write(output/'metric_protocol.json',protocol)
    return evaluator,valid,inventory


def evaluate_set(evaluator, destination, mask_root, mask_paths, labels, serialized, kept, gt_path, cache, cache_key):
    destination.mkdir(parents=True,exist_ok=True)
    manifest=destination/'pred_inst_sem_mapping.txt'
    import os
    lines=[f'{os.path.relpath(mask_paths[i],destination)} {int(labels[i])} {serialized[i]}' for i in kept]
    manifest.write_text('\n'.join(lines)+'\n')
    if cache_key in cache:
        return {**cache[cache_key], 'manifest':str(manifest),'manifest_sha256':sha256_file(manifest),'cache_hit':True}
    start=time.perf_counter()
    gt2pred,pred2gt=evaluator['assign_instances_for_scan'](str(mask_root),str(manifest),str(gt_path))
    matches={str(Path(gt_path).resolve()):{'gt':gt2pred,'pred':pred2gt}}
    ap,trace=trace_released_matches(evaluator,matches)
    averages=evaluator['compute_averages'](ap)
    trace_path=destination/'released_trace.json.gz';write_gzip(trace_path,trace)
    # Preserve exact evaluator relationships and candidate filenames for downstream object attribution.
    match_path=destination/'released_matches.json.gz';write_gzip(match_path,matches)
    result={'released':_plain(averages),'trace_path':str(trace_path),'matches_path':str(match_path),
        'trace_parity':trace['parity'],'manifest':str(manifest),'manifest_sha256':sha256_file(manifest),
        'evaluator_prediction_count':sum(len(v) for v in pred2gt.values()),
        'evaluation_seconds':time.perf_counter()-start,'cache_hit':False}
    cache[cache_key]=result
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--conditions',nargs='+')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();config=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=args.resume)
    overall=time.perf_counter(); evaluator,valid,inventory=bind_protocol(config,args.output)
    prediction_manifest=json.loads((args.predictions/'prediction_manifest.json').read_text())
    if prediction_manifest['status']!='COMPLETE_PREDICTION' or prediction_manifest['GT_input']:
        raise ValueError('completed GT-free predictions required')
    meshes=[PlyData.read(config[k])['vertex'].data for k in ('gt_instance_map','gt_semantic_map')]
    inst,sem=meshes
    if len(inst)!=len(sem) or any(not np.array_equal(inst[k],sem[k]) for k in ('x','y','z')):
        raise ValueError('evaluation coordinate/order mismatch')
    exports=load_exports(Path(config['evaluator_root'])/'scripts/eval_sem_seg.py')
    gt_path=exports['map_gt_mesh']({'inst_mesh_f':config['gt_instance_map'],'sem_mesh_f':config['gt_semantic_map'],'res_folder':str(args.output)})
    if not np.array_equal(np.load(gt_path),np.load(config['reference_gt_ids'])):
        raise ValueError('original GT conversion parity failed')
    gt_ids=np.load(gt_path);gt_values,gt_counts=np.unique(gt_ids,return_counts=True)
    large=json.loads((ROOT/'artifacts/static_ovmap/room0_spaceformer/large_artifacts.json').read_text())
    expected={x['path']:x['sha256'] for x in large}
    inventory[config['evaluation_projection']]=check(config['evaluation_projection'],expected[config['evaluation_projection']])
    historical=json.loads(Path(config['historical_evaluation']).read_text())
    with np.load(config['historical_t1'],allow_pickle=False) as data:reference_coord=data['coord']
    conditions=args.conditions or config['prediction_conditions'];rows=[]
    if not set(conditions)<=set(config['prediction_conditions']):raise ValueError('unknown frozen condition')
    for run in config['runs']:
        pred=args.predictions/run['id'];out=args.output/run['id'];out.mkdir(exist_ok=args.resume)
        run_start=time.perf_counter();coord=np.load(pred/'coord.npy',mmap_mode='r');bank=np.load(pred/'masks.npy',mmap_mode='r')
        records=json.loads((pred/'candidates.json').read_text());n=len(records)
        inventory[str(pred/'masks.npy')]=check(pred/'masks.npy')
        projection_start=time.perf_counter()
        if np.array_equal(coord,reference_coord):
            with np.load(config['evaluation_projection'],allow_pickle=False) as data:
                nearest,distance,matched=[data[k] for k in ('nearest','distance_squared','matched')]
            reused=True
        else:
            import open3d as o3d
            tree=o3d.core.nns.NearestNeighborSearch(o3d.core.Tensor(np.ascontiguousarray(coord,dtype=np.float32)));tree.knn_index()
            near,dist=tree.knn_search(o3d.core.Tensor(np.column_stack([inst[c] for c in ('x','y','z')]).astype(np.float32)),1)
            nearest,distance=near.numpy().ravel(),dist.numpy().ravel();matched=distance<.05**2;reused=False
        if len(nearest)!=len(gt_ids) or not np.array_equal(matched,distance<.05**2) or np.any(nearest<0) or np.any(nearest>=len(coord)):
            raise ValueError('invalid fixed projection')
        np.savez_compressed(out/'projection.npz',nearest=nearest,distance_squared=distance,matched=matched)
        projection_seconds=time.perf_counter()-projection_start
        original_regions=np.load(pred/'regions.npz')['region'];regions=np.where(matched,original_regions[nearest],-1)
        np.save(out/'regions.npy',regions)
        mask_dir=out/'overlapping_masks';mask_dir.mkdir(exist_ok=args.resume); paths={};sizes=np.zeros(n,dtype=np.int64)
        overlap_counts=np.zeros((n,len(gt_values)),dtype=np.int64)
        for i in range(n):
            path=mask_dir/f'candidate_{i:04d}.npy';paths[i]=path
            mask=bank[i,nearest]&matched
            if not path.exists():np.save(path,mask)
            elif not np.array_equal(np.load(path,mmap_mode='r'),mask):raise ValueError('cached projected mask differs')
            sizes[i]=int(mask.sum());v,c=np.unique(gt_ids[mask],return_counts=True)
            overlap_counts[i,np.searchsorted(gt_values,v)]=c
        np.savez_compressed(out/'candidate_gt_overlap.npz',gt_ids=gt_values,gt_sizes=gt_counts,
            projected_sizes=sizes,intersections=overlap_counts)
        cache={};unique_cache={}
        for condition in conditions:
            destination=out/condition;existing=destination/'metrics.json'
            if args.resume and existing.exists():
                rows.append(json.loads(existing.read_text()));continue
            stage_start=time.perf_counter();destination.mkdir(exist_ok=args.resume)
            doc=json.loads((pred/(condition+'.json')).read_text());kept=np.array(doc['kept'],dtype=np.int64)
            labels=np.array(doc['labels'],dtype=np.int64);serialized=doc['rank_scores_serialized'];parsed=np.array(list(map(float,serialized)))
            if not np.isin(labels,valid).all():raise ValueError('class outside semantic vocabulary')
            key=(tuple(kept),tuple(labels[kept]),tuple(serialized[i] for i in kept))
            overlapping=evaluate_set(evaluator,destination/'overlapping',out,paths,labels,serialized,kept,gt_path,cache,key)
            source_owners=np.load(doc['owner_path'],mmap_mode='r')
            owners=np.where(matched,source_owners[nearest],0)
            semantic=np.zeros(len(owners),dtype=np.int64);covered=owners>0;semantic[covered]=labels[owners[covered]-1]
            np.save(destination/'owners.npy',owners);np.save(destination/'semantic.npy',semantic)
            # Legacy projected-mask-first assignment, independently recomputed without GT filtering.
            legacy=np.zeros(len(owners),dtype=np.int64)
            assignment=np.array(doc['assignment_priorities']);groups=np.array(doc['assignment_groups'])
            for i in kept[np.lexsort((kept,-assignment[kept],groups[kept]))]:
                mask=np.load(paths[int(i)],mmap_mode='r');legacy[mask&(legacy==0)]=int(i)+1
            if not np.array_equal(owners,legacy):raise ValueError('source assignment and projection do not commute')
            regional=regional_confusions(sem['label'],semantic,regions,valid)
            direct=semantic_metrics(sem['label'],semantic,valid)
            for field in ('semantic_miou','semantic_macc'):
                if not np.isclose(regional['metrics'][field],direct[field],rtol=0,atol=1e-15):raise ValueError('regional/global semantic mismatch')
            write(destination/'regional_confusion.json',regional)
            unique_dir=out/'unique_masks'/doc['owner_cache_key'];unique_dir.mkdir(parents=True,exist_ok=True)
            unique_paths={};unique_sizes=np.zeros(n,dtype=np.int64)
            for i in kept:
                path=unique_dir/f'candidate_{i:04d}.npy';unique_paths[i]=path;mask=owners==i+1
                if not path.exists():np.save(path,mask)
                unique_sizes[i]=int(mask.sum())
            unique=evaluate_set(evaluator,destination/'unique',out,unique_paths,labels,serialized,kept,gt_path,
                unique_cache,(doc['owner_cache_key'],key))
            high=projected_instance_metrics(owners,inst['label'],{int(i)+1:float(parsed[i]) for i in kept})
            ledger=[]
            for i,r in enumerate(doc['ledger']):
                flags=[]
                if r['source_area']==0:flags.append('source_empty')
                if sizes[i]==0:flags.append('projected_empty')
                elif sizes[i]<100:flags.append('projected_too_small')
                if int(labels[i]) not in evaluator['VALID_CLASS_IDS']:flags.append('invalid_instance_class')
                if i not in set(kept.tolist()):flags.append('not_retained')
                if unique_sizes[i]==0:unique_flag='unique_empty'
                elif unique_sizes[i]<100:unique_flag='unique_too_small'
                else:unique_flag=None
                ledger.append({**r,'projected_area':int(sizes[i]),'unique_projected_area':int(unique_sizes[i]),
                    'evaluation_filter_flags':flags,'unique_export_outcome':unique_flag,
                    'GT_overlap_reference':str(out/'candidate_gt_overlap.npz')})
            write_gzip(destination/'evaluation_ledger.json.gz',ledger)
            if run==config['runs'][0] and condition=='AT_U11':
                expected_row=historical['rows'][0]
                for field in ('all_ap','all_ap_50%','all_ap_25%'):
                    if overlapping['released'][field]!=expected_row['released_semantic_instance'][field]:raise ValueError('historical T1 released AP parity failed')
                for field in ('semantic_miou','semantic_macc'):
                    if not np.isclose(direct[field],expected_row['semantic'][field],rtol=0,atol=1e-15):raise ValueError('historical T1 semantic parity failed')
                write(destination/'historical_metric_parity.json',{'status':'PASS','AP_tolerance':0,'semantic_atol':1e-15})
            row={'scene':config['scene'],'run':run['id'],'condition':condition,'status':'COMPLETE',
                'ovi_readout_id':doc['ovi_readout_id'],'output_representation':'overlapping candidate AP and separately identified disjoint unique map',
                'candidate_count':n,'kept_count':len(kept),'overlapping':overlapping,'unique':unique,
                'semantic':direct,'unique_high_iou_canonical_diagnostic':high,
                'projected_eligible_count':int(sum(sizes[i]>=100 and labels[i] in evaluator['VALID_CLASS_IDS'] for i in kept)),
                'unique_eligible_count':int(sum(unique_sizes[i]>=100 and labels[i] in evaluator['VALID_CLASS_IDS'] for i in kept)),
                'unmatched_evaluation_vertices':int((~matched).sum()),'new_model_inferences':0,
                'extra_3D_pretraining':True,'training_scene_exclusion':'UNVERIFIED',
                'projection_assignment_commutation':'EXACT','regional_confusion_sum':'EXACT',
                'prediction_condition_seconds':doc['condition_seconds'],'evaluation_and_export_seconds':time.perf_counter()-stage_start,
                'evaluation_ledger':str(destination/'evaluation_ledger.json.gz')}
            write(existing,row);rows.append(row)
            print(run['id'],condition,'AP',overlapping['released']['all_ap'],'mIoU',direct['semantic_miou'],'uniqueAP',unique['released']['all_ap'],flush=True)
        write(out/'evaluation_run.json',{'status':'COMPLETE_REQUESTED_CONDITIONS','conditions':conditions,
            'projection_reused_after_exact_coordinate_check':reused,'projection_seconds':projection_seconds,'seconds':time.perf_counter()-run_start})
    write(args.output/'performance.json',rows)
    write(args.output/'evaluation_manifest.json',{'status':'COMPLETE_REQUESTED_CONDITIONS','conditions':conditions,
        'command':sys.argv,'input_inventory':inventory,'source_sha256':{str(f):sha256_file(f) for f in [Path(__file__),ROOT/'src/static_ovmap/released_trace.py',ROOT/'src/static_ovmap/attribution_regions.py']},
        'seconds':time.perf_counter()-overall})


if __name__=='__main__':main()
