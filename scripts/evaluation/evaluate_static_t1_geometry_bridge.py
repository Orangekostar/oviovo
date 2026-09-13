#!/usr/bin/env python3
"""Historical emitted-registry bridge; evaluator metadata is compatibility-only."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from plyfile import PlyData

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.static_ovmap.attribution_bridge import load_historical_registry
from src.static_ovmap.fusion_attribution import serialize_scores
from src.evaluation.static_projected_instances import projected_instance_metrics
from scripts.evaluation.diagnose_static_t1_attribution import bind_protocol,evaluate_set,write,check
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','predictions','evaluation','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();config=json.loads(args.config.read_text());args.output.mkdir(parents=True,exist_ok=False)
    evaluator,valid,inventory=bind_protocol(config,args.output);overall=time.perf_counter()
    original=json.loads(Path(config['historical_native_instances_receipt']).read_text())
    argv=dict(zip(original['command'][1::2],original['command'][2::2]))
    instance_path=argv['--projected-instance-map'];inventory[instance_path]=check(instance_path,original['input_sha256'][instance_path])
    native_mesh=PlyData.read(instance_path)['vertex'].data
    gt_sem=PlyData.read(config['gt_semantic_map'])['vertex'].data
    gt_inst=PlyData.read(config['gt_instance_map'])['vertex'].data
    if len(native_mesh)!=len(gt_sem) or any(not np.array_equal(native_mesh[k],gt_sem[k]) for k in ('x','y','z')):
        raise ValueError('historical native projection is on a different evaluation domain')
    semantic=np.load(config['historical_s1a_semantic'])
    metrics=json.loads(Path(config['historical_s1a_metrics']).read_text())
    inventory[config['historical_s1a_manifest']]=check(config['historical_s1a_manifest'],metrics['prediction_manifest_sha256'])
    inventory[config['historical_s1a_semantic']]=check(config['historical_s1a_semantic'])
    registry=load_historical_registry(config['historical_s1a_manifest'],native_mesh['label'],semantic)
    labels=registry['labels'];original_scores=registry['scores_full_precision'];ids=registry['owner_ids'];kept=np.arange(len(ids))
    readout=json.loads(Path(config['readout']).read_text())['observations']
    for owner,label in zip(ids,labels):
        if readout[str(int(owner))]['class_id']!=int(label):raise ValueError('emitted registry/readout class mismatch')
    native_reference=semantic_metrics(gt_sem['label'],semantic,valid)
    registry_semantic=semantic_metrics(gt_sem['label'],registry['registry_semantic'],valid)
    write(args.output/'registry.json',{'native_owner_ids':ids.tolist(),'labels':labels.tolist(),
        'original_scores_full_precision':original_scores.tolist(),'original_scores_serialized':registry['scores_serialized'],
        'full_precision_score_origin':'reconstructed from exact historical emitted masks using original map_pred_mesh area/max-same-class rule; .6f matches original manifest',
        'compatibility_only_not_prediction_eligibility':True,
        'positive_semantics_outside_emitted_registry':registry['positive_semantics_outside_emitted_registry'],
        'historical_full_semantic_reference':native_reference,'frozen_registry_native_semantic':registry_semantic,
        'semantic_export_registry_effect':{k:registry_semantic[k]-native_reference[k] for k in ['semantic_miou','semantic_macc']}})
    rows=[]
    for run in config['runs']:
        pred=args.predictions/run['id'];out=args.output/run['id'];out.mkdir();cache={}
        native_common=np.load(pred/'native_owners.npy',mmap_mode='r')
        projection=np.load(args.evaluation/run['id']/'projection.npz');near=projection['nearest'];matched=projection['matched']
        common_owner=np.zeros(len(native_common),dtype=np.int64);native_owner=np.zeros(len(native_mesh),dtype=np.int64)
        source_areas=np.zeros(len(ids),dtype=np.int64)
        for i,owner in enumerate(ids):
            mask=native_common==owner;source_areas[i]=int(mask.sum());common_owner[mask]=i+1
            native_owner[native_mesh['label']==owner]=i+1
        transferred_owner=np.where(matched,common_owner[near],0)
        np.save(out/'source_unique_owners.npy',common_owner)
        core=json.loads((pred/'candidates.json').read_text());core_ids={r['native_owner_id'] for r in core if r['source']=='OVI'}
        registry_difference={'core_only_native_owners':sorted(core_ids-set(ids.tolist())),
            'bridge_only_native_owners':sorted(set(ids.tolist())-core_ids)}
        write(out/'registry_difference.json',registry_difference)
        for condition in config['geometry_bridge_conditions']:
            start=time.perf_counter();dest=out/condition;dest.mkdir()
            owners=native_owner if condition=='AT_GEO_NATIVE' else transferred_owner
            scores=source_areas.astype(float) if condition=='AT_GEO_AREA' else original_scores
            serialized,parsed=serialize_scores(scores)
            mask_dir=out/('native_masks' if condition=='AT_GEO_NATIVE' else 'transfer_masks');mask_dir.mkdir(exist_ok=True)
            paths={};ledger=[]
            for i,owner in enumerate(ids):
                mask=owners==i+1;path=mask_dir/f'candidate_{i:04d}.npy';paths[i]=path
                if not path.exists():np.save(path,mask)
                ledger.append({'candidate_id':f"room0/{run['id']}/OVI/{int(owner)}",'canonical_index':i,
                    'native_owner_id':int(owner),'registry':'historical_emitted_61_variable_not_hardcoded',
                    'class_id':int(labels[i]),'original_projected_area':int(registry['original_projected_areas'][i]),
                    'source_area':int(source_areas[i]),'projected_area':int(mask.sum()),
                    'rank_score_full_precision':float(scores[i]),'rank_score_serialized':serialized[i],
                    'rank_score_parsed':float(parsed[i]),'frozen_assignment_priority':float(original_scores[i]),
                    'output_owner_id':i+1,'source_empty':bool(source_areas[i]==0),
                    'projected_empty':not bool(mask.any()),'projected_too_small':bool(0<mask.sum()<100),
                    'invalid_instance_class':int(labels[i]) not in evaluator['VALID_CLASS_IDS']})
            result=evaluate_set(evaluator,dest/'overlapping',out,paths,labels,serialized,kept,
                config['reference_gt_ids'],cache,(condition=='AT_GEO_NATIVE',tuple(serialized)))
            output_sem=np.zeros(len(owners),dtype=np.int64);covered=owners>0;output_sem[covered]=labels[owners[covered]-1]
            if condition=='AT_GEO_NATIVE':
                if not np.array_equal(output_sem,registry['registry_semantic']):raise ValueError('native registry reconstruction failed')
                if result['released']!=metrics['released_semantic_instance']:raise ValueError('exact historical emitted AP parity failed')
            semantic_result=semantic_metrics(gt_sem['label'],output_sem,valid)
            np.save(dest/'owners.npy',owners);np.save(dest/'semantic.npy',output_sem);write(dest/'ledger.json',ledger)
            canonical=projected_instance_metrics(owners,gt_inst['label'],{i+1:float(parsed[i]) for i in kept})
            row={'scene':'room0','run':run['id'],'condition':condition,'status':'COMPLETE',
                'ovi_readout_id':'sha256:'+check(config['readout'])['sha256'],
                'output_representation':'historical emitted registry, disjoint masks; separate full historical semantic reference',
                'candidate_count':len(ids),'kept_count':len(ids),'projected_eligible_count':result['evaluator_prediction_count'],
                'unique_eligible_count':result['evaluator_prediction_count'],'overlapping':result,
                'unique':{**result,'reuse_reason':'masks are already exactly disjoint; no ownership intervention'},
                'semantic':semantic_result,'historical_full_semantic_reference':native_reference if condition=='AT_GEO_NATIVE' else None,
                'unique_high_iou_canonical_diagnostic':canonical,'registry_difference':registry_difference,
                'positive_semantics_excluded_by_original_emitted_registry':registry['positive_semantics_outside_emitted_registry'],
                'assignment_priority_rule':'original full-precision exported native area scores, frozen across all three cells; masks disjoint',
                'source_area_definition':'point count on returned T0 rows; not square metres',
                'new_model_inferences':0,'extra_3D_pretraining':True,'training_scene_exclusion':'UNVERIFIED',
                'evaluation_and_export_seconds':time.perf_counter()-start}
            write(dest/'metrics.json',row);rows.append(row)
            print(run['id'],condition,result['released']['all_ap'],semantic_result['semantic_miou'],flush=True)
    write(args.output/'performance.json',rows)
    write(args.output/'manifest.json',{'status':'COMPLETE','command':sys.argv,'input_inventory':inventory,
        'seconds':time.perf_counter()-overall,'scope':'geometry/correspondence/export compatibility; not improved reconstruction'})


if __name__=='__main__':main()
