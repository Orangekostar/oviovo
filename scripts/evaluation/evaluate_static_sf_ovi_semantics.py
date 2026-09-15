#!/usr/bin/env python3
"""Label-only evaluator adapter with complete payload reuse proofs."""
import argparse
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from plyfile import PlyData
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.cached_semantic_transfer import METHODS,payload_key,assert_fixed
from src.static_ovmap.attribution_regions import regional_confusions
from scripts.evaluation.diagnose_static_t1_attribution import bind_protocol,evaluate_set,write
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
from scripts.evaluation.run_static_sf_ovi_semantics import read,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);old=read(ROOT/cfg['parent_configs'][0]);root=Path(cfg['new_output_root']);pred=root/'predictions';out=root/'evaluation';out.mkdir(exist_ok=False)
    start=time.perf_counter();evaluator,valid,inventory=bind_protocol(old,out);protocol=read(out/'metric_protocol.json')
    semgt=PlyData.read(old['gt_semantic_map'])['vertex']['label'];projection=np.load(old['evaluation_projection']);near,matched=projection['nearest'],projection['matched']
    binding=read(pred/'input_binding.json');cache={};rows=[];references=[];evaluated=0
    for run in cfg['runs']:
        pool=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run;ref=Path(cfg['recorded_roots_to_verify']['reference_evaluation_root'])/run
        baseline=read(pool/'AT_O_AREA.json');owner=np.load(baseline['owner_path']);kept=baseline['kept'];masks=np.load(pool/'masks.npy',mmap_mode='r')
        for path in [pool/'masks.npy',pool/'coord.npy',pool/'AT_O_AREA.json',Path(baseline['owner_path'])]:
            if sha256_file(path)!=binding['inventory'][str(path)]['sha256']:raise ValueError('bound source changed')
        oldproj=np.load(ref/'projection.npz')
        if any(not np.array_equal(oldproj[k],projection[k]) for k in ['nearest','matched','distance_squared']):raise ValueError('projection changed')
        projected=np.where(matched,owner[near],0);reference_owners=np.load(ref/'AT_O_AREA/owners.npy')
        if not np.array_equal(projected,reference_owners):raise ValueError('reference geometry differs')
        dest=out/run;dest.mkdir();np.save(dest/'owners.npy',projected);paths={};sizes=np.bincount(projected,minlength=len(masks)+1)[1:]
        for i in kept:
            path=ref/'overlapping_masks'/f'candidate_{i:04d}.npy'
            if not np.array_equal(masks[i,near]&matched,projected==i+1) or not np.array_equal(np.load(path),projected==i+1):raise ValueError('active/unique mask mismatch')
            paths[i]=path
        old_o=read(ref/'AT_O_AREA/metrics.json');old_e1=read(Path(cfg['recorded_roots_to_verify']['previous_od_root'])/'evaluation'/run/'OD_E1_SEMANTIC/metrics.json')
        e1doc=read(Path(cfg['recorded_roots_to_verify']['previous_od_root'])/'predictions'/run/'OD_E1_SEMANTIC.json')
        assert_fixed(baseline,{**e1doc,'owner_path':baseline['owner_path']})
        assert np.array_equal(np.load(e1doc['owner_path']),owner) and e1doc['labels']==baseline['labels']
        references.extend([{**old_o,'execution_role':'REUSED_REFERENCE'},{**old_e1,'execution_role':'REUSED_REFERENCE'}])
        common={'owner_sha256':binding['inventory'][baseline['owner_path']]['sha256'],
                'coordinates_sha256':binding['inventory'][str(pool/'coord.npy')]['sha256'],
                'kept':kept,'rank_strings':[baseline['rank_scores_serialized'][i] for i in kept],
                'protocol':protocol,'mask_paths_sha256':{str(i):sha256_file(paths[i]) for i in kept}}
        for reference in [old_o,old_e1]:
            manifest=Path(reference['unique']['manifest']);lines=manifest.read_text().splitlines()
            if len(lines)!=len(kept):raise ValueError('reference manifest size differs')
            for i,line in zip(kept,lines):
                path,label,rank=line.split()
                if int(label)!=baseline['labels'][i] or rank!=baseline['rank_scores_serialized'][i] or sha256_file(manifest.parent/path)!=common['mask_paths_sha256'][str(i)]:
                    raise ValueError('reference mask/class/rank payload differs')
        basekey=payload_key({**common,'labels':[baseline['labels'][i] for i in kept]});cache[basekey]=old_o['unique']
        regions=np.where(matched,np.load(pool/'regions.npz')['region'][near],-1)
        for method in METHODS:
            begin=time.perf_counter();doc=read(pred/run/(method+'.json'))
            if doc['status']=='BLOCKED':rows.append({'run':run,'condition':method,'status':'BLOCKED','blocker':doc['blocker']});continue
            assert_fixed(baseline,doc)
            if doc['owner_sha256']!=common['owner_sha256']:raise ValueError('method owner changed')
            labels=np.asarray(doc['labels']);key=payload_key({**common,'labels':[int(labels[i]) for i in kept]})
            target=dest/method;target.mkdir();hit=key in cache
            unique=evaluate_set(evaluator,target/'unique',dest,paths,labels,doc['rank_scores_serialized'],kept,old['reference_gt_ids'],cache,key)
            evaluated+=not hit
            semantic=np.zeros(len(projected),np.int64);covered=projected>0;semantic[covered]=labels[projected[covered]-1];np.save(target/'semantic.npy',semantic)
            direct=semantic_metrics(semgt,semantic,valid);confusion=regional_confusions(semgt,semantic,regions,valid);write(target/'regional_confusion.json',confusion)
            assert np.isclose(direct['semantic_miou'],confusion['metrics']['semantic_miou'],atol=1e-15,rtol=0)
            entries=[i for i in kept if labels[i] in evaluator['VALID_CLASS_IDS'] and baseline['labels'][i] not in evaluator['VALID_CLASS_IDS']]
            exits=[i for i in kept if labels[i] not in evaluator['VALID_CLASS_IDS'] and baseline['labels'][i] in evaluator['VALID_CLASS_IDS']]
            row={'run':run,'condition':method,'status':'COMPLETE','execution_role':'REUSED_IDENTICAL_PAYLOAD' if hit else 'NEW_PAYLOAD_EVALUATION',
                 'unique':unique,'semantic':direct,'geometry_diagnostic':old_o['unique_high_iou_canonical_diagnostic'],
                 'active_candidate_AP':'IDENTICAL_DISJOINT_MASKS_REUSE_UNIQUE','fixed_owners_masks_ranks':'EXACT',
                 'payload_key':key,'input_method_key':doc['method_key'],'subset_entries':entries,'subset_exits':exits,
                 'unique_eligible_count':sum(int(sizes[i]>=100 and labels[i] in evaluator['VALID_CLASS_IDS']) for i in kept),
                 'changed_receivers':doc['changed_receivers'],'new_inference':0,'seconds':time.perf_counter()-begin,
                 'actual_released_evaluation_seconds':0. if hit else unique['evaluation_seconds']}
            write(target/'metrics.json',row);rows.append(row)
            print(run,method,'uAP',unique['released']['all_ap'],'mIoU',direct['semantic_miou'],row['execution_role'],flush=True)
    write(out/'performance.json',rows);write(out/'reused_references.json',references)
    write(out/'evaluation_manifest.json',{'status':'COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL','conditions':len(rows),
          'actual_new_payload_evaluations':evaluated,'new_inference':0,'seconds':time.perf_counter()-start,'source':identity(Path(__file__)),
          'command':[sys.executable,*sys.argv],'inventory':inventory})


if __name__=='__main__':main()
