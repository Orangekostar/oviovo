#!/usr/bin/env python3
"""New-family evaluator adapter; original released matching and old invariants retained."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from plyfile import PlyData
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.candidate_semantics import array_digest,subset_transition
from src.static_ovmap.object_arbitration import validate_final
from src.static_ovmap.final_instance_ranking import final_source_ranks
from src.static_ovmap.attribution_regions import regional_confusions
from src.evaluation.static_projected_instances import projected_instance_metrics
from scripts.evaluation.diagnose_static_t1_attribution import bind_protocol,evaluate_set,write,write_gzip
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
from scripts.evaluation.run_static_object_decoupling import read



def evaluation_cache_key(run, method, family, doc, bank_sha256, protocol):
    payload = {'run': run, 'method': method, 'family': family,
               'active_ids': doc['kept'], 'labels': doc['labels'],
               'serialized_ranks': doc['rank_scores_serialized'],
               'owner_sha256': doc['owner_sha256'], 'final_masks': doc['final_mask_hashes'],
               'bank_sha256': bank_sha256, 'protocol': protocol}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);old=read(ROOT/cfg['existing_asset_configs'][0])
    root=Path(cfg['runtime_assets_to_resolve']['new_output_root']);pred=root/'predictions';out=root/'evaluation';out.mkdir(exist_ok=False)
    assert read(pred/'prediction_manifest.json')['status']=='COMPLETE'
    evaluator,valid,inventory=bind_protocol(old,out)
    gt=np.load(old['reference_gt_ids']);semgt=PlyData.read(old['gt_semantic_map'])['vertex']['label'];instgt=PlyData.read(old['gt_instance_map'])['vertex']['label']
    authoritative=np.load(old['evaluation_projection']);rows=[]
    for run in cfg['runs']:
        pool=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run
        ref=Path(cfg['recorded_roots_to_verify']['reference_evaluation_root'])/run
        masks=np.load(pool/'masks.npy',mmap_mode='r');records=read(pool/'candidates.json');o=read(pool/'AT_O_AREA.json')
        projection=np.load(ref/'projection.npz')
        for key in ['nearest','matched','distance_squared']:
            if not np.array_equal(projection[key],authoritative[key]):raise ValueError('fixed projection changed')
        near,matched=projection['nearest'],projection['matched'];regions=np.where(matched,np.load(pool/'regions.npz')['region'][near],-1)
        bank_sha=sha256_file(pool/'masks.npy')
        bound=read(pred/'input_binding.json')['inventory'][str(pool/'masks.npy')]['sha256']
        if bank_sha!=bound:raise ValueError('frozen bank changed after prediction')
        previous={}
        for entry in cfg['primary_methods']:
            method=entry['id'];start=time.perf_counter();doc=read(pred/run/(method+'.json'))
            owners_source=np.load(doc['owner_path']);kept=doc['kept'];labels=np.asarray(doc['labels']);serialized=doc['rank_scores_serialized']
            if sha256_file(doc['owner_path'])!=doc['owner_sha256']:raise ValueError('owner payload changed')
            validate_final(owners_source,masks,kept)
            for i in kept:
                if array_digest(owners_source==i+1)!=doc['final_mask_hashes'][str(i)]:raise ValueError('mask payload changed')
            if method=='OD_E1_SEMANTIC':
                assert np.array_equal(owners_source,np.load(o['owner_path'])) and kept==o['kept'] and serialized==o['rank_scores_serialized']
            elif method=='OD_E2_OBJECT':
                assert doc['labels']==o['labels'] and serialized==o['rank_scores_serialized']
            else:
                other='OD_E2_OBJECT' if method=='OD_E3_OBJECT_SEMANTIC' else 'OD_E3_OBJECT_SEMANTIC'
                pdoc,powners=previous[other]
                assert np.array_equal(owners_source,powners) and kept==pdoc['kept']
                if method=='OD_E3_OBJECT_SEMANTIC':assert serialized==pdoc['rank_scores_serialized']
                else:
                    assert doc['labels']==pdoc['labels']
                    assert [float(serialized[i]) for i in kept]==final_source_ranks(owners_source,kept)
            previous[method]=(doc,owners_source)
            dest=out/run/method;dest.mkdir(parents=True)
            owners=np.where(matched,owners_source[near],0)
            semantic=np.zeros(len(owners),np.int64);covered=owners>0;semantic[covered]=labels[owners[covered]-1]
            np.save(dest/'owners.npy',owners);np.save(dest/'semantic.npy',semantic)
            confusion=regional_confusions(semgt,semantic,regions,valid);direct=semantic_metrics(semgt,semantic,valid)
            assert np.isclose(confusion['metrics']['semantic_miou'],direct['semantic_miou'],rtol=0,atol=1e-15)
            write(dest/'regional_confusion.json',confusion)
            paths={};unique_paths={};(dest/'active_masks').mkdir();(dest/'unique_masks').mkdir()
            sizes=np.bincount(owners,minlength=len(masks)+1)[1:];overlap_sizes={}
            for i in kept:
                paths[i]=dest/'active_masks'/f'candidate_{i:04d}.npy'
                projected=masks[i,near]&matched;np.save(paths[i],projected);overlap_sizes[i]=int(projected.sum())
                unique_paths[i]=dest/'unique_masks'/f'candidate_{i:04d}.npy';np.save(unique_paths[i],owners==i+1)
            # No U00 result reuse: active source masks/classes/ranks are actually evaluated.
            overlap_key=evaluation_cache_key(run,method,'active',doc,bank_sha,read(out/'metric_protocol.json'))
            unique_key=evaluation_cache_key(run,method,'unique',doc,bank_sha,read(out/'metric_protocol.json'))
            overlap=evaluate_set(evaluator,dest/'overlapping',dest,paths,labels,serialized,kept,old['reference_gt_ids'],{},overlap_key)
            unique=evaluate_set(evaluator,dest/'unique',dest,unique_paths,labels,serialized,kept,old['reference_gt_ids'],{},unique_key)
            high=projected_instance_metrics(owners,instgt,{i+1:float(serialized[i]) for i in kept})
            ledger=[]
            for i in kept:
                original=records[i]['original_class_id'];subset=labels[i] in evaluator['VALID_CLASS_IDS'];oldsubset=original in evaluator['VALID_CLASS_IDS']
                ledger.append({**records[i],'final_class_id':int(labels[i]),'source_final_area':int(np.count_nonzero(owners_source==i+1)),
                  'projected_area':overlap_sizes[i],'unique_projected_area':int(sizes[i]),'serialized_rank':serialized[i],
                  'instance_subset_transition':subset_transition(original,int(labels[i]),evaluator['VALID_CLASS_IDS']),
                  'original_subset_eligible':bool(oldsubset),'final_subset_eligible':bool(subset)})
            write_gzip(dest/'evaluation_ledger.json.gz',ledger)
            row={'run':run,'condition':method,'status':'COMPLETE','execution_role':'NEW_EXPERIMENT','active_count':len(kept),
                 'active_source_counts':{s:sum(records[i]['source']==s for i in kept) for s in ['OVI','SpaCeFormer']},
                 'overlapping':overlap,'unique':unique,'semantic':direct,'unique_high_iou_canonical_diagnostic':high,
                 'source_unknown_fraction':float(np.mean(owners_source==0)),'projected_unknown_fraction':float(np.mean(owners==0)),
                 'unique_empty_count':int(sum(sizes[i]==0 for i in kept)),
                 'unique_small_nonempty_count':int(sum(0<sizes[i]<100 for i in kept)),
                 'invalid_instance_class_count':int(sum(labels[i] not in evaluator['VALID_CLASS_IDS'] for i in kept)),
                 'subset_entries':sum(r['instance_subset_transition']=='entry' for r in ledger),
                 'subset_exits':sum(r['instance_subset_transition']=='exit' for r in ledger),
                 'cache_keys':{'active':overlap_key,'unique':unique_key},
                 'seconds':time.perf_counter()-start,'regional_confusion_sum':'EXACT',
                 'evaluation_ledger':str(dest/'evaluation_ledger.json.gz')}
            write(dest/'metrics.json',row);rows.append(row)
            print(run,method,'AP',unique['released']['all_ap'],'mIoU',direct['semantic_miou'],flush=True)
    write(out/'performance.json',rows);write(out/'evaluation_manifest.json',{'status':'COMPLETE','cells':len(rows),'inventory':inventory,'command':[sys.executable,*sys.argv]})


if __name__=='__main__':main()
