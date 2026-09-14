#!/usr/bin/env python3
"""Family-aware local-map evaluator; original metric implementations stay unchanged."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from plyfile import PlyData
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.local_ownership import method_family,validate_assignment
from src.static_ovmap.attribution_regions import regional_confusions
from src.evaluation.static_projected_instances import projected_instance_metrics
from scripts.evaluation.diagnose_static_t1_attribution import bind_protocol,evaluate_set,write,write_gzip
from scripts.evaluation.evaluate_static_ovmap_readout import semantic_metrics
from scripts.evaluation.run_static_local_ownership import read,identity


def candidate_manifest_parity(reference,paths,labels,serialized,kept):
    reference=Path(reference);old=[];new=[]
    for line in reference.read_text().splitlines():
        path,label,score=line.split();old.append((sha256_file(reference.parent/path),int(label),score))
    for i in kept:new.append((sha256_file(paths[i]),int(labels[i]),serialized[i]))
    if old!=new:raise ValueError('U00 candidate mask/class/serialized-rank identity differs')
    return {'exact':True,'canonical_entries':len(new),'records':[{'mask_sha256':h,'class_id':c,'serialized_rank':s} for h,c,s in new],
        'path_rule':'relative path strings may differ; canonical sequence/mask file bytes/class/serialized rank equal'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','predictions','output']:p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();config=read(args.config);old_config=read(config['reference_config']);args.output.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter();evaluator,valid,inventory=bind_protocol(old_config,args.output)
    gt= np.load(old_config['reference_gt_ids']);semgt=PlyData.read(old_config['gt_semantic_map'])['vertex']['label'];instgt=PlyData.read(old_config['gt_instance_map'])['vertex']['label']
    binding=read(args.predictions/'input_binding.json');rows=[];baselines=[]
    for run in config['runs']:
        source=Path(config['predictions'])/run;reference=Path(config['reference_evaluation'])/run;dest=args.output/run;dest.mkdir()
        bank=np.load(source/'masks.npy',mmap_mode='r');coord=np.load(source/'coord.npy',mmap_mode='r');old_doc=read(source/'AT_U00.json')
        bound=next(r for r in binding['runs'] if r['run']==run)
        for name in ['masks.npy','coord.npy','AT_U00.json','regions.npz']:
            if sha256_file(source/name)!=bound['sources'][name]:raise ValueError('prediction inputs changed after source binding')
        projection_start=time.perf_counter();projection= np.load(reference/'projection.npz');near=projection['nearest'];matched=projection['matched'];distance=projection['distance_squared']
        authoritative=np.load(old_config['evaluation_projection']);
        if any(not np.array_equal(projection[k],authoritative[k]) for k in ['nearest','matched','distance_squared']):raise ValueError('fixed historical projection changed')
        if len(near)!=len(gt) or not np.array_equal(matched,distance<.05**2) or np.any(near<0) or np.any(near>=len(coord)):raise ValueError('invalid fixed projection')
        inventory[str(reference/'projection.npz')]=identity(reference/'projection.npz')
        # Original coordinate binding was checked exactly against T0; original reference uses the same verified pool.
        original=np.load(source/'regions.npz');regions=np.where(matched,original['region'][near],-1);np.save(dest/'regions.npy',regions)
        multiplicity=np.zeros(len(coord),np.int32)
        for i in old_doc['kept']:multiplicity+=bank[i]
        projection_seconds=time.perf_counter()-projection_start
        labels=np.array(old_doc['labels']);kept=np.array(old_doc['kept']);serialized=old_doc['rank_scores_serialized'];parsed=np.array(list(map(float,serialized)))
        paths={i:reference/'overlapping_masks'/f'candidate_{i:04d}.npy' for i in kept};sizes=np.zeros(len(bank),np.int64)
        for i in kept:
            projected=np.load(paths[i],mmap_mode='r')
            if not np.array_equal(projected,bank[i,near]&matched):raise ValueError('reused overlapping mask is not frozen common-domain projection')
            sizes[i]=int(projected.sum())
        base=read(reference/'AT_U00/metrics.json');parity=candidate_manifest_parity(base['overlapping']['manifest'],paths,labels,serialized,kept)
        write(dest/'candidate_manifest_parity.json',parity)
        for condition in ['AT_O_AREA','AT_U00','AT_U11']:
            r=read(reference/condition/'metrics.json');baselines.append({**r,'execution_role':'REUSED_REFERENCE_NOT_NEW_EXPERIMENT'})
        cache={'U00':base['overlapping']};first_manifest=None
        for method in config['methods']:
            family=method_family(method);doc=read(args.predictions/run/(method+'.json'))
            if doc['status']=='BLOCKED':rows.append({'run':run,'condition':method,'status':'BLOCKED','reason':doc['reason']});continue
            start=time.perf_counter();target=dest/method;target.mkdir()
            if doc['assignment_family']!=family:raise ValueError('method/family mismatch')
            for k in ['kept','labels','rank_scores','rank_scores_serialized','ovi_readout_id']:
                if doc[k]!=old_doc[k]:raise ValueError('local intervention changed '+k)
            if sha256_file(doc['owner_path'])!=doc['owner_sha256']:raise ValueError('source owner file changed')
            owners_source=np.load(doc['owner_path'],mmap_mode='r')
            verification=validate_assignment(family,bank,kept,owners_source,near,matched,
                priorities=np.array(doc['assignment_priorities']) if family=='global_priority' else None,
                groups=np.array(doc['assignment_groups']) if family=='global_priority' else None)
            owners=verification.pop('projected_owners');raw_source=np.load(old_doc['owner_path'],mmap_mode='r')
            if np.any((owners_source!=raw_source)&(multiplicity<2)):raise ValueError('changed ownership outside active U00 conflicts')
            verification['changed_only_U00_multicandidate']=True
            semantic=np.zeros(len(owners),np.int64);covered=owners>0;semantic[covered]=labels[owners[covered]-1]
            np.save(target/'owners.npy',owners);np.save(target/'semantic.npy',semantic)
            c=regional_confusions(semgt,semantic,regions,valid);direct=semantic_metrics(semgt,semantic,valid)
            for field in ['semantic_miou','semantic_macc']:
                if not np.isclose(c['metrics'][field],direct[field],rtol=0,atol=1e-15):raise ValueError('regional/global semantic mismatch')
            write(target/'regional_confusion.json',c)
            overlap=evaluate_set(evaluator,target/'overlapping',dest,paths,labels,serialized,kept,old_config['reference_gt_ids'],cache,'U00')
            manifest=Path(overlap['manifest']).read_bytes()
            if first_manifest is not None and manifest!=first_manifest:raise ValueError('new candidate manifests differ bytewise')
            first_manifest=manifest;overlap={**overlap,'historical_evaluation_seconds':overlap['evaluation_seconds'],'evaluation_seconds':0.,'cache_hit':True,'reuse_proof':str(dest/'candidate_manifest_parity.json')}
            export_start=time.perf_counter();unique_dir=target/'unique_masks';unique_dir.mkdir();unique_paths={};unique_sizes=np.bincount(owners,minlength=len(bank)+1)[1:]
            for i in kept:
                path=unique_dir/f'candidate_{i:04d}.npy';np.save(path,owners==i+1);unique_paths[i]=path
            export_seconds=time.perf_counter()-export_start
            unique=evaluate_set(evaluator,target/'unique',dest,unique_paths,labels,serialized,kept,old_config['reference_gt_ids'],{},method)
            high=projected_instance_metrics(owners,instgt,{int(i)+1:float(parsed[i]) for i in kept})
            ledger=[]
            for i,r in enumerate(doc['ledger']):
                flags=[]
                if sizes[i]==0:flags.append('projected_empty')
                elif sizes[i]<100:flags.append('projected_too_small')
                if labels[i] not in evaluator['VALID_CLASS_IDS']:flags.append('invalid_instance_class')
                ledger.append({**r,'projected_area':int(sizes[i]),'unique_projected_area':int(unique_sizes[i]),
                    'owned_fraction':float(unique_sizes[i]/sizes[i]) if sizes[i] else None,
                    'evaluation_filter_flags':flags,'unique_export_outcome':'unique_empty' if unique_sizes[i]==0 else 'unique_too_small' if unique_sizes[i]<100 else 'size_eligible'})
            write_gzip(target/'evaluation_ledger.json.gz',ledger)
            row={'scene':'room0','run':run,'condition':method,'status':'COMPLETE','execution_role':'NEW_EXPERIMENT','assignment_family':family,
                'ovi_readout_id':doc['ovi_readout_id'],'candidate_count':len(bank),'kept_count':len(kept),'projected_eligible_count':base['projected_eligible_count'],
                'unique_eligible_count':int(sum(unique_sizes[i]>=100 and labels[i] in evaluator['VALID_CLASS_IDS'] for i in kept)),
                'unique_empty_count':int((unique_sizes[kept]==0).sum()),'unique_small_nonempty_count':int(((unique_sizes[kept]>0)&(unique_sizes[kept]<100)).sum()),
                'overlapping':overlap,'unique':unique,'semantic':direct,'unique_high_iou_canonical_diagnostic':high,
                'validation':verification,'regional_confusion_sum':'EXACT','candidate_manifest_identity':'EXACT','unique_rank_rule':'unchanged U00 raw source area serialized .6f, never owned-area rescoring',
                'unmatched_evaluation_vertices':int((~matched).sum()),'new_model_inferences':0,'extra_3D_pretraining':True,'training_scene_exclusion':'UNVERIFIED',
                'source_summary':str(args.predictions/run/'summary.json'),'evaluation_ledger':str(target/'evaluation_ledger.json.gz'),
                'costs':{'projection_cache_load_seconds':projection_seconds,'unique_mask_export_seconds':export_seconds,'released_unique_evaluation_seconds':unique['evaluation_seconds'],'total_condition_seconds':time.perf_counter()-start}}
            write(target/'metrics.json',row);rows.append(row)
            print(run,method,'uniqueAP',unique['released']['all_ap'],'mIoU',direct['semantic_miou'],flush=True)
        write(dest/'manifest_isolation.json',{'three_new_manifests_byte_identical':first_manifest is not None,'U00_canonical_mask_class_rank_identity':True})
    write(args.output/'performance.json',rows);write(args.output/'reused_baselines.json',baselines)
    write(args.output/'evaluation_manifest.json',{'status':'COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'PARTIAL','rows':len(rows),'command':[sys.executable,*sys.argv],
        'input_inventory':inventory,'source_sha256':{str(f.relative_to(ROOT)):sha256_file(f) for f in [Path(__file__),ROOT/'src/static_ovmap/local_ownership.py',ROOT/'src/static_ovmap/released_trace.py']},
        'seconds':time.perf_counter()-begin,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


if __name__=='__main__':main()
