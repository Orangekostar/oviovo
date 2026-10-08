"""Matched OLD/NEW semantic decisions from the same two frozen FC views."""

import numpy as np


def paired_labels(old_class,class_ids,old_scores,new_scores,margin=.01,anchor_success=True):
    ids=list(map(int,class_ids))
    if old_class not in ids or len(set(ids))!=len(ids) or len(old_scores)!=2 or len(new_scores)!=2:raise ValueError('two fixed views and exact vocabulary required')
    if not anchor_success or any(s is None for s in [*old_scores,*new_scores]):return old_class,old_class,False
    decisions=[]
    for source in (old_scores,new_scores):
        scores=np.asarray(source,np.float64)
        if scores.shape!=(2,len(ids)) or not np.isfinite(scores).all():raise ValueError('finite full-vocabulary cosine arrays required')
        average=scores.mean(0);winner=int(average.argmax())
        decisions.append(ids[winner] if ids[winner]!=old_class and average[winner]-average[ids.index(old_class)]>=margin else old_class)
    return decisions[0],decisions[1],True


def prepare(binding):
    """Lock OLD/NEW masks on the two preselected original RGB views."""
    from pathlib import Path
    from .binding import load_scene
    from .common import arrays_record,canonical_digest,verified,write,_array_digest
    from .geometry import qualified_mask
    from .query_plan import load_observation
    root=Path(binding['output_root']);scenes={}
    for scene in binding['scenes']:
        inputs=load_scene(binding,scene);plan=verified(root/'query_plan'/scene/'receipt.json')
        regions={};targets={};segments={}
        for query in plan['queries']:
            owner=query['owner'];seg=verified(root/'segmentation'/scene/'SAMV'/str(owner)/'receipt.json')
            if seg['status']!='COMPLETE':raise RuntimeError('unexecuted segmentor is not semantic unavailability')
            inputs.index.identity(seg['raw_arrays']['path'],seg['raw_arrays']);segments[owner]=seg
        dependencies=[segments[q['owner']]['identity'] for q in plan['queries']]
        key=canonical_digest({'plan':plan['identity'],'SAMV':dependencies})
        path=root/'semantic_masks'/scene/'manifest.json'
        if path.exists():
            previous=verified(path)
            if previous['input_identity']!=key:raise ValueError('paired-mask inputs changed')
            for region in previous['regions'].values():inputs.index.identity(region['mask']['path'],region['mask'])
            scenes[scene]=previous;continue
        for query in plan['queries']:
            owner=query['owner'];seg=segments[owner]
            with np.load(query['domain']['path'],allow_pickle=False) as arrays:domain=arrays[f'owner_{owner}']
            records={'OLD':[],'NEW':[]}
            with np.load(seg['raw_arrays']['path'],allow_pickle=False) as arrays:
                for fid in query['semantic_frame_ids']:
                    slot=query['frame_ids'].index(fid);frame=query['frames'][slot]
                    source,valid=load_observation(frame,len(inputs.xyz))
                    old=np.zeros(valid.shape,bool);old[valid]=inputs.g1.owner_ids[source[valid]]==owner
                    inside=np.zeros(valid.shape,bool);inside[valid]=domain[source[valid]]
                    new=arrays[f'mask_{slot}']&valid&inside
                    for name,mask in (('OLD',old),('NEW',new)):
                        rid=f'{owner}:{name}:{fid}';qualified,bbox,pixels=qualified_mask(mask)
                        mask_record=arrays_record(root/'semantic_masks'/scene/(rid+'.npz'),{'mask':mask},inputs.index)
                        regions[rid]={'region_id':rid,'owner':owner,'source':name,'frame_id':fid,
                            'RGB':frame['rgb'],'depth':frame['depth'],'pose_c2w':frame['pose_c2w'],
                            'intrinsics':frame['intrinsics'],'mask':mask_record,'mask_digest':_array_digest(mask),
                            'pixels':pixels,'bbox':bbox,'qualified':qualified,
                            'original_size':frame['image_size_hw'],'query_identity':query['identity'],
                            'source_observation':frame['arrays'],'segmentation_identity':seg['identity'],
                            'target_domain_sha256':query['target_domain_sha256'],'NEW_clipped_to_old_owner':False}
                        records[name].append(rid)
            targets[str(owner)]={'old_class':query['class'],'regions':records,
                'anchor_adherent':seg['outcome']['anchor_adherent'],'query_identity':query['identity']}
        value={'status':'PAIRED_MASKS_LOCKED','scene':scene,'regions':regions,'targets':targets,
               'input_identity':key,
               'GT_used':False,'mandatory_pair_reselected':False,'new_AnyUp':0}
        if path.exists():
            previous=verified(path)
            if previous['input_identity']!=value['input_identity']:raise ValueError('paired-mask inputs changed')
            scenes[scene]=previous
        else:scenes[scene]=write(path,value)
        inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    return scenes


def decide(binding):
    from pathlib import Path
    from .binding import load_scene
    from .common import verified,write
    root=Path(binding['output_root']);result={}
    for scene in binding['scenes']:
        manifest=verified(root/'semantic_masks'/scene/'manifest.json');acquired=verified(root/'readout'/scene/'receipt.json')
        if acquired['status']!='COMPLETE':raise RuntimeError('missing/crashed FC is an execution block')
        inputs=load_scene(binding,scene);rows={}
        for owner,target in manifest['targets'].items():
            scores={name:[acquired['regions'][rid]['scores'] for rid in ids] for name,ids in target['regions'].items()}
            old,new,eligible=paired_labels(target['old_class'],inputs.valid_ids,scores['OLD'],scores['NEW'],
                anchor_success=target['anchor_adherent'])
            raw={name:all(s is not None for s in values) for name,values in scores.items()}
            rows[owner]={'old_G1_class':target['old_class'],'OLD_class':old,'NEW_class':new,
                'common_update_eligible':eligible,'raw_source_available':raw,'anchor_adherent':target['anchor_adherent'],
                'full_cosines':scores,'equal_weight_means':{name:np.asarray(values,np.float64).mean(0).tolist() if raw[name] else None for name,values in scores.items()},
                'regions':target['regions'],'joint_keep':not eligible,
                'reason':'MATCHED_BOTH_SOURCES_TWO_VIEWS' if eligible else 'JOINT_SCIENTIFIC_UNAVAILABILITY_KEEP',
                'actual_old_class_change':old!=target['old_class'],'actual_new_class_change':new!=target['old_class']}
        result[scene]=write(root/'readout'/scene/'decisions.json',{'status':'PAIRED_DECISIONS_LOCKED','scene':scene,
            'manifest_identity':manifest['identity'],'acquisition_identity':acquired['identity'],
            'class_ids':inputs.valid_ids,'targets':rows,'paired_eligible':sum(r['common_update_eligible'] for r in rows.values()),
            'margin':.01,'old_raw_zero_veto':False,'GT_used':False})
    return write(root/'readout/summary.json',{'status':'READOUT_COMPLETE','scenes':{s:v['identity'] for s,v in result.items()}})


def acquire(binding):
    from pathlib import Path
    import os,subprocess,time
    from .common import REPO,write
    prepare(binding);root=Path(binding['output_root']);folder=root/'readout/invocations'/str(time.time_ns());folder.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=binding['gpu'],PYTHONPATH=str(REPO/'src'),
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    argv=[binding['FC_python'],'-m','static_ovmap.samv_local_probe.fc_worker','--root',str(root)]
    begin=time.perf_counter()
    with (folder/'worker.log').open('w') as log:process=subprocess.run(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
    write(folder/'process.json',{'argv':argv,'exit_code':process.returncode,'elapsed_seconds':time.perf_counter()-begin,'log':str(folder/'worker.log')})
    if process.returncode:raise RuntimeError('paired FC worker failed: '+str(folder/'worker.log'))
    return decide(binding)
