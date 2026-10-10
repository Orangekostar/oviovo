"""GT-free retained AMG masks -> exact mesh-row subsets -> common frozen readouts."""
import os
from pathlib import Path
import subprocess
import time
import numpy as np
from .common import REPO,ConsumptionIndex,PathResolver,arrays_record,canonical_digest,read,verified,write
from .holdout2 import require_transfer

def run(binding,fc,*,sam2_root=None,sam2_python=None):
    import torch
    from static_ovmap.learned_object_readout.data import _mesh,proposal_sites,site_observation,instance_arrays,_label_map
    from static_ovmap.learned_object_readout.observations import full_qualified,choose_views
    from static_ovmap.disagreement_query.physical_support import canonical_sites,quadrature,direct_visibility
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
    from .data import ObjectLoader
    from .models import teacher
    from .recognition import load_head
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);lock=require_transfer(binding)
    folder=root/'real_proposals';path=folder/'receipt.json'
    if path.exists():
        old=verified(path)
        if old['predictor_lock']!=lock['identity']:raise ValueError('Proposal diagnostic predictor changed')
        return old
    started=time.perf_counter();index=ConsumptionIndex(root/'real_proposals/verifications.json')
    plan=verified(root/'H2_plan.json');spec=binding['specification']['data']['real_proposal']
    resolver=PathResolver(binding['path_map'])
    assets=resolver.rewrite(verified(resolver.resolve('/mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001/model_assets.json')))
    repo=Path(sam2_root or (Path(assets['asset_root'])/'SAM-V/submodules/sam2')).resolve()
    python=sam2_python or assets['environments']['sam2'];checkpoint=assets['weights']['sam2_1_hiera_large']
    index.identity(checkpoint['path'],checkpoint)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if head!=spec['sam2_commit']:raise ValueError('Pinned SAM2 checkout differs')
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=repo,text=True)
    if dirty:raise ValueError('Pinned SAM2 tracked sources are modified')
    frames=[dict(f,scene=s) for s,bank in plan['real_proposal_frames'].items() for f in bank]
    if len(frames)>8:raise ValueError('AMG image cap exceeded')
    for f in frames:index.identity(f['rgb']['path'],f['rgb'])
    settings={k:spec[k] for k in ('points_per_side','points_per_batch','pred_iou_thresh','stability_score_thresh',
              'stability_score_offset','mask_threshold','box_nms_thresh','crop_n_layers','crop_nms_thresh','crop_overlap_ratio',
              'crop_n_points_downscale_factor','min_mask_region_area','output_mode','use_m2m','multimask_output')}
    job=write(folder/'generator_job.json',dict(repo=str(repo),config=spec['sam2_config'],checkpoint=checkpoint['path'],
              settings=settings,frames=frames,output_root=str(folder),predictor_lock=lock['identity'],GT_free=True))
    generator_lock=folder/'generator_lock.json'
    if generator_lock.exists():generated=verified(generator_lock)
    else:
        cmd=[python,'-u',str(Path(__file__).with_name('sam2_worker.py')),'--job',str(folder/'generator_job.json')]
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=binding['gpu']
        with (folder/'generator.log').open('w') as stream:result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            write(folder/'generator_failure.json',dict(status='BLOCKED_MODEL_INTERFACE',exit_code=result.returncode,argv=cmd,missing_outcomes=None))
            raise RuntimeError('SAM2 generator failed; cannot report successful empty proposals')
        raw=read(folder/'generator_raw.json')
        for row in raw['frames'].values():row['arrays']=index.identity(row['arrays'])
        generated=write(generator_lock,dict(raw,job=job['identity'],checkpoint=checkpoint,repo_HEAD=head,argv=cmd,
                                            tracked_sources_clean=True,
                                            inputs_locked_before_GT_matching=True))
    split=resolver.rewrite(verified(root/'split_manifest.json'));selected={s['scene']:s for s in split['selected']}
    inputs_lock=folder/'input_lock.json';contexts={};meshes={};objects=[];coverage=[]
    if inputs_lock.exists():
        locked_inputs=verified(inputs_lock);objects=locked_inputs['objects'];coverage=locked_inputs['coverage']
    else:
        for scene in plan['real_proposal_frames']:
            source=selected[scene];index.identity(source['files']['_vh_clean_2.ply'])
            xyz,faces=_mesh(source['files']['_vh_clean_2.ply']);meshes[scene]=(xyz,faces)
            manifest=verified(parent/'data/generated'/scene/'manifest.json');bank=manifest['frames'];frame_values=[]
            for f in bank:
                index.identity(f['arrays']['path'],f['arrays'])
                with np.load(f['arrays']['path'],allow_pickle=False) as a:
                    frame_values.append((a['source_rows'].copy(),a['source_valid'].copy(),a['depth_m'].copy()))
            projector=SourceRowProjector(xyz,faces)
            for seed in plan['real_proposal_frames'][scene]:
                key=scene+':'+str(seed['frame_id']);row=generated['frames'][key]
                index.identity(row['arrays']['path'],row['arrays'])
                with np.load(row['arrays']['path'],allow_pickle=False) as a:masks=a['masks'].copy()
                source_frame=next(i for i,f in enumerate(bank) if f['frame_id']==seed['frame_id'])
                source_rows,valid,_=frame_values[source_frame]
                for mi,mask in enumerate(masks):
                    proposal_key=key+':'+str(mi);rows=np.unique(source_rows[valid&mask]);rows=rows[rows>=0]
                    c=dict(key=proposal_key,scene=scene,seed_frame=seed['frame_id'],mask_index=mi,generated_mask_pixels=int(mask.sum()),
                           lifted_source_rows=len(rows),supported=False,reason='NO_LIFTABLE_VISIBLE_SOURCE_ROWS')
                    if not len(rows):coverage.append(c);continue
                    support=np.zeros(len(xyz),bool);support[rows]=True
                    reprojections=[]
                    for sr,v,depth in frame_values:
                        projected=np.zeros(v.shape,bool);projected[v]=support[sr[v]];reprojections.append(projected)
                    qualified=[i for i,m in enumerate(reprojections) if full_qualified(m)]
                    c['qualified_views']=len(qualified)
                    support_array=arrays_record(folder/'supports'/(proposal_key.replace(':','_')+'.npz'),
                                               dict(source_rows=rows,original_sam_mask=mask),index)
                    c['support']=support_array
                    if len(qualified)<2:c['reason']='FEWER_THAN_TWO_GEOMETRIC_QUALIFIED_VIEWS';coverage.append(c);continue
                    # Only proposal-vs-context flags; no annotation owners or labels.
                    flags=support.astype(np.int64);physical=canonical_sites(xyz,faces,flags)
                    inside=physical['owners']==1
                    planned,area=quadrature(physical['xyz'][inside],physical['area'][inside],4096)
                    footprints=[direct_visibility(planned,bank[i],frame_values[i][2],reprojections[i],projector.mesh.scene)[1] for i in qualified]
                    chosen=[qualified[i] for i in choose_views([bank[i]['frame_id'] for i in qualified],
                                    [reprojections[i] for i in qualified],footprints,area)]
                    sites=proposal_sites(physical,flags,support);points=physical['xyz'][sites];nv=len(chosen)
                    uv=np.zeros((1,nv,64,2),np.float64);meta=np.zeros((1,nv,64,8),np.float32);visible=np.zeros((1,nv,64),bool)
                    site_xyz=np.zeros((1,64,3),np.float64);available=np.zeros((1,64),bool)
                    site_xyz[0,:len(points)]=points;available[0,:len(points)]=True;bbox=(xyz[support].min(0),xyz[support].max(0))
                    for vi,fi in enumerate(chosen):
                        u,m,v=site_observation(points,bank[fi],frame_values[fi][2],reprojections[fi],projector.mesh.scene,bbox)
                        uv[0,vi,:len(points)]=u;meta[0,vi,:len(points)]=m;visible[0,vi,:len(points)]=v
                    arr=arrays_record(folder/'inputs'/(proposal_key.replace(':','_')+'.npz'),
                        dict(masks=np.stack([reprojections[i] for i in chosen])[None],uv=uv,metadata=meta,local_valid=visible,
                             site_xyz=site_xyz,site_available=available),index)
                    obj=dict(key=proposal_key,scene=scene,family=scene.split('_')[0],views=[bank[i] for i in chosen],inputs=arr,
                             support=support_array,seed_frame=seed['frame_id'],mask_index=mi,GT_used_for_support=False)
                    objects.append(obj);c.update(supported=True,reason='COMMON_MULTIVIEW_SUPPORTED',selected_views=len(chosen));coverage.append(c)
        locked_inputs=write(inputs_lock,dict(status='PROPOSAL_INPUTS_LOCKED',objects=objects,coverage=coverage,
                   generator_lock=generated['identity'],GT_used=False,oracle_union=False))
    feature_path=folder/'features.json'
    if feature_path.exists():features=verified(feature_path)
    else:
        inherited=verified(parent/'features/train-dev.json');frames={};before=fc.images
        for obj in objects:
            for frame in obj['views']:
                key=obj['scene']+':'+str(frame['frame_id'])
                if key not in frames:frames[key]=inherited['frames'].get(key) or fc.capture(frame)
        if len(frames)>384:raise ValueError('Real-proposal context image cap exceeded')
        features=write(feature_path,dict(status='COMPLETE',frames=frames,physical_images=len(frames),new_FC_encodings=fc.images-before))
    output_path=folder/'output_lock.json'
    if output_path.exists():outputs=verified(output_path)
    else:
        loader=ObjectLoader(fc,features);heads={n:load_head(r['checkpoint']['path'],fc.device)[0] for n,r in lock['heads'].items()}
        methods=['FC8',*heads];records=[]
        with torch.no_grad():
            for obj in objects:
                x,_,_=loader.load(obj,0,8);zs=[teacher(x),*[h(x,fc.phi)['embedding'] for h in heads.values()]]
                scores=np.stack([(z@fc.text_tensor.T).cpu().numpy() for z in zs])
                arr=arrays_record(folder/'scores'/(obj['key'].replace(':','_')+'.npz'),dict(full200_cosines=scores,
                         embeddings=np.stack([z.cpu().numpy() for z in zs]),valid_ids=np.asarray(fc.ids)),index)
                records.append(dict(key=obj['key'],scene=obj['scene'],arrays=arr,predictions=[fc.ids[int(s.argmax())] for s in scores]))
        outputs=write(output_path,dict(status='HEAD_OUTPUTS_LOCKED_BEFORE_GT_MATCHING',methods=methods,records=records,
                                       inputs=locked_inputs['identity'],predictors=lock['identity']))
    # Annotation ownership first enters here, after masks, inputs and predictions lock.
    matches=[];gt_support={}
    classes=verified(root/'class_split.json');tsv=Path(binding['prior_data_root'])/'scannetv2-labels.combined.tsv'
    for scene in plan['real_proposal_frames']:
        s=selected[scene];segments=np.asarray(read(s['files']['_vh_clean_2.0.010000.segs.json'])['segIndices'])
        groups=read(s['files']['.aggregation.json'])['segGroups']
        owners,_,registry=instance_arrays(segments,groups,_label_map(tsv),classes['official_ids'])
        counts={o:int((owners==o).sum()) for o in registry}
        registry={o:r for o,r in registry.items() if counts[o]>=100}
        gt_support[scene]=(owners,registry,counts)
    by_key={r['key']:r for r in outputs['records']}
    for obj in objects:
        with np.load(obj['support']['path'],allow_pickle=False) as a:rows=a['source_rows']
        owners,registry,counts=gt_support[obj['scene']];seen,cnt=np.unique(owners[rows],return_counts=True)
        ious=[(int(o),int(n)/(len(rows)+counts[int(o)]-int(n))) for o,n in zip(seen,cnt) if int(o) in registry]
        best=sorted(ious,key=lambda v:(-v[1],v[0]))[0] if ious else (None,0.)
        matched=best[1]>.25
        matches.append(dict(key=obj['key'],scene=obj['scene'],GT_owner=best[0] if matched else None,IoU=best[1],
                       class_id=registry[best[0]]['class_id'] if matched else None,high_quality=best[1]>.5,
                       predictions=by_key[obj['key']]['predictions'],undefined_target=not matched))
    metrics={}
    for j,name in enumerate(outputs['methods']):
        metrics[name]={}
        for quality in ('all_matched','IoU_(.25,.5]','IoU_>.5'):
            subset=[r for r in matches if not r['undefined_target'] and (quality=='all_matched' or r['high_quality']==(quality=='IoU_>.5'))]
            per_gt={}
            for r in subset:per_gt.setdefault((r['scene'],r['GT_owner']),[]).append(int(r['predictions'][j]==r['class_id']))
            metrics[name][quality]=dict(proposals=len(subset),unique_GT=len(per_gt),
                proposal_top1=float(np.mean([r['predictions'][j]==r['class_id'] for r in subset])) if subset else None,
                equal_GT_top1=float(np.mean([np.mean(v) for v in per_gt.values()])) if per_gt else None)
    index.write_memo(root/'real_proposals/verifications.json')
    return write(path,dict(status='COMPLETE',predictor_lock=lock['identity'],generator=generated['identity'],input_lock=locked_inputs['identity'],
                output_lock=outputs['identity'],generated=sum(r['generated'] for r in generated['frames'].values()),retained=len(coverage),
                liftable=sum(r['lifted_source_rows']>0 for r in coverage),multiview=len(objects),matched=sum(not r['undefined_target'] for r in matches),
                matches=matches,metrics=metrics,coverage=coverage,unsupported_are_common=True,new_training_examples=0,
                prediction_only_retention=True,DEV_diagnostic_not_new_confirmation=True,elapsed_seconds=time.perf_counter()-started))
