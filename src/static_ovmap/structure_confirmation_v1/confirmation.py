"""GT-free sensor/mesh geometry and fixed SAM supports before any GT parsing."""
from pathlib import Path
import subprocess
import os
import time
import numpy as np
from .common import (verified,write,arrays_record,ConsumptionIndex,unit_key,existing,
    immutable_record,reserve,DependencyError,PathResolver,workload_budget)

def geometry(binding,*,allow_annotation_parse=False):
    if allow_annotation_parse:raise ValueError('Geometry has no annotation parser')
    from PIL import Image
    from static_ovmap.learned_object_readout.sensor import SensorReader
    from static_ovmap.learned_object_readout.data import _mesh
    from static_ovmap.minimal_instance_repair.observations import pose_banks,SourceRowProjector
    root=Path(binding['output_root']);acquired=verified(root/'confirmation/acquisition.json')
    if acquired['status']!='COMPLETE':raise DependencyError('All fixed CONF12 raw files required')
    index=ConsumptionIndex(root/'confirmation/geometry_verifications.json');scenes=verified(root/'role_lock.json')['roles']['CONF'];locks={}
    for scene in scenes:
        raw=acquired['files'][scene];files={s:r['identity'] for s,r in raw.items()}
        for suffix in ('.sens','_vh_clean_2.ply'):index.identity(files[suffix]['path'],files[suffix])
        identity=unit_key('confirm_geometry',dict(acquisition=acquired['identity'],sensor=files['.sens'],mesh=files['_vh_clean_2.ply']),producers=('confirmation',),limit=96)
        folder=root/'confirmation/geometry'/scene;path=folder/'manifest.json';old=existing(path,identity)
        if old:locks[scene]=old['identity'];continue
        reserve(root);begin=time.perf_counter();sensor=SensorReader(files['.sens']['path'])
        chosen,bins=pose_banks([f for f in sensor.frames if f['finite_pose']],limit=96)
        bank_order=[f['frame_id'] for f in chosen];chosen=sorted(chosen,key=lambda f:int(f['frame_id']))
        if not chosen:raise ValueError('No finite frames: '+scene)
        xyz,faces=_mesh(files['_vh_clean_2.ply']['path']);projector=SourceRowProjector(xyz,faces);frames=[]
        for original in chosen:
            fid=int(original['frame_id']);decoded=sensor.decode(fid);pose=np.asarray(original['pose_c2w'])
            source_rows,valid=projector.project(decoded['intrinsics'],pose,decoded['depth_m']);valid&=decoded['color_valid']
            image_path=folder/'frames'/f'{fid:06d}.png';image_path.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(decoded['rgb']).save(image_path)
            array=arrays_record(image_path.with_suffix('.npz'),dict(depth_m=decoded['depth_m'],color_valid=decoded['color_valid'],source_rows=source_rows,source_valid=valid),index)
            frames.append(dict(scene=scene,frame_id=fid,pose_c2w=pose.tolist(),intrinsics=decoded['intrinsics'].tolist(),image_size_hw=list(decoded['depth_m'].shape),convention=decoded['convention'],rgb=index.identity(image_path),arrays=array))
        ids=list(dict.fromkeys(int(np.floor(j*(len(frames)-1)/3)) for j in range(4)))
        value=write(path,dict(status='COMPLETE',input_identity=identity,scene=scene,frames=frames,AMG_seed_frames=[frames[i] for i in ids],original_pose_bank_order=bank_order,render_order=[f['frame_id'] for f in frames],pose_bin_assignments=bins,calibration=sensor.calibration(),mesh=files['_vh_clean_2.ply'],mesh_vertices=len(xyz),mesh_faces=len(faces),queried_rays=projector.queried_rays,elapsed_seconds=time.perf_counter()-begin,RGB_decoded=len(frames),annotation_files_parsed=0))
        locks[scene]=value['identity'];print('CONF_GEOMETRY',scene,len(frames),flush=True)
    index.write_memo(root/'confirmation/geometry_verifications.json')
    return immutable_record(root/'confirmation/geometry_lock.json',dict(status='COMPLETE',scenes=locks,frames=sum(len(verified(root/'confirmation/geometry'/s/'manifest.json')['frames']) for s in scenes),annotations_parsed=0))

def proposals(binding):
    from static_ovmap.frozen_fc_routing.resources import queued_gpu_lock
    from static_ovmap.learned_object_readout.features import execution_config
    root=Path(binding['output_root']);geometry_lock=verified(root/'confirmation/geometry_lock.json');predictors=verified(root/'predictor_lock.json');spec=binding['task_specification']['sam2']
    resolver=PathResolver(binding['path_map']);index=ConsumptionIndex(root/'confirmation/SAM_verifications.json')
    asset_path=resolver.resolve('/mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001/model_assets.json')
    assets=resolver.rewrite(verified(asset_path));index.identity(asset_path)
    checkout=Path(binding.get('task_sam2_root') or Path(assets['asset_root'])/'SAM-V/submodules/sam2')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()
    if head!=spec['commit'] or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip():raise ValueError('Pinned SAM2 checkout differs')
    checkpoint=assets['weights']['sam2_1_hiera_large'];index.identity(checkpoint['path'],checkpoint)
    worker=Path(__file__).parents[1]/'preservation_routing/sam2_worker.py';worker_id=index.identity(worker)
    identity=unit_key('confirm_proposals',dict(geometry=geometry_lock['identity'],predictors=predictors['identity'],checkpoint=checkpoint,worker=worker_id,settings=spec),producers=('confirmation',))
    path=root/'confirmation/SAM/generator_lock.json';old=existing(path,identity)
    if old:
        budget=workload_budget(root,binding['task_specification']);count=old['counts']['image_encoder_calls']
        if budget.counts['SAM_images']==0:budget.consume('SAM_images',count)
        elif budget.counts['SAM_images']!=count:raise ValueError('SAM budget ledger differs from measured hooks')
        return real_inputs(binding,old)
    scenes=verified(root/'role_lock.json')['roles']['CONF'];frames=[f for s in scenes for f in verified(root/'confirmation/geometry'/s/'manifest.json')['AMG_seed_frames']]
    if len(frames)>48:raise ValueError('SAM image budget exceeded')
    for f in frames:index.identity(f['rgb']['path'],f['rgb'])
    folder=path.parent;job_path=folder/'generator_job.json';observed_path=folder/'observed_exit.json'
    payload=dict(status='LOCKED',input_identity=identity,repo=str(checkout),config=spec['config'],checkpoint=checkpoint['path'],settings=spec['settings'],frames=frames,output_root=str(folder),predictor_lock=predictors['identity'],GT_free=True)
    prior_job=verified(job_path) if job_path.exists() else None
    if prior_job and prior_job['input_identity']!=identity:raise ValueError('Fixed SAM job identity changed')
    job=prior_job or write(job_path,payload)
    command=[binding.get('task_sam2_python') or assets['environments']['sam2'],'-u',str(worker),'--job',str(folder/'generator_job.json')]
    if prior_job:
        observed=verified(observed_path) if observed_path.exists() else None
        if not observed or observed['actual_exit_code']!=0 or not (folder/'generator_raw.json').exists():
            raise DependencyError('Prior SAM job did not complete; preserve outputs and do not repeat its bounded image encodings: '+str(folder))
    else:
        with queued_gpu_lock(execution_config(binding)['gpu_lock']):
            with (folder/'generator.log').open('w') as stream:result=subprocess.run(command,env={**os.environ,'CUDA_VISIBLE_DEVICES':binding['gpu'],'OMP_NUM_THREADS':'1'},stdout=stream,stderr=subprocess.STDOUT)
        write(observed_path,dict(status='COMPLETE' if result.returncode==0 else 'FAILED',command=command,actual_exit_code=result.returncode,planned_maximum_image_encodings=len(frames),partial_encoding_count=None if result.returncode else len(frames)))
        if result.returncode:raise DependencyError('SAM2 worker failed; image count is unknown and automatic rerun is disabled: '+str(folder/'generator.log'))
    raw=read_raw(folder/'generator_raw.json')
    if raw['counts']['image_encoder_calls']!=len(frames) or raw['counts']['image_encoder_calls']>48:raise ValueError('Measured SAM image encodings differ from fixed plan')
    budget=workload_budget(root,binding['task_specification']);count=raw['counts']['image_encoder_calls']
    if budget.counts['SAM_images']==0:budget.consume('SAM_images',count)
    elif budget.counts['SAM_images']!=count:raise ValueError('SAM budget ledger differs from measured hooks')
    for row in raw['frames'].values():row['arrays']=index.identity(row['arrays'])
    value=write(path,dict(raw,input_identity=identity,job=job['identity'],checkpoint=checkpoint,repo_HEAD=head,annotations_parsed=0))
    index.write_memo(root/'confirmation/SAM_verifications.json')
    return real_inputs(binding,value)

def read_raw(path):
    import json
    return json.loads(Path(path).read_text())

def real_inputs(binding,generated):
    from static_ovmap.learned_object_readout.data import _mesh,proposal_sites,site_observation
    from static_ovmap.learned_object_readout.observations import full_qualified,choose_views
    from static_ovmap.disagreement_query.physical_support import canonical_sites,quadrature,direct_visibility
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
    root=Path(binding['output_root']);index=ConsumptionIndex(root/'confirmation/input_verifications.json')
    geometry_lock=verified(root/'confirmation/geometry_lock.json');identity=unit_key('real_inputs',dict(geometry=geometry_lock['identity'],SAM=generated['identity']),producers=('confirmation',))
    path=root/'confirmation/real_input_lock.json';old=existing(path,identity)
    if old:return old
    objects=[]
    for scene in verified(root/'role_lock.json')['roles']['CONF']:
        manifest=verified(root/'confirmation/geometry'/scene/'manifest.json');bank=manifest['frames'];xyz,faces=_mesh(manifest['mesh']['path']);projector=SourceRowProjector(xyz,faces);frame_arrays=[]
        for f in bank:
            index.identity(f['arrays']['path'],f['arrays'])
            with np.load(f['arrays']['path'],allow_pickle=False) as a:frame_arrays.append((a['source_rows'].copy(),a['source_valid'].copy(),a['depth_m'].copy()))
        for seed in manifest['AMG_seed_frames']:
            row=generated['frames'][scene+':'+str(seed['frame_id'])];index.identity(row['arrays']['path'],row['arrays'])
            with np.load(row['arrays']['path'],allow_pickle=False) as a:masks=a['masks'].copy()
            fi=next(i for i,f in enumerate(bank) if f['frame_id']==seed['frame_id']);sr,valid,_=frame_arrays[fi]
            for mi,mask in enumerate(masks):
                digest=row['metadata'][mi]['digest'];key=scene+':'+str(seed['frame_id'])+':'+digest;rows=np.unique(sr[valid&mask]);rows=rows[rows>=0]
                support=np.zeros(len(xyz),bool);support[rows]=True
                support_record=arrays_record(root/'confirmation/real_supports'/scene/(str(seed['frame_id'])+'_'+digest+'.npz'),dict(source_rows=rows,original_sam_mask=mask),index)
                obj=dict(key=key,scene=scene,family=scene.split('_')[0],seed_frame=seed['frame_id'],mask_index=mi,mask_digest=digest,source_mask=row['metadata'][mi],support=support_record,mesh=manifest['mesh'],mesh_vertices=len(xyz),supported=False,views=[],inputs=None)
                projections=[]
                for source,good,_ in frame_arrays:
                    m=np.zeros(good.shape,bool);m[good]=support[source[good]];projections.append(m)
                qualified=[i for i,m in enumerate(projections) if full_qualified(m)];obj['qualified_views']=len(qualified)
                if len(qualified)<2:obj['reason']='NO_MULTIVIEW_SUPPORT';objects.append(obj);continue
                flags=support.astype(np.int64);physical=canonical_sites(xyz,faces,flags);inside=physical['owners']==1
                planned,area=quadrature(physical['xyz'][inside],physical['area'][inside],4096)
                footprints=[direct_visibility(planned,bank[i],frame_arrays[i][2],projections[i],projector.mesh.scene)[1] for i in qualified]
                chosen=[qualified[i] for i in choose_views([bank[i]['frame_id'] for i in qualified],[projections[i] for i in qualified],footprints,area)]
                sites=proposal_sites(physical,flags,support);points=physical['xyz'][sites];nv=len(chosen)
                uv=np.zeros((1,nv,64,2),np.float64);metadata=np.zeros((1,nv,64,8),np.float32);local_valid=np.zeros((1,nv,64),bool);site_xyz=np.zeros((1,64,3),np.float64);site_available=np.zeros((1,64),bool)
                site_xyz[0,:len(points)]=points;site_available[0,:len(points)]=True
                bbox=(xyz[rows].min(0),xyz[rows].max(0))
                for vi,i in enumerate(chosen):
                    p,m,v=site_observation(points,bank[i],frame_arrays[i][2],projections[i],projector.mesh.scene,bbox)
                    rounded=np.floor(p+.5).astype(np.int64);visible=np.flatnonzero(v);v[visible]&=frame_arrays[i][1][rounded[visible,1],rounded[visible,0]]
                    uv[0,vi,:len(points)]=p;metadata[0,vi,:len(points)]=m;local_valid[0,vi,:len(points)]=v
                arr=arrays_record(root/'confirmation/real_inputs'/scene/(str(seed['frame_id'])+'_'+digest+'.npz'),dict(masks=np.asarray([[projections[i] for i in chosen]],bool),uv=uv,metadata=metadata,local_valid=local_valid,site_xyz=site_xyz,site_available=site_available),index)
                obj.update(supported=True,reason='MULTIVIEW_INPUT_AVAILABLE',inputs=arr,views=[bank[i] for i in chosen]);objects.append(obj)
        print('CONF_REAL_INPUTS',scene,sum(o['scene']==scene for o in objects),flush=True)
    if len(objects)>768:raise ValueError('Retained proposal cap exceeded')
    for owner,obj in enumerate(objects,1):obj['owner']=owner
    index.write_memo(root/'confirmation/input_verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,objects=objects,retained_proposals=len(objects),supported=sum(o['supported'] for o in objects),annotations_parsed=0,raw_supports_retained=True,overlapping_masks_retained=True,data_scope='PROVIDED_RECONSTRUCTION_SPARSE_SAM_PROPOSALS'))

def run(binding,*,phase):
    if phase=='confirm_geometry':return geometry(binding)
    if phase=='confirm_proposals':return proposals(binding)
    if phase=='confirm_features':
        from .features import run
        return run(binding)
    raise ValueError('Unexpected confirmation phase')
