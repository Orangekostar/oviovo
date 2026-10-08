"""GT-free frozen targets, exact source observations and shared JPEG windows."""

from pathlib import Path
import time

import numpy as np
from scipy.ndimage import find_objects

from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.minimal_instance_repair.observations import SourceRowProjector,pose_banks

from .binding import load_scene
from .common import (REPO, _array_digest, archive, arrays_record, canonical_digest,
                     verified, write, read)
from .geometry import (choose_window,edit_domain,interior_points,qualified_mask,
                       select_targets,transform_points)


def _valid_observation(source,valid,hw,count):
    if source.shape!=tuple(hw) or valid.shape!=source.shape or valid.dtype!=bool or not np.issubdtype(source.dtype,np.integer):
        raise ValueError('observer raster shape/dtype differs from original capture')
    if np.any(source[~valid]!=-1) or np.any(source[valid]<0) or np.any(source[valid]>=count):
        raise ValueError('invalid source-row sentinel/bounds')


def observations(binding,inputs):
    root=Path(binding['output_root']);scene=inputs.scene;dest=root/'observations'/scene
    index=inputs.index;frames,assignments=pose_banks(inputs.capture['frames'])
    mesh=canonical_digest(inputs.g1.geometry.to_dict())
    new_operator=index.identity(__file__)
    key=canonical_digest({'mesh':mesh,'capture':inputs.capture['identity'],
        'observer':binding['specification']['observer'],'frames':[int(f['frame_id']) for f in frames],
        'operator':new_operator})
    prior_path=dest/'receipt.json'
    if prior_path.exists():
        prior=verified(prior_path)
        if prior['input_identity']!=key:raise ValueError('observation dependency changed; invalidate affected descendants')
        rows=prior['frames']
        for frame,row in zip(frames,rows):
            index.identity(row['rgb']['path'],row['rgb']);index.identity(row['depth']['path'],row['depth'])
            index.identity(row['arrays']['path'],row['arrays'])
        if [r['frame_id'] for r in rows]!=[int(f['frame_id']) for f in frames]:raise ValueError('pose representatives changed')
        return frames,prior
    begin=time.perf_counter();old_root=Path(binding['minimal_root']);old_dir=old_root/'observations'/scene
    old_rows={};reuse_reason=None;proof=None
    try:
        old_binding=verified(old_root/'source_binding.json');index.identity(old_root/'source_binding.json')
        old=verified(old_dir/'receipt.json');index.identity(old_dir/'receipt.json')
        supports=verified(old_dir/'supports.json');index.identity(old_dir/'supports.json')
        if supports['mesh_identity']!=mesh:raise ValueError('old cache mesh differs')
        old_repo=Path(old_binding['spec']['path']).resolve().parents[2]
        operators=[]
        for filename in ('observations.py','support_units.py'):
            operator=index.identity(old_repo/'src/static_ovmap/minimal_instance_repair'/filename)
            current=index.identity(REPO/'src/static_ovmap/minimal_instance_repair'/filename)
            if operator['sha256']!=current['sha256']:raise ValueError('old observer operator differs')
            operators.append(operator)
        cfg=old_binding['specification']['observer']
        if (cfg['max_frames']!=32 or cfg['pose_translation_m']!=.2 or cfg['pose_rotation_degrees']!=15.
                or cfg['depth_absolute_tolerance_m']!=.02 or cfg['depth_relative_tolerance']!=.02
                or cfg['normalize_rays'] or cfg['pixel_coordinates']!='integer'
                or cfg['ray_camera_z']!=1. or cfg['minimum_depth_m']!=1e-6):
            raise ValueError('old observer rule differs')
        dependency=canonical_digest({'capture':inputs.capture['identity'],'geometry':mesh,
            'units':supports['identity'],'observer':cfg,'operators':operators})
        if old['input_identity']!=dependency:raise ValueError('old observer input proof differs')
        old_rows={int(r['frame_id']):r for r in old['frames']}
        if list(old_rows)!=[int(f['frame_id']) for f in frames]:raise ValueError('old frame selection differs')
        proof={'original_receipt':index.identity(old_dir/'receipt.json'),'input_identity':dependency,
               'mesh_identity':mesh,'operators':operators,'panoptic_pixels_read':False}
    except (FileNotFoundError,ValueError,KeyError) as exc:
        old_rows={};reuse_reason=str(exc)
    rows=[];projector=None;new_frames=0
    for frame in frames:
        fid=int(frame['frame_id']);rgb=index.identity(inputs.capture_path.parent/frame['rgb_path'],{'sha256':frame['rgb_sha256']})
        depth=index.identity(inputs.capture_path.parent/frame['depth_path'],{'sha256':frame['depth_sha256']})
        row=old_rows.get(fid);source_record=None;source_kind='RECOMPUTED_ORIGINAL_FULL_MESH'
        if row:
            leaf=canonical_digest({'observer':proof['input_identity'],'frame_id':fid,'bank':frame['bank'],
                'pose':frame['pose_c2w'],'intrinsics':frame['intrinsics'],'rgb':rgb,'depth':depth,
                'panoptic':row['panoptic_input']})
            if row['status']!='COMPLETE' or row['input_identity']!=leaf:
                raise ValueError('old cache frame/camera/depth proof differs')
            source_record=index.identity(row['arrays']['path'],row['arrays']);source_kind='EXACT_PARENT_OBSERVATION'
            with np.load(source_record['path'],allow_pickle=False) as arr:
                source,valid=arr['source_rows'],arr['valid']
            if _array_digest(source)!=row['source_row_sha256'] or _array_digest(valid)!=row['valid_sha256']:
                raise ValueError('actual parent observation raster differs')
        else:
            if projector is None:projector=SourceRowProjector(inputs.xyz,inputs.faces)
            with np.load(depth['path'],allow_pickle=False) as arr:depth_values=arr['depth_m']
            source,valid=projector.project(frame['intrinsics'],frame['pose_c2w'],depth_values)
            source_record=arrays_record(dest/'frames'/f'{fid:06d}.npz',{'source_rows':source,'valid':valid},index)
            new_frames+=1
        _valid_observation(source,valid,frame['image_size_hw'],len(inputs.xyz))
        rows.append({'frame_id':fid,'arrays':source_record,'source_row_sha256':_array_digest(source),
            'valid_sha256':_array_digest(valid),'valid_pixels':int(valid.sum()),'rgb':rgb,'depth':depth,
            'pose_c2w':frame['pose_c2w'],'intrinsics':frame['intrinsics'],
            'image_size_hw':frame['image_size_hw'],'source_kind':source_kind})
    result=write(prior_path,{'status':'COMPLETE','scene':scene,'input_identity':key,'frames':rows,
        'pose_assignments':assignments,'parent_reuse_proof':proof,'cache_rejection_reason':reuse_reason,
        'new_observer_frames':new_frames,'new_rays':0 if projector is None else projector.queried_rays,
        'elapsed_seconds':time.perf_counter()-begin,'GT_used':False,'new_maps':0})
    index.write_memo(root/'inputs'/scene/'verifications.json');return frames,result


def load_observation(row,count):
    with np.load(row['arrays']['path'],allow_pickle=False) as arrays:source,valid=arrays['source_rows'],arrays['valid']
    _valid_observation(source,valid,row['image_size_hw'],count);return source,valid


def canonical_jpeg(frame,path,index):
    """Write once using the prescribed Torch bilinear and uint8 truncation."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    image=np.asarray(Image.open(frame['rgb']['path']).convert('RGB'))
    if image.dtype!=np.uint8 or image.shape[:2]!=tuple(frame['image_size_hw']):raise ValueError('original RGB alignment differs')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tensor=torch.from_numpy(image.copy()).permute(2,0,1).unsqueeze(0).float()
    output=F.interpolate(tensor,size=(1024,1024),mode='bilinear',align_corners=False)[0]
    output=output.clamp(0,255).to(torch.uint8).permute(1,2,0).numpy()
    if not path.exists():Image.fromarray(output).save(path,format='JPEG',quality=95)
    decoded=np.asarray(Image.open(path).convert('RGB'))
    return {'file':index.identity(path),'decoded_rgb_sha256':_array_digest(decoded),
            'shape':list(decoded.shape),'canonical_operator':'torch.nn.functional.interpolate',
            'torch_version':torch.__version__,'jpeg_quality':95,'jpeg_subsampling':'PIL_DEFAULT'}


def plan_scene(binding,scene,count=6):
    inputs=load_scene(binding,scene);root=Path(binding['output_root']);index=inputs.index
    frames,observed=observations(binding,inputs);dest=root/'query_plan'/scene
    key=canonical_digest({'G1':inputs.g1.prediction_key,'D2':inputs.d2.prediction_key,
        'D2_probability_parity':inputs.d2_probability_parity,'observations':observed['identity'],
        'profile':count,'operator':index.identity(__file__),
        'geometry_operator':index.identity(Path(__file__).with_name('geometry.py'))})
    if (dest/'receipt.json').exists():
        old=verified(dest/'receipt.json')
        if old['input_identity']!=key:raise ValueError('changed query plan requires explicit scoped invalidation')
        index.identity(old['domain']['path'],old['domain'])
        for query in old['queries']:
            for f in query['frames']:index.identity(f['canonical']['file']['path'],f['canonical']['file'])
        return old
    begin=time.perf_counter();labels=owner_labels(inputs.g1);d2=set(owner_labels(inputs.d2))
    observation_table={int(r['frame_id']):r for r in observed['frames']};images={};views={o:[] for o in labels}
    for frame in frames:
        fid=int(frame['frame_id']);source,valid=load_observation(observation_table[fid],len(inputs.xyz))
        image=np.zeros(source.shape,np.int64);image[valid]=inputs.g1.owner_ids[source[valid]];images[fid]=image
        sizes=np.bincount(image.ravel());boxes=find_objects(image)
        for owner in labels:
            area=int(sizes[owner]) if owner<len(sizes) else 0
            box=boxes[owner-1] if owner<=len(boxes) else None
            qualified=bool(area>=100 and box is not None and box[0].stop-box[0].start>=2 and box[1].stop-box[1].start>=2)
            views[owner].append({'frame_id':fid,'pixels':area,'qualified':qualified})
    inventory=[]
    for owner,cls in sorted(labels.items()):
        maximum=max(v['pixels'] for v in views[owner]);anchor=min(views[owner],key=lambda v:(-v['pixels'],v['frame_id']))['frame_id']
        qualified=[v['frame_id'] for v in views[owner] if v['qualified']]
        points=interior_points(images[anchor]==owner) if cls in inputs.valid_ids and len(qualified)>=2 else []
        pool='INCUMBENT' if owner in d2 else 'RECOVERED';p=inputs.probabilities.get(str(owner),inputs.probabilities.get(owner))
        probabilities=None if p is None else p['probabilities'];margin=None
        if probabilities is not None:
            sorted_probs=np.sort(np.asarray(probabilities,np.float64));margin=float(sorted_probs[-1]-sorted_probs[-2])
        reason=('INVALID_G1_CLASS' if cls not in inputs.valid_ids else 'FEWER_THAN_TWO_QUALIFIED_OLD_VIEWS' if len(qualified)<2
                else 'UNPROMPTABLE' if not points else 'D2_PROBABILITIES_UNAVAILABLE' if pool=='INCUMBENT' and margin is None else None)
        inventory.append({'owner':owner,'class':cls,'pool':pool,'margin':margin,'maximum_area':maximum,
            'eligible':reason is None,'exclusion_reason':reason,'views':views[owner],
            'anchor_frame_id':anchor,'points_xy_original':points,'qualified_frame_ids':qualified,
            'old_support_sha256':_array_digest(np.flatnonzero(inputs.g1.owner_ids==owner))})
    selected=select_targets(inventory);owners=[r['owner'] for r in selected]
    domains,editable,protected=edit_domain(inputs.xyz,inputs.faces,inputs.g1.owner_ids,owners)
    domain=arrays_record(dest/'edit_domain.npz',{'editable':editable,'protected':protected,
        **{f'owner_{o}':v for o,v in domains.items()}},index)
    queries=[];prompts={}
    for target in selected:
        owner=target['owner'];chosen,anchor_slot,second_slot=choose_window(frames,target['qualified_frame_ids'],target['anchor_frame_id'],count)
        window=dest/'canonical'/str(owner);window_frames=[]
        for slot,fid in enumerate(chosen):
            f=observation_table[fid];canonical=canonical_jpeg(f,window/f'{slot:05d}.jpg',index)
            window_frames.append({**f,'slot':slot,'canonical':canonical})
        anchor=window_frames[anchor_slot];source,valid=load_observation(anchor,len(inputs.xyz))
        point_rows=[]
        for x,y in target['points_xy_original']:
            row=int(source[y,x])
            if not valid[y,x] or inputs.g1.owner_ids[row]!=owner:raise ValueError('automatic prompt is not inside its old target')
            if row in prompts and prompts[row]!=owner:raise ValueError('conflicting source-row prompt owners')
            prompts[row]=owner;point_rows.append(row)
        support=inputs.xyz[inputs.g1.owner_ids==owner]
        query={'scene':scene,'owner':owner,'class':target['class'],'pool':target['pool'],
            'pool_priority_value':target['margin'] if target['pool']=='INCUMBENT' else target['maximum_area'],
            'old_support_sha256':target['old_support_sha256'],'G1_prediction_key':inputs.g1.prediction_key,
            'frame_ids':chosen,'frames':window_frames,'canonical_directory':str(window),
            'anchor_slot':anchor_slot,'second_slot':second_slot,'anchor_frame_id':chosen[anchor_slot],
            'semantic_frame_ids':[chosen[anchor_slot],chosen[second_slot]],
            'points_xy_original':target['points_xy_original'],
            'points_xy_canonical':transform_points(target['points_xy_original'],anchor['image_size_hw']).tolist(),
            'point_labels':[1]*len(point_rows),'prompt_source_rows':point_rows,
            'AABB_min':(support.min(0)-.30).tolist(),'AABB_max':(support.max(0)+.30).tolist(),
            'domain':domain,'target_domain_sha256':_array_digest(domains[owner]),
            'effective_window_length':count,'GT_used':False}
        query=write(dest/'targets'/f'{owner}.json',query);queries.append(query)
    result=write(dest/'receipt.json',{'status':'QUERIES_LOCKED','scene':scene,'input_identity':key,
        'observation_receipt':str(root/'observations'/scene/'receipt.json'),
        'observation_identity':observed['identity'],'inventory':inventory,'queries':queries,
        'selected_owner_ids':owners,'eligible_count':sum(r['eligible'] for r in inventory),
        'domain':domain,'prompt_owners':prompts,'effective_window_length':count,
        'elapsed_seconds':time.perf_counter()-begin,'canonical_RGB_written_once':True,
        'GT_used':False,'models_used_by_selection':False})
    index.write_memo(root/'inputs'/scene/'verifications.json')
    print('QUERIES_LOCKED',scene,len(queries),'eligible',result['eligible_count'],flush=True)
    return result


def plan(binding,count=6):
    return {scene:plan_scene(binding,scene,count) for scene in binding['scenes']}
