"""Post-prediction GT sidecars and a separate fixed-original diagnostic domain."""
import hashlib
from pathlib import Path
import time
import numpy as np
from .common import (verified,write,arrays_record,ConsumptionIndex,read_document,
    unit_key,existing,immutable_record,CONDITIONS)

def run(binding,*,phase='confirm_gt_prepare'):
    from static_ovmap.learned_object_readout.data import (_mesh,instance_arrays,_label_map,vertex_area,
        perturb_support,proposal_sites,site_observation)
    from static_ovmap.learned_object_readout.observations import full_qualified,choose_views
    from static_ovmap.disagreement_query.physical_support import canonical_sites,quadrature,direct_visibility
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
    root=Path(binding['output_root']);predictions=verified(root/'confirmation/real_prediction_lock.json')
    if predictions['status']!='COMPLETE':raise ValueError('All real predictions must lock before GT parse')
    scenes=verified(root/'role_lock.json')['roles']['CONF']
    if predictions['all_12_scenes']!=scenes:raise ValueError('Real prediction scene set differs')
    acquisition=verified(root/'confirmation/acquisition.json');features=verified(root/'confirmation/features/manifest.json');classes=verified(root/'class_split.json');index=ConsumptionIndex(root/'confirmation/GT/verifications.json')
    identity=unit_key('confirm_gt_prepare',dict(real_predictions=predictions['identity'],acquisition=acquisition['identity'],features=features['identity']),producers=('gt_diagnostic',),selection='SCV1-GT_BEFORE_VISIBILITY',limit=64)
    path=root/'confirmation/GT/input_lock.json';old=existing(path,identity)
    if old:return old
    tsv=acquisition['label_tsv'];index.identity(tsv['path'],tsv);label_map=_label_map(tsv['path']);all_objects=[];manifests={};began=time.perf_counter()
    for scene in scenes:
        geometry=verified(root/'confirmation/geometry'/scene/'manifest.json');folder=root/'confirmation/GT'/scene;mp=folder/'manifest.json'
        scene_id=unit_key('GT_family',dict(parent=identity,geometry=geometry['identity']),producers=('gt_diagnostic',),scene=scene)
        prior=existing(mp,scene_id)
        if prior:all_objects.extend(prior['objects']);manifests[scene]=prior['identity'];continue
        xyz,faces=_mesh(geometry['mesh']['path']);raw=acquisition['files'][scene]
        seg=raw['_vh_clean_2.0.010000.segs.json']['identity'];agg=raw['.aggregation.json']['identity']
        segments=np.asarray(read_document(seg['path'],index,sealed=False,expected=seg)['segIndices']);groups=read_document(agg['path'],index,sealed=False,expected=agg)['segGroups']
        if len(segments)!=len(xyz):raise ValueError('GT/mesh vertex indexing differs')
        owners,labels,registry=instance_arrays(segments,groups,label_map,classes['official_ids']);area=vertex_area(xyz,faces);family=scene.split('_')[0]
        if len(registry)>999:raise ValueError('Official encoded GT has more than999 instances')
        ordered_owners=sorted(registry,key=lambda o:registry[o]['original_group_id']);official=np.zeros(len(xyz),np.int64)
        for suffix,owner in enumerate(ordered_owners,1):official[owners==owner]=registry[owner]['class_id']*1000+suffix
        target=arrays_record(folder/'targets/official_mesh_GT.npz',dict(official_instance_ids=official,owners=owners,semantic_labels=labels,vertex_area=area),index)
        legal=[o for o in registry if int((owners==o).sum())>=100 and area[owners==o].sum()>0]
        ordered=sorted(legal,key=lambda o:(hashlib.sha256(('SCV1-GT|'+family+'|'+str(registry[o]['original_group_id'])).encode()).hexdigest(),registry[o]['original_group_id']))[:64]
        selection=immutable_record(folder/'original_selection.json',dict(status='COMPLETE',scene=scene,selected=[dict(owner=o,**registry[o]) for o in ordered],legal_originals=len(legal),hash_prefix='SCV1-GT',selected_before_visibility=True,replaced_after_visibility=False,real_prediction_lock=predictions['identity']))
        physical=canonical_sites(xyz,faces,owners);projector=SourceRowProjector(xyz,faces);bank=geometry['frames'];frame_arrays=[]
        for f in bank:
            index.identity(f['arrays']['path'],f['arrays'])
            with np.load(f['arrays']['path'],allow_pickle=False) as a:frame_arrays.append((a['source_rows'].copy(),a['source_valid'].copy(),a['depth_m'].copy()))
        objects=[]
        for owner in ordered:
            row=registry[owner];rows=np.flatnonzero(owners==owner);direction=int(hashlib.sha256((family+'|'+str(row['original_group_id'])+'|direction').encode()).hexdigest(),16)%6
            supports=perturb_support(xyz,area,owners,owner,direction);projections=[[] for _ in CONDITIONS];common=[]
            for fi,(sr,valid,_) in enumerate(frame_arrays):
                for ci,condition in enumerate(CONDITIONS):
                    mask=np.zeros(valid.shape,bool);mask[valid]=supports[condition][sr[valid]];projections[ci].append(mask)
                if all(full_qualified(projections[ci][-1]) for ci in range(4)):common.append(fi)
            obj=dict(key=scene+':GT:'+str(row['original_group_id']),scene=scene,family=family,original_group_id=row['original_group_id'],original_local_id=owner,class_id=row['class_id'],base=row['class_id'] in classes['base_ids'],supported=False,inputs=None,views=[],direction=direction,common_qualified_views=len(common),original_vertex_count=len(rows),original_area=float(area[rows].sum()),support_vertices={c:int(supports[c].sum()) for c in CONDITIONS},no_op_conditions=[c for c in CONDITIONS[1:] if np.array_equal(supports[c],supports['clean'])],data_scope='GT_DERIVED_DIAGNOSTIC')
            truth=arrays_record(folder/'targets'/f'{owner:06d}.npz',dict(class_id=np.asarray(row['class_id'],np.int64),source_rows=rows),index);obj['targets']=truth
            if len(common)<2:obj['reason']='NO_TWO_COMMON_FULL_VIEWS';objects.append(obj);continue
            own=physical['owners']==owner;planned,planned_area=quadrature(physical['xyz'][own],physical['area'][own],4096)
            footprints=[direct_visibility(planned,bank[i],frame_arrays[i][2],projections[0][i],projector.mesh.scene)[1] for i in common]
            chosen=[common[i] for i in choose_views([bank[i]['frame_id'] for i in common],[projections[0][i] for i in common],footprints,planned_area)]
            nv=len(chosen);uv=np.zeros((4,nv,64,2),np.float64);metadata=np.zeros((4,nv,64,8),np.float32);valid=np.zeros((4,nv,64),bool);site_xyz=np.zeros((4,64,3),np.float64);site_available=np.zeros((4,64),bool)
            for ci,condition in enumerate(CONDITIONS):
                sites=proposal_sites(physical,owners,supports[condition]);points=physical['xyz'][sites];site_xyz[ci,:len(points)]=points;site_available[ci,:len(points)]=True
                bbox=(xyz[supports[condition]].min(0),xyz[supports[condition]].max(0))
                for vi,i in enumerate(chosen):
                    p,m,v=site_observation(points,bank[i],frame_arrays[i][2],projections[ci][i],projector.mesh.scene,bbox)
                    rounded=np.floor(p+.5).astype(np.int64);visible=np.flatnonzero(v);v[visible]&=frame_arrays[i][1][rounded[visible,1],rounded[visible,0]]
                    uv[ci,vi,:len(points)]=p;metadata[ci,vi,:len(points)]=m;valid[ci,vi,:len(points)]=v
            array=arrays_record(folder/'inputs'/f'{owner:06d}.npz',dict(masks=np.asarray([[projections[ci][i] for i in chosen] for ci in range(4)],bool),uv=uv,metadata=metadata,local_valid=valid,site_xyz=site_xyz,site_available=site_available),index)
            obj.update(inputs=array,views=[bank[i] for i in chosen],supported=True,reason='COMMON_MULTIVIEW_INPUT_AVAILABLE');objects.append(obj)
        result=write(mp,dict(status='COMPLETE',input_identity=scene_id,scene=scene,selection=selection['identity'],objects=objects,GT_targets=target,GT_registry=registry,annotation_sources=dict(segments=seg,aggregation=agg),annotations_only_after_real_lock=predictions['identity'],original_objects=len(objects),supported_originals=sum(o['supported'] for o in objects),no_65th_replacement=True))
        all_objects.extend(objects);manifests[scene]=result['identity'];print('CONF_GT_INPUTS',scene,len(objects),sum(o['supported'] for o in objects),flush=True)
    index.write_memo(root/'confirmation/GT/verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,objects=all_objects,scenes=manifests,original_objects=len(all_objects),base_objects=sum(o['base'] for o in all_objects),heldout_objects=sum(not o['base'] for o in all_objects),supported_originals=sum(o['supported'] for o in all_objects),abstention_originals=sum(not o['supported'] for o in all_objects),elapsed_seconds=time.perf_counter()-began,data_scope='GT_DERIVED_DIAGNOSTIC',original_denominator_preserved=True,GT_target_files_separate=True))
