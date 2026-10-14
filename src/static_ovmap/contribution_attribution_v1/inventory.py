"""Bounded role-specific parent inventory; JSON identities precede migration."""
from pathlib import Path
from .common import (read_document,write,ConsumptionIndex,PathResolver,unit_identity,cached_record,HISTORICAL,family,DependencyError,canonical_digest)
from .binding import load_binding

def scene_data(binding,scene,role,index):
    resolver=PathResolver(binding['path_map'])
    if role=='FUSION_DEV':
        bb=Path(binding['backbone_root']);base=read_document(bb/'resolved_inputs.json',index)
        receipt=read_document(bb/'readouts'/scene/'BB00_NATIVE/receipt.json',index)
        d2=read_document(receipt['predictions']['D2'],index,sealed=False)
        data=dict(dataset=base['scenes'][scene]['dataset'],config=receipt['config'],sources={n:dict(path=p) for n,p in receipt['sources'].items()},temperatures=d2['metadata']['temperatures'],predictions=receipt['predictions'],scene=scene,parent_readout_receipt=str(bb/'readouts'/scene/'BB00_NATIVE/receipt.json'))
    else:
        compact=Path(binding['compact_root']);parent=read_document(compact/'resolved_inputs.json',index)
        context=compact/'contexts'/(scene+'.json')
        data=read_document(context,index) if context.exists() else parent['scenes'][scene]
    return resolver.rewrite(data)

def scene_sources(binding,scene,role,index):
    data=scene_data(binding,scene,role,index)
    sources={n:read_document(item['path'],index) for n,item in data['sources'].items()}
    ids=sources['N']['valid_ids']
    if set(sources)!={'N','Q','F'} or any(s['valid_ids']!=ids for s in sources.values()):raise ValueError('Source vocabulary/domain differs')
    owners=set(sources['N']['objects'])
    if any(set(s['objects'])!=owners for s in sources.values()):raise ValueError('Source incumbent owner registry differs')
    return data,sources

def run(root):
    root=Path(root);binding=load_binding(root);spec=binding['task_specification'];index=ConsumptionIndex(root/'inventory_verifications.json')
    rows=[];missing=[]
    for role,scenes in (('FUSION_DEV',spec['fusion_dev']),('REGRESSION',[s for names in spec['cohorts'].values() for s in names])):
        for scene in scenes:
            try:
                data,sources=scene_sources(binding,scene,role,index)
                config=read_document(data['config'],index,sealed=False)
                native=read_document(data['predictions']['NATIVE_READOUT'],index,sealed=False)
                geometry=config['scenes'][scene]
                for key in ('projection','annotations'):index.identity(geometry[key])
                projection=read_document(geometry['projection'],index,sealed=False)
                projection_arrays=Path(geometry['projection']).parent/projection.get('arrays',{}).get('path','projection.npz')
                index.identity(projection_arrays)
                rows.append(dict(status='COMPLETE',role=role,scene=scene,family=family(scene),dataset=data['dataset'],data=data,
                    native_manifest=data['predictions']['NATIVE_READOUT'],native_prediction_key=native['prediction_key'],geometry_identity=native['geometry'],
                    owner_partition_digest=native['owner_ids']['sha256'],semantic_digest=native['semantic_labels']['sha256'],rank_digest=canonical_digest(native['instance_ranks']),
                    sources={n:dict(path=data['sources'][n]['path'],identity=s['identity'],valid_ids=s['valid_ids'],model_identity=s.get('model_identity'),text_identity=s.get('text_identity')) for n,s in sources.items()},
                    source_availability={n:sum(bool(o['available']) for o in s['objects'].values()) for n,s in sources.items()},temperatures=data['temperatures'],incumbent_owners=len(sources['N']['objects']),
                    supervision='INHERITED_FROZEN_MODELS',request_view_scope='HISTORICAL_SOURCES_NOT_NEW_FC8',exposure='EXPOSED',measurement_kind='REUSED_EXACT',scorer_context_file=data['config']))
            except (DependencyError,FileNotFoundError) as exc:
                missing.append(dict(role=role,scene=scene,status='PARTIAL_DEPENDENCY',reason=str(exc)))
    parent_rows=[];parent_pools={}
    area=Path(binding['area_root']);resolver=PathResolver(binding['path_map'])
    for cohort,scenes in spec['cohorts'].items():
        parent_pools[cohort]={}
        for scene in scenes:
            try:
                receipt=read_document(area/'evaluation'/scene/'receipt.json',index)
                for row in receipt['rows']:parent_rows.append(dict(resolver.rewrite(row),experiment_ref='AREA_V2',measurement_kind='REUSED_EXACT'))
            except (DependencyError,FileNotFoundError) as exc:missing.append(dict(role='HISTORY',scene=scene,reason=str(exc)))
        for alias,method in HISTORICAL.items():
            file=area/'pools'/cohort/(method+'.json')
            if file.exists():parent_pools[cohort][alias]=resolver.rewrite(read_document(file,index))
    fgstore=read_document(Path(binding['fg_parent_root'])/'evaluation/store.json',index)
    status='PARTIAL_DEPENDENCY' if any(r['role']!='FUSION_DEV' for r in missing) else 'COMPLETE'
    key=unit_identity('inventory',index.entries(),producers=('common','binding','inventory'),binding=binding['identity'])
    previous=cached_record(root/'inventory.json',key)
    if previous:return previous
    result=write(root/'inventory.json',dict(status=status,input_identity=key,role_scenes=rows,missing=missing,fusion_dev_available=all(any(r['scene']==s and r['role']=='FUSION_DEV' for r in rows) for s in spec['fusion_dev']),historical_rows=parent_rows,historical_pools=parent_pools,FG_rows=resolver.rewrite(fgstore['scene_metrics']),FG_pools=resolver.rewrite(fgstore['pooled_metrics']),direct_inputs=index.entries(),paper_protocol_difference=dict(local_voxel_size_m=.01,OVI_MAP_paper_description_m=.1,protocol_match_claim=False)))
    index.write_memo(root/'inventory_verifications.json')
    return result
