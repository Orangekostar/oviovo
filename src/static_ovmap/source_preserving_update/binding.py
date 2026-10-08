"""Immutable parent binding and GT-free narrow scene input adapter."""

from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess

import numpy as np

from static_ovmap.backbone_wave1.readouts import fuse_readout
from static_ovmap.composition_study.object_evidence import owner_labels
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.minimal_instance_repair.binding import seal,validate_baseline
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.module_validation.scannet_study import load_prediction
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,PathResolver,read

REPO=Path(__file__).resolve().parents[3]
BASELINES={'SU00_D2':'IR00_D2','SU01_G1':'IR01_G1'}


def free_space(root,minimum=10):
    available=shutil.disk_usage(root).free/2**30
    if available<minimum:raise OSError(f'new store has {available:.3f} GiB free; {minimum} required')
    return available


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value


def bind(spec_path,parent_root,output_root,*,storage_root=None,gpu=None,path_map=None):
    spec_path=Path(spec_path).resolve();spec=read(spec_path)
    if spec_path.read_bytes()!=(REPO/'docs/paper/static_ovmap/source_preserving_update_v1/spec/PROTOCOL_SPEC.json').read_bytes():
        raise ValueError('numeric protocol must be verbatim')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch']:raise ValueError('incorrect isolated branch')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping)
    parent_logical=Path(parent_root).absolute();parent=Path(resolver.resolve(parent_root)).resolve()
    logical=Path(output_root).absolute();root=Path(storage_root).resolve() if storage_root else logical.resolve()
    if root==parent or root.is_relative_to(parent):raise ValueError('new writes cannot target the parent')
    root.mkdir(parents=True,exist_ok=True)
    if storage_root and logical.resolve()!=root:
        if logical.exists() or logical.is_symlink():raise ValueError('storage override differs from existing output')
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(root,target_is_directory=True)
    free_space(root)
    index=ConsumptionIndex(root/'input_verifications.json')
    def doc(path,*,verified=True,expected=None):
        p=Path(resolver.resolve(path));index.identity(p,expected);v=read(p)
        if verified:_verified_identity(v)
        return v
    parent_source=doc(parent/'source_binding.json');store=doc(parent/'result_store.json')
    publication=doc(parent/'publication/final.json');assets=doc(parent/'assets.json')
    if (publication['status']!='PUSH_VERIFIED' or publication['local_HEAD']!=spec['base_commit']
            or publication['remote_HEAD']!=spec['base_commit'] or publication['full_resume_exit_code']!=0
            or publication['result_store_identity']!=store['identity']):
        raise ValueError('final post-correction parent publication does not attest the required base/store')
    if (store['scene_method_coverage']!=234 or store['full_cohort_pool_coverage']!=18
            or store['status']!='SCIENCE_COMPLETE' or store['cohorts']!=spec['cohorts']):
        raise ValueError('parent must have complete 234/18 frozen cohorts')
    source=resolver.rewrite(parent_source);scenes={};baseline_rows={m:{} for m in BASELINES}
    for cohort,names in spec['cohorts'].items():
        for scene in names:
            inherited=source['scenes'][scene]
            context=doc(inherited['context']['path'],expected=inherited['context'])
            predictions=doc(parent/'predictions'/scene/'receipt.json')
            if predictions['status']!='PREDICTIONS_LOCKED':raise ValueError('parent predictions are incomplete')
            for method,old in BASELINES.items():
                row=next(r for r in store['scene_metrics'] if r['scene']==scene and r['method']==old)
                if row['status']!='COMPLETE':raise ValueError('incomplete parent baseline row')
                baseline_rows[method][scene]=resolver.rewrite({**row,'method':method,'source_method':old})
            scenes[scene]={**inherited,'cohort':cohort,
                'parent_prediction_receipt':index.identity(parent/'predictions'/scene/'receipt.json'),
                'parent_predictions':resolver.rewrite(predictions['methods']),
                'parent_plan':index.identity(parent/'recognition'/scene/'plan.json'),
                'parent_decisions':index.identity(parent/'recognition'/scene/'decisions.json')}
    pools={cohort:{m:resolver.rewrite({**store['pooled_metrics'][cohort][old],
                'method':m,'source_method':old,'reuse_kind':'EXACT_PARENT_ORDERED_POOL'})
             for m,old in BASELINES.items()} for cohort in spec['cohorts']}
    result=seal({'status':'BOUND_ACTUAL_PARENT','protocol':spec['task_id'],'spec':index.identity(spec_path),
        'specification':spec,'cohorts':spec['cohorts'],'branch':branch,'base_commit':spec['base_commit'],
        'parent_logical_root':str(parent_logical),'parent_root':str(parent),'logical_root':str(logical),
        'output_root':str(root),'path_map':mapping,'parent_identity':parent_source['identity'],
        'parent_store_identity':store['identity'],'parent_source_binding':index.identity(parent/'source_binding.json'),
        'parent_result_store':index.identity(parent/'result_store.json'),'parent_publication':index.identity(parent/'publication/final.json'),
        'assets':resolver.rewrite(assets),'reference':source['reference'],'FC_python':source['FC_python'],
        'gpu':str(gpu if gpu is not None else source['gpu']),'scenes':scenes,
        'baseline_rows':baseline_rows,'baseline_pools':pools,'parent_reread_settings':source['specification']['reread'],
        'new_maps':0,'deployment':'N0_UNCHANGED','GT_used_by_predictor':False})
    dest=root/'source_binding.json'
    if dest.exists():
        previous=read(dest);_verified_identity(previous)
        if previous!=result:raise ValueError('bound input identity changed; preserve records and invalidate affected descendants')
    else:atomic_write_json(dest,result)
    index.write_memo(root/'input_verifications.json')
    print('BOUND',result['identity'],'26 scenes / 52 baseline rows / 4 pools',flush=True)
    return result


def load_binding(root):
    binding=read(Path(root).resolve()/'source_binding.json');_verified_identity(binding)
    index=ConsumptionIndex(Path(binding['output_root'])/'input_verifications.json')
    for field in ('spec','parent_source_binding','parent_result_store','parent_publication'):
        index.identity(binding[field]['path'],binding[field])
    return binding


def load_scene(binding,scene):
    row=binding['scenes'][scene];root=Path(binding['output_root'])
    resolver=PathResolver(binding['path_map']);index=ConsumptionIndex(root/'inputs'/scene/'verifications.json')
    index.identity(row['context']['path'],row['context']);original=read(row['context']['path']);_verified_identity(original)
    data=resolver.rewrite(original);capture_path=Path(data['capture_manifest']);index.identity(capture_path)
    capture=read(capture_path);_verified_identity(capture)
    if capture['identity']!=row['capture_identity']:raise ValueError('bound native capture changed')
    surface=capture_path.parent/capture['surface']['path'];index.identity(surface,capture['surface'])
    with np.load(surface,allow_pickle=False) as arrays:
        xyz,faces,raw=arrays['surface_xyz'],arrays['surface_faces'],arrays['original_owner']
    payloads={}
    for old in ('IR00_D2','IR01_G1'):
        path=row['parent_predictions'][old]['manifest'];index.identity(path)
        manifest=read(path);index.identity(Path(path).parent/manifest['arrays']['path'],manifest['arrays'])
        payload=load_prediction(path)
        if payload.prediction_key!=row['parent_predictions'][old]['prediction_key']:raise ValueError('parent payload changed')
        baseline='SU00_D2' if old=='IR00_D2' else 'SU01_G1'
        score_path=binding['baseline_rows'][baseline][scene]['evaluation_receipt'];index.identity(score_path)
        validate_baseline(payload,binding['baseline_rows'][baseline][scene],read(score_path))
        payloads[old]=payload
    d2,g1=payloads.values()
    if (d2.geometry!=g1.geometry or _array_digest(xyz)!=d2.geometry.xyz_sha256
            or _array_digest(faces)!=d2.geometry.faces_sha256 or capture['tsdf']['sha256']!=d2.geometry.tsdf_sha256):
        raise ValueError('fixed source geometry changed')
    active=d2.owner_ids>0
    if not np.array_equal(d2.owner_ids[active],g1.owner_ids[active]) or not np.array_equal(d2.semantic_labels[active],g1.semantic_labels[active]):
        raise ValueError('G1 changed original D2 incumbents')
    ids=list(map(int,data['models']['native']['valid_ids']));sources={}
    for name,item in data['sources'].items():
        index.identity(item['path']);value=read(item['path']);_verified_identity(value)
        if value['identity']!=item['identity'] or value['valid_ids']!=ids:raise ValueError('N/Q/F class ordering or identity changed')
        sources[name]=value
    if sources['F']['model_identity']!=data['FC_physical_model_identity']:raise ValueError('historical FC model changed')
    labels,probabilities=fuse_readout(sources,data['temperatures'],'D2',ids,owner_labels(d2))
    if labels!=owner_labels(d2):raise ValueError('real D2 class reconstruction differs')
    audit_path=Path(row['parent_predictions']['IR00_D2']['manifest']).parent.parent/'existing_D2_probability_audit.json'
    index.identity(audit_path);audit=read(audit_path)
    if audit['decision_identity']!=d2.metadata['existing_D2_decision_identity'] or set(audit['owners'])!=set(probabilities):
        raise ValueError('parent full D2 probability lineage differs')
    error=0.
    for owner,p in probabilities.items():
        saved=audit['owners'][owner]
        if saved['label']!=p['label'] or (saved['probabilities'] is None)!=(p['probabilities'] is None):
            raise ValueError('parent D2 fallback/class mismatch')
        if p['probabilities'] is not None:
            delta=np.max(np.abs(np.asarray(p['probabilities'])-np.asarray(saved['probabilities'])))
            error=max(error,float(delta))
            if delta>1e-12:raise ValueError('parent D2 full probability mismatch')
    index.identity(data['config']);config=resolver.rewrite(read(data['config']))
    projection_path=Path(config['scenes'][scene]['projection']);index.identity(projection_path);projection=read(projection_path)
    if projection['identity']!=d2.geometry.projection_identity:raise ValueError('frozen target projection changed')
    path=projection_path.with_name('projection.npz');index.identity(path,{'sha256':projection['sha256']})
    with np.load(path,allow_pickle=False) as arrays:nearest,matched=arrays['nearest'],arrays['matched']
    plan=read(row['parent_plan']['path']);index.identity(row['parent_plan']['path'],row['parent_plan']);_verified_identity(plan)
    # Only original selected supports are needed; no residual registry/seeding/proposals are rebuilt.
    units={}
    mesh=canonical_digest(d2.geometry.to_dict())
    for selected in plan['incumbents']:
        owner=int(selected['owner']);rows=np.flatnonzero(d2.owner_ids==owner)
        digest=canonical_digest({'mesh':mesh,'sorted_source_rows':_array_digest(rows)})
        if digest!=selected['support_hash']:raise ValueError('parent selected original support changed')
        units[f'I:{owner}']=SimpleNamespace(support_hash=digest,rows=rows)
    # construct_partition operations=[] uses this only for its empty moved-row audit.
    unit_set=SimpleNamespace(units=units,vertex_weights=np.zeros(len(raw),np.float64))
    result=SimpleNamespace(scene=scene,data=data,capture=capture,capture_path=capture_path,
        xyz=xyz,faces=faces,raw=raw,d2=d2,g1=g1,sources=sources,probabilities=probabilities,
        valid_ids=ids,nearest=nearest,matched=matched,units=unit_set,index=index,
        d2_probability_parity={'maximum_absolute_error':error,'owners':len(probabilities),
            'parent_probability_audit':index.identity(audit_path),'actual_operator':'backbone_wave1.readouts.fuse_readout'})
    index.write_memo(root/'inputs'/scene/'verifications.json')
    return result
