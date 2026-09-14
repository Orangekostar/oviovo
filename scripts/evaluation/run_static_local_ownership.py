#!/usr/bin/env python3
"""Source-only U00 local ownership from saved model output and archived observations."""
import argparse
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
from scipy.sparse import save_npz,load_npz
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.fusion_attribution import resolve_native_owners
from src.static_ovmap.local_ownership import (method_family,cache_identity,build_membership_atoms,
    solve_local_unaries,build_evidence_graph,solve_spatial_ownership,validate_source_owners)
from src.static_ovmap.ownership_evidence import (select_observation_frames,project_shared_visibility,
    atom_histogram,frame_local_evidence,REASONS)


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def arrays(path):
    with np.load(path,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def identity(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha256_file(path)}
def array_hash(x):
    import hashlib
    x=np.ascontiguousarray(x);return hashlib.sha256(str(x.dtype).encode()+str(x.shape).encode()+x.tobytes()).hexdigest()


def bind_inputs(config,output):
    begin=time.perf_counter();inventory={};old=read(config['reference_config']);previous=read(Path(config['predictions'])/'prediction_manifest.json')
    def bind(path,expected=None):
        key=str(Path(path).resolve())
        if key not in inventory:inventory[key]=identity(path)
        if expected and inventory[key]['sha256']!=expected:raise ValueError('bound source changed: '+str(path))
        return inventory[key]['sha256']
    for path in [config['reference_config'],Path(config['predictions'])/'prediction_manifest.json',config['geometry_support']]:bind(path)
    for name in ['readout','native_projection']:
        path=old[name];bind(path,previous['input_inventory'][path]['sha256'])
    native=arrays(old['native_projection'])['owners'];readout=read(old['readout'])['observations']
    runs=[];reference=None
    for name in config['runs']:
        pred=Path(config['predictions'])/name;doc=read(pred/'AT_U00.json');records=read(pred/'candidates.json')
        sources={k:bind(pred/k) for k in ['masks.npy','coord.npy','native_owners.npy','candidates.json','AT_U00.json','regions.npz','region_registry.json']}
        coord=np.load(pred/'coord.npy',mmap_mode='r');bank=np.load(pred/'masks.npy',mmap_mode='r');owners=np.load(pred/'native_owners.npy',mmap_mode='r')
        entry=next(r for r in old['runs'] if r['id']==name);receipt=read(entry['receipt']);bind(entry['receipt']);bind(entry['identity'],receipt['identity_sha256']);bind(entry['t0'],receipt['payload_sha256']['T0.npz'])
        with np.load(entry['t0'],allow_pickle=False) as d:
            if not np.array_equal(coord,d['coord']):raise ValueError('T0 returned coordinate rows changed')
            sfm=d['masks'];sfl=d['class_ids'];sfq=d['query_ids']
        if reference is not None and not np.array_equal(coord,reference):raise ValueError('shared observation reuse requires exact coordinate equality')
        reference=np.asarray(coord)
        if not np.array_equal(owners,native):raise ValueError('native geometry provenance changed')
        canonical=[]
        for i,r in enumerate(records):
            if i!=r['canonical_index'] or r['prediction_run_id']!=name:raise ValueError('candidate/run identity mismatch')
            if r['source']=='OVI':
                o=r['native_owner_id'];expected=owners==o;label=readout[str(o)]['class_id']
                if r['source_query_ids']!=readout[str(o)]['selected_query_ids']:raise ValueError('S1a query binding differs')
            else:
                row=r['original_t0_row'];expected=sfm[row];label=int(sfl[row])
                if r['spaceformer_query_id']!=int(sfq[row]):raise ValueError('model query identity changed')
            if not np.array_equal(bank[i],expected) or doc['labels'][i]!=label or r['original_class_id']!=label:raise ValueError('U00 source masks/classes differ')
            area=int(bank[i].sum(dtype=np.int64))
            if doc['rank_scores_serialized'][i]!=f'{area:.6f}' or doc['ledger'][i]['borrowed_from'] is not None or doc['ledger'][i]['suppressed_by'] is not None:raise ValueError('not raw-rank unmodified U00')
            canonical.append({'index':i,'candidate_id':r['candidate_id'],'mask_sha256':array_hash(bank[i]),'class_id':label,'serialized_rank':doc['rank_scores_serialized'][i]})
        if doc['kept']!=[i for i in range(len(bank)) if bank[i].any()]:raise ValueError('U00 nonempty candidate registry changed')
        del sfm
        runs.append({'run':name,'sources':sources,'t0_sha256':receipt['payload_sha256']['T0.npz'],'canonical_manifest':canonical,
            'candidate_identity_key':cache_identity(canonical),'coordinate_equal':True,'ovi_readout_id':doc['ovi_readout_id']})
    observation_error=None;selection={};frames=[];camera=None;poses=None
    try:
        graph=read(config['graph_receipt']);bind(config['graph_receipt']);support=read(config['geometry_support'])
        for path in [config['intrinsics'],str(Path(config['scene_root'])/'traj.txt')]:bind(path,graph['input_sha256'][path])
        argv=dict(zip(graph['command'][1::2],graph['command'][2::2]))
        if Path(argv['--mask-dir']).resolve()!=Path(config['mask_dir']).resolve():raise ValueError('frame prediction producer differs')
        for flag in ['--mask-export-source','--mask-run-log']:bind(argv[flag],graph['input_sha256'][argv[flag]])
        camera=read(config['intrinsics'])['camera'];poses=np.loadtxt(Path(config['scene_root'])/'traj.txt').reshape(-1,4,4)
        p=config['parameters'];frames,selection=select_observation_frames(support['input_frame_ids'],poses,p['max_frames'],p['translation_m'],p['rotation_degrees'])
        original={r['frame_id']:r for r in graph['frame_hashes']};selection['frame_files']=[]
        for frame in frames:
            mask=Path(config['mask_dir'])/f'frame{frame:06d}.png';depth=Path(config['scene_root'])/'results'/f'depth{frame:06d}.png'
            bind(mask,original[frame]['mask_sha256']);bind(depth,original[frame]['depth_sha256'])
            selection['frame_files'].append({'frame_id':frame,'mask':str(mask),'depth':str(depth),'mask_sha256':inventory[str(mask.resolve())]['sha256'],'depth_sha256':inventory[str(depth.resolve())]['sha256']})
        selection['camera']=camera;selection['poses_sha256']=inventory[str((Path(config['scene_root'])/'traj.txt').resolve())]['sha256']
        selection['observation_origin']='archived CropFormer predicted frame instance IDs; correlated with OVI frontend, not GT'
    except FileNotFoundError as exc:observation_error=str(exc)
    result={'status':'BOUND' if observation_error is None else 'LOCAL_OBSERVATIONS_BLOCKED','runs':runs,'inventory':inventory,
        'observation_error':observation_error,'selection':selection,'GT_input':False,'new_model_inferences':0,
        'preparation_seconds':time.perf_counter()-begin,'config_sha256':bind(output/'frozen_config.json')}
    write(output/'input_binding.json',result);return result,frames,camera,poses


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();config=read(args.config);p=config['parameters']
    for method in config['methods']:method_family(method)
    args.output.mkdir(parents=True,exist_ok=False);write(args.output/'frozen_config.json',config)
    start=time.perf_counter();binding,frames,camera,poses=bind_inputs(config,args.output)
    sources={str(f.relative_to(ROOT)):identity(f)['sha256'] for f in [Path(__file__),ROOT/'src/static_ovmap/local_ownership.py',ROOT/'src/static_ovmap/ownership_evidence.py']}
    shared_key=cache_identity({'coordinates':binding['runs'][0]['sources']['coord.npy'],'selection':binding['selection'],'parameters':p,'source':sources['src/static_ovmap/ownership_evidence.py']})
    shared=args.output/'shared';shared.mkdir();projection_summaries=[];projection_seconds=0
    if not binding['observation_error']:
        coord=np.load(Path(config['predictions'])/config['runs'][0]/'coord.npy',mmap_mode='r')
        # One real input smoke checks actual image shape/depth convention before the full selected-frame pass.
        entry=binding['selection']['frame_files'][0]
        with Image.open(entry['depth']) as im:depth=np.asarray(im).astype(float)/camera['scale']
        with Image.open(entry['mask']) as im:masks=np.asarray(im)
        smoke=project_shared_visibility(coord[:p['chunk']],poses[entry['frame_id']],camera,depth,masks,p['depth_tolerance'],p['chunk'])
        write(args.output/'smoke.json',{'status':'PASS','scope':'first 65536 real source points and first selected real frame; I/O only, not full-scene performance','frame_id':entry['frame_id'],'image_shape':list(depth.shape),'positive_pixels':smoke['positive_instance_pixels'],'coordinate_units':'metres; camera-z; camera-to-world pose; np.rint ties-to-even pixels'})
        for entry in binding['selection']['frame_files']:
            begin=time.perf_counter();frame=entry['frame_id']
            with Image.open(entry['depth']) as im:depth=np.asarray(im).astype(float)/camera['scale']
            with Image.open(entry['mask']) as im:masks=np.asarray(im)
            v=project_shared_visibility(coord,poses[frame],camera,depth,masks,p['depth_tolerance'],p['chunk'])
            np.savez_compressed(shared/f'frame_{frame:06d}.npz',point_ids=v['point_ids'],mask_ids=v['mask_ids'],pixel_ids=v['pixel_ids'])
            elapsed=time.perf_counter()-begin;projection_seconds+=elapsed
            projection_summaries.append({'frame_id':frame,'seconds':elapsed,**{k:v[k] for k in ['rendered_pixels','depth_consistent_pixels','positive_instance_pixels','unknown_consistent_pixels']}})
            print('shared projection',frame,v['positive_instance_pixels'],flush=True)
        write(shared/'summary.json',{'key':shared_key,'frames':projection_summaries,'seconds':projection_seconds,'cache_hit':False})
    summaries=[]
    for run in config['runs']:
        begin=time.perf_counter();pred=Path(config['predictions'])/run;dest=args.output/run;dest.mkdir()
        bank=np.load(pred/'masks.npy',mmap_mode='r');coord=np.load(pred/'coord.npy',mmap_mode='r');doc=read(pred/'AT_U00.json')
        kept=np.array(doc['kept']);areas=np.array(doc['rank_scores']);groups=np.array([r['source']!='OVI' for r in doc['ledger']],np.int64)
        t=time.perf_counter();fallback=resolve_native_owners(bank,kept,areas,source_priority=groups);fill_seconds=time.perf_counter()-t
        owners_by_method={'LO_U00_OVI_FILL':fallback};solver_summaries={'LO_U00_OVI_FILL':{'seconds':fill_seconds,'rule':'same U00 OVI first then SF; raw source area/canonical tie'}}
        run_binding=next(x for x in binding['runs'] if x['run']==run)
        atom_key=evidence_key=unary_key=graph_key=None;atom_seconds=evidence_seconds=graph_seconds=0;statistics={}
        if not binding['observation_error']:
            t=time.perf_counter();atoms=build_membership_atoms(coord,bank,kept,fallback,p['cell_m'],p['connectivity_m'],p['chunk']);atom_seconds=time.perf_counter()-t
            np.savez_compressed(dest/'atoms.npz',**atoms);atom_key=cache_identity({'input':run_binding,'cell':p['cell_m'],'radius':p['connectivity_m'],'atom_arrays':{k:array_hash(v) for k,v in atoms.items()},'source':sources['src/static_ovmap/local_ownership.py']})
            print(run,'atoms',len(atoms['counts']),'incidences',len(atoms['candidates']),'seconds',atom_seconds,flush=True)
            t=time.perf_counter();evidence_dir=dest/'evidence';evidence_dir.mkdir();n=len(atoms['candidates']);views=np.zeros(n,np.int32);mass=np.zeros(n);agreement=np.zeros(n);observed=np.zeros(len(atoms['counts']),np.int32);rejections={name:0 for name in REASONS.values()};hist_paths=[]
            for frame in frames:
                v=arrays(shared/f'frame_{frame:06d}.npz');hist,labels=atom_histogram(atoms['point_to_atom'],v['point_ids'],v['mask_ids'],len(atoms['counts']))
                hist_path=evidence_dir/f'hist_{frame:06d}.npz';save_npz(hist_path,hist);hist_paths.append(hist_path)
                e=frame_local_evidence(atoms,hist,labels,len(bank),p['chunk'],p['min_intersection'],p['min_iou'],p['min_gap'])
                views+=e['usable'];mass+=e['r'];agreement+=e['r']*e['p'];observed+=e['atom_valid_pixel_count']>0
                counts=np.bincount(e['reason'],minlength=len(REASONS))
                for code,name in REASONS.items():rejections[name]+=int(counts[code])
                ii=np.flatnonzero(e['usable']);np.savez_compressed(evidence_dir/f'correspondences_{frame:06d}.npz',incidence_index=ii,r=e['r'][ii],p=e['p'][ii],matched_mask=e['matched_mask'][ii],reason=e['reason'],atom_valid_pixel_count=e['atom_valid_pixel_count'],candidate_mask_table=e['candidate_mask_table'],mask_ids=labels)
                print(run,'evidence',frame,'usable',len(ii),flush=True)
            scores=np.full(n,np.nan);seen=views>0;scores[seen]=(1+agreement[seen])/(2+mass[seen]);evidence_seconds=time.perf_counter()-t
            np.savez_compressed(dest/'evidence.npz',views=views,support_mass=mass,agreement=agreement,scores=scores,observed_views=observed)
            evidence_key=cache_identity({'atom_key':atom_key,'shared_key':shared_key,'parameters':p,'source':sources['src/static_ovmap/ownership_evidence.py'],'evidence_sha256':sha256_file(dest/'evidence.npz')})
            t=time.perf_counter();unaries=solve_local_unaries(atoms,views,scores,p['min_views'],p['beta']);local_seconds=time.perf_counter()-t
            np.savez_compressed(dest/'unaries.npz',**unaries);unary_key=cache_identity({'evidence_key':evidence_key,'unaries_sha256':sha256_file(dest/'unaries.npz'),'beta':p['beta'],'min_views':p['min_views']})
            owners_by_method['LO_U00_LOCAL']=unaries['labels'][atoms['point_to_atom']]+1;solver_summaries['LO_U00_LOCAL']={'seconds':local_seconds,'unary_key':unary_key}
            t=time.perf_counter();graph=build_evidence_graph(atoms,(load_npz(path) for path in hist_paths),p['neighbors'],p['graph_radius'],p['graph_sigma'],p['epsilon'],p['threads'],p['chunk']);graph_seconds=time.perf_counter()-t
            np.savez_compressed(dest/'graph.npz',**graph);graph_key=cache_identity({'atom_key':atom_key,'evidence_key':evidence_key,'graph_sha256':sha256_file(dest/'graph.npz'),'parameters':p})
            t=time.perf_counter();spatial=solve_spatial_ownership(atoms,unaries,graph,p['spatial_lambda'],p['max_sweeps']);spatial_seconds=time.perf_counter()-t
            owners_by_method['LO_U00_SPATIAL']=spatial.pop('labels')[atoms['point_to_atom']]+1
            solver_summaries['LO_U00_SPATIAL']={'seconds':spatial_seconds,'unary_key':unary_key,**spatial}
            aa=np.repeat(np.arange(len(atoms['counts'])),np.diff(atoms['indptr']));challenger=(atoms['candidates']!=atoms['fallback'][aa])&(views>=p['min_views']);qualified=np.zeros(len(atoms['counts']),bool);qualified[aa[challenger]]=True
            np.savez_compressed(dest/'evidence_diagnostics.npz',challenger_qualified=qualified,observed_views=observed)
            statistics={'atom_count':len(atoms['counts']),'incidence_count':n,'source_points':len(coord),'observed_atoms':int((observed>0).sum()),'observed_source_point_fraction':float(atoms['counts'][observed>0].sum()/len(coord)), 'challenger_qualified_atoms':int(qualified.sum()),'challenger_qualified_point_fraction':float(atoms['counts'][qualified].sum()/len(coord)), 'no_challenger_fallback_point_fraction':float(atoms['counts'][~qualified].sum()/len(coord)), 'rejection_reason_counts':rejections,'graph_candidate_edges':graph['candidate_edge_count'],'graph_supported_edges':len(graph['edges']),'usable_view_count_histogram':{str(k):int(v) for k,v in zip(*np.unique(views,return_counts=True))}}
        for method in config['methods']:
            if method not in owners_by_method:
                summary={'run':run,'condition':method,'status':'BLOCKED','reason':binding['observation_error']};write(dest/(method+'.json'),summary);summaries.append(summary);continue
            owners=owners_by_method[method];checks=validate_source_owners(bank,kept,owners)
            if method!='LO_U00_OVI_FILL':
                multiplicity=np.load(pred/'regions.npz')['multiplicity'] if 'multiplicity' in np.load(pred/'regions.npz').files else bank.sum(axis=0)
                if np.any((owners!=fallback)&(multiplicity<2)):raise ValueError('local ownership changed outside U00 conflict')
            key=cache_identity({'candidate_identity':run_binding,'shared_key':shared_key,'atom_key':atom_key,'evidence_key':evidence_key,'unary_key':unary_key,'graph_key':graph_key if method=='LO_U00_SPATIAL' else None,'method':method,'assignment_family':method_family(method),'source_sha256':sources,'parameters':p})
            owner_path=dest/(method+'_owners.npy');np.save(owner_path,owners);count=np.bincount(owners,minlength=len(bank)+1)[1:]
            ledger=[]
            for i,row in enumerate(doc['ledger']):
                r={k:v for k,v in row.items() if not k.startswith('assignment_')};r.update(condition_id=method,owned_source_point_count=int(count[i]));ledger.append(r)
            record={k:doc[k] for k in ['scene','prediction_run_id','ovi_readout_id','kept','labels','rank_scores','rank_scores_serialized','extra_3D_pretraining']}
            record.update(condition=method,status='COMPLETE_PREDICTION',assignment_family=method_family(method),owner_path=str(owner_path),owner_sha256=sha256_file(owner_path),owner_cache_key=key,GT_input=False,new_model_inferences=0,ledger=ledger,canonical_manifest_key=run_binding['candidate_identity_key'],atom_key=atom_key,evidence_key=evidence_key,unary_key=unary_key,graph_key=graph_key if method=='LO_U00_SPATIAL' else None,source_validation=checks,solver=solver_summaries[method],source_sha256=sources)
            if method_family(method)=='global_priority':record.update(assignment_priorities=areas.tolist(),assignment_groups=groups.tolist())
            write(dest/(method+'.json'),record);summaries.append({'run':run,'condition':method,'status':'COMPLETE_PREDICTION','owner_path':str(owner_path),'changed_from_U00_FILL_points':int((owners!=fallback).sum()),'solver':solver_summaries[method]})
        write(dest/'summary.json',{'run':run,'statistics':statistics,'costs':{'atom_seconds':atom_seconds,'evidence_seconds':evidence_seconds,'graph_seconds':graph_seconds,'solvers':solver_summaries,'total_seconds':time.perf_counter()-begin},'keys':{'atom':atom_key,'evidence':evidence_key,'unary':unary_key,'graph':graph_key},'cache_hit':False})
        print(run,'COMPLETE',statistics,flush=True)
    write(args.output/'prediction_manifest.json',{'status':'COMPLETE' if not binding['observation_error'] else 'PARTIAL_REQUIRED_ASSET_MISSING','conditions':summaries,'source_sha256':sources,'command':[sys.executable,*sys.argv],'environment':{'python':sys.version,'numpy':np.__version__,'thread_env':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}},'seconds':time.perf_counter()-start,'shared_projection_seconds':projection_seconds,'preparation_seconds':binding['preparation_seconds'],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'new_model_inferences':0,'training_calls':0})


if __name__=='__main__':main()
