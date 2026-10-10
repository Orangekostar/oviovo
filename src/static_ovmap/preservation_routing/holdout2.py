"""Four prelocked official families; annotation parsing occurs only after nominations."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import time
from .common import ConsumptionIndex,PathResolver,arrays_record,canonical_digest,read,verified,write

def require_transfer(binding):
    root=Path(binding['output_root']);lock=verified(root/'predictor_lock.json')
    if lock['status']!='PREDICTORS_LOCKED' or not lock['activated_full_path']:
        raise ValueError('New confirmation annotations forbidden without qualified, frozen predictors')
    main=verified(root/'R_nomination_seed17.json');repeat=verified(root/'R_nomination_seed29.json')
    if main['Rstar'] is None or repeat['Rstar']!=main['Rstar']:raise ValueError('Both independent R qualifications required')
    if lock['R_nominations']!={'17':main['identity'],'29':repeat['identity']}:raise ValueError('R lock changed')
    for seed in (17,29):
        nom=verified(root/f'G_nomination_seed{seed}.json')
        if nom['identity']!=lock['G_nominations'][str(seed)]['identity']:raise ValueError('G predictor lock changed')
    return lock

def run(binding,fc):
    import torch
    from static_ovmap.learned_object_readout.inventory import inventory_roots,RAW_SUFFIXES
    from static_ovmap.module_validation.scannet_download import DownloaderLayout,transfer
    from static_ovmap.learned_object_readout.data import prepare_family
    from .data import ObjectLoader
    from .recognition import evaluate,load_head
    root=Path(binding['output_root']);lock=require_transfer(binding);plan=verified(root/'H2_plan.json')
    if plan['identity']!=lock['H2_plan']:raise ValueError('H2 family names changed')
    path=root/'holdout2/receipt.json'
    if path.exists():
        previous=verified(path)
        if previous['status']!='PARTIAL_CONFIRMATION_BLOCKED':
            if previous['predictor_lock']!=lock['identity']:raise ValueError('Completed H2 predictors differ')
            return previous
        if previous['selected_names']!=plan['selected']:raise ValueError('Blocked H2 names differ')
        archive=path.parent/'failed_attempts'/f'{time.time_ns()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,archive)
    started=time.perf_counter();hroot=root/'holdout2';hroot.mkdir(parents=True,exist_ok=True)
    index=ConsumptionIndex(root/'holdout2/verifications.json')
    acquisition=PathResolver(binding['path_map']).rewrite(read(Path(binding['prior_data_root'])/'acquisition_plan.json'))
    script=acquisition['layout']['script_path'];index.identity(script,{'sha256':acquisition['layout']['script_sha256']})
    layout=DownloaderLayout.from_script(script)
    inventory=inventory_roots([str(hroot/'data/scannet'),*binding['data_roots'],str(Path(binding['lr_parent_root'])/'data/scannet')])
    selected=[]
    for scene in plan['selected']:
        existing=inventory['scans'].get(scene)
        if existing and existing['complete']:files=existing['files']
        else:
            files={suffix:str(hroot/'data/scannet/scans'/scene/(scene+suffix)) for suffix in RAW_SUFFIXES}
            try:
                with ThreadPoolExecutor(max_workers=4) as pool:
                    jobs={s:pool.submit(transfer,layout.scene_url(scene,s),Path(p)) for s,p in files.items()}
                    downloads={s:f.result() for s,f in jobs.items()}
                write(hroot/'downloads'/(scene+'.json'),dict(scene=scene,downloads=downloads,prelocked_plan=plan['identity']))
            except Exception as exc:
                write(path,dict(status='PARTIAL_CONFIRMATION_BLOCKED',scene=scene,error=repr(exc),missing_outcomes=None,
                     selected_names=plan['selected'],no_outcome_based_replacement=True,elapsed_seconds=time.perf_counter()-started))
                raise
        selected.append(dict(scene=scene,family=scene.split('_')[0],files=files))
    for n in ('class_split.json',):shutil.copyfile(root/n,hroot/n)
    tsv=Path(binding['prior_data_root'])/'scannetv2-labels.combined.tsv';dest=hroot/'data/scannet'/tsv.name
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(tsv,dest)
    shadow={**binding,'output_root':str(hroot)}
    # Same production generator and camera registration, with explicit new H2 role.
    manifests=[prepare_family(shadow,s,'H2') for s in selected]
    items=[o for m in manifests for o in m['objects']]
    feature_path=hroot/'features.json'
    if feature_path.exists():features=verified(feature_path)
    else:
        frames={}
        before=fc.images
        for obj in items:
            for f in obj['views']:
                key=obj['scene']+':'+str(f['frame_id'])
                if key not in frames:frames[key]=fc.capture(f)
        if len(frames)>384 or fc.images-before>384:raise ValueError('H2 FC image cap exceeded')
        features=write(feature_path,dict(status='COMPLETE',frames=frames,physical_images=len(frames),
                       physical_encodings=fc.images-before,raw_C=1536,projected_D=768))
    results={};loader=ObjectLoader(fc,features)
    if items:
        fc_row=evaluate(None,loader,items,fc,base_only=False,save_path=hroot/'recognition/FC8.json');results['FC8']=fc_row['identity']
        for name,row in lock['heads'].items():
            head,_=load_head(row['checkpoint']['path'],fc.device)
            result=evaluate(head,loader,items,fc,base_only=False,reference=[r for r in fc_row['records'] if r['base']],save_path=hroot/'recognition'/(name+'.json'))
            results[name]=result['identity']
    novel=[o for o in items if not o['base']]
    index.write_memo(root/'holdout2/verifications.json')
    return write(path,dict(status='COMPLETE' if len(items)>=32 else 'INSUFFICIENT_H2_SUPPORT',predictor_lock=lock['identity'],
          plan=plan['identity'],selected_names=plan['selected'],family_manifests=[m['identity'] for m in manifests],
          original_objects=len(items),variant_records=4*len(items),families=len(manifests),recognition=results,
          novel_original_objects=len(novel),novel_families=len({o['family'] for o in novel}),
          novel_status='SUFFICIENT' if len(novel)>=20 and len({o['family'] for o in novel})>=2 else 'INSUFFICIENT_NOVEL_SUPPORT',
          proposal_recognition_only=True,independent_whole_map_AP=False,features=features['identity'],elapsed_seconds=time.perf_counter()-started))
