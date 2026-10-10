"""Conditional stage graph; no missing required stage becomes a no-op result."""
from pathlib import Path
from .common import ConsumptionIndex,canonical_digest,objects,verified,write

def lock(binding):
    root=Path(binding['output_root']);main=verified(root/'R_nomination_seed17.json');repeat=verified(root/'R_nomination_seed29.json')
    ready=main['Rstar'] is not None and repeat['Rstar'] is not None
    index=ConsumptionIndex(root/'training/verifications.json');heads={}
    for seed,nom in ((17,main),(29,repeat)):
        for branch,checkpoint in nom.get('checkpoints',{}).items():
            index.identity(checkpoint['path'],checkpoint)
            heads[branch if seed==17 else 'R29_'+branch]=dict(seed=seed,branch=branch,checkpoint=checkpoint)
    nominations={};map_heads={}
    if ready:
        if main['Rstar']!=repeat['Rstar']:raise ValueError('Repeat R architecture differs')
        for seed in (17,29):
            nom=verified(root/f'G_nomination_seed{seed}.json');nominations[str(seed)]=nom
            prefix='' if seed==17 else 'R29_'
            map_heads[prefix+'PR_RSTAR']=prefix+main['Rstar']
            for branch,checkpoint in nom['checkpoints'].items():
                index.identity(checkpoint['path'],checkpoint)
                heads[prefix+branch]=dict(seed=seed,branch=branch,checkpoint=checkpoint)
                map_heads[prefix+branch]=prefix+branch
        if nominations['17']['architecture']!=nominations['29']['architecture']:raise ValueError('Repeat G architecture switched')
    status='PREDICTORS_LOCKED' if ready else 'COMPLETE_NO_2D_FOUNDATION' if main['Rstar'] is None else 'COMPLETE_2D_NOT_REPEATED'
    value=dict(status=status,heads=heads,map_heads=map_heads,Rstar=main['Rstar'],G_nominations=nominations,
               R_nominations={'17':main['identity'],'29':repeat['identity']},H2_plan=verified(root/'H2_plan.json')['identity'],
               activated_full_path=ready,old_H_role='EXPOSED_POST_NOMINATION_REGRESSION_ONLY',
               new_maps_expected=390 if ready else 0,new_pools_expected=30 if ready else 0,
               DEV_nominee=nominations['17']['architecture'] if ready else main['Rstar'],deployment='N0_UNCHANGED')
    path=root/'predictor_lock.json'
    if path.exists():
        old=verified(path)
        if {k:v for k,v in old.items() if k!='identity'}!=value:raise ValueError('Frozen nominations changed')
        return old
    return write(path,value)

def diagnose(binding,fc,loader):
    from .data import ObjectLoader
    from .recognition import evaluate,load_head
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);locked=verified(root/'predictor_lock.json')
    path=root/'diagnostics/receipt.json'
    if path.exists():
        old=verified(path)
        if old['predictor_lock']!=locked['identity']:raise ValueError('Diagnostic predictor lock changed')
        return old
    old_loader=ObjectLoader(fc,verified(parent/'features/holdout.json'));items=objects(binding,'holdout')
    baseline=evaluate(None,old_loader,items,fc,base_only=False,save_path=root/'recognition/old_H/FC8.json')
    rows={'FC8':baseline['identity']};heads={}
    for name,row in locked['heads'].items():
        h,_=load_head(row['checkpoint']['path'],fc.device);heads[name]=h
        r=evaluate(h,old_loader,items,fc,base_only=False,reference=[r for r in baseline['records'] if r['base']],
                   save_path=root/'recognition/old_H'/(name+'.json'))
        rows[name]=r['identity'];print('OLD_H',name,r['metrics']['A'],flush=True)
    # Separate phases share the 32 GiB host grid budget, rather than retaining both banks.
    old_loader.cache.clear();old_loader.bytes=0
    strata={}
    for role in ('train','dev'):
        role_items=objects(binding,role)
        baseline_path=root/'recognition'/role/'FC8.json'
        reference=evaluate(None,loader,role_items,fc,base_only=False,save_path=baseline_path)
        strata[role]={'FC8':reference['identity']}
        for name,head in heads.items():
            dest=root/'recognition'/role/(name+'.json')
            r=verified(dest) if dest.exists() else evaluate(head,loader,role_items,fc,
                  base_only=False,reference=[r for r in reference['records'] if r['base']],save_path=dest)
            strata[role][name]=r['identity']
            print('POST_LOCK_RECOGNITION',role,name,r['metrics']['A'],flush=True)
    from .mechanisms import interventions
    details=interventions(binding,fc,loader,heads)
    from .export import run
    exported=run(binding,fc,loader,heads)
    return write(path,dict(status='COMPLETE',predictor_lock=locked['identity'],old_H=rows,TRAIN_DEV_strata=strata,mechanisms=details,
              old_H_original_objects=len(items),old_H_exposed=True,used_for_selection=False,
              export_roundtrip=exported['identity'],real_proposals='ACTIVATED' if locked['activated_full_path'] else 'NOT_TRIGGERED_NO_FOUNDATION'))

def gpu_dispatch(binding,phase,fc,loader,*,sam2_root=None,sam2_python=None):
    root=Path(binding['output_root'])
    if phase in ('train_G','repeat_G'):
        from .geometry_training import train_G
        return train_G(binding,fc,loader,17 if phase=='train_G' else 29)
    if phase=='diagnose':
        result=diagnose(binding,fc,loader)
        if verified(root/'predictor_lock.json')['activated_full_path']:
            loader.cache.clear();loader.bytes=0
            from .real_proposals import run
            run(binding,fc,sam2_root=sam2_root,sam2_python=sam2_python)
        return result
    if phase=='holdout2':
        from .holdout2 import run
        return run(binding,fc)
    if phase=='predict':
        from .prediction import run
        return run(binding,fc)
    raise ValueError('Unknown GPU stage '+phase)
