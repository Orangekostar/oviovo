"""Task-owned fixed nomination lock and post-lock diagnostics on every completed path."""
from pathlib import Path
from .common import ConsumptionIndex,immutable_write,objects,verified,write
from .selection import path_counts

def lock(binding):
    root=Path(binding['output_root']);main=verified(root/'nomination_seed17.json');repeat=verified(root/'nomination_seed29.json')
    nominee=main['nominee'];ready=bool(nominee is not None and repeat['same_nominee_qualified'])
    if nominee is not None and repeat['nominee']!=nominee:raise ValueError('Repeat architecture switched')
    index=ConsumptionIndex(root/'training/verifications.json');heads={}
    for seed,nom in ((17,main),(29,repeat)):
        expected=4 if seed==17 or nominee is not None else 0
        if len(nom['checkpoints'])!=expected:raise ValueError('Activated arm coverage incomplete')
        for branch,item in nom['checkpoints'].items():
            index.identity(item['path'],item)
            r=verified(root/f'training/seed{seed}/{branch}/receipt.json')
            if r['status']!='COMPLETE' or r['completed_steps']!=2000:raise ValueError('Incomplete selected training arm')
            name=branch if seed==17 else 'S29_'+branch
            heads[name]=dict(seed=seed,branch=branch,checkpoint=item,selected_step=r['selected_step'],config=r['config'],
                scientific_input_identity=r['input_identity'],qualified=r['qualified'])
    updates,rows,pools=path_counts(nominee,ready)
    value=dict(status='PREDICTORS_LOCKED',source_binding=binding['identity'],heads=heads,
        map_heads={k:k for k in heads} if ready else {},DEV_nominee=nominee,
        nominations={'17':main['identity'],'29':repeat['identity']},repeat_activated=nominee is not None,
        activated_full_path=ready,scientific_updates_expected=updates,new_maps_expected=rows,new_pools_expected=pools,
        H2_plan=verified(root/'H2_plan.json')['identity'],old_H_role='EXPOSED_POST_NOMINATION_REGRESSION_ONLY',
        enabled_diagnostics=binding['specification']['diagnostics'],MA_dependency=False,R_dependency=False,deployment='N0_UNCHANGED')
    return immutable_write(root/'predictor_lock.json',value)

def require_transfer(binding):
    root=Path(binding['output_root']);locked=verified(root/'predictor_lock.json')
    if not locked['activated_full_path']:raise ValueError('Confirmation requires the same DEV nominee to qualify twice')
    for seed in (17,29):
        nom=verified(root/f'nomination_seed{seed}.json')
        if nom['identity']!=locked['nominations'][str(seed)] or not nom['qualified'][locked['DEV_nominee']]:raise ValueError('Frozen nomination changed')
    return locked

def diagnose(binding,fc,loader):
    from .data import CachedObjectLoader
    from .recognition import evaluate,load_head
    from .mechanisms import Collector
    from .diagnostics import summarize_rows
    from .export import run as export
    root=Path(binding['output_root']);parent=Path(binding['lr_parent_root']);locked=verified(root/'predictor_lock.json')
    path=root/'diagnostics/receipt.json'
    if path.exists():
        old=verified(path)
        if old['predictor_lock']!=locked['identity']:raise ValueError('Diagnostic predictor lock changed')
        return old
    heads={n:load_head(r['checkpoint']['path'],fc.device)[0] for n,r in locked['heads'].items()}
    rows={};mechanisms={};summary=[]
    for role in ('train','dev','old_H'):
        items=objects(binding,role)
        if role=='old_H':
            loader.clear()
            current=CachedObjectLoader(fc,verified(parent/'features/holdout.json'))
        else:current=loader
        basepath=root/'recognition'/role/'FC8.json'
        baseline=verified(basepath) if basepath.exists() else evaluate(None,current,items,fc,base_only=False,save_path=basepath)
        rows[role]={'FC8':baseline['identity']};methods={'FC8':None,**heads}
        for name,head in methods.items():
            destination=root/'recognition'/role/(name+'.json');diag=root/'diagnostics'/role/name
            if head is None:value=baseline
            elif destination.exists() and (diag/'receipt.json').exists():
                value=verified(destination);mechanisms[f'{role}/{name}']=verified(diag/'receipt.json')['identity']
            else:
                collector=Collector(binding,fc,current,role,name,base_only=role!='old_H')
                value=evaluate(head,current,items,fc,base_only=False,reference=[r for r in baseline['records'] if r['base']],
                               save_path=destination,observer=collector)
                mechanisms[f'{role}/{name}']=collector.finish(diag)['identity']
            rows[role][name]=value['identity'];flatten=[];ref={r['key']:r for r in baseline['records']}
            for r in value['records']:
                for c,p in enumerate(r['predictions']):
                    old=ref[r['key']]['predictions'][c];correct=int(p==r['class_id'])
                    flatten.append(dict(key=r['key'],family=r['family'],class_id=r['class_id'],base=r['base'],condition=c,
                        method=name,correct=correct,NLL=r['NLL'][c],correction=int(old!=r['class_id'] and correct),harm=int(old==r['class_id'] and not correct)))
            summary += [dict(role=role,**r) for r in summarize_rows(flatten)]
            print('POST_LOCK',role,name,value['metrics']['A'],flush=True)
        if role=='old_H':current.clear()
    # Reuse training banks for exact export reload, with selected weights for every trained arm.
    exported=export(binding,fc,loader,heads)
    write(root/'diagnostics/recognition_summary.json',dict(status='COMPLETE',rows=summary,used_for_selection=False))
    return write(path,dict(status='COMPLETE',predictor_lock=locked['identity'],recognition=rows,mechanisms=mechanisms,
        export_roundtrip=exported['identity'],old_H_original_objects=101,old_H_exposed=True,used_for_selection=False))
