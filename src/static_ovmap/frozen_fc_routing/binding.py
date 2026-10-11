"""Bind immutable PR -> LR -> source-update artifacts without old task binders."""
import copy
import shutil
import subprocess
from pathlib import Path
from .common import REPO,ConsumptionIndex,PathResolver,fixed_spec,verified,read,write,immutable_write
from static_ovmap.learned_object_readout.binding import load_scene as numerical_scene

def validate_lineage(spec,pr,pr_pub,lr,lr_pub,split):
    lin=spec['lineage']
    for store,pub,commit,exit_field in ((pr,pr_pub,lin['pr_commit'],'all_exit_code'),(lr,lr_pub,lin['lr_commit'],'full_resume_exit_code')):
        if not (pub['status']=='PUSH_VERIFIED' and pub['local_HEAD']==pub['remote_HEAD']==commit and
                pub['result_store_identity']==store['identity'] and pub[exit_field]==0):
            raise ValueError('Published parent result/commit/exit mismatch')
    if (pr['status']!=lin['pr_terminal'] or pr['scientific_optimizer_updates']!=8000 or
        pr['scene_method_coverage']!=0 or pr['full_cohort_pool_coverage']!=0):raise ValueError('PR terminal identity differs')
    if (lr['scene_method_coverage']!=286 or lr['full_cohort_pool_coverage']!=22 or len(lr['scene_metrics'])!=286 or
        sum(map(len,lr['pooled_metrics'].values()))!=22):raise ValueError('LR complete coverage differs')
    if split['identity']!=lin['parent_split_identity']:raise ValueError('Original split changed')

def load_scene(binding,scene):
    rows=binding['baseline_rows']
    inherited=dict(binding,baseline_rows=dict(rows,SU00_D2=rows['REF_D2'],SU01_G1=rows['REF_G1']))
    return numerical_scene(inherited,scene)

def bind(spec_path,parent_root,output_root,*,storage_root=None,path_map=None,gpu=None,data_roots=()):
    spec=fixed_spec(spec_path);mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping);parent=Path(resolver.resolve(parent_root)).resolve()
    root=Path(storage_root or output_root).resolve();logical=Path(output_root).absolute()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch']:raise ValueError('Incorrect task branch')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if remote not in ('git@github.com:Orangekostar/oviovo.git','https://github.com/Orangekostar/oviovo.git'):raise ValueError('Incorrect task origin')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    source=resolver.rewrite(verified(parent/'source_binding.json'));lrroot=Path(source['lr_parent_root']).resolve()
    for old in (parent,lrroot,Path(source['lr_source_root']).resolve()):
        if root==old or root.is_relative_to(old):raise ValueError('Parent artifacts are read-only')
    root.mkdir(parents=True,exist_ok=True)
    if root!=logical.resolve():
        if logical.exists() or logical.is_symlink():raise ValueError('Output relocation differs')
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(root,target_is_directory=True)
    index=ConsumptionIndex(root/'input_verifications.json')
    names=['source_binding.json','result_store.json','publication/final.json','execution_observed.json',
           'split_manifest.json','class_split.json','H2_plan.json','predictor_lock.json']
    pr={n:verified(parent/n) for n in names};prids={n:index.identity(parent/n) for n in names}
    lrnames=['source_binding.json','result_store.json','publication/final.json','execution_observed.json',
             'split_manifest.json','class_split.json','data_profile.json','features/train-dev.json','features/holdout.json',
             'features/regression.json','regression/manifest.json','training/seed17/receipt.json','training/seed29/receipt.json']
    lr={n:verified(lrroot/n) for n in lrnames};lrids={n:index.identity(lrroot/n) for n in lrnames}
    validate_lineage(spec,pr['result_store.json'],pr['publication/final.json'],lr['result_store.json'],lr['publication/final.json'],lr['split_manifest.json'])
    if any(d['execution_observed.json']['exit_code']!=0 or d['execution_observed.json']['result_store_identity']!=d['result_store.json']['identity'] for d in (pr,lr)):
        raise ValueError('Actual parent exit/result identity differs')
    if source['parent_identity']!=lr['source_binding.json']['identity'] or source['parent_store_identity']!=lr['result_store.json']['identity']:
        raise ValueError('PR -> LR edge changed')
    if source['lr_source_root']!=resolver.rewrite(lr['source_binding.json'])['parent_root']:raise ValueError('LR -> source-update edge changed')
    if sum(lr[f'training/seed{s}/receipt.json']['scientific_updates'] for s in (17,29))!=16000:raise ValueError('Parent LR update count changed')
    if source['cohorts']!=spec['cohorts']:raise ValueError('Ordered cohorts changed')
    for n in ('split_manifest.json','class_split.json','data_profile.json','features/train-dev.json'):
        dest=root/n;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=(lrroot/n).read_bytes():raise ValueError('Exact copied LR manifest differs')
        if not dest.exists():shutil.copyfile(lrroot/n,dest)
    for relative,target in (('external/upstream_sources',lrroot/'external/upstream_sources'),('data/generated',lrroot/'data/generated')):
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.resolve()!=target.resolve():raise ValueError('Inherited source link changed')
        if not dest.exists():dest.symlink_to(target,target_is_directory=True)
    memo=root/'features/verifications.json'
    if not memo.exists():shutil.copyfile(parent/'features/verifications.json',memo)
    b=copy.deepcopy(source);b.pop('identity',None)
    for k in ('parent_local_head','parent_clean'):b.pop(k,None)
    b.update(protocol=spec['task_id'],specification=spec,branch=branch,base_commit=spec['base_commit'],
             spec=index.identity(spec_path),output_root=str(root),logical_root=str(logical),pr_parent_root=str(parent),
             lr_parent_root=str(lrroot),path_map=mapping,gpu=str(source['gpu'] if gpu is None else gpu),
             parent_identity=pr['source_binding.json']['identity'],parent_store_identity=pr['result_store.json']['identity'],
             parent_release_commit=pr['publication/final.json']['local_HEAD'],bound_PR=prids,bound_LR=lrids,
             parent_scientific_updates=dict(PR=8000,LR=16000),new_maps=0,deployment='N0_UNCHANGED',
             R_qualification_required=False,MA_inference_dependency=False)
    b['data_roots']=list(dict.fromkeys([*map(str,data_roots),*source['data_roots']]))
    b['baseline_rows']={m:{r['scene']:resolver.rewrite(r) for r in lr['result_store.json']['scene_metrics'] if r['method']==old}
                        for m,old in spec['map_baselines'].items()}
    if any(len(v)!=26 for v in b['baseline_rows'].values()):raise ValueError('Actual LR baseline coverage incomplete')
    if (root/'source_binding.json').exists():
        old=verified(root/'source_binding.json')
        if Path(old['logical_root']).resolve()==root:b['logical_root']=old['logical_root']
    else:
        free=shutil.disk_usage(root).free/2**30
        if free<40:raise OSError('Initial free disk below40GiB')
        write(root/'storage_preflight.json',dict(free_GiB=free,minimum_GiB=40))
    result=immutable_write(root/'source_binding.json',b)
    write(root/'dependency_manifest.json',dict(PR=prids,LR=lrids,FC=source['reference']['fc'],
          source_binding=result['identity'],controller_python=spec['controller_python_hint'],FC_python=b['FC_python'],
          old_checkout_HEAD_required=False,MA_inference_dependency=False))
    immutable_write(root/'run_registry.json',dict(status='BOUND',protocol=spec['task_id'],
          source_binding=result['identity'],spec=b['spec'],branch=branch,base_commit=spec['base_commit'],
          origin=remote,worktree=str(REPO),output_root=str(root),logical_root=b['logical_root'],
          controller_python=spec['controller_python_hint'],FC_python=b['FC_python'],GPU=b['gpu'],
          phases=spec['phases'],deployment='N0_UNCHANGED',publication_receipt='publication/final.json'))
    index.write_memo(root/'input_verifications.json');return result
