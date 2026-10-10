"""Bind real completed LR inputs without invoking its older experiment binder."""
import copy
import re
import shutil
import subprocess
from pathlib import Path
from .common import REPO,ConsumptionIndex,PathResolver,fixed_spec,read,verified,write
from static_ovmap.learned_object_readout.binding import load_scene as parent_load_scene

def load_scene(binding,scene):
    # The inherited source loader validates the same payloads under its old keys.
    rows=binding['baseline_rows']
    inherited=dict(binding,baseline_rows=dict(rows,SU00_D2=rows['PR00_D2'],SU01_G1=rows['PR01_G1']))
    return parent_load_scene(inherited,scene)

def bind(spec_path,parent_root,output_root,*,storage_root=None,path_map=None,gpu=None,data_roots=(),**unused):
    spec=fixed_spec(spec_path);mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping);parent=Path(resolver.resolve(parent_root)).resolve()
    root=Path(storage_root or output_root).resolve();logical=Path(output_root).absolute()
    if root==parent or root.is_relative_to(parent):raise ValueError('LR parent is read-only')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch']:raise ValueError('Task branch mismatch')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if remote not in ('git@github.com:Orangekostar/oviovo.git','https://github.com/Orangekostar/oviovo.git'):
        raise ValueError('Task repository origin mismatch')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    root.mkdir(parents=True,exist_ok=True)
    if root!=logical.resolve():
        if logical.exists() or logical.is_symlink():raise ValueError('Output relocation changed')
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(root,target_is_directory=True)
    index=ConsumptionIndex(parent/'input_verifications.json')
    names=['source_binding.json','result_store.json','split_manifest.json','class_split.json','data_profile.json',
           'features/train-dev.json','features/holdout.json','features/regression.json','regression/manifest.json',
           'dev_nomination.json','repeat_nomination.json','publication/final.json','execution_observed.json',
           'training/seed17/receipt.json','training/seed29/receipt.json']
    bound={n:verified(parent/n) for n in names};identities={n:index.identity(parent/n) for n in names}
    source=resolver.rewrite(bound['source_binding.json']);store=bound['result_store.json'];pub=bound['publication/final.json']
    if (store['scene_method_coverage']!=286 or store['full_cohort_pool_coverage']!=22 or
        len(store['scene_metrics'])!=286 or pub['status']!='PUSH_VERIFIED' or
        pub['local_HEAD']!=pub['remote_HEAD'] or pub['result_store_identity']!=store['identity'] or
        pub['full_resume_exit_code']!=0 or bound['execution_observed.json']['exit_code']!=0):
        raise ValueError('Actual completed LR coverage/publication is incomplete')
    if pub['local_HEAD']!=spec['base_commit']:raise ValueError('Unexpected parent publication; bind exact scientific identities explicitly')
    if bound['split_manifest.json']['identity']!=spec['data']['parent_split_identity']:raise ValueError('Split identity differs')
    if source['cohorts']!=spec['cohorts']:raise ValueError('Full ordered cohorts changed')
    if sum(map(len,store['pooled_metrics'].values()))!=22:
        raise ValueError('Actual parent ordered pools incomplete')
    updates={str(seed):bound[f'training/seed{seed}/receipt.json']['scientific_updates'] for seed in (17,29)}
    if updates!={'17':10000,'29':6000}:raise ValueError('Parent successful scientific updates differ')
    parent_wt=parent.parent/'worktree'
    cleanliness=subprocess.check_output(['git','status','--porcelain'],cwd=parent_wt,text=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=parent_wt,text=True).strip()
    if head!=spec['base_commit'] or cleanliness:raise ValueError('Parent local checkout differs or is dirty')
    for name in ('split_manifest.json','class_split.json','data_profile.json','features/train-dev.json'):
        dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=(parent/name).read_bytes():raise ValueError('Exact copied parent manifest differs')
        if not dest.exists():shutil.copyfile(parent/name,dest)
    ext=root/'external/upstream_sources';ext.parent.mkdir(parents=True,exist_ok=True)
    if not ext.exists():ext.symlink_to(parent/'external/upstream_sources',target_is_directory=True)
    generated=root/'data/generated';generated.parent.mkdir(parents=True,exist_ok=True)
    if not generated.exists():generated.symlink_to(parent/'data/generated',target_is_directory=True)
    # Existing validated filesystem stamps permit hash-once reuse of immutable grids.
    memo=parent/'features/verifications.json';dest=root/'features/verifications.json'
    if memo.exists() and not dest.exists():shutil.copyfile(memo,dest)
    b=copy.deepcopy(source);b.pop('identity',None)
    b['parent_spec']=b['spec'];b['spec']=index.identity(spec_path)
    logical_name=str(logical)
    if (root/'source_binding.json').exists():
        prior=verified(root/'source_binding.json')
        if Path(prior['logical_root']).resolve()==root:logical_name=prior['logical_root']
    b.update(protocol=spec['task_id'],specification=spec,branch=branch,base_commit=spec['base_commit'],
             output_root=str(root),logical_root=logical_name,lr_parent_root=str(parent),
             lr_source_root=source['parent_root'],path_map=mapping,gpu=str(gpu if gpu is not None else source['gpu']),
             parent_identity=bound['source_binding.json']['identity'],parent_store_identity=store['identity'],
             parent_release_commit=pub['local_HEAD'],bound_parent=identities,parent_scientific_updates=updates,
             parent_local_head=head,parent_clean=True,new_maps=0,deployment='N0_UNCHANGED')
    b['data_roots']=list(dict.fromkeys([*map(str,data_roots),*source['data_roots']]))
    b['baseline_rows']={m:{r['scene']:resolver.rewrite(r) for r in store['scene_metrics'] if r['method']==old}
                        for m,old in spec['baseline_methods'].items()}
    path=root/'source_binding.json'
    if path.exists():
        previous=verified(path)
        if {k:v for k,v in previous.items() if k!='identity'}!=b:raise ValueError('Task binding changed')
        result=previous
    else:
        available=shutil.disk_usage(root).free/2**30
        if available<40:raise OSError('Initial free space below40GiB')
        write(root/'storage_preflight.json',dict(free_GiB=available,minimum_GiB=40))
        result=write(path,b)
    write(root/'dependency_manifest.json',dict(parent=identities,FC=source['reference']['fc'],
          source_binding=result['identity'],controller_python=spec['controller_python_hint'],FC_python=b['FC_python'],
          parent_clean=True,parent_scientific_updates=updates))
    write(root/'method_registry.json',dict(R=spec['R_methods'],G=spec['G_methods'],baselines=spec['baseline_methods'],
          aliases={'PR_G0':'same-seed PR_RSTAR'},deployment='N0_UNCHANGED'))
    index.write_memo(root/'input_verifications.json')
    return result
