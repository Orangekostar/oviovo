"""Read-only four-scene adapter for the published source-update lineage."""

from functools import lru_cache
from pathlib import Path
import subprocess
import time

from static_ovmap.source_preserving_update.binding import load_scene as parent_scene

from .common import (COHORTS, REPO, ConsumptionIndex, PathResolver, canonical_digest,
                     free_space, read, verified, write)


def bind(spec_path, parent_root, output_root, *, storage_root=None, path_map=None, gpu=None):
    spec_path=Path(spec_path).resolve();spec=read(spec_path)
    if spec_path.read_bytes()!=(REPO/'docs/paper/static_ovmap/samv_local_probe_v1/spec/PROTOCOL_SPEC.json').read_bytes():
        raise ValueError('the fixed numerical protocol must remain verbatim')
    if spec['cohorts']!=COHORTS:raise ValueError('incorrect four-scene scope')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch']:raise ValueError('incorrect task branch')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping);parent=Path(resolver.resolve(parent_root)).resolve()
    logical=Path(output_root).absolute();root=Path(storage_root).resolve() if storage_root else logical.resolve()
    if root==parent or root.is_relative_to(parent):raise ValueError('parent store is read-only')
    root.mkdir(parents=True,exist_ok=True);free_space(root)
    if storage_root and logical.resolve()!=root:
        if logical.exists() or logical.is_symlink():raise ValueError('existing logical root points elsewhere')
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(root,target_is_directory=True)
    index=ConsumptionIndex(root/'input_verifications.json')
    docs={}
    for name in ('source_binding.json','result_store.json','publication/final.json'):
        index.identity(parent/name);docs[name]=verified(parent/name)
    source,store,publication=[docs[n] for n in ('source_binding.json','result_store.json','publication/final.json')]
    if (publication['status']!='PUSH_VERIFIED' or publication['local_HEAD']!=spec['base_commit']
            or publication['remote_HEAD']!=spec['base_commit'] or publication['full_resume_exit_code']!=0
            or publication['result_store_identity']!=store['identity']):
        raise ValueError('required published parent lineage is not established')
    if store['status']!='SCIENCE_COMPLETE' or store['scene_method_coverage']!=234 or store['full_cohort_pool_coverage']!=18:
        raise ValueError('published parent is incomplete')
    inherited=resolver.rewrite(source);names=[s for v in COHORTS.values() for s in v]
    rows={m:{} for m in ('SU00_D2','SU01_G1')};scenes={}
    for cohort,subset in COHORTS.items():
        for scene in subset:
            index.identity(parent/'predictions'/scene/'receipt.json')
            lock=verified(parent/'predictions'/scene/'receipt.json')
            if lock['status']!='PREDICTIONS_LOCKED':raise ValueError('parent predictions are not locked')
            scenes[scene]={**inherited['scenes'][scene],'cohort':cohort,
                'source_update_prediction_receipt':index.identity(parent/'predictions'/scene/'receipt.json')}
            for method in rows:
                row=next(r for r in store['scene_metrics'] if r['scene']==scene and r['method']==method)
                if row['status']!='COMPLETE' or lock['methods'][method]['prediction_key']!=row['prediction_identity']:
                    raise ValueError('parent baseline prediction/scoring identity differs')
                rows[method][scene]=resolver.rewrite(row)
    # The inherited narrow scene loader uses only these fields; no parent orchestration runs.
    value={k:inherited[k] for k in ('reference','assets','FC_python')}
    value.update(status='BOUND_ACTUAL_PARENT',protocol=spec['task_id'],spec=index.identity(spec_path),
        specification=spec,cohorts=COHORTS,scenes=scenes,baseline_rows=rows,path_map=mapping,
        branch=branch,base_commit=spec['base_commit'],parent_root=str(parent),
        minimal_root=inherited['parent_root'],logical_root=str(logical),output_root=str(root),
        parent_source_binding=index.identity(parent/'source_binding.json'),
        parent_result_store=index.identity(parent/'result_store.json'),
        parent_publication=index.identity(parent/'publication/final.json'),
        parent_identity=source['identity'],parent_store_identity=store['identity'],
        gpu=str(gpu if gpu is not None else inherited['gpu']),new_maps=0,
        GT_used_by_predictor=False,deployment='N0_UNCHANGED')
    value['reference']={**value['reference'],'contexts':{s:value['reference']['contexts'][s] for s in names}}
    from static_ovmap.minimal_instance_repair.binding import seal
    value=seal(value);dest=root/'source_binding.json'
    if dest.exists() and verified(dest)!=value:raise ValueError('changed bound inputs require scoped descendant invalidation')
    if not dest.exists():write(dest,value)
    index.write_memo(root/'input_verifications.json')
    _BINDINGS[value['identity']]=value
    return value


_BINDINGS={}


def load_binding(root):
    value=verified(Path(root).resolve()/'source_binding.json');index=ConsumptionIndex(Path(value['output_root'])/'input_verifications.json')
    for name in ('spec','parent_source_binding','parent_result_store','parent_publication'):
        index.identity(value[name]['path'],value[name])
    index.write_memo(Path(value['output_root'])/'input_verifications.json')
    _BINDINGS[value['identity']]=value;return value


@lru_cache(maxsize=4)
def _scene(identity,scene):
    return parent_scene(_BINDINGS[identity],scene)


def load_scene(binding,scene):
    if scene not in binding['scenes']:raise ValueError('scene outside fixed probe')
    _BINDINGS[binding['identity']]=binding;return _scene(binding['identity'],scene)


def assets(binding,assets_root=None,samv_python=None,sam2_python=None):
    root=Path(binding['output_root']);asset_root=Path(assets_root or root.parent/'assets').resolve()
    index=ConsumptionIndex(root/'asset_verifications.json');inventory=read(asset_root/'acquisition.json')
    index.identity(asset_root/'acquisition.json');code=inventory['code'];spec=binding['specification']['assets']
    heads={'SAM-V':spec['samv']['commit'],'SAM-V/submodules/sam-hq':spec['samv']['sam_hq_gitlink'],
           'SAM-V/submodules/vggt':spec['samv']['vggt_gitlink'],'SAM-V/submodules/sam2':spec['sam2']['commit']}
    for directory,expected in heads.items():
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=asset_root/directory,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=asset_root/directory,text=True)
        if actual!=expected or dirty:raise ValueError(f'asset source revision differs: {directory}')
    begin=time.perf_counter();weights={}
    for item in inventory['assets']:
        if item['status']!='success':raise ValueError('required checkpoint was not acquired')
        record=index.identity(asset_root/item['local_path'],{'sha256':item['sha256']})
        if record['bytes']!=item['size_bytes']:raise ValueError('checkpoint size mismatch')
        weights[item['id']]={**record,'source':item['source'],'license_refs':item['license_refs']}
    licences=[index.identity(asset_root/r['path'],r) for r in inventory['licenses_and_notices']]
    envs={'samv':str(Path(samv_python or root.parent/'envs/samv/bin/python').resolve()),
          'sam2':str(Path(sam2_python or root.parent/'envs/sam2/bin/python').resolve())}
    value={'status':'ASSETS_BOUND','asset_root':str(asset_root),'heads':heads,'weights':weights,
           'licenses':licences,'environments':envs,'acquisition':index.identity(asset_root/'acquisition.json'),
           'FC':binding['assets'],'weights_published':False,'hash_seconds':time.perf_counter()-begin}
    index.write_memo(root/'asset_verifications.json')
    dest=root/'model_assets.json'
    if dest.exists():
        old=verified(dest)
        # The measured verification wall is not a model/request dependency.
        def stable(v):
            result={k:item for k,item in v.items() if k not in ('identity','hash_seconds')}
            result['environments']={k:str(Path(p).resolve()) for k,p in result['environments'].items()}
            return result
        if canonical_digest(stable(old))!=canonical_digest(stable(value)):
            raise ValueError('actual asset identities changed; invalidate affected descendants')
        return old
    return write(dest,value)
