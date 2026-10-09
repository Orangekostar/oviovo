"""Read-only full-parent adapter and explicit predictor/evaluation separation."""
from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

from static_ovmap.source_preserving_update.binding import load_scene as parent_scene

from .common import BASELINES,REPO,ConsumptionIndex,PathResolver,read,verified,write


def predictor_namespace(value):
    # No target labels, projection arrays, parent selected units or evaluator rows.
    frame_keys=('frame_id','pose_c2w','intrinsics','image_size_hw','rgb_path','rgb_sha256','depth_path','depth_sha256')
    capture={'identity':value.capture['identity'],
             'frames':[{k:f[k] for k in frame_keys if k in f} for f in value.capture['frames']]}
    return SimpleNamespace(scene=value.scene,xyz=value.xyz,faces=value.faces,raw=value.raw,
        owner_ids=value.g1.owner_ids,semantic_labels=value.g1.semantic_labels,
        d2_owner_ids=value.d2.owner_ids,geometry=value.g1.geometry,sources=value.sources,
        probabilities=value.probabilities,valid_ids=value.valid_ids,
        temperatures=value.data['temperatures'],model_identity=value.data['FC_physical_model_identity'],
        capture=capture,capture_path=value.capture_path,index=value.index)


def bind(spec_path,parent_root,output_root,*,storage_root=None,path_map=None,gpu=None):
    spec_path=Path(spec_path).resolve();spec=read(spec_path)
    if spec_path.read_bytes()!=(REPO/'docs/paper/static_ovmap/disagreement_query_v1/spec/PROTOCOL_SPEC.json').read_bytes():
        raise ValueError('fixed specification must remain verbatim')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch']:raise ValueError('incorrect isolated task branch')
    subprocess.run(['git','merge-base','--is-ancestor',spec['base_commit'],'HEAD'],cwd=REPO,check=True)
    mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping);parent=Path(resolver.resolve(parent_root)).resolve()
    logical=Path(output_root).absolute();root=Path(storage_root).resolve() if storage_root else logical.resolve()
    if root==parent or root.is_relative_to(parent):raise ValueError('parent is read-only')
    root.mkdir(parents=True,exist_ok=True)
    if storage_root and logical.resolve()!=root:
        if logical.exists() or logical.is_symlink():raise ValueError('output relocation differs')
        logical.parent.mkdir(parents=True,exist_ok=True);logical.symlink_to(root,target_is_directory=True)
    index=ConsumptionIndex(root/'input_verifications.json');docs={}
    for name in ('source_binding.json','result_store.json','publication/final.json'):
        index.identity(parent/name);docs[name]=verified(parent/name)
    source,store,pub=[docs[k] for k in ('source_binding.json','result_store.json','publication/final.json')]
    release=spec['parent_full_release_commit']
    if (pub['status']!='PUSH_VERIFIED' or pub['local_HEAD']!=release or pub['remote_HEAD']!=release
        or pub['full_resume_exit_code']!=0 or pub['result_store_identity']!=store['identity']):
        raise ValueError('required full-parent publication is not verified')
    if store['status']!='SCIENCE_COMPLETE' or store['scene_method_coverage']!=234 or store['full_cohort_pool_coverage']!=18:
        raise ValueError('incomplete full-parent coverage')
    inherited=resolver.rewrite(source);rows={m:{} for m in BASELINES.values()};scenes={}
    for cohort,names in spec['cohorts'].items():
        for scene in names:
            path=parent/'predictions'/scene/'receipt.json';index.identity(path);lock=verified(path)
            if lock['status']!='PREDICTIONS_LOCKED':raise ValueError('parent prediction not locked')
            scenes[scene]={**inherited['scenes'][scene],'cohort':cohort,
                'source_update_prediction_receipt':index.identity(path)}
            for method in rows:
                row=next(r for r in store['scene_metrics'] if r['scene']==scene and r['method']==method)
                if row['status']!='COMPLETE' or lock['methods'][method]['prediction_key']!=row['prediction_identity']:
                    raise ValueError('parent baseline prediction and score differ')
                rows[method][scene]=resolver.rewrite(row)
    value={k:inherited[k] for k in ('reference','assets','FC_python')}
    value.update(status='BOUND_ACTUAL_PARENT',protocol=spec['task_id'],spec=index.identity(spec_path),
        specification=spec,cohorts=spec['cohorts'],screen_cohorts=spec['screen_cohorts'],scenes=scenes,
        baseline_rows=rows,baseline_lineage={'DQ00_D2':['SU00_D2','IR00_D2'],'DQ01_G1':['SU01_G1','IR01_G1']},
        path_map=mapping,branch=branch,base_commit=spec['base_commit'],parent_root=str(parent),
        minimal_root=inherited['parent_root'],logical_root=str(logical),output_root=str(root),
        parent_source_binding=index.identity(parent/'source_binding.json'),
        parent_result_store=index.identity(parent/'result_store.json'),parent_publication=index.identity(parent/'publication/final.json'),
        parent_identity=source['identity'],parent_store_identity=store['identity'],
        gpu=str(gpu if gpu is not None else inherited['gpu']),new_maps=0,GT_used_by_predictor=False,deployment='N0_UNCHANGED')
    from static_ovmap.minimal_instance_repair.binding import seal
    value=seal(value);dest=root/'source_binding.json'
    if dest.exists() and verified(dest)!=value:raise ValueError('bound inputs changed')
    if not dest.exists():write(dest,value)
    if not (root/'storage_preflight.json').exists():
        free=shutil.disk_usage(root).free/2**30
        if free<20:raise OSError('at least 20 GiB required')
        write(root/'storage_preflight.json',{'free_GiB':free,'minimum_GiB':20,'root':str(root)})
    index.write_memo(root/'input_verifications.json');_BINDINGS[value['identity']]=value
    return value


_BINDINGS={}


def load_binding(root):
    value=verified(Path(root).resolve()/'source_binding.json')
    index=ConsumptionIndex(Path(value['output_root'])/'input_verifications.json')
    for name in ('spec','parent_source_binding','parent_result_store','parent_publication'):
        index.identity(value[name]['path'],value[name])
    index.write_memo(Path(value['output_root'])/'input_verifications.json')
    _BINDINGS[value['identity']]=value;return value


@lru_cache(maxsize=1)
def _scene(identity,scene):
    original=parent_scene(_BINDINGS[identity],scene)
    return predictor_namespace(original),original


def load_scene(binding,scene):
    if scene not in binding['scenes']:raise ValueError('scene outside fixed full cohort')
    _BINDINGS[binding['identity']]=binding
    return _scene(binding['identity'],scene)
