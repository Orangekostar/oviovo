"""Minimal read-only FG/LR binding, without any old binder or trainer."""
import copy
from pathlib import Path
import shutil
import subprocess
from .common import REPO, BASE, DependencyError, require, verified, write, read, PathResolver, ConsumptionIndex, digest_file

def minimal_fc_dependencies(source,resolver):
    """Follow the one context FrozenFC consumes; P3 contexts remain lazy."""
    scene=source['cohorts']['scannet_cf18'][0]; context_path=Path(source['scenes'][scene]['context']['path'])
    paths=[context_path,Path(source['FC_python'])]
    execution=source['assets']['execution_config']; assets=Path(execution['assets_root']); operators=Path(execution['fc']['operator_root'])
    paths.extend([assets/'ovrcoat/ovrcoat.pth',assets/'fc_frozen/model/open_clip_model.safetensors',operators/'ovrcoat/modeling/backbone/clip.py',operators/'ovrcoat/modeling/transformer_decoder/fcclip_transformer_decoder.py',operators/'ovrcoat/ovrcoat.py'])
    if context_path.exists():
        context=resolver.rewrite(verified(context_path))
        paths.extend([Path(context['config']),Path(context['FC_text']['path'])])
        model_root=context.get('FC_model_reference_root',context.get('fc_root'))
        if model_root is None: raise ValueError('Frozen FC context lacks model reference')
        receipt=Path(model_root)/'receipt.json'; paths.append(receipt)
        if receipt.exists():
            paths.extend(Path(resolver.resolve(row['path'])) for row in read(receipt)['inputs'] if '/assets/fc_frozen/' in row['path'])
    return paths

def preflight(spec,parent_root,output_root,path_map=None,gpu=None):
    root=Path(output_root).resolve(); mapping=read(path_map) if isinstance(path_map,(str,Path)) else (path_map or {})
    resolver=PathResolver(mapping); parent=Path(resolver.resolve(str(parent_root))).resolve()
    if root==parent or root.is_relative_to(parent) or root.is_relative_to(REPO/'artifacts'): raise ValueError('Output cannot mutate a parent or Git artifact directory')
    root.mkdir(parents=True,exist_ok=True)
    storage=root/'storage_preflight.json'
    if not storage.exists():
        free=shutil.disk_usage(root).free/2**30; write(storage,dict(free_GiB=free,minimum_GiB=10))
        if free<10: raise OSError('Initial free space below10GiB')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if remote.removesuffix('.git') not in ('git@github.com:Orangekostar/oviovo','https://github.com/Orangekostar/oviovo'): raise ValueError('Incorrect origin owner/repo')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    if not (parent/'source_binding.json').exists():
        missing=[str(parent/'source_binding.json')]
        write(root/'dependency_manifest.json',dict(status='BLOCKED_DEPENDENCY',missing=missing,checked_parent=str(parent),bounded_candidates=spec['parent_root_candidates'],no_substitutions=True))
        write(root/'preflight.json',dict(status='BLOCKED_DEPENDENCY',missing=missing,new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0))
        raise DependencyError(missing)
    original=verified(parent/'source_binding.json'); source=resolver.rewrite(copy.deepcopy(original)); lr=Path(source['lr_parent_root']).resolve()
    for key in ('lr_parent_root','parent_root','pr_parent_root','lr_source_root','observation_root','minimal_root','prior_data_root'):
        if key in source and (root==Path(source[key]).resolve() or root.is_relative_to(Path(source[key]).resolve())): raise ValueError('Inherited experiment roots are read-only: '+key)
    registry=parent/'export/weights_registry.json'; pinned=False
    if not registry.exists():
        archive='artifacts/static_ovmap/frozen_fc_routing_v1/export/weights_registry.json'
        contents=subprocess.check_output(['git','show',BASE+':'+archive],cwd=REPO)
        registry=root/'parent_export/weights_registry.json'; registry.parent.mkdir(parents=True,exist_ok=True); registry.write_bytes(contents); pinned=True
    registered=verified(registry); index=ConsumptionIndex(root/'input_verifications.json')
    required=[lr/'split_manifest.json',lr/'class_split.json',lr/'features/train-dev.json',lr/'external/upstream_sources/MaskAdapter/fcclip/fcclip.py']
    required += [parent/'recognition/dev'/f'{key}.json' for key in ('FC8','FG2_SURFACE','S29_FG2_SURFACE')]
    missing=[str(p) for p in required if not p.exists()]
    if missing:
        write(root/'dependency_manifest.json',dict(status='BLOCKED_DEPENDENCY',parent_identity=original['identity'],missing=missing,registry=registered['identity']))
        raise DependencyError(missing)
    split=verified(lr/'split_manifest.json'); classes=verified(lr/'class_split.json'); features=verified(lr/'features/train-dev.json')
    if len(classes['base_ids'])!=160 or len(classes['heldout_ids'])!=40: raise ValueError('Fixed class split changed')
    objects=[]; manifest_identities={}
    for role,count in (('train',622),('dev',69)):
        rows=[]
        for scene in split['roles'][role]:
            manifest=lr/'data/generated'/scene/'manifest.json'; require([manifest]); value=verified(manifest)
            manifest_identities[scene]=value['identity']
            rows.extend(resolver.rewrite(value['objects']))
        if sum(bool(o['base']) for o in rows)!=count: raise ValueError('Fixed original object coverage changed')
        if len({o['family'] for o in rows if o['base']})!=(24 if role=='train' else 4): raise ValueError('Fixed physical-family coverage changed')
        objects.extend(rows)
    needed=[o['inputs']['path'] for o in objects if o['base']]
    frames=resolver.rewrite(features['frames'])
    for obj in objects:
        if not obj['base']: continue
        needed += [frames[obj['scene']+':'+str(v['frame_id'])]['arrays']['path'] for v in obj['views'][:8]]
    for key in ('FC8','FG2_SURFACE','S29_FG2_SURFACE'):
        value=verified(parent/'recognition/dev'/f'{key}.json'); needed.append(resolver.rewrite(value['arrays'])['path'])
    # AnyUp assets and the other25 map contexts are not consumed by P0–P2.
    needed += minimal_fc_dependencies(source,resolver)
    missing=[str(p) for p in dict.fromkeys(needed) if not Path(p).exists()]
    weights={}
    for seed in (17,29):
        key=spec['heads'][str(seed)]['registry_key']; row=registered['heads'][key]; item=row['weights']; file=registry.parent/item['relative_path']
        if pinned:
            archived=Path('artifacts/static_ovmap/frozen_fc_routing_v1/export')/item['relative_path']
            file.parent.mkdir(parents=True,exist_ok=True); file.write_bytes(subprocess.check_output(['git','show',BASE+':'+str(archived)],cwd=REPO))
        if not file.exists(): missing.append(str(file)); continue
        if digest_file(file)!=spec['heads'][str(seed)]['sha256'] or row['seed']!=seed or row['selected_step']!=spec['heads'][str(seed)]['selected_step']: raise ValueError('Fixed exported head identity changed')
        weights[key]=dict(row,weights=dict(item,path=str(file.resolve())))
    p3=[lr/'regression/manifest.json',lr/'features/regression.json',parent/'recognition/regression.json',parent/'evaluation/store.json']
    write(root/'dependency_manifest.json',dict(status='BLOCKED_DEPENDENCY' if missing else 'COMPLETE',missing=missing,P3_top_level={str(p):p.exists() for p in p3},P3_contexts={key:Path(row['context']['path']).exists() for key,row in source['scenes'].items()},parent_identity=original['identity'],lr_root=str(lr),registry_identity=registered['identity'],pinned_registry=pinned))
    if missing: raise DependencyError(missing)
    for name in ('class_split.json','split_manifest.json','features/train-dev.json'):
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and dest.read_bytes()!=(lr/name).read_bytes(): raise ValueError('Copied parent manifest changed')
        if not dest.exists(): shutil.copyfile(lr/name,dest)
    for name,target in (('external/upstream_sources',lr/'external/upstream_sources'),('data/generated',lr/'data/generated')):
        dest=root/name; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() or dest.exists():
            if dest.resolve()!=target.resolve(): raise ValueError('Read-only dependency link changed')
        else: dest.symlink_to(target,target_is_directory=True)
    binding=copy.deepcopy(source); binding.pop('identity',None)
    binding.update(task_id=spec['task_id'],new_spec=spec,fg_parent_root=str(parent),lr_parent_root=str(lr),output_root=str(root),path_map=mapping,gpu=str(source['gpu'] if gpu is None else gpu),parent_FG_identity=original['identity'],parent_registry_identity=registered['identity'],parent_registry_path=str(registry),export_heads=weights,base_commit=BASE,consumed_manifest_identities=dict(split=split['identity'],classes=classes['identity'],features=features['identity'],objects=manifest_identities))
    if ',' in binding['gpu'] or not binding['gpu']: raise ValueError('One inherited GPU device required')
    result=write(root/'binding.json',binding); write(root/'preflight.json',dict(status='COMPLETE',binding_identity=result['identity'],P3_top_level_only=True,real_FC_forward_not_yet_run=True))
    index.write_memo(root/'input_verifications.json'); return result

def runtime_registry(binding,fc):
    spec=binding['new_spec']; root=Path(binding['output_root']); text=fc.session.text_identity
    path=Path(text['path']); require([path])
    if fc.model_key!=spec['frozen_fc_identity'] or path.stat().st_size!=spec['frozen_text_bytes'] or digest_file(path)!=spec['frozen_text_sha256']: raise ValueError('Fixed FC/text physical identity differs')
    import numpy as np
    with np.load(path,allow_pickle=False) as arrays:
        if not np.array_equal(arrays['valid_ids'],fc.ids): raise ValueError('Actual frozen category order changed')
    heads={}
    for name,row in binding['export_heads'].items():
        source=Path(row['weights']['path']); dest=root/'export/weights'/(name+'.pt'); dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists(): shutil.copyfile(source,dest)
        expected=spec['heads'][str(row['seed'])]['sha256']
        if digest_file(dest)!=expected: raise ValueError('Runtime exported head bytes changed')
        heads[name]=dict(row,weights=dict(path=str(dest),relative_path='weights/'+dest.name,bytes=dest.stat().st_size,sha256=expected))
    registry=write(root/'export/runtime_registry.json',dict(status='COMPLETE',heads=heads,FC_identity=fc.model_key,text_identity=text,parent_registry_identity=binding['parent_registry_identity'],no_new_weights=True))
    from static_ovmap.frozen_fc_routing.export import load_exported
    return {seed:load_exported(root/'export/runtime_registry.json',spec['heads'][str(seed)]['registry_key'],fc.device,FC_identity=fc.model_key,text_identity=text) for seed in (17,29)}

def load_binding(root):
    require([Path(root)/'binding.json']); result=verified(Path(root)/'binding.json')
    identities=result.get('consumed_manifest_identities')
    if identities:
        lr=Path(result['lr_parent_root']); parent=Path(result['fg_parent_root'])
        expected={parent/'source_binding.json':result['parent_FG_identity'],lr/'split_manifest.json':identities['split'],lr/'class_split.json':identities['classes'],lr/'features/train-dev.json':identities['features']}
        expected.update({lr/'data/generated'/scene/'manifest.json':identity for scene,identity in identities['objects'].items()})
        require(expected)
        for path,identity in expected.items():
            if verified(path)['identity']!=identity: raise ValueError('Bound numerical manifest changed; rerun preflight before downstream phases: '+str(path))
    return result
