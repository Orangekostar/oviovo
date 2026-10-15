"""Immutable role/source binding; inventory never opens confirmation annotations."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import subprocess
from .common import (REPO,TASK,BASE,ConsumptionIndex,PathResolver,read_document,
    verified,write,immutable_record,digest_file,canonical_digest,reserve,family,objects)

DEFAULT_FG='/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001'
DEFAULT_ATTR='/home/ww/crove/ovmap-contribution-attribution-v1/attempt_001'

def bind(args,spec):
    root=Path(args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    mapping=json.loads(Path(args.path_map).read_text()) if args.path_map else {}
    resolver=PathResolver(mapping);fg=Path(resolver.resolve(str(args.parent_root or DEFAULT_FG))).resolve()
    attr=Path(resolver.resolve(str(args.attribution_root or DEFAULT_ATTR))).resolve()
    index=ConsumptionIndex(root/'input_verifications.json')
    original=read_document(fg/'source_binding.json',index);source=resolver.rewrite(original)
    lr=Path(source['lr_parent_root']);parents=[fg,attr,lr,Path(source['pr_parent_root'])]
    if any(root==p or root.is_relative_to(p) for p in parents):raise ValueError('OUT overlaps a readonly parent')
    if subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()!=spec['branch']:raise ValueError('Unexpected research branch')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if remote not in ('git@github.com:Orangekostar/oviovo.git','https://github.com/Orangekostar/oviovo.git'):raise ValueError('Unexpected origin')
    split=read_document(lr/'split_manifest.json',index);classes=read_document(lr/'class_split.json',index)
    profile=read_document(lr/'data_profile.json',index)
    plan_path=spec['confirmation']['plan_path']
    blob=subprocess.check_output(['git','rev-parse',BASE+':'+plan_path],cwd=REPO,text=True).strip()
    if blob!=spec['confirmation']['plan_git_blob']:raise ValueError('Sealed confirmation plan blob differs')
    plan=json.loads(gzip.decompress(subprocess.check_output(['git','show',BASE+':'+plan_path],cwd=REPO)))
    selected=plan['scene_nomination']['selected']
    if len(selected)!=12 or len({family(s) for s in selected})!=12:raise ValueError('Exactly twelve fixed confirmation families required')
    roles=dict(TRAIN=split['roles']['train'],DEV=split['roles']['dev'],REG=[s for v in spec['map_regression']['cohorts'].values() for s in v],CONF=selected)
    names=list(roles)
    if any({family(s) for s in roles[a]} & {family(s) for s in roles[b]} for i,a in enumerate(names) for b in names[i+1:]):raise ValueError('Role family overlap')
    if source['cohorts']!=spec['map_regression']['cohorts']:raise ValueError('Regression order differs')
    for role,n,f in [('train',622,24),('dev',69,4)]:
        if profile['roles'][role]['base_objects']!=n or profile['roles'][role]['families']!=f:raise ValueError('Original support differs: '+role)
    if len(classes['base_ids'])!=160 or len(classes['heldout_ids'])!=40:raise ValueError('Class partition differs')
    free=reserve(root,initial=not (root/'source_binding.json').exists())
    role_lock=immutable_record(root/'role_lock.json',dict(status='COMPLETE',roles=roles,confirmation_plan_blob=blob,split_identity=split['identity'],classes_identity=classes['identity'],independence=spec['confirmation']['independence']))
    immutable_record(root/'roles.json',{k:v for k,v in role_lock.items() if k!='identity'})
    binding=copy.deepcopy(source);binding.pop('identity',None)
    binding.update(task_id=TASK,task_source_base=BASE,output_root=str(root),path_map=mapping,
        fg_parent_root=str(fg),attribution_root=str(attr),task_parent_binding=original['identity'],
        task_parent_binding_file=index.identity(fg/'source_binding.json'),task_spec=str(Path(args.spec).resolve()),
        task_spec_sha256=digest_file(args.spec),task_specification=spec,role_lock=role_lock['identity'],
        gpu=str(source['gpu'] if args.gpu is None else args.gpu),
        confirmation_data_root=str(Path(args.scannet_root).resolve()) if args.scannet_root else None,
        task_sam2_root=str(Path(args.sam2_root).resolve()) if args.sam2_root else None,task_sam2_python=args.sam2_python)
    result=immutable_record(root/'source_binding.json',binding)
    for name in ('class_split.json','split_manifest.json','features/train-dev.json'):
        src=lr/name;dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);index.identity(src)
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise ValueError('Inherited copy differs: '+name)
        if not dst.exists():shutil.copyfile(src,dst)
    for relative,target in [('external/upstream_sources',lr/'external/upstream_sources'),('data/generated',lr/'data/generated')]:
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() and dest.resolve()!=target.resolve():raise ValueError('Readonly link differs')
        if not dest.exists():dest.symlink_to(target,target_is_directory=True)
    items={r:objects(result,r) for r in ('train','dev')}
    if [len(items[r]) for r in ('train','dev')]!=[785,87]:raise ValueError('Original draw index domain changed')
    index.write_memo(root/'input_verifications.json')
    return write(root/'preflight.json',dict(status='COMPLETE',binding=result['identity'],role_lock=role_lock['identity'],free_bytes=free,source_grid_encodings=0,confirmation_annotations_parsed=0))

def load_binding(root):
    b=verified(Path(root)/'source_binding.json')
    if b['task_id']!=TASK or digest_file(b['task_spec'])!=b['task_spec_sha256']:raise ValueError('Binding/spec changed')
    if digest_file(Path(b['fg_parent_root'])/'source_binding.json')!=b['task_parent_binding_file']['sha256']:raise ValueError('FG source binding changed')
    return b
