"""New role binding preserves inherited FG fields and consumes no old binder."""
import copy
from pathlib import Path
import shutil
import subprocess
from .common import (REPO,TASK,BASE,ConsumptionIndex,PathResolver,read,read_document,digest_file,
                     write,immutable_record,reserve,DependencyError,family)

DEFAULT_FG='/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001'
DEFAULT_AREA='/home/ww/ovimap-area-fallback-v2/attempt_001'
DEFAULT_COMPACT='/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001'
DEFAULT_BB='/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001'
DEFAULT_W2='/home/ww/crove/fg2-branch-utility-wave2-v1/attempt_001'

def preflight(args,spec):
    root=Path(args.output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    mapping=read(args.path_map) if args.path_map else {}
    resolver=PathResolver(mapping)
    fg=Path(resolver.resolve(str(args.parent_root or DEFAULT_FG))).resolve()
    index=ConsumptionIndex(root/'input_verifications.json')
    original=read_document(fg/'source_binding.json',index)
    source=resolver.rewrite(original)
    lr=Path(source['lr_parent_root']).resolve()
    area=Path(resolver.resolve(str(args.area_root or DEFAULT_AREA))).resolve()
    area_experiment=read_document(area/'experiment.json',index) if (area/'experiment.json').exists() else None
    compact=Path(resolver.resolve(str(args.compact_root or (area_experiment['parent'] if area_experiment else DEFAULT_COMPACT)))).resolve()
    bb=Path(resolver.resolve(str(args.backbone_root or DEFAULT_BB))).resolve()
    wave2=Path(resolver.resolve(str(args.wave2_root or DEFAULT_W2))).resolve()
    for parent in (fg,lr,area,compact,bb,wave2):
        if root==parent or root.is_relative_to(parent):raise ValueError('Task OUT must be outside every readonly parent')
    remote=subprocess.check_output(['git','remote','get-url','origin'],cwd=REPO,text=True).strip()
    if remote not in ('git@github.com:Orangekostar/oviovo.git','https://github.com/Orangekostar/oviovo.git'):raise ValueError('Wrong task origin')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if branch!=spec['branch'] and not branch.startswith(spec['branch']+'-'):raise ValueError('Wrong task branch')
    split=read_document(lr/'split_manifest.json',index)
    classes=read_document(lr/'class_split.json',index)
    profile=read_document(lr/'data_profile.json',index)
    if source['cohorts']!=spec['cohorts']:raise ValueError('Parent ordered regression cohorts differ')
    for role,count,families in (('train',622,24),('dev',69,4)):
        if profile['roles'][role]['base_objects']!=count or profile['roles'][role]['families']!=families:raise ValueError('Recognition role support differs: '+role)
    if len(classes['base_ids'])!=160 or len(classes['heldout_ids'])!=40:raise ValueError('Parent class split differs')
    regression=[s for scenes in spec['cohorts'].values() for s in scenes]
    if {family(s) for s in spec['fusion_dev']} & {family(s) for s in regression}:raise ValueError('Fusion DEV overlaps regression family')
    free=reserve(root,initial=not (root/'source_binding.json').exists())
    roles=dict(FUSION_DEV=spec['fusion_dev'],REGRESSION=spec['cohorts'],S_TRAIN=split['roles']['train'],S_DEV=split['roles']['dev'],CONFIRM_NEXT='NAMES_ONLY_NOT_RUN_BY_CONTRACT')
    role_lock=immutable_record(root/'role_lock.json',dict(status='COMPLETE',task=TASK,roles=roles,recognition_support=profile['roles'],class_split_identity=classes['identity'],split_identity=split['identity'],family_overlap=[],exposure='EXPOSED_EXPLORATORY_NOT_CONFIRMATORY'))
    binding=copy.deepcopy(source);binding.pop('identity',None)
    binding.update(task_id=TASK,task_source_base=BASE,task_spec=str(args.spec.resolve()),task_spec_sha256=digest_file(args.spec),task_specification=spec,
                   output_root=str(root),path_map=mapping,gpu=str(source['gpu'] if args.gpu is None else args.gpu),
                   fg_parent_root=str(fg),lr_parent_root=str(lr),area_root=str(area),compact_root=str(compact),backbone_root=str(bb),wave2_root=str(wave2),
                   task_parent_binding=original['identity'],task_parent_binding_file=index.identity(fg/'source_binding.json'),role_lock=role_lock['identity'],task_branch=branch,
                   task_runtime_arguments=dict(parent_root=str(fg),output_root=str(root),wave2_root=str(wave2),area_root=str(area),compact_root=str(compact),backbone_root=str(bb),path_map=mapping,gpu=str(source['gpu'] if args.gpu is None else args.gpu)),
                   task_new_optimizer_updates=0,task_new_rgb_encodings=0,task_new_sam_proposals=0,task_new_geometry_maps=0)
    result=immutable_record(root/'source_binding.json',binding)
    for name in ('class_split.json','split_manifest.json','features/train-dev.json'):
        src=lr/name;dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.read_bytes()!=src.read_bytes():raise ValueError('Inherited manifest copy differs: '+name)
        if not dst.exists():shutil.copyfile(src,dst)
    for relative,target in (('external/upstream_sources',lr/'external/upstream_sources'),('data/generated',lr/'data/generated')):
        dest=root/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink() and dest.resolve()!=target.resolve():raise ValueError('Inherited readonly link differs')
        if not dest.exists():dest.symlink_to(target,target_is_directory=True)
    index.write_memo(root/'input_verifications.json')
    return write(root/'preflight.json',dict(status='COMPLETE',binding=result['identity'],role_lock=role_lock['identity'],free_bytes=free,parent_roots={k:binding[k] for k in ('fg_parent_root','lr_parent_root','area_root','compact_root','backbone_root','wave2_root')},inherited_parent_root_semantics_preserved=binding['parent_root']==source['parent_root'],new_optimizer_updates=0,new_rgb_encodings=0,new_sam_proposals=0,new_map_construction=0))

def load_binding(root):
    from .common import verified
    binding=verified(Path(root)/'source_binding.json')
    if binding['task_id']!=TASK or digest_file(binding['task_spec'])!=binding['task_spec_sha256']:raise ValueError('Task binding/spec identity changed')
    if digest_file(Path(binding['fg_parent_root'])/'source_binding.json')!=binding['task_parent_binding_file']['sha256']:raise ValueError('Consumed FG parent binding changed')
    return binding
