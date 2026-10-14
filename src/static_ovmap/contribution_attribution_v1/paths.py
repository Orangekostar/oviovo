"""Finite CLI root defaults and immutable task-resume arguments."""
from pathlib import Path
from .common import ConsumptionIndex,PathResolver,read,read_document,verified,DependencyError
from .binding import DEFAULT_FG,DEFAULT_AREA,DEFAULT_COMPACT,DEFAULT_BB,DEFAULT_W2

FG_CANDIDATES=(DEFAULT_FG,'/mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001')
ROOT_FIELDS=dict(parent_root='fg_parent_root',area_root='area_root',compact_root='compact_root',backbone_root='backbone_root',wave2_root='wave2_root')

def resolve_defaults(args,spec):
    locked=args.output_root/'source_binding.json'
    if locked.exists():
        binding=verified(locked);mapping=read(args.path_map) if args.path_map else binding['path_map']
        if mapping!=binding['path_map']:raise ValueError('Resume path-map differs from the recorded task binding')
        resolver=PathResolver(mapping)
        for option,field in ROOT_FIELDS.items():
            value=getattr(args,option)
            if value is not None and Path(resolver.resolve(str(value))).resolve()!=Path(binding[field]).resolve():
                raise ValueError('Resume root differs from the recorded task binding: '+option)
            setattr(args,option,Path(binding[field]))
        if args.gpu is not None and str(args.gpu)!=binding['gpu']:raise ValueError('Resume GPU differs from the recorded task binding')
        args.gpu=binding['gpu'];return
    mapping=read(args.path_map) if args.path_map else {};resolver=PathResolver(mapping);index=ConsumptionIndex()
    candidates=(str(args.parent_root),) if args.parent_root else FG_CANDIDATES
    failures=[];source=None
    for candidate in candidates:
        parent=Path(resolver.resolve(candidate)).resolve()
        try:
            original=read_document(parent/'source_binding.json',index)
            if original['specification']['task_id']!='ovimap-frozen-fc-routing-v1' or original['cohorts']!=spec['cohorts']:
                raise ValueError('FG identity or ordered cohorts differ')
            source=resolver.rewrite(original);args.parent_root=parent;break
        except (DependencyError,ValueError,KeyError) as exc:
            failures.append(dict(root=str(parent),reason=str(exc)))
            if args.parent_root:raise
    if source is None:raise DependencyError('No identity-matching FG root in finite candidates: '+str(failures))
    if args.area_root is None:
        reference=source.get('reference',{}).get('v2_results_root')
        candidate=Path(resolver.resolve(reference)).resolve() if reference else None
        args.area_root=candidate if candidate and (candidate/'experiment.json').is_file() else Path(resolver.resolve(DEFAULT_AREA)).resolve()
    if args.compact_root is None:
        path=Path(resolver.resolve(str(args.area_root)))/'experiment.json'
        experiment=read_document(path,index) if path.is_file() else None
        args.compact_root=Path(resolver.resolve(experiment['parent'] if experiment else DEFAULT_COMPACT)).resolve()
    if args.backbone_root is None:args.backbone_root=Path(resolver.resolve(DEFAULT_BB)).resolve()
    if args.wave2_root is None:args.wave2_root=Path(resolver.resolve(DEFAULT_W2)).resolve()
    index.write_memo(args.output_root/'resolution_verifications.json')
