"""Bind STRUCT_ROOT, preserving original identities before physical rewrites."""
from pathlib import Path
from .common import (REPO, DOCS, read, write, verified, digest_file,
                     ConsumptionIndex, PathResolver, DependencyError, source_identity,save_memo)

def bind(args, spec):
    root = Path(args.output_root).resolve(); root.mkdir(parents=True, exist_ok=True)
    resolver = PathResolver(read(args.path_map) if args.path_map else {})
    parent = Path(resolver.resolve(args.parent_root)).resolve()
    if root == parent or root.is_relative_to(parent):
        raise ValueError('OUT must be outside immutable STRUCT_ROOT')
    original_spec = REPO / DOCS / 'prompt/02_冻结执行配置.json'
    if Path(args.spec).read_bytes() != original_spec.read_bytes():
        raise ValueError('Frozen specification changed')
    index = ConsumptionIndex(root / 'input_verifications.json')
    for path in (Path(args.spec), parent / 'source_binding.json', parent / 'class_split.json'):
        if not path.is_file():
            raise DependencyError('Missing binding dependency: ' + str(path))
        index.identity(path)
    original = verified(parent / 'source_binding.json')
    inherited = resolver.rewrite(original)
    if args.gpu is not None and int(args.gpu) != int(inherited['gpu']):
        raise ValueError('Only inherited GPU is permitted')
    class_split = verified(parent / 'class_split.json')
    (root / 'class_split.json').write_bytes((parent / 'class_split.json').read_bytes())
    value = dict(status='COMPLETE', task_id=spec['task_id'], task_specification=spec,
                 STRUCT_ROOT=str(parent), output_root=str(root), path_map=read(args.path_map) if args.path_map else {},
                 parent_binding_identity=original['identity'], inherited=inherited, gpu=inherited['gpu'],
                 old_class_split_identity=class_split['identity'], source_identity=source_identity(),
                 roles=dict(TRAIN='UNCHANGED', DEV='UNCHANGED', CONF12='A_EXPLORE_EXPOSED',
                            REG26='EXPOSED_REGRESSION', FINAL_NOVEL40='PREVIOUSLY_OBSERVED_CONF_LABELS'))
    path = root / 'source_binding.json'
    if path.exists():
        old = verified(path)
        for k in ('source_identity', 'parent_binding_identity', 'task_specification', 'path_map'):
            if old[k] != value[k]:
                raise ValueError('Bound numerical identity changed: ' + k)
        value = old
    else:
        value = write(path, value)
    save_memo(index,root / 'input_verifications.json')
    return value
