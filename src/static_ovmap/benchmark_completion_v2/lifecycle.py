"""Stdlib terminal recovery: validate identities before importing science modules."""
from pathlib import Path
import sys

from .common import read, verified, write, digest_file, canonical_digest


def producer_identity():
    folder = Path(__file__).parent
    return canonical_digest({p.name: digest_file(p) for p in sorted(folder.glob('*.py'))})


def parameter_identity(args):
    root = Path(args.output_root).resolve()
    paths = read(root / 'task_paths.json') if (root / 'task_paths.json').is_file() else {}
    assets = read(args.asset_config) if args.asset_config else read(paths['asset_config']) if paths.get('asset_config') else {}
    mapping = read(args.path_map) if args.path_map else assets.get('path_map', paths.get('path_map', {}))
    constraints = read(root / 'runtime_constraints.json') if (root / 'runtime_constraints.json').is_file() else {}
    maximum = args.max_gpu_hours if args.max_gpu_hours is not None else constraints.get('max_gpu_hours')
    return canonical_digest(dict(output_root=str(root), semantic_root=args.semantic_root or paths.get('semantic_root'),
        structure_root=args.structure_root or paths.get('structure_root'), path_map=mapping,
        assets=assets, gpu=args.gpu, max_gpu_hours=maximum, specification=digest_file(args.spec)))


def terminal_resume(args):
    """Returns None for live work; terminal path only reads existing receipts."""
    path = Path(args.output_root) / 'terminal.json'
    if not args.resume or args.phase != 'all' or not path.is_file() or args.refresh_assets:
        return None
    terminal = verified(path)
    if terminal['lifecycle'] not in ('CLOSED', 'PARTIAL_CLOSED'):
        return None
    if terminal['producer_identity'] != producer_identity() or terminal['parameter_identity'] != parameter_identity(args):
        raise ValueError('Terminal task identity changed; use an explicit affected repair or --refresh-assets for new assets')
    for descriptor in terminal['receipts']:
        row = verified(descriptor['path'])
        if row['identity'] != descriptor['identity']:
            raise ValueError('Terminal scientific receipt changed: ' + descriptor['path'])
    scientific_imports = sorted(name for name in sys.modules if name.split('.')[0] in ('numpy', 'torch', 'scipy', 'open3d'))
    if scientific_imports:
        raise ValueError('Terminal resume imported scientific modules: ' + repr(scientific_imports))
    print('TERMINAL_RESUME', terminal['lifecycle'], 'new_workers=0 scientific_imports=0 writes=0', flush=True)
    return int(terminal['exit_code'])


def close(args, *, flags, receipts, remaining_jobs):
    if any(row.get('runnable') for row in remaining_jobs):
        raise ValueError('Runnable unfinished scientific jobs cannot be declared terminal')
    root = Path(args.output_root)
    # A fully executed negative result is a completed experiment. Claim support
    # remains an independent scientific outcome, never a success-code gate.
    complete = all(value for key, value in flags.items() if key != 'claim_supported')
    return write(root / 'terminal.json', dict(lifecycle='CLOSED' if complete else 'PARTIAL_CLOSED',
        producer_identity=producer_identity(), parameter_identity=parameter_identity(args),
        completion_flags=flags, exit_code=0 if complete else 20,
        receipts=[dict(path=str(p), identity=verified(p)['identity']) for p in receipts],
        remaining_jobs=remaining_jobs, publication_is_not_scientific_completion=True))
