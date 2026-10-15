"""Task-scoped SAM linker isolation from the controller's Conda libraries."""
import contextlib
import os
from pathlib import Path
from .common import verified,PathResolver,DependencyError

@contextlib.contextmanager
def isolated_linker(binding):
    resolver=PathResolver(binding['path_map'])
    assets=resolver.rewrite(verified(resolver.resolve('/mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001/model_assets.json')))
    python=Path(binding.get('task_sam2_python') or assets['environments']['sam2'])
    libraries={p.resolve() for p in python.parent.parent.glob('lib/python*/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12')}
    if len(libraries)!=1:raise DependencyError('SAM environment must provide one unambiguous nvJitLink library')
    library=str(next(iter(libraries)));old=os.environ.get('LD_PRELOAD')
    # Preload affects the new child only; the controller's loaded libraries stay intact.
    os.environ['LD_PRELOAD']=library+(':'+old if old else '')
    try:yield
    finally:
        if old is None:os.environ.pop('LD_PRELOAD',None)
        else:os.environ['LD_PRELOAD']=old
