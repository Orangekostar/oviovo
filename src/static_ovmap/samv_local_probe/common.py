"""Small content-bound records shared by the isolated probe workers."""

from pathlib import Path
import shutil

from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.minimal_instance_repair.binding import seal
from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest,_write_npz
from static_ovmap.recovery_wave2.binding import ConsumptionIndex,PathResolver,read

REPO=Path(__file__).resolve().parents[3]
METHODS=('SV00_G1','SV01_SAM2_GEOM','SV02_SAMV_GEOM','SV03_OLDMASK_FC','SV04_SAMV_FC','SV05_COMBINED')
COHORTS={'replica_probe2':['office1','room0'],'cf_probe2':['scene0011_00','scene0050_00']}


def request_key(query,model,asset_identity):
    return canonical_digest({'ordered_query':query,'model':model,'assets':asset_identity})


def write(path,value):
    result=seal(value);atomic_write_json(path,result);return result


def verified(path):
    result=read(path);_verified_identity(result);return result


def free_space(root,minimum=30):
    available=shutil.disk_usage(root).free/2**30
    if available<minimum:raise OSError(f'{available:.3f} GiB free; {minimum} GiB required; use explicit storage relocation')
    return available


def arrays_record(path,values,index):
    _write_npz(Path(path),values)
    return index.identity(path)


def archive(root,paths,reason):
    """Only superseded records owned by this new task can be moved."""
    import time
    root=Path(root).resolve();dest=root/'history'/str(time.time_ns());moved=[]
    for path in map(Path,paths):
        if not path.exists():continue
        if not path.resolve().is_relative_to(root):raise ValueError('cannot invalidate inherited data')
        target=dest/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        path.rename(target);moved.append(str(path.relative_to(root)))
    write(dest/'invalidation.json',{'reason':reason,'moved':moved})
    return moved
