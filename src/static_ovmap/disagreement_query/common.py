"""Task-owned content records; inherited inputs remain read-only."""
from pathlib import Path

from static_ovmap.samv_local_probe.common import (
    ConsumptionIndex, PathResolver, _array_digest, archive, arrays_record,
    canonical_digest, read, verified, write,
)
from static_ovmap.source_preserving_update.binding import plain

REPO=Path(__file__).resolve().parents[3]
METRICS=('apall','ap50','ap25','miou','macc')
POLICIES=('AREA','COVERAGE','VERIFY','DISAGREEMENT')
BASELINES={'DQ00_D2':'SU00_D2','DQ01_G1':'SU01_G1'}


def producer(index,*names):
    return [index.identity(Path(__file__).with_name(n)) for n in names]


def unchanged_receipt(path,key,index=None):
    path=Path(path)
    if not path.exists():return None
    row=verified(path)
    if row['input_identity']!=key:raise ValueError(f'changed dependency: {path}; invalidate only its new-task descendants')
    if index is not None:
        for dep in row.get('dependencies',[]):index.identity(dep['path'],dep)
    return row
