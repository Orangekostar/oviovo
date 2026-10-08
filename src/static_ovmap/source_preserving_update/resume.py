"""Archive only new-task descendants of a changed leaf, retaining lineage."""

from pathlib import Path
import time

from static_ovmap.module_validation.contracts import atomic_write_json
from .binding import seal


def invalidate_descendants(binding,scene,stage,reason):
    root=Path(binding['output_root']).resolve()
    stages=('prepare','coarse','predict','evaluate','diagnose')
    start=stages.index(stage)
    paths=[]
    if start<=0:paths.extend([root/'evidence'/scene,root/'evidence_manifest.json'])
    if start<=1:paths.extend([root/'coarse'/scene,root/'eligibility'/(scene+'.json'),root/'coarse/summary.json'])
    if start<=2:paths.extend([root/'decisions'/scene,root/'predictions'/scene,root/'predictions/summary.json'])
    if start<=3:paths.extend([root/'evaluation'/scene,root/'pools',root/'result_store.json'])
    paths.extend([root/'diagnostics'/scene,root/'diagnostics/summary.json',root/'next_stage_assessment.json',
                  root/'selection.json',root/'costs/summary.json',root/'tables'])
    archive=root/'history'/(str(time.time_ns())+'_'+scene+'_'+stage)
    moved=[]
    for path in paths:
        if not path.exists():continue
        if not path.resolve().is_relative_to(root):raise ValueError('invalidation cannot change inherited data')
        target=archive/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True)
        path.rename(target);moved.append(str(path.relative_to(root)))
    atomic_write_json(archive/'invalidation.json',seal({'scene':scene,'stage':stage,'reason':reason,'archived':moved}))
    return moved
