"""Preserve stale new-task leaves and invalidate only their actual descendants."""

from pathlib import Path
import shutil
import time

from static_ovmap.module_validation.contracts import atomic_write_json

from .binding import seal


def archive_paths(root, paths, reason):
    root = Path(root).resolve()
    history = root/'history'/'invalidations'/str(time.time_ns())
    archived = []
    for relative in paths:
        relative = Path(relative)
        path = root/relative
        if relative.is_absolute() or '..' in relative.parts or not path.resolve().is_relative_to(root):
            raise ValueError('invalidation cannot modify a parent or external artifact')
        if not path.exists():
            continue
        target = history/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(path),str(target))
        archived.append({'relative_path':str(relative),'archived':str(target)})
    result = seal({'status':'SCOPED_DESCENDANTS_INVALIDATED','reason':reason,'archived':archived,
                   'parent_artifacts_modified':False,'historical_work_deleted':False})
    if archived:
        atomic_write_json(history/'invalidation.json',result)
    return result


def invalidate_descendants(binding, scene, phase, reason, *, leaf=None):
    if scene not in binding['scenes']:
        raise ValueError('scoped invalidation requires a bound scene')
    paths = []
    if leaf is not None:
        paths.extend(leaf)
    if phase=='observe':
        paths.extend([f'observations/{scene}/receipt.json',f'proposals/{scene}/receipt.json',
                      f'diagnostics/{scene}/opportunity.json','diagnostics/summary.json'])
    if phase=='propose':
        paths.extend([f'proposals/{scene}/receipt.json',f'diagnostics/{scene}/opportunity.json','diagnostics/summary.json'])
    if phase in ('observe','propose','recognition_plan'):
        paths.append(f'recognition/{scene}/plan.json')
    if phase in ('observe','propose','recognition_plan','recognition_frame'):
        paths.extend([f'recognition/{scene}/decisions.json','recognition/summary.json','pilots/summary.json'])
    if phase in ('observe','propose','recognition_plan','recognition_frame','predict'):
        paths.append(f'predictions/{scene}')
    if phase in ('observe','propose','recognition_plan','recognition_frame','predict','evaluate'):
        paths.extend([f'evaluation/{scene}/receipt.json',f'evaluation/{scene}/rows',
            'result_store.json','selection.json',f'pools/{binding["scenes"][scene]["cohort"]}',
            'timing/binding.json','timing/summary.json'])
    else:
        raise ValueError('unknown dependency stage: '+phase)
    return archive_paths(binding['output_root'],list(dict.fromkeys(paths)),reason)
