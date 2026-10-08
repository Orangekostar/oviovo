"""Resume orchestration without changing already validated scientific kernels."""

from pathlib import Path

from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.recovery_wave2.binding import read, ConsumptionIndex

from .binding import load_scene
from .resume import invalidate_descendants


def observe(binding, scene):
    from .observations import observe_scene, pose_banks, load_panoptic
    root = Path(binding['output_root'])
    target = root/'observations'/scene/'receipt.json'
    if not target.exists():
        return observe_scene(binding,scene)
    inputs = load_scene(binding,scene)
    cfg = binding['specification']['observer']
    module = Path(__file__).with_name('observations.py')
    dependency = canonical_digest({'capture':inputs.capture['identity'],'geometry':inputs.units.mesh_identity,
        'units':inputs.units.identity,'observer':cfg,
        'operators':[inputs.index.identity(module),inputs.index.identity(module.with_name('support_units.py'))]})
    selected,_ = pose_banks(inputs.capture['frames'],limit=cfg['max_frames'],
                            translation=cfg['pose_translation_m'],degrees=cfg['pose_rotation_degrees'])
    changed = []
    for frame in selected:
        fid = int(frame['frame_id'])
        path = target.parent/'frames'/f'{fid:06d}.json'
        try:
            _,panoptic = load_panoptic(inputs.capture_path.parent,frame,inputs.index,binding['path_map'])
        except FileNotFoundError:
            panoptic = None
        key = canonical_digest({'observer':dependency,'frame_id':fid,'bank':frame['bank'],
            'pose':frame['pose_c2w'],'intrinsics':frame['intrinsics'],
            'rgb':inputs.index.identity(inputs.capture_path.parent/frame['rgb_path'],{'sha256':frame['rgb_sha256']}),
            'depth':inputs.index.identity(inputs.capture_path.parent/frame['depth_path'],{'sha256':frame['depth_sha256']}),
            'panoptic':panoptic})
        if not path.exists() or read(path)['input_identity']!=key:
            changed.extend(str(p.relative_to(root)) for p in (path,path.with_suffix('.npz')))
        else:
            inputs.index.identity(path.with_suffix('.npz'),read(path)['arrays'])
    previous = read(target)
    if not changed and previous['input_identity']==dependency and len(previous['frames'])==len(selected):
        return previous
    invalidate_descendants(binding,scene,'observe','changed observer leaves',leaf=changed)
    return observe_scene(binding,scene)


def propose(binding, scene):
    from .proposals import propose_scene
    root = Path(binding['output_root'])
    target = root/'proposals'/scene/'receipt.json'
    if target.exists():
        inputs = load_scene(binding,scene)
        key = canonical_digest({'observer':read(root/'observations'/scene/'receipt.json')['identity'],
            'supports':inputs.units.identity,'support':binding['specification']['support'],
            'repair':binding['specification']['repair'],'producer':inputs.index.identity(Path(__file__).with_name('proposals.py')),
            'verification':inputs.index.identity(Path(__file__).with_name('verification.py'))})
        if read(target)['input_identity']!=key:
            invalidate_descendants(binding,scene,'propose','changed scene proposal inputs')
    return propose_scene(binding,scene)


def recognition_frame(binding, plan, fid, session, assets):
    root = Path(binding['output_root'])
    path = root/'recognition'/plan['scene']/'frames'/(str(fid)+'.json')
    if not path.exists() or read(path).get('status')!='COMPLETE':
        return
    index = ConsumptionIndex()
    module = Path(__file__).with_name('recognition_worker.py')
    producer = [index.identity(module),index.identity(module.parents[1]/'evidence_exploration/timing.py'),
        index.identity(module.parents[1]/'evidence_exploration/anyup_adapter.py'),
        index.identity(module.parents[1]/'cvpr_compact/area_fallback.py')]
    operators = canonical_digest({'producer':producer,'assets':assets['identity'],'model':session.model_key})
    group = plan['frames'][str(fid)]
    key = canonical_digest({'group':group,'regions':{r:plan['regions'][r] for r in group['regions']},
                            'operators':operators,'text':session.text_identity})
    if read(path)['input_identity']!=key:
        leaf = [str(path.relative_to(root)),str(Path(read(path)['vectors']['path']).relative_to(root))]
        invalidate_descendants(binding,plan['scene'],'recognition_frame','changed selected frame content',leaf=leaf)
