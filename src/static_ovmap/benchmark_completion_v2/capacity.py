"""Scene-boundary disk estimates; metadata only, with no change to model inputs."""
from pathlib import Path
import shutil

from .common import StageFailure, read, verified, write, source_identity


def visual_inputs_complete(root, benchmark, scene):
    root=Path(root)
    paths=[root/'benchmark'/part/benchmark/scene/name for part,name in (
        ('bundles','bundle.json'),('F','receipt.json'),('NQ','sources.json'))]
    return all(path.is_file() and verified(path).get('status')=='DONE' for path in paths)


def estimate_scene(binding, benchmark, scene):
    root=Path(binding['output_root'])
    inputs=[]
    mesh_bytes=0
    if benchmark=='scannet200_val':
        from .acquisition import scannet_scene
        from static_ovmap.module_validation.scannet_frames import sensor_header
        acquired=scannet_scene(binding,scene)
        sensor=Path(acquired['files']['.sens']['path'])
        with sensor.open('rb') as stream:
            header=sensor_header(stream)
        rgb_pixels=header['color_width']*header['color_height']
        depth_pixels=header['depth_width']*header['depth_height']
        frames=min(200,header['source_frame_count'])
        mesh_bytes=Path(acquired['files']['_vh_clean_2.ply']['path']).stat().st_size
        inputs=[dict(path=str(sensor),bytes=sensor.stat().st_size),
                dict(path=acquired['files']['_vh_clean_2.ply']['path'],bytes=mesh_bytes)]
    elif benchmark=='replica48':
        roots=(binding.get('inherited') or {}).get('data_roots',[])
        folder=next((Path(p) for p in roots if (Path(p)/scene/'traj.txt').is_file()
                     and (Path(p)/'cam_params.json').is_file()),None)
        if folder is None:
            raise StageFailure('Replica scene disk estimate needs its original camera metadata',missing=[scene+'/cam_params.json'])
        path=folder/'cam_params.json'
        camera=read(path)
        camera=camera.get('camera',camera)
        rgb_pixels=depth_pixels=int(camera['w'])*int(camera['h'])
        frames=200
        inputs=[dict(path=str(path),bytes=path.stat().st_size)]
    else:
        raise ValueError('Disk estimate outside the locked benchmark scope')
    if min(rgb_pixels,depth_pixels,frames)<=0:
        raise ValueError('Scene disk estimate requires positive original image dimensions')
    # Planning allowance: original RGB/depth plus128 bytes per selected pixel
    # for mapping/surface/projection copies;16 copies of the evaluation mesh.
    # These are conservative storage assumptions, not measured memory/FPS.
    planned=frames*(3*rgb_pixels+2*depth_pixels+128*max(rgb_pixels,depth_pixels))+16*mesh_bytes
    previous=[verified(path)['task_owned_bytes'] for path in (root/'resources/storage').glob('*/*.json')]
    observed=max(previous,default=0)
    return dict(estimated_bytes=max(planned,observed),pixel_plan_bytes=planned,
        maximum_completed_scene_bytes=observed,selected_frame_upper_bound=frames,
        original_rgb_pixels=rgb_pixels,original_depth_pixels=depth_pixels,inputs=inputs,
        estimate_is_measured_consumption=False,GT_labels_read=False,
        rule='MAX(ORIGINAL_DIMENSIONS_PIXEL_COPY_ALLOWANCE,COMPLETED_TASK_SCENE_BYTES)')


def check_scene(binding, benchmark, scene):
    root=Path(binding['output_root'])
    if visual_inputs_complete(root,benchmark,scene):
        return dict(status='DONE',visual_inputs_already_complete=True)
    from .runtime import check_gpu_budget
    check_gpu_budget(binding)
    estimate=estimate_scene(binding,benchmark,scene)
    reserve=int(binding['specification']['runtime']['minimal_disk_reserve_GiB']*2**30)
    required=reserve+2*estimate['estimated_bytes']
    free=shutil.disk_usage(root).free
    path=root/'resources/capacity'/benchmark/(scene+'.json')
    record=write(path,dict(status='DONE' if free>=required else 'BLOCKED_RESOURCE',
        benchmark=benchmark,scene=scene,free_bytes=free,required_free_bytes=required,
        minimal_reserve_bytes=reserve,next_scene_buffer_multiplier=2,estimate=estimate,
        producer=source_identity(('capacity',))))
    if free<required:
        raise StageFailure('Scene-boundary disk reserve is below20GiB plus twice the next-scene estimate',
                           status='BLOCKED_RESOURCE',missing=[str(path)])
    return record


def observe_scene(binding, benchmark, scene):
    root=Path(binding['output_root'])
    path=root/'resources/storage'/benchmark/(scene+'.json')
    if path.is_file():
        return verified(path)
    directories=[root/'benchmark'/part/benchmark/scene for part in
                 ('frames','native','bundles','F','NQ','predictions','evaluation','GT')]
    if benchmark=='scannet200_val':
        directories.append(root/'assets/datasets/scannet/scans'/scene)
    sizes={str(folder.relative_to(root)):sum(p.stat().st_size for p in folder.rglob('*')
             if p.is_file() and not p.is_symlink()) for folder in directories if folder.is_dir() and not folder.is_symlink()}
    return write(path,dict(status='DONE',benchmark=benchmark,scene=scene,task_owned_bytes=sum(sizes.values()),
        directory_bytes=sizes,raw_shared_assets_scanned=False,GT_label_values_read=False,
        collection='KNOWN_TASK_SCENE_DIRECTORIES;FILESYSTEM_SIZES_ONLY'))
