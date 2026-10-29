"""GT-free all-owner geometric supports and the original FC eight-view selection."""
from pathlib import Path
import time
import numpy as np

from .common import (verified, read, write, save_arrays, ConsumedInputs, Resolver,
                     canonical_digest, source_identity, StageFailure)


def _mesh(path):
    from plyfile import PlyData
    data = PlyData.read(str(path))
    xyz = np.column_stack([data['vertex'][name] for name in ('x', 'y', 'z')]).astype(np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('Evaluation geometry must contain finite original XYZ vertices')
    return xyz


def target_geometry(binding, benchmark, scene):
    from .acquisition import scannet_scene
    root = Path(binding['output_root'])
    if benchmark == 'scannet200_val':
        acquired = scannet_scene(binding, scene)
        return Path(acquired['files']['_vh_clean_2.ply']['path'])
    config = binding['asset_config'].get('datasets', {}).get('replica48', {})
    folder = config.get('evaluation_mesh_root')
    options = [Path(folder) / (scene + '.ply')] if folder else []
    options.append(root / 'assets/datasets/replica48/geometry' / (scene + '.ply'))
    found = next((path for path in options if path.is_file()), None)
    if found is None:
        raise StageFailure('Replica48 author evaluation geometry/point ordering is absent',
            missing=[str(path) for path in options] + ['author replica_3d/' + scene + '.pth coordinate sidecar'])
    return found


def build(binding, benchmark, scene):
    from .scene_bundle import prepare_native_scene
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector
    from static_ovmap.learned_object_readout.observations import full_qualified, choose_views
    from static_ovmap.disagreement_query.physical_support import canonical_sites, quadrature, direct_visibility
    from static_ovmap.module_validation.scannet_study import freeze_projection, project_values
    from static_ovmap.module_validation.native_capture import _array_digest

    root = Path(binding['output_root'])
    target = target_geometry(binding, benchmark, scene)
    native = prepare_native_scene(binding, benchmark, scene)
    capture_path = Path(native['capture_manifest']['path'])
    capture = read(capture_path)
    index = ConsumedInputs(root)
    from .benchmark import template_context
    expected_extension = template_context(binding, benchmark)['native_extension']
    actual_extension = capture['native_extension']
    if actual_extension['sha256'] != expected_extension['sha256']:
        raise ValueError('Actual native extension differs from the inherited numerical implementation')
    index.identity(actual_extension['path'], expected_extension)
    surface_path = capture_path.parent / capture['surface']['path']
    index.identity(surface_path, capture['surface'])
    target_record = index.identity(target)
    identity = canonical_digest(dict(native=native['identity'], target=target_record['sha256'],
        producer=source_identity(('native_inputs',)), selection='original_choose_views_max8;all_owners;one_view_valid'))
    folder = root / 'benchmark/bundles' / benchmark / scene
    receipt = folder / 'bundle.json'
    if receipt.exists():
        previous = verified(receipt)
        if previous['input_identity'] != identity:
            raise ValueError('Completed SceneBundle changed; preserve before affected repair')
        for row in previous['arrays'].values():
            index.identity(row['path'], row)
        return previous
    start = time.monotonic()
    with np.load(surface_path, allow_pickle=False) as arrays:
        xyz, faces, owners = (arrays[name].copy() for name in ('surface_xyz', 'surface_faces', 'original_owner'))
    if xyz.shape != (len(owners), 3) or (owners < 0).any():
        raise ValueError('Original native all-owner partition has invalid dimensions')
    owner_ids = np.unique(owners[owners > 0])
    target_xyz = _mesh(target)
    projection = freeze_projection(xyz, target_xyz, folder / 'projection')
    target_owners = project_values(owners, projection['nearest'], projection['matched'])
    indices = [np.flatnonzero(target_owners == owner) for owner in owner_ids]
    indptr = np.r_[0, np.cumsum([len(row) for row in indices])].astype(np.int64)
    mask_indices = np.concatenate(indices).astype(np.int64) if indices else np.empty(0, np.int64)
    geometry_path = folder / 'geometry.npz'
    save_arrays(geometry_path, source_xyz=xyz, source_faces=faces, source_owners=owners,
        owners=owner_ids, target_xyz=target_xyz, target_nearest=projection['nearest'],
        target_matched=projection['matched'], indptr=indptr, indices=mask_indices)
    geometry = index.identity(geometry_path)
    projector = SourceRowProjector(xyz, faces)
    physical = canonical_sites(xyz, faces, owners)
    bank, projections, depth_bank = [], [], []
    frame_records = {}
    for frame in capture['frames']:
        fid = int(frame['frame_id'])
        depth_path = capture_path.parent / frame['depth_path']
        depth_record = index.identity(depth_path, dict(sha256=frame['depth_sha256']))
        with np.load(depth_path, allow_pickle=False) as arrays:
            depth = arrays['depth_m'].astype(np.float32)
        metadata = dict(frame_id=fid, pose_c2w=frame['pose_c2w'], intrinsics=frame['intrinsics'],
                        image_size_hw=frame['image_size_hw'])
        rgb = index.identity(capture_path.parent / frame['rgb_path'], dict(sha256=frame['rgb_sha256']))
        source_rows, valid = projector.project(metadata['intrinsics'], np.asarray(metadata['pose_c2w']), depth)
        array_path = folder / 'frame_projection' / (str(fid) + '.npz')
        save_arrays(array_path, source_rows=source_rows, valid=valid)
        frame_records[str(fid)] = dict(metadata, rgb=rgb, depth=depth_record,
                                       projection=index.identity(array_path))
        bank.append(metadata)
        projections.append((source_rows, valid))
        depth_bank.append(depth)
    objects, requests = [], {}
    for owner in owner_ids:
        owner = int(owner)
        masks = []
        for source_rows, valid in projections:
            mask = np.zeros(valid.shape, bool)
            mask[valid] = owners[source_rows[valid]] == owner
            masks.append(mask)
        qualified = [i for i, mask in enumerate(masks) if full_qualified(mask)]
        physical_owner = physical['owners'] == owner
        chosen = []
        if qualified and physical_owner.any():
            points, area = quadrature(physical['xyz'][physical_owner], physical['area'][physical_owner], 4096)
            footprints = [direct_visibility(points, bank[i], depth_bank[i], masks[i], projector.mesh.scene)[1] for i in qualified]
            chosen = [qualified[i] for i in choose_views([bank[i]['frame_id'] for i in qualified],
                [masks[i] for i in qualified], footprints, area, maximum=8)]
        ids = [bank[i]['frame_id'] for i in chosen]
        input_path = folder / 'object_masks' / (str(owner) + '.npz')
        # A zero-view object remains explicit; no artificial mask/vector is made.
        selected_masks = np.stack([masks[i] for i in chosen]) if chosen else np.empty((0, 0, 0), bool)
        save_arrays(input_path, masks=selected_masks, frame_ids=np.asarray(ids, np.int64))
        descriptor = index.identity(input_path)
        views = []
        for vi, i in enumerate(chosen):
            frame = frame_records[str(bank[i]['frame_id'])]
            request_id = canonical_digest(dict(scene=scene, owner=owner, frame=bank[i]['frame_id'],
                rgb=frame['rgb']['sha256'], mask=_array_digest(masks[i]), geometry=geometry['sha256']))
            request = dict(request_id=request_id, owner=owner, frame_id=bank[i]['frame_id'],
                image_sha256=frame['rgb']['sha256'], visible_pixels=int(masks[i].sum()),
                target_mask_sha256=_array_digest(masks[i]), input_masks=descriptor, mask_index=vi,
                rgb=frame['rgb'], view_rank=vi)
            requests[request_id] = request
            views.append(request_id)
        objects.append(dict(owner=owner, key=scene + ':native:' + str(owner), views=views,
            qualified_views=len(qualified), chosen_views=len(chosen), original_surface_points=int((owners == owner).sum()),
            evaluation_support_points=int((target_owners == owner).sum()), masks=descriptor,
            supported=bool(chosen), reason='F_VIEW_AVAILABLE' if chosen else 'NO_QUALIFIED_F_VIEW'))
        print('BUNDLE_OWNER', scene, owner, len(chosen), 'qualified', len(qualified), flush=True)
    protocol = verified(root / 'protocol_lock.json')
    scope = protocol['scopes'][benchmark]
    result = write(receipt, dict(status='DONE', input_identity=identity, scene=scene, benchmark=benchmark,
        format='OVMAP_SCENE_BUNDLE_V1', native=native['identity'], capture=native['capture_manifest'],
        deferred_metadata=native['deferred_metadata'], arrays=dict(geometry=geometry), frames=frame_records,
        objects=objects, requests=requests, text_ids=scope['text_ids'], text_names=scope['text_names'],
        model_lock=verified(root / 'model_lock.json')['identity'], protocol_lock=protocol['identity'],
        native_owner_count=len(objects), owners_with_F_view=sum(row['supported'] for row in objects),
        native_extension=dict(actual=actual_extension, expected=expected_extension),
        zero_F_view_owners=sum(not row['supported'] for row in objects),
        one_F_view_owners=sum(row['chosen_views'] == 1 for row in objects),
        evaluation_geometry=target_record, projection_identity=projection['identity'],
        projection_rule='AUTHOR_OVI_Open3D_FP32_1NN_STRICT_SQUARED_DISTANCE_LT_.05**2',
        unmatched_evaluation_points=int((~projection['matched']).sum()),
        duplicate_nearest_source_rows=int(len(projection['nearest'][projection['matched']]) - len(np.unique(projection['nearest'][projection['matched']]))),
        all_original_owners_retained=True, owner_domain_digest=_array_digest(owner_ids),
        maximum_F_views=8, F_selection='UNCHANGED_PARENT_CHOOSE_VIEWS_AREA_INCREMENT',
        G1_recovery=False, GT_labels_read=False, GT_labels_sidecar_only=True,
        elapsed_seconds=time.monotonic() - start))
    index.save()
    return result
