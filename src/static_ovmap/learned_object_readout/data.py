"""Supervised proposal generation; annotation targets never enter model inputs."""
import csv
import hashlib
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree


def instance_arrays(segment_ids, groups, label_map, valid_ids):
    segments = np.asarray(segment_ids)
    owners = np.zeros(len(segments), np.int64)
    labels = np.zeros(len(segments), np.int64)
    registry = {}
    for group in groups:
        label = label_map.get(group['label'])
        if label not in valid_ids:
            continue
        owner = int(group['id']) + 1
        if owner <= 0 or owner in registry:
            raise ValueError('Aggregation group mapping must be positive and reversible')
        mask = np.isin(segments, group['segments'])
        if np.any(owners[mask] != 0):
            raise ValueError('Overlapping official instance segments')
        owners[mask] = owner; labels[mask] = int(label)
        registry[owner] = dict(original_group_id=int(group['id']), class_id=int(label))
    return owners, labels, registry


def vertex_area(xyz, faces):
    xyz, faces = np.asarray(xyz, np.float64), np.asarray(faces, np.int64)
    triangles = xyz[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1) / 2
    mass = np.zeros(len(xyz), np.float64)
    np.add.at(mass, faces.ravel(), np.repeat(areas / 3, 3))
    return mass


def perturb_support(xyz, area, owners, owner, direction):
    xyz, area, owners = np.asarray(xyz), np.asarray(area, np.float64), np.asarray(owners)
    rows = np.flatnonzero(owners == owner)
    if not len(rows) or area[rows].sum() <= 0 or direction not in range(6):
        raise ValueError('Valid annotated surface and fixed signed axis required')
    axis, sign = direction // 2, (1 if direction % 2 == 0 else -1)
    order = rows[np.lexsort((rows, sign * xyz[rows, axis]))]
    mass = np.cumsum(area[order]); end = int(np.searchsorted(mass, .70 * area[rows].sum(), side='left')) + 1
    clean = owners == owner
    truncated = np.zeros(len(xyz), bool); truncated[order[:end]] = True
    other = np.flatnonzero((owners > 0) & (owners != owner) & (area > 0))
    donor = np.zeros(len(xyz), bool)
    if len(other):
        distance = cKDTree(xyz[rows]).query(xyz[other], workers=1)[0]
        close = distance <= .05
        candidates = other[close]
        candidates = candidates[np.lexsort((candidates, distance[close]))]
        eligible = np.cumsum(area[candidates]) <= .20 * area[rows].sum()
        donor[candidates[eligible]] = True
    return dict(clean=clean, truncate=truncated, append=clean | donor, truncate_append=truncated | donor)


CONDITIONS = ('clean', 'truncate', 'append', 'truncate_append')


def proposal_sites(physical, owners, support, proposal_limit=48, context_limit=16):
    from .observations import site_indices
    coordinates = physical['coordinate_ids']
    count = len(physical['coordinates'])
    selected_codes = owners[support] * count + physical['row_to_coordinate'][support]
    codes = physical['owners'] * count + coordinates
    selected = np.isin(codes, selected_codes)
    inside = np.flatnonzero(selected)
    outside = np.flatnonzero(~selected)
    if not len(inside):
        return np.empty(0, np.int64)
    chosen = inside[site_indices(physical['xyz'][inside], physical['area'][inside], proposal_limit)]
    if len(outside):
        distance = cKDTree(physical['xyz'][inside]).query(physical['xyz'][outside], workers=1)[0]
        candidates = outside[distance <= .05]
        context = candidates[site_indices(physical['xyz'][candidates], physical['area'][candidates], context_limit)]
        chosen = np.r_[chosen, context]
    return chosen.astype(np.int64)


def site_observation(points, frame, depth, full, ray_scene, bbox):
    from static_ovmap.disagreement_query.physical_support import direct_visibility
    points = np.asarray(points, np.float64)
    visible, _, rounded, _ = direct_visibility(points, frame, depth, full, ray_scene)
    pose, k = np.asarray(frame['pose_c2w']), np.asarray(frame['intrinsics'])
    inv = np.linalg.inv(pose)
    camera = points @ inv[:3, :3].T + inv[:3, 3]
    q = camera @ k.T
    uv = np.zeros((len(points), 2), np.float64)
    front = camera[:, 2] > 1e-6
    uv[front] = q[front, :2] / q[front, 2, None]
    h, w = full.shape
    valid = visible & (uv[:, 0] >= 0) & (uv[:, 0] <= w-1) & (uv[:, 1] >= 0) & (uv[:, 1] <= h-1)
    quality = np.zeros((len(points), 8), np.float32)
    rows = np.flatnonzero(valid)
    if len(rows):
        x0, y0 = np.floor(uv[rows]).astype(np.int64).T
        x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
        dx, dy = (uv[rows] - np.column_stack((x0, y0))).T
        occupancy = ((1-dx)*(1-dy)*full[y0, x0] + dx*(1-dy)*full[y0, x1] +
                     (1-dx)*dy*full[y1, x0] + dx*dy*full[y1, x1])
        measured = depth[rounded[rows, 1], rounded[rows, 0]]
        residual = np.minimum(1, np.abs(measured-camera[rows, 2]) / np.maximum(.02, .02*camera[rows, 2]))
        direction = pose[:3, 3] - points[rows]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-6)
        lo, hi = bbox
        centered = (points[rows] - .5*(lo+hi)) / max(float(np.linalg.norm(hi-lo)), 1e-6)
        quality[rows] = np.column_stack((occupancy, residual, direction, centered)).astype(np.float32)
    return uv, quality, valid


def _label_map(path):
    result = {}
    with Path(path).open() as handle:
        for row in csv.DictReader(handle, delimiter='\t'):
            # Match official point_indices_from_group: first raw_category row.
            result.setdefault(row['raw_category'], int(row['id']))
    return result


def _mesh(path):
    from plyfile import PlyData
    value = PlyData.read(path)
    vertices = value['vertex'].data
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    faces = np.stack(value['face'].data['vertex_indices']).astype(np.int64)
    if not np.isfinite(xyz).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError('Original unaligned triangular ScanNet mesh required')
    return xyz, faces


def prepare_family(binding, selected, role):
    from PIL import Image
    from static_ovmap.minimal_instance_repair.observations import SourceRowProjector, pose_banks
    from static_ovmap.disagreement_query.physical_support import canonical_sites, quadrature, direct_visibility
    from .common import ConsumptionIndex, arrays_record, canonical_digest, read, verified, write
    from .observations import full_qualified, choose_views
    from .sensor import SensorReader
    root = Path(binding['output_root']); scene = selected['scene']; family = selected['family']
    if role == 'holdout':
        main, repeat = root / 'dev_nomination.json', root / 'repeat_nomination.json'
        if not main.exists() or not repeat.exists() or verified(main)['status'] != 'NOMINATED' or verified(repeat)['status'] != 'NOMINATED':
            raise ValueError('H annotations require both seed nominations')
    classes = verified(root / 'class_split.json')
    dest = root / 'data/generated' / scene
    index = ConsumptionIndex(root / 'data/verifications.json')
    files = {k: Path(v) for k, v in selected['files'].items()}
    dependencies = {k: index.identity(v) for k, v in files.items()}
    input_identity = canonical_digest(dict(files=dependencies, role=role, classes=classes['identity'],
                  spec=binding['spec']['sha256'], producer=index.identity(__file__)))
    manifest_path = dest / 'manifest.json'
    if manifest_path.exists():
        previous = verified(manifest_path)
        if previous['input_identity'] != input_identity:
            raise ValueError('Supervised family generation inputs changed')
        return previous
    started = time.perf_counter()
    sensor = SensorReader(files['.sens'])
    finite = [frame for frame in sensor.frames if frame['finite_pose']]
    frames, bins = pose_banks(finite, limit=96)
    xyz, faces = _mesh(files['_vh_clean_2.ply'])
    segments = np.asarray(read(files['_vh_clean_2.0.010000.segs.json'])['segIndices'])
    groups = read(files['.aggregation.json'])['segGroups']
    if len(segments) != len(xyz):
        raise ValueError('Mesh and official segment annotations differ')
    label_tsv = root / 'data/scannet/scannetv2-labels.combined.tsv'
    index.identity(label_tsv)
    owners, _, registry = instance_arrays(segments, groups, _label_map(label_tsv), classes['official_ids'])
    area = vertex_area(xyz, faces)
    physical = canonical_sites(xyz, faces, owners)
    projector = SourceRowProjector(xyz, faces)
    arrays_record(dest / 'source_geometry_and_targets.npz', dict(xyz=xyz, faces=faces, owners=owners, vertex_area=area), index)
    generated, frame_arrays = [], []
    for original in frames:
        fid = int(original['frame_id'])
        decoded = sensor.decode(fid)
        frame = dict(frame_id=fid, pose_c2w=original['pose_c2w'], intrinsics=decoded['intrinsics'].tolist(),
                     image_size_hw=list(decoded['depth_m'].shape), convention=decoded['convention'])
        source_rows, valid = projector.project(decoded['intrinsics'], np.asarray(frame['pose_c2w']), decoded['depth_m'])
        valid &= decoded['color_valid']
        path = dest / 'frames' / (f'{fid:06d}.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(decoded['rgb']).save(path)
        record = arrays_record(path.with_suffix('.npz'), dict(depth_m=decoded['depth_m'], color_valid=decoded['color_valid'],
                              source_rows=source_rows, source_valid=valid), index)
        frame.update(rgb=index.identity(path), arrays=record)
        generated.append(frame)
        frame_arrays.append((source_rows, valid, decoded['depth_m']))
    objects, excluded = [], []
    ordered = sorted(registry, key=lambda owner: hashlib.sha256((family+'|'+str(registry[owner]['original_group_id'])).encode()).hexdigest())
    for owner in ordered:
        rows = np.flatnonzero(owners == owner)
        if len(rows) < 100 or area[rows].sum() <= 0:
            excluded.append(dict(owner=owner, reason='INSUFFICIENT_ORIGINAL_VERTICES_OR_AREA'))
            continue
        direction = int(hashlib.sha256((family+'|'+str(registry[owner]['original_group_id'])+'|direction').encode()).hexdigest(), 16) % 6
        supports = perturb_support(xyz, area, owners, owner, direction)
        masks = [[] for _ in CONDITIONS]
        common = []
        for fi, (source_rows, valid, _) in enumerate(frame_arrays):
            for ci, condition in enumerate(CONDITIONS):
                mask = np.zeros(valid.shape, bool)
                mask[valid] = supports[condition][source_rows[valid]]
                masks[ci].append(mask)
            if all(full_qualified(masks[ci][-1]) for ci in range(4)):
                common.append(fi)
        if len(common) < 2:
            excluded.append(dict(owner=owner, reason='NO_TWO_COMMON_FULL_VIEWS', common_views=len(common)))
            continue
        owner_sites = physical['owners'] == owner
        planned_xyz, planned_area = quadrature(physical['xyz'][owner_sites], physical['area'][owner_sites], 4096)
        footprints = []
        for fi in common:
            footprint = direct_visibility(planned_xyz, generated[fi], frame_arrays[fi][2], masks[0][fi], projector.mesh.scene)[1]
            footprints.append(footprint)
        bank = choose_views([generated[i]['frame_id'] for i in common], [masks[0][i] for i in common], footprints, planned_area)
        chosen = [common[i] for i in bank]
        nviews = len(chosen)
        uv = np.zeros((4, nviews, 64, 2), np.float64)
        quality = np.zeros((4, nviews, 64, 8), np.float32)
        local_valid = np.zeros((4, nviews, 64), bool)
        membership = np.full((4, nviews, 64), -1, np.int8)
        site_xyz = np.zeros((4, 64, 3), np.float64)
        site_available = np.zeros((4, 64), bool)
        for ci, condition in enumerate(CONDITIONS):
            selected_sites = proposal_sites(physical, owners, supports[condition])
            points = physical['xyz'][selected_sites]
            site_xyz[ci, :len(points)] = points
            site_available[ci, :len(points)] = True
            support_points = xyz[supports[condition]]
            bbox = (support_points.min(0), support_points.max(0))
            truth = (physical['owners'][selected_sites] == owner).astype(np.int8)
            for vi, fi in enumerate(chosen):
                pixels, metadata, visible = site_observation(points, generated[fi], frame_arrays[fi][2], masks[ci][fi], projector.mesh.scene, bbox)
                safe = np.floor(pixels + .5).astype(np.int64)
                ids = np.flatnonzero(visible)
                visible[ids] &= frame_arrays[fi][1][safe[ids, 1], safe[ids, 0]]
                uv[ci, vi, :len(points)] = pixels
                quality[ci, vi, :len(points)] = metadata
                local_valid[ci, vi, :len(points)] = visible
                membership[ci, vi, :len(points)] = np.where(visible, truth, -1)
        predictor = arrays_record(dest / 'objects' / f'{owner:06d}.inputs.npz',
                   dict(masks=np.stack([[masks[ci][fi] for fi in chosen] for ci in range(4)]),
                        uv=uv, metadata=quality, local_valid=local_valid, site_xyz=site_xyz, site_available=site_available), index)
        targets = arrays_record(dest / 'objects' / f'{owner:06d}.targets.npz',
                  dict(membership=membership, class_id=np.asarray(registry[owner]['class_id'], np.int64)), index)
        objects.append(dict(key=scene+':'+str(owner), scene=scene, family=family, role=role,
               original_group_id=registry[owner]['original_group_id'], original_local_id=owner,
               class_id=registry[owner]['class_id'], base=registry[owner]['class_id'] in classes['base_ids'],
               views=[generated[fi] for fi in chosen], inputs=predictor, targets=targets,
               support_vertices={c: int(supports[c].sum()) for c in CONDITIONS},
               original_vertex_count=len(rows), original_area=float(area[rows].sum()), direction=direction,
               no_donor=bool(np.array_equal(supports['append'], supports['clean'])),
               no_op_conditions=[c for c in CONDITIONS[1:] if np.array_equal(supports[c], supports['clean'])]))
        if len(objects) == 64:
            break
    result = write(manifest_path, dict(status='COMPLETE', scene=scene, family=family, role=role,
             input_identity=input_identity, objects=objects, excluded=excluded, frames=generated,
             calibration=sensor.calibration(), finite_frames=len(finite), nonfinite_frames=len(sensor.frames)-len(finite),
             pose_representatives=len(frames), base_objects=sum(o['base'] for o in objects),
             original_objects=len(objects), annotation_targets_separate=True, predictor_GT_fields=False,
             physical_support=physical['audit'], ray_depth_valid_pixels=sum(int(a[1].sum()) for a in frame_arrays),
             supervision='GT_DERIVED_PROPOSAL_RECOGNITION', elapsed_seconds=time.perf_counter()-started))
    index.write_memo(root / 'data/verifications.json')
    print('PREPARED', scene, role, len(objects), 'base', result['base_objects'], flush=True)
    return result


def prepare_dataset(binding, *, holdout=False):
    from .common import verified, write
    from .sensor import SensorReader
    root = Path(binding['output_root'])
    split = verified(root / 'split_manifest.json')
    roles = ('holdout',) if holdout else ('train', 'dev')
    # Header/calibration inventory is permitted for H; never parse its annotations here.
    if not holdout:
        calibration = {row['scene']: SensorReader(row['files']['.sens']).calibration() for row in split['selected']}
        write(root / 'data/calibration_inventory.json', dict(status='COMPLETE', scenes=calibration, holdout_annotation_reads=0))
    by_scene = {row['scene']: row for row in split['selected']}
    result = {role: [prepare_family(binding, by_scene[scene], role) for scene in split['roles'][role]] for role in roles}
    objects = {role: [obj for family in result[role] for obj in family['objects']] for role in roles}
    for role in roles:
        if any(family['base_objects'] < 1 for family in result[role]):
            raise ValueError('BLOCKED_TRAINING_DATA: chosen family has no usable original base object')
        required = dict(train=240, dev=32, holdout=32)[role]
        if sum(obj['base'] for obj in objects[role]) < required:
            raise ValueError(f'BLOCKED_TRAINING_DATA: {role} original base support below{required}')
    if not holdout and len({obj['class_id'] for obj in objects['train'] if obj['base']}) < 8:
        raise ValueError('BLOCKED_TRAINING_DATA: fewer than8 usable training base classes')
    profile = dict(status='COMPLETE', split=split['identity'], roles={role: dict(original_objects=len(objects[role]),
             base_objects=sum(o['base'] for o in objects[role]), base_categories=len({o['class_id'] for o in objects[role] if o['base']}),
             families=len(result[role])) for role in roles}, supervision='GT_DERIVED_PROPOSALS',
             holdout_annotations_processed=holdout, family_manifests=[r['identity'] for rows in result.values() for r in rows])
    write(root / ('holdout_data_profile.json' if holdout else 'data_profile.json'), profile)
    return objects, result
