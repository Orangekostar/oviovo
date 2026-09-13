"""GT-free visibility witnesses from native geometry and measured RGB-D poses."""
import numpy as np


def native_pixel_rays(intrinsics, camera_to_world, width, height):
    """Match native depth_to_3d: integer (u,v), camera depth is camera-z."""
    k, pose = np.asarray(intrinsics), np.asarray(camera_to_world)
    y, x = np.indices((height, width), dtype=np.float64)
    camera = np.stack([(x-k[0,2])/k[0,0], (y-k[1,2])/k[1,1], np.ones_like(x)], axis=-1)
    directions = camera @ pose[:3,:3].T
    rays = np.empty((height,width,6), dtype=np.float32)
    rays[...,:3] = pose[:3,3]
    rays[...,3:] = directions
    return rays


def supported_owner_pixels(owners, predicted_depth, measured_depth, tolerance=.05):
    owners, predicted_depth, measured_depth = map(np.asarray, (owners, predicted_depth, measured_depth))
    if owners.shape != predicted_depth.shape or owners.shape != measured_depth.shape:
        raise ValueError('owner and depth images must align')
    valid = ((owners > 0) & np.isfinite(predicted_depth) & np.isfinite(measured_depth)
             & (measured_depth > 0) & (predicted_depth > 0)
             & (np.abs(predicted_depth - measured_depth) < tolerance))
    ids, counts = np.unique(owners[valid], return_counts=True)
    return dict(zip(map(int, ids), map(int, counts)))


def independent_support_frames(evidence, poses, min_pixels=100,
                               translation_m=.05, rotation_degrees=5.):
    result = {}
    for frame, counts in sorted(evidence.items()):
        pose = np.asarray(poses[frame])
        for owner, pixels in counts.items():
            if pixels < min_pixels:
                continue
            entry = result.setdefault(owner, {'support_frame_ids': [], 'independent_frame_ids': []})
            entry['support_frame_ids'].append(frame)
            distinct = True
            for previous in entry['independent_frame_ids']:
                other = np.asarray(poses[previous])
                translation = float(np.linalg.norm(pose[:3, 3] - other[:3, 3]))
                angle = np.degrees(np.arccos(np.clip((np.trace(pose[:3, :3].T @ other[:3, :3])-1)/2, -1, 1)))
                if translation < translation_m and angle < rotation_degrees:
                    distinct = False
                    break
            if distinct:
                entry['independent_frame_ids'].append(frame)
    return result
