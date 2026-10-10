"""Selected-frame Python3 port of the pinned ScanNet sensor format.

Source: ScanNet/ScanNet@3830fce7f8b2e48ef047ef7fd76ea5f62903f51c,
SensReader/python/SensorData.py. Preserve sensor poses, scale and extrinsics;
do not apply the annotation exporter's axis alignment to only one modality.
"""
import io
from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image


def change_world_frame(xyz, poses, transform):
    transform = np.asarray(transform, np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError('Explicit finite world transform required')
    xyz = np.asarray(xyz, np.float64)
    return xyz @ transform[:3, :3].T + transform[:3, 3], [transform @ np.asarray(p) for p in poses]


def register_depth_rgb(rgb, raw_depth, kd, kc, e_cd, pose_wd, depth_shift):
    rgb = np.asarray(rgb)
    raw_depth = np.asarray(raw_depth)
    kd, kc, e_cd, pose_wd = [np.asarray(x, np.float64) for x in (kd, kc, e_cd, pose_wd)]
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or raw_depth.ndim != 2:
        raise ValueError('RGB uint8 HWC and measured depth grid required')
    if kd.shape != (3, 3) or kc.shape != (3, 3) or e_cd.shape != (4, 4) or pose_wd.shape != (4, 4):
        raise ValueError('Explicit camera calibration dimensions required')
    if not all(np.isfinite(x).all() for x in (kd, kc, e_cd, pose_wd)) or not np.isfinite(depth_shift) or depth_shift <= 0:
        raise ValueError('BLOCKED_CALIBRATION: finite camera and positive sensor depth scale required')
    depth = raw_depth.astype(np.float32) / np.float32(depth_shift)
    h, w = depth.shape; hc, wc = rgb.shape[:2]
    yy, xx = np.indices((h, w))
    pixels = np.column_stack((xx.ravel(), yy.ravel(), np.ones(h * w)))
    points = (pixels @ np.linalg.inv(kd).T) * depth.ravel()[:, None]
    color = points @ e_cd[:3, :3].T + e_cd[:3, 3]
    q = color @ kc.T
    valid = np.isfinite(depth.ravel()) & (depth.ravel() > 1e-6) & np.isfinite(color).all(1) & (color[:, 2] > 1e-6)
    uv = np.zeros((h * w, 2), np.float64)
    uv[valid] = q[valid, :2] / q[valid, 2, None]
    valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= wc - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= hc - 1)
    selected = np.flatnonzero(valid)
    rounded = np.floor(uv[selected] + .5).astype(np.int64)
    zbuffer = np.full(hc * wc, np.inf)
    flat = rounded[:, 1] * wc + rounded[:, 0]
    np.minimum.at(zbuffer, flat, color[selected, 2])
    visible = color[selected, 2] <= zbuffer[flat] + np.maximum(.02, .02 * color[selected, 2])
    valid[selected[~visible]] = False
    selected = selected[visible]
    aligned = np.zeros((h * w, 3), np.uint8)
    if len(selected):
        coords = uv[selected]
        x0, y0 = np.floor(coords).astype(np.int64).T
        x1, y1 = np.minimum(x0 + 1, wc - 1), np.minimum(y0 + 1, hc - 1)
        dx, dy = (coords - np.column_stack((x0, y0))).T
        colors = ((1-dx)[:, None]*(1-dy)[:, None]*rgb[y0, x0] +
                  dx[:, None]*(1-dy)[:, None]*rgb[y0, x1] +
                  (1-dx)[:, None]*dy[:, None]*rgb[y1, x0] +
                  dx[:, None]*dy[:, None]*rgb[y1, x1])
        aligned[selected] = np.floor(colors + .5).astype(np.uint8)
    return dict(rgb=aligned.reshape(h, w, 3), depth_m=depth, color_valid=valid.reshape(h, w),
                pose_wd=pose_wd, pose_wc=pose_wd @ np.linalg.inv(e_cd), intrinsics=kd,
                color_uv=uv.reshape(h, w, 2), convention='DEPTH_GRID_RGB_REGISTERED_E_DEPTH_TO_COLOR')


class SensorReader:
    """Index compressed offsets/poses without retaining whole sensor payloads."""
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as handle:
            def unpack(fmt):
                size = struct.calcsize('<' + fmt)
                value = handle.read(size)
                if len(value) != size:
                    raise ValueError('Incomplete sensor stream')
                return struct.unpack('<' + fmt, value)
            if unpack('I')[0] != 4:
                raise ValueError('BLOCKED_CALIBRATION: supported sensor version4 required')
            name_size = unpack('Q')[0]
            if name_size > 65536:
                raise ValueError('Invalid sensor header')
            self.sensor_name = handle.read(name_size).decode('utf8', errors='replace')
            matrices = [np.asarray(unpack('16f'), np.float64).reshape(4, 4) for _ in range(4)]
            self.kc, self.ec, self.kd, self.ed = matrices
            if not np.allclose(self.ec, np.eye(4), atol=1e-6, rtol=0):
                raise ValueError('BLOCKED_CALIBRATION: nonidentity color extrinsic needs explicit adapter')
            self.color_type, self.depth_type = unpack('ii')
            self.wc, self.hc, self.wd, self.hd = unpack('4I')
            self.depth_shift = unpack('f')[0]
            count = unpack('Q')[0]
            if min(self.wc, self.hc, self.wd, self.hd) <= 0 or self.depth_shift <= 0 or count > 1000000:
                raise ValueError('Invalid sensor dimensions/scale/frame count')
            if self.color_type not in (0, 1, 2) or self.depth_type not in (0, 1):
                raise ValueError('BLOCKED_RUNTIME: unsupported sensor compression')
            self.frames = []
            byte_count = self.path.stat().st_size
            for frame_id in range(count):
                pose = np.asarray(unpack('16f'), np.float64).reshape(4, 4)
                tc, td, colors, depths = unpack('4Q')
                color_offset = handle.tell()
                depth_offset = color_offset + colors
                end = depth_offset + depths
                if end > byte_count:
                    raise ValueError('Incomplete sensor frame')
                self.frames.append(dict(frame_id=frame_id, pose_c2w=pose.tolist(),
                     color_offset=color_offset, depth_offset=depth_offset, color_bytes=colors,
                     depth_bytes=depths, timestamp_color=tc, timestamp_depth=td,
                     finite_pose=bool(np.isfinite(pose).all())))
                handle.seek(end)

    def decode(self, frame_id):
        frame = self.frames[int(frame_id)]
        if not frame['finite_pose']:
            raise ValueError('Nonfinite sensor poses are never replaced by identity')
        with self.path.open('rb') as handle:
            handle.seek(frame['color_offset']); color = handle.read(frame['color_bytes'])
            handle.seek(frame['depth_offset']); depth = handle.read(frame['depth_bytes'])
        if self.color_type == 0:
            rgb = np.frombuffer(color, np.uint8).reshape(self.hc, self.wc, 3).copy()
        else:
            rgb = np.asarray(Image.open(io.BytesIO(color)).convert('RGB')).copy()
        if self.depth_type == 1:
            depth = zlib.decompress(depth)
        raw = np.frombuffer(depth, dtype='<u2').reshape(self.hd, self.wd).copy()
        if rgb.shape != (self.hc, self.wc, 3):
            raise ValueError('Sensor image header and decoder disagree')
        return register_depth_rgb(rgb, raw, self.kd[:3, :3], self.kc[:3, :3], self.ed,
                                  np.asarray(frame['pose_c2w']), self.depth_shift)

    def calibration(self):
        return dict(k_color=self.kc.tolist(), k_depth=self.kd.tolist(), e_color=self.ec.tolist(),
                    e_depth_to_color=self.ed.tolist(), depth_shift=self.depth_shift,
                    sensor_name=self.sensor_name, frame_count=len(self.frames),
                    color_compression=self.color_type, depth_compression=self.depth_type,
                    original_color_hw=[self.hc, self.wc], original_depth_hw=[self.hd, self.wd],
                    pose_convention='WORLD_FROM_DEPTH', mesh_frame='ORIGINAL_UNALIGNED')
