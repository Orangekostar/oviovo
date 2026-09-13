"""Lossless native partitions with explicit compact-format ID remapping."""
import hashlib

import numpy as np


def _wide_ids(values):
    values = np.asarray(values)
    if (values.ndim != 1 or not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0) or (values.size and int(values.max()) > np.iinfo(np.int64).max)):
        raise ValueError('one-dimensional nonnegative IDs within int64 required')
    return values.astype(np.int64, copy=False)


def remap_partition(original, components):
    original = _wide_ids(original)
    mapping = {0: 0}
    for component in components:
        if not component or any(int(n) <= 0 or int(n) in mapping for n in component):
            raise ValueError('components must contain each positive owner exactly once')
        if len(set(component)) != len(component):
            raise ValueError('duplicate owner in component')
        mapping.update({int(n): min(component) for n in component})
    if set(mapping)-{0} != set(map(int, np.unique(original)))-{0}:
        raise ValueError('components must cover exactly the original native owners')
    keys = np.array(sorted(mapping), dtype=np.int64)
    values = np.array([mapping[int(k)] for k in keys], dtype=np.int64)
    changed = values[np.searchsorted(keys, original)]
    return changed, partition_change_ledger(original, changed)


def partition_change_ledger(original, changed):
    original, changed = _wide_ids(original), _wide_ids(changed)
    if original.shape != changed.shape or np.any((original == 0) != (changed == 0)):
        raise ValueError('partition changes must preserve native vertices and unknown background')
    positions = np.flatnonzero(changed != original)
    ledger = {'positions': positions, 'original_ids': original[positions],
              'changed_sha256': hashlib.sha256(changed.tobytes()).hexdigest(),
              'original_sha256': hashlib.sha256(original.tobytes()).hexdigest(),
              'vertex_count': len(original)}
    return ledger


def restore_partition(changed, ledger):
    changed = _wide_ids(changed)
    if (len(changed) != ledger['vertex_count']
            or hashlib.sha256(changed.tobytes()).hexdigest() != ledger['changed_sha256']):
        raise ValueError('restoration ledger does not match this partition')
    original = changed.copy()
    original[ledger['positions']] = ledger['original_ids']
    if hashlib.sha256(original.tobytes()).hexdigest() != ledger['original_sha256']:
        raise ValueError('restoration ledger is corrupt')
    return original


def compact_export(partition, dtype):
    partition = _wide_ids(partition)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.integer):
        raise ValueError('integer export dtype required')
    owners = np.unique(partition[partition > 0])
    if len(owners) > np.iinfo(dtype).max:
        raise ValueError('too many owners for requested export dtype; use a wider format')
    keys = np.concatenate((np.array([0], dtype=np.int64), owners))
    encoded = np.searchsorted(keys, partition).astype(dtype)
    return encoded, {i: int(owner) for i, owner in enumerate(keys)}


def write_partition_mesh(points, faces, partition, path):
    """Write original geometry with uint32 instance_id and reversible RGB identity.

    This is a native partition artifact, not a geometry reconstruction. RGB encodes
    compact identity; it is not an appearance texture or a semantic class color.
    """
    from plyfile import PlyData, PlyElement

    points, faces, partition = np.asarray(points), np.asarray(faces), _wide_ids(partition)
    if (points.shape != (len(partition), 3) or points.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(points).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer) or np.any(faces < 0)
            or np.any(faces >= len(points)) or np.any(faces > np.iinfo(np.int32).max)
            or (partition.size and int(partition.max()) > np.iinfo(np.uint32).max)):
        raise ValueError('geometry or native IDs cannot be exported losslessly in this PLY schema')
    encoded, inverse = compact_export(partition, np.uint32)
    if len(inverse) > 2**24:
        raise ValueError('too many owners for collision-free 24-bit RGB identity')
    vertex = np.empty(len(points), dtype=[(c, points.dtype) for c in ('x', 'y', 'z')]
                       + [(c, 'u1') for c in ('red', 'green', 'blue')] + [('instance_id', 'u4')])
    for axis, name in enumerate(('x', 'y', 'z')):
        vertex[name] = points[:, axis]
    for shift, name in enumerate(('red', 'green', 'blue')):
        vertex[name] = (encoded >> (8*shift)) & 255
    vertex['instance_id'] = partition
    triangles = np.empty(len(faces), dtype=[('vertex_indices', 'i4', (3,))])
    triangles['vertex_indices'] = faces
    PlyData([PlyElement.describe(vertex, 'vertex'), PlyElement.describe(triangles, 'face')],
            text=False, byte_order='<').write(path)
    return {'vertex_count': len(points), 'face_count': len(faces), 'instance_id_dtype': 'uint32',
            'coordinates': 'exact_input_values', 'faces': 'exact_input_indices',
            'rgb_to_native_id': {f'{i & 255},{(i >> 8) & 255},{(i >> 16) & 255}': owner
                                 for i, owner in inverse.items() if i > 0},
            'omitted_source_attributes': 'non-coordinate source attributes, including original normals and old owner colors'}
