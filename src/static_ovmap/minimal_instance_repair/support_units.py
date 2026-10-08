"""Whole original-incumbent and omitted-residual supports, without target labels."""

from dataclasses import dataclass

import numpy as np

from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.recovery_registry import build_registry


def vertex_areas(xyz, faces):
    xyz, faces = np.asarray(xyz), np.asarray(faces)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all()
            or faces.ndim != 2 or faces.shape[1] != 3
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0) or np.any(faces >= len(xyz))):
        raise ValueError('invalid fixed mesh for physical area')
    weights = np.zeros(len(xyz), np.float64)
    for start in range(0, len(faces), 262144):
        f = faces[start:start+262144]
        triangles = xyz[f].astype(np.float64)
        area = np.linalg.norm(np.cross(triangles[:,1]-triangles[:,0], triangles[:,2]-triangles[:,0]), axis=1)/2
        if not np.isfinite(area).all():
            raise ValueError('nonfinite physical triangle area')
        for vertex in range(3):
            np.add.at(weights, f[:,vertex], area/3)
    if not weights.sum() > 0:
        raise ValueError('fixed surface has zero triangle area')
    return weights


@dataclass(frozen=True)
class SupportUnit:
    unit_id: str
    kind: str
    owner: int
    rows: np.ndarray
    support_hash: str
    area: float
    raw_rows: int

    def record(self):
        return {'unit_id': self.unit_id, 'kind': self.kind, 'owner': self.owner,
                'source_rows': len(self.rows), 'support_hash': self.support_hash,
                'physical_area': self.area, 'raw_source_rows': self.raw_rows}


@dataclass
class UnitSet:
    units: dict
    seeds: list
    registry: dict
    vertex_weights: np.ndarray
    mesh_identity: str

    @property
    def identity(self):
        return canonical_digest(self.record())

    def record(self):
        return {'mesh_identity': self.mesh_identity, 'registry_identity': self.registry['identity'],
                'seeds': self.seeds, 'units': {k:v.record() for k,v in self.units.items()}}


def build_units(xyz, faces, raw, painted, geometry, settings, *, inherited_registry=None):
    raw, painted = np.asarray(raw), np.asarray(painted)
    registry = build_registry(xyz, raw, painted,
        minimum_rows=settings['minimum_residual_source_rows'], candidate_cap=settings['candidate_cap_inherited'])
    if inherited_registry is not None and registry != inherited_registry:
        raise ValueError('actual omitted-owner registry differs from the inherited registry')
    mesh = canonical_digest(geometry.to_dict())
    weights = vertex_areas(xyz, faces)
    units = {}
    def add(kind, owner, rows, raw_count):
        rows = np.asarray(rows, np.int64)
        rows.flags.writeable = False
        name = f'{kind}:{owner}'
        digest = canonical_digest({'mesh':mesh, 'sorted_source_rows':_array_digest(rows)})
        units[name] = SupportUnit(name, kind, owner, rows, digest, float(weights[rows].sum()), raw_count)
    for owner in sorted(map(int, np.unique(painted[painted > 0]))):
        rows = np.flatnonzero(painted == owner)
        add('I', owner, rows, len(rows))
    for row in registry['candidates']:
        owner = row['raw_owner']
        add('C', owner, np.flatnonzero((raw == owner) & (painted == 0)), row['raw_source_rows'])
    residuals = [u for u in units.values() if u.kind == 'C']
    seeds = [u.unit_id for u in sorted(residuals, key=lambda u:(-u.area,-len(u.rows),u.support_hash))[:settings['max_repair_seeds']]]
    return UnitSet(units, seeds, registry, weights, mesh)
