"""Small task receipts; reuse the established parent content primitives."""
from pathlib import Path

from static_ovmap.samv_local_probe.common import (
    ConsumptionIndex, PathResolver, _array_digest, arrays_record,
    canonical_digest, read, verified, write,
)

REPO = Path(__file__).resolve().parents[3]
METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')
BASELINES = {'LR00_D2': 'SU00_D2', 'LR01_G1': 'SU01_G1'}
PHASES = ('bind', 'data', 'features', 'engineer', 'train', 'nominate', 'repeat',
          'holdout', 'predict', 'evaluate', 'report')


def fixed_spec(path):
    path = Path(path).resolve()
    original = REPO / 'docs/paper/static_ovmap/learned_object_readout_v1/PROTOCOL_SPEC.json'
    if path.read_bytes() != original.read_bytes():
        raise ValueError('Scientific specification must remain verbatim; use CLI path overrides')
    return read(path)


def receipt(root, phase, inputs, **values):
    path = Path(root) / 'phases' / (phase + '.json')
    return write(path, dict(phase=phase, input_identity=canonical_digest(inputs), **values))


def existing(root, phase, inputs):
    path = Path(root) / 'phases' / (phase + '.json')
    if not path.exists():
        return None
    row = verified(path)
    if row['input_identity'] != canonical_digest(inputs):
        raise ValueError(f'{phase} inputs changed; preserve completed work and explicitly invalidate task descendants')
    return row if row['status'] == 'COMPLETE' else None
