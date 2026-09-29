"""Task-local binding and explicit plain/gzip JSON access."""

import gzip
import json
import subprocess
from pathlib import Path

from src.static_ovmap.composition_study.io import write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex
from src.static_ovmap.module_validation.contracts import canonical_digest

ROOT = Path(__file__).resolve().parents[3]
INDEX = InputIndex()


def read_json_or_gzip(path):
    path = Path(path)
    if not path.exists() and path.suffix != '.gz' and path.with_suffix(path.suffix + '.gz').exists():
        path = path.with_suffix(path.suffix + '.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def bind(spec_path, parent_path, output_root):
    spec, parent = read_json_or_gzip(spec_path), read_json_or_gzip(parent_path)
    if canonical_digest({k: v for k, v in parent.items() if k != 'identity'}) != parent['identity']:
        raise ValueError('wave1 binding identity changed')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != spec['branch']:
        raise ValueError('paired study requires prescribed isolated branch')
    subprocess.run(['git', 'merge-base', '--is-ancestor', spec['base_commit'], 'HEAD'], cwd=ROOT, check=True)
    for path in (spec_path, parent_path, parent['reviewer_binding']):
        INDEX.identity(path)
    scenes = spec['datasets']['calibration'] + spec['datasets']['replica']
    if set(scenes) != set(parent['scenes']):
        raise ValueError('bound scene registry changed')
    result = {'schema': 1, 'status': 'BOUND', 'spec': str(Path(spec_path).resolve()),
              'wave1_binding': str(Path(parent_path).resolve()), 'wave1_identity': parent['identity'],
              'reviewer_binding': parent['reviewer_binding'], 'output_root': str(Path(output_root).resolve()),
              'scenes': parent['scenes'], 'branch': branch, 'base': spec['base_commit'],
              'inputs': INDEX.entries(), 'deployment': 'N0_UNCHANGED'}
    result['identity'] = canonical_digest(result)
    write_once(Path(output_root) / 'binding.json', result)
    return result
