"""Task identities and bounded IO; parent records remain immutable."""
import gzip
import json
from pathlib import Path
import numpy as np
from static_ovmap.contribution_attribution_v1.common import (
    write, verified, plain, digest_file, canonical_digest, ConsumptionIndex,
    PathResolver, DependencyError,
)

REPO = Path(__file__).resolve().parents[3]
TASK = 'ovmap-semantic-update-evidence-v1'
BASE = '553758bda89031b05916ac81a2981b597dfb3bda'
DOCS = Path('docs/paper/static_ovmap/semantic_update_evidence_v1')
ARTIFACTS = Path('artifacts/static_ovmap/semantic_update_evidence_v1')
METHODS = ('B_FC', 'S_LL', 'G_LL', 'V_LL', 'MV_ATTN')
SEEDS = (17, 29)
CONDITIONS = ('clean', 'truncate', 'append', 'truncate_append')

def read(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as f:
        return json.load(f)

def gzip_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
        f.write(json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                           separators=(',', ':'), allow_nan=False).encode())

def save_arrays(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays); temp.replace(path)

def save_memo(index,path):
    """Retain this task's already consumed identities across cached bind calls."""
    path=Path(path)
    old=read(path)['entries'] if path.exists() else []
    temporary=path.with_name(path.name+'.current')
    index.write_memo(temporary)
    current=read(temporary)['entries'];temporary.unlink()
    by_path={r['identity']['path']:r for r in old}
    by_path.update({r['identity']['path']:r for r in current})
    write(path,dict(entries=[by_path[p] for p in sorted(by_path)]))

def source_identity():
    names = ('common', 'binding', 'readers', 'metrics', 'calibration', 'diagnostics')
    return canonical_digest({n: digest_file(Path(__file__).with_name(n + '.py')) for n in names})
