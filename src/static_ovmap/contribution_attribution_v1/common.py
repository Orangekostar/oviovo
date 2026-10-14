"""Task-owned atomic records and independent numerical producer identities."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
import numpy as np
from static_ovmap.frozen_fc_routing.common import verified, read, ConsumptionIndex, PathResolver
from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest

REPO = Path(__file__).resolve().parents[3]
TASK = 'ovmap-contribution-attribution-final-v1'
BASE = 'f4245ec0654577634fefb60f3344c6ca9c9335e8'
FG_BASE = 'bec4176d9a6784a89ea293a9803e5cf6fab178ec'
METRICS = ('apall', 'ap50', 'ap25', 'miou', 'macc')
PHASES = ('preflight','inventory','history','fusion_dev_predict','fusion_dev_score','freeze','main_predict','main_score','surface','attribute','rank','stability','qualitative','nominate')
HISTORICAL = {'A0':'CT_A0_NATIVE','A1':'CT_A1_E','A2':'CT_A2_R','A3':'CT_A3_ER','A4':'CT_A4_NATIVE_RECOVERY','A5':'CT_A5_FC_ONLY','U2':'CT_H_U2','G3':'CT_G3'}

class DependencyError(RuntimeError):
    pass

def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()

def plain(value):
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple,np.ndarray)): return [plain(v) for v in value]
    if isinstance(value, (float,np.floating)): return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.bool_): return bool(value)
    return value

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    result = plain({k:v for k,v in value.items() if k != 'identity'})
    result['identity'] = canonical_digest(result)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)
    return result

def immutable_record(path, value):
    path = Path(path)
    if path.exists():
        old = verified(path)
        if {k:v for k,v in old.items() if k!='identity'} != plain({k:v for k,v in value.items() if k!='identity'}):
            raise ValueError('Immutable record differs: '+str(path))
        return old
    return write(path, value)

def cached_record(path, identity):
    path = Path(path)
    if not path.exists(): return None
    value = verified(path)
    if value.get('input_identity') != identity:
        raise ValueError('Sealed unit identity changed; preserve old unit before affected-stage repair: '+str(path))
    return value if value.get('status') in ('COMPLETE','ALL_PREDICTIONS_LOCKED','NOT_SELECTED_NO_FUSION_DEV','NOT_RUN_BY_CONTRACT') else None

def source_digest(names=None):
    folder = REPO/'src/static_ovmap/contribution_attribution_v1'
    files = sorted(folder.glob('*.py')) if names is None else [folder/(name+'.py') for name in names]
    files = [p for p in files if p.name not in ('reporting.py','publication.py')]
    return canonical_digest({str(p.relative_to(REPO)):digest_file(p) for p in files})

def unit_identity(phase, inputs, **parameters):
    return canonical_digest(dict(task=TASK,schema=1,phase=phase,producer=source_digest(parameters.pop('producers',None)),inputs=inputs,parameters=parameters))

def fixed_spec(path):
    source = REPO/'docs/paper/static_ovmap/contribution_attribution_v1/prompt/02_冻结执行配置.json'
    if Path(path).read_bytes()!=source.read_bytes(): raise ValueError('Frozen task specification differs')
    return read(path)

def read_document(path,index=None,*,sealed=True,expected=None):
    path=Path(path)
    if not path.is_file(): raise DependencyError('Missing required input: '+str(path))
    if index is not None: index.identity(path,expected)
    if path.suffix=='.gz':
        with gzip.open(path,'rt') as stream: value=json.load(stream)
        if sealed and value.get('identity')!=canonical_digest({k:v for k,v in value.items() if k!='identity'}):
            raise ValueError('Compressed original identity differs: '+str(path))
        return value
    return verified(path) if sealed else read(path)

def reserve(path, initial=False):
    free=shutil.disk_usage(path).free
    minimum=(10 if initial else 5)*2**30
    if free<minimum: raise OSError(f'OUT free space below {minimum//2**30}GiB: '+str(path))
    return free

def family(scene):
    return scene.split('_')[0] if scene.startswith('scene') else scene
