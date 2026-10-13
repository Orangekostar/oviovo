"""Task-owned identities and atomic units; inherited JSON is verified first."""
from pathlib import Path
import hashlib
import json
from static_ovmap.frozen_fc_routing.common import verified, write, read, canonical_digest, PathResolver, ConsumptionIndex, arrays_record as parent_arrays_record, _array_digest, FORWARD_KEYS

REPO=Path(__file__).resolve().parents[3]
BASE='bec4176d9a6784a89ea293a9803e5cf6fab178ec'
METHODS=('C1_FULL_VIEW_CENTER','C2_SITE_HUBER_CONSENSUS','M1_CORRESPONDENCE_GRAPH')
SEEDS=(17,29)
LAMBDAS=(0,.125,.25,.5)
REFERENCES=('REF_D2','REF_G1','FC8','FG2_seed17','FG2_seed29')

class DependencyError(RuntimeError):
    def __init__(self,missing):
        self.missing=list(missing); super().__init__('Missing exact dependencies: '+', '.join(map(str,self.missing)))

def require(paths):
    missing=[str(p) for p in paths if not Path(p).exists()]
    if missing: raise DependencyError(missing)

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(4*2**20),b''): h.update(part)
    return h.hexdigest()

def source_digest():
    files=sorted((REPO/'src/static_ovmap/surface_bias_wave1_v2').glob('*.py'))
    files += [REPO/'scripts/evaluation/run_fg2_surface_bias_wave1_v2.py',REPO/'configs/static_ovmap/surface_bias_wave1_v2.json']
    return canonical_digest({str(p.relative_to(REPO)):digest_file(p) for p in files})

def cached_unit(path,identity):
    if not Path(path).exists(): return None
    old=verified(path)
    if old.get('input_identity')!=identity: raise ValueError('Preserve exposed unit; input/code identity changed: '+str(path))
    return old if old['status']=='COMPLETE' else None

def unit_identity(binding,phase,**extra):
    names={'diagnose':('common','binding','solver','diagnostics'), 'recognition':('common','binding','prediction'), 'readout':('common','binding','prediction'), 'map_predict':('common','binding','prediction','solver','adapter'), 'evaluate_scene':('common','binding','evaluation')}.get(phase,('common','binding','solver','adapter','prediction','selection','diagnostics','evaluation'))
    if phase in ('recognition','readout') and extra.get('method') not in ('FG2','FC8'): names=(*names,'solver','adapter')
    codes={name:digest_file(REPO/'src/static_ovmap/surface_bias_wave1_v2'/(name+'.py')) for name in names}
    return canonical_digest(dict(binding=binding['identity'],numerical_sources=codes,phase=phase,**extra))

def arrays_record(path,values,index):
    import shutil
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(p.parent).free<5*2**30: raise OSError('Stop new array writes below5GiB reserve')
    return parent_arrays_record(p,values,index)

def fixed_spec(path):
    original=REPO/'docs/paper/static_ovmap/surface_bias_wave1_v2/surface_bias_wave1_v2.json'
    if Path(path).read_bytes()!=original.read_bytes(): raise ValueError('Frozen protocol changed; use runtime path overrides')
    return read(path)

def atomic_text(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp'); temporary.write_text(text); temporary.replace(path)
