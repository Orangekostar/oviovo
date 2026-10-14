"""Task identities, bounded array writes and sealed units."""
from pathlib import Path
import hashlib
import shutil
from static_ovmap.frozen_fc_routing.common import (verified,write,read,canonical_digest,PathResolver,ConsumptionIndex,arrays_record as parent_arrays_record,_array_digest,FORWARD_KEYS,objects,immutable_write)
from static_ovmap.surface_bias_wave1_v2.common import DependencyError,require,digest_file,atomic_text
REPO=Path(__file__).resolve().parents[3]
SOURCE='0f96c5e09c3ced9262105c1352cc428643f5676e'
FG_BASE='bec4176d9a6784a89ea293a9803e5cf6fab178ec'
SEEDS=(17,29)
PREFIXES=(2,4,8)
METHODS=('STATIC','COUPLED','NO_NEW_CORRESPONDENCE','WITH_NEW_CORRESPONDENCE')
REFERENCES=('REF_D2','REF_G1','FC8','FG2_seed17','FG2_seed29')
CONDITIONS=('clean','truncate','append','truncate_append')

def fixed_spec(path):
    frozen=REPO/'docs/paper/static_ovmap/branch_utility_wave2_v1/prompt/experiment_spec.json'
    if Path(path).read_bytes()!=frozen.read_bytes():raise ValueError('Frozen wave2 specification changed')
    return read(path)

def source_digest(names=None):
    folder=REPO/'src/static_ovmap/branch_utility_wave2_v1'
    files=sorted(folder.glob('*.py')) if names is None else [folder/(n+'.py') for n in names]
    files+=[REPO/'configs/static_ovmap/branch_utility_wave2_v1.json']
    return canonical_digest({str(p.relative_to(REPO)):digest_file(p) for p in files})

def unit_identity(binding,phase,**extra):
    names={'capture':('common','binding','subsets','decomposition','action_bank','features','capture'), 'training':('common','utility_model','training'), 'map_predict':('common','subsets','decomposition','action_bank','features','utility_model','prediction'), 'evaluate_scene':('common','evaluation')}.get(phase)
    return canonical_digest(dict(binding=binding['identity'],sources=source_digest(names),phase=phase,**extra))

def cached_unit(path,identity):
    path=Path(path)
    if not path.exists():return None
    row=verified(path)
    if row.get('input_identity')!=identity:raise ValueError('Sealed unit differs; preserve output and use a new attempt: '+str(path))
    if row.get('status')!='COMPLETE':return None
    def visit(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=set(value):ConsumptionIndex().identity(value['path'],value)
            else:
                for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    visit(row)
    return row

def arrays_record(path,values,index):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free<5*2**30:raise OSError('Stop large writes below5GiB reserve')
    return parent_arrays_record(path,values,index)

def items(binding,role):
    return sorted([o for o in objects(binding,role) if o['base']],key=lambda o:o['key'])
