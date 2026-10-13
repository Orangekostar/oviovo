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
    path=Path(path)
    if not path.exists(): return None
    old=verified(path)
    if old.get('input_identity')!=identity:
        # A source repair invalidates this unit only; preserve its exposed JSON
        # and its local array output before the replacement can be computed.
        preserve_unit(path,old,identity,'INPUT_OR_ENGINEERING_IDENTITY_CHANGED')
        return None
    if old['status'] not in ('COMPLETE','UNSUPPORTED_STRICT_DIAGNOSTIC'): return None
    descriptors=[]
    def visit(value):
        if isinstance(value,dict):
            if {'path','bytes','sha256'}<=set(value): descriptors.append(value)
            else:
                for child in value.values(): visit(child)
        elif isinstance(value,list):
            for child in value: visit(child)
    visit(old)
    if any(not Path(row['path']).is_file() for row in descriptors): return None
    index=ConsumptionIndex()
    for row in descriptors: index.identity(row['path'],row)
    # Prediction/evaluation receipts refer to further sealed records, which
    # must exist as well; a receipt alone is not a complete cached result.
    def records(value):
        if isinstance(value,dict):
            for key,child in value.items():
                if key in ('manifest','evaluation_receipt','per_class_receipt') and isinstance(child,str):
                    if not Path(child).is_file(): return False
                    if key=='evaluation_receipt':
                        score=read(child)
                        if score.get('status')!='COMPLETE' or value.get('evaluation_identity',score.get('identity'))!=score.get('identity'): return False
                    elif key=='per_class_receipt': verified(child)
                    elif value.get('prediction_key'):
                        from static_ovmap.module_validation.scannet_study import load_prediction
                        payload=load_prediction(child)
                        if payload.prediction_key!=value['prediction_key']: return False
                elif isinstance(child,(dict,list)) and not records(child): return False
        elif isinstance(value,list):
            return all(records(child) for child in value)
        return True
    return old if records(old) else None

def preserve_unit(path,old,new_identity,reason):
    import shutil
    path=Path(path); folder=path.parent/'superseded'; folder.mkdir(parents=True,exist_ok=True)
    token=old.get('input_identity',old['identity']); saved=folder/(path.stem+'_'+token+'.json')
    if saved.exists() and saved.read_bytes()!=path.read_bytes(): raise ValueError('Superseded exposure identity collision')
    if not saved.exists(): shutil.copyfile(path,saved)
    array=old.get('arrays'); mapping={}
    if array and Path(array['path']).is_file() and Path(array['path']).resolve().parent==path.parent.resolve():
        original=Path(array['path']); destination=folder/(path.stem+'_'+token+original.suffix)
        if not destination.exists(): shutil.copyfile(original,destination)
        mapping[str(original)]=str(destination)
    write(folder/(path.stem+'_'+token+'_invalidation.json'),dict(reason=reason,old_input_identity=old.get('input_identity'),new_input_identity=new_identity,original_receipt=str(path),preserved_receipt=str(saved),preserved_arrays=mapping))
    path.unlink()

def unit_identity(binding,phase,**extra):
    names={'diagnose':('common','binding','data','solver','diagnostics'), 'recognition':('common','binding','data','prediction'), 'readout':('common','binding','data','prediction'), 'map_predict':('common','binding','data','prediction','solver','adapter'), 'evaluate_scene':('common','binding','evaluation')}.get(phase,('common','binding','data','solver','adapter','prediction','selection','diagnostics','evaluation'))
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
