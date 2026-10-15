"""Finite, already authorized CONF12 acquisition, with one retry per file."""
import ast
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
from static_ovmap.module_validation import scannet_download as original
from static_ovmap.learned_object_readout.inventory import inventory_roots
from .common import (verified,read_document,write,ConsumptionIndex,canonical_digest,
    unit_key,existing,DependencyError,reserve)

AUTHORIZATION='User explicitly confirmed prior ScanNet authorization and terms agreement'

def remote_identity(url):
    if not url.startswith('https://'):raise ValueError('HTTPS required')
    for attempt in range(2):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as response:
                if not response.url.startswith('https://'):raise ValueError('HTTPS downgrade')
                size=int(response.headers.get('Content-Length','0'))
                if size<=0:raise ValueError('Unknown remote size: '+url)
                return dict(url=url,size_bytes=size,etag=response.headers.get('ETag'),last_modified=response.headers.get('Last-Modified'))
        except urllib.error.HTTPError as e:
            if e.code not in (408,429,500,502,503,504) or attempt:raise
        except (OSError,TimeoutError):
            if attempt:raise
        time.sleep(2)

def bounded_transfer(url,path,remote):
    # Exact original resume/hash/curl behavior; only lower its retry allowance.
    tree=ast.parse(inspect.getsource(original.transfer));changes=0
    for node in ast.walk(tree):
        if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='attempt':
            if ast.dump(node.iter)!=ast.dump(ast.parse('range(4)',mode='eval').body):raise ValueError('Parent retry interface changed')
            node.iter.args[0]=ast.Constant(2);changes+=1
        if isinstance(node,ast.Compare) and isinstance(node.left,ast.Name) and node.left.id=='attempt':
            if not(len(node.comparators)==1 and isinstance(node.comparators[0],ast.Constant) and node.comparators[0].value==3):raise ValueError('Parent retry guard changed')
            node.comparators[0]=ast.Constant(1);changes+=1
    if changes!=2:raise ValueError('Expected two resource-only retry edits')
    # Persist before starting curl, so process restarts cannot reset the file's retry allowance.
    import subprocess
    from types import SimpleNamespace
    attempt_path=Path(str(path)+'.task_attempts.json')
    def counted_run(command,**kwargs):
        previous=verified(attempt_path) if attempt_path.exists() else dict(remote=remote,attempts=[])
        if previous['remote']!=remote:raise ValueError('Transfer attempt remote identity changed')
        attempts=previous['attempts']
        if len(attempts)>=2:raise DependencyError('File already used its initial GET and one retry: '+str(path))
        attempts.append(dict(status='STARTED',attempt=len(attempts)+1,actual_exit_code=None))
        write(attempt_path,dict(status='IN_PROGRESS',remote=remote,attempts=attempts))
        result=subprocess.run(command,**kwargs)
        attempts[-1].update(status='EXIT_OBSERVED',actual_exit_code=result.returncode)
        write(attempt_path,dict(status='COMPLETE' if result.returncode==0 else 'FAILED',remote=remote,attempts=attempts))
        return result
    scope=dict(vars(original));scope['remote_identity']=remote_identity
    scope['subprocess']=SimpleNamespace(run=counted_run,CalledProcessError=subprocess.CalledProcessError)
    exec(compile(ast.fix_missing_locations(tree),original.__file__,'exec'),scope)
    return scope['transfer'](url,Path(path),remote)

def run(binding,*,phase='confirm_acquire',allow_annotation_parse=False):
    if allow_annotation_parse:raise ValueError('Acquisition must not parse annotations')
    root=Path(binding['output_root']);predictors=verified(root/'predictor_lock.json');roles=verified(root/'role_lock.json')
    spec=binding['task_specification'];scenes=roles['roles']['CONF'];index=ConsumptionIndex(root/'confirmation/acquisition_verifications.json')
    prior=Path(binding['prior_data_root']);authority=read_document(prior/'acquisition_lock.json',index,sealed=False)
    if authority.get('authorization')!=AUTHORIZATION:raise PermissionError('Exact prior ScanNet acceptance lock required')
    previous_plan=read_document(prior/'acquisition_plan.json',index,sealed=False)
    script=Path(previous_plan['layout']['script_path']);index.identity(script,{'sha256':previous_plan['layout']['script_sha256']})
    layout=original.DownloaderLayout.from_script(script)
    roots=list(binding['data_roots'])
    if binding.get('confirmation_data_root'):roots.append(binding['confirmation_data_root'])
    roots.append(str(root/'confirmation/raw'))
    inventory=inventory_roots(list(dict.fromkeys(roots)));suffixes=spec['confirmation']['raw_suffixes']
    source_id=unit_key('confirm_acquire',dict(predictors=predictors['identity'],roles=roles['identity'],authority=index.identity(prior/'acquisition_lock.json'),script=index.identity(script)),producers=('acquisition',),roots=roots)
    path=root/'confirmation/acquisition.json';old=existing(path,source_id)
    if old:return old
    write(root/'confirmation/local_inventory.json',dict(status='COMPLETE',inventory=inventory,selected=scenes,annotation_files_parsed=0))
    files={};pending=[];errors=[]
    for scene in scenes:
        local=inventory['scans'].get(scene,{}).get('files',{})
        files[scene]={}
        for suffix in suffixes:
            candidate=Path(local.get(suffix,'/__absent__'))
            if candidate.is_file() and candidate.stat().st_size:
                files[scene][suffix]=dict(source='EXISTING_AUTHORIZED_LOCAL',identity=index.identity(candidate))
            else:pending.append(dict(scene=scene,suffix=suffix,path=str(root/'confirmation/raw/scans'/scene/(scene+suffix)),url=layout.scene_url(scene,suffix)))
    # Resolve all sizes before transferring any byte; never replace a failed family.
    def head(row):
        try:return {**row,'remote':remote_identity(row['url'])}
        except Exception as e:return {**row,'error_type':type(e).__name__,'error':str(e)}
    with ThreadPoolExecutor(max_workers=2) as pool:remote_rows=list(pool.map(head,pending))
    errors.extend(r for r in remote_rows if 'error' in r)
    planned=sum(r['remote']['size_bytes'] for r in remote_rows if 'remote' in r)
    free=shutil.disk_usage(root).free
    # Conservative inherited ScanNet grid estimate; replaced by the first formal frame's measurement.
    features=verified(root/'features/train-dev.json');max_pixels=max(int(f['raw_hw'][0])*int(f['raw_hw'][1]) for f in features['frames'].values())
    grid_estimate=max_pixels*(1536+768)*4*1152
    planning=write(root/'confirmation/download_plan.json',dict(status='PARTIAL' if errors else 'LOCKED',input_identity=source_id,files=remote_rows,planned_download_bytes=planned,grid_preliminary_bytes=grid_estimate,grid_estimate_source=features['identity'],available_bytes=free,reserve_bytes=20*2**30,max_concurrency=2,max_GET_retries_per_file=1,layout=vars(layout),annotation_files_parsed=0))
    if planned>120*2**30:errors.append(dict(error='New download exceeds 120GiB',planned_bytes=planned))
    if free<planned+grid_estimate+20*2**30:errors.append(dict(error='Download plus grid estimate violates 20GiB reserve',required_bytes=planned+grid_estimate+20*2**30,free_bytes=free))
    if errors:
        write(path,dict(status='PARTIAL',input_identity=source_id,planning=planning['identity'],files=files,missing=errors,annotations_parsed=0,confirmation_scores=None))
        raise DependencyError('CONF12 acquisition planning blocked; exact failures: '+str(path))
    def fetch(row):
        try:
            reserve(root);receipt=bounded_transfer(row['url'],row['path'],row['remote'])
            return dict(scene=row['scene'],suffix=row['suffix'],source='AUTHORIZED_OFFICIAL_DOWNLOAD',download=receipt,identity=index.identity(row['path'],{'sha256':receipt['sha256']}))
        except Exception as e:return dict(scene=row['scene'],suffix=row['suffix'],error_type=type(e).__name__,error=str(e))
    with ThreadPoolExecutor(max_workers=2) as pool:
        for item in pool.map(fetch,remote_rows):
            if 'error' in item:errors.append(item)
            else:files[item['scene']][item['suffix']]=item
    source=prior/'scannetv2-labels.combined.tsv';index.identity(source)
    dest=root/'confirmation/raw/scannetv2-labels.combined.tsv';dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.read_bytes()!=source.read_bytes():raise ValueError('Inherited official TSV differs')
    if not dest.exists():shutil.copyfile(source,dest)
    index.write_memo(root/'confirmation/acquisition_verifications.json')
    value=write(path,dict(status='PARTIAL' if errors else 'COMPLETE',input_identity=source_id,planning=planning['identity'],files=files,label_tsv=index.identity(dest),missing=errors,annotations_parsed=0,new_download_bytes=planned))
    if errors:raise DependencyError('Required confirmation files missing: '+str(path))
    return value
