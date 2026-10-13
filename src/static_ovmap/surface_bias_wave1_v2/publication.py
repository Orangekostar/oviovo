"""Two normal commits; C2 proves C1, final C2 proof stays outside Git."""
from pathlib import Path
import subprocess
from .common import REPO, write, read, digest_file, source_digest, atomic_text

def git(repo,*args):
    result=subprocess.run(['git',*args],cwd=repo,text=True,capture_output=True)
    if result.returncode: raise RuntimeError('git '+args[0]+' failed: '+(result.stderr+result.stdout)[-3000:])
    return result.stdout.strip()

def remote_head(repo,branch):
    line=git(repo,'ls-remote','origin','refs/heads/'+branch)
    return line.split()[0] if line else None

def push_verified(repo,branch,commit):
    failure=None
    for attempt in range(2):
        try:
            git(repo,'push','-u','origin','HEAD:refs/heads/'+branch)
            remote=remote_head(repo,branch)
            if remote!=commit: raise RuntimeError('Full local/remote SHA differs')
            return dict(commit=commit,remote_SHA=remote,push_attempts=attempt+1,status='PUSH_VERIFIED')
        except RuntimeError as exc:
            failure=exc
            if any(s in str(exc) for s in ('non-fast-forward','Permission denied','Authentication failed','Please tell me who you are')): break
    raise failure

def check_files(repo,files):
    total=0; checked=[]
    for name in sorted(set(files)):
        p=Path(repo)/name
        if p.is_symlink() or not p.is_file(): raise ValueError('Publish only task-owned regular files: '+name)
        size=p.stat().st_size
        if size>40*2**20: raise ValueError('Task40MiB file budget exceeded: '+name)
        total+=size; checked.append(dict(path=name,bytes=size,sha256=digest_file(p)))
    if total>150*2**20: raise ValueError('Task150MiB total budget exceeded')
    return dict(files=checked,total_bytes=total)

def publish_local_protocol(repo,out,files,branch,*,receipt,handoff,release_identity=None):
    repo=Path(repo); out=Path(out); out.mkdir(parents=True,exist_ok=True); final=out/'publication/final.json'
    if final.exists():
        old=read(final)
        if old.get('status')=='PUSH_VERIFIED' and git(repo,'rev-parse','HEAD')==old['C2'] and remote_head(repo,old['branch'])==old['C2'] and not git(repo,'status','--porcelain','--',*files): return old
        raise ValueError('Existing publication differs; preserve its immutable release')
    transaction=out/'publication/transaction.json'
    if transaction.exists():
        state=read(transaction); state.pop('identity',None); branch=state['branch']; checked=state['file_inventory']
        if state.get('release_identity')!=release_identity: raise ValueError('Publication source/spec/artifact identity changed')
        if git(repo,'branch','--show-current')!=branch or git(repo,'rev-parse','HEAD')!=state.get('C2',state['C1']): raise ValueError('Pending publication branch/head changed')
        for item in checked['files']:
            if item['path']==handoff: continue
            if digest_file(repo/item['path'])!=item['sha256']: raise ValueError('Pending C1 artifact changed: '+item['path'])
    else:
        checked=check_files(repo,files); remote=remote_head(repo,branch)
        if remote and remote!=git(repo,'rev-parse','HEAD'):
            requested=branch; suffix=2
            while remote_head(repo,requested+f'-{suffix:02d}') is not None: suffix+=1
            branch=requested+f'-{suffix:02d}'
        current=git(repo,'branch','--show-current')
        if current!=branch: git(repo,'checkout','-b',branch)
        git(repo,'add','--',*files); git(repo,'diff','--cached','--check')
        staged=[p for p in git(repo,'diff','--cached','--name-only','-z').split('\0') if p]
        if not set(staged)<=set(files): raise ValueError('Unrelated staged files must not enter task commit')
        if staged: git(repo,'commit','-m','Implement frozen FG2 surface bias calibration and report measured scope')
        state=dict(stage='C1_LOCAL',branch=branch,C1=git(repo,'rev-parse','HEAD'),file_inventory=checked,release_identity=release_identity); write(transaction,state)
    C1=state['C1']
    if state['stage']=='C1_LOCAL':
        first=push_verified(repo,branch,C1); state.update(stage='C1_PUSH_VERIFIED',first_push=first); write(transaction,state)
    if state['stage']=='C1_PUSH_VERIFIED':
        record=dict(state['first_push'],branch=branch,scope='PRECEDING_CODE_REPORTS_RESULTS_COMMIT_C1_ONLY',file_inventory=checked,no_new_weights=True,release_identity=release_identity)
        write(repo/receipt,record); handoff_path=repo/handoff
        if f'Verified result commit C1: `{C1}`' not in handoff_path.read_text(): atomic_text(handoff_path,handoff_path.read_text()+f'\nVerified result commit C1: `{C1}`. [Immutable results](https://github.com/Orangekostar/oviovo/blob/{C1}/docs/paper/static_ovmap/surface_bias_wave1_v2/RESULTS.md).\n\nPublication of C1 was verified; the committed receipt proves C1 only. The final C2/local/remote equality receipt is external at OUT/publication/final.json.\n')
        git(repo,'add','--',receipt,handoff)
        if {p for p in git(repo,'diff','--cached','--name-only','-z').split('\0') if p}!={receipt,handoff}: raise ValueError('C2 may contain only C1 receipt and handoff')
        git(repo,'commit','-m','Record verified FG2 calibration C1 publication and handoff'); state.update(stage='C2_LOCAL',C2=git(repo,'rev-parse','HEAD')); write(transaction,state)
    C2=state['C2']; second=push_verified(repo,branch,C2)
    result=dict(status='PUSH_VERIFIED',branch=branch,C1=C1,C2=C2,local_HEAD=C2,remote_HEAD=second['remote_SHA'],C1_receipt=receipt,commit2_scope='COMMIT1_PUSH_RECEIPT_AND_HANDOFF_ONLY')
    write(final,result); return result

def run(root):
    root=Path(root); paths=[]
    for folder in ('src/static_ovmap/surface_bias_wave1_v2','docs/paper/static_ovmap/surface_bias_wave1_v2','artifacts/static_ovmap/surface_bias_wave1_v2'):
        paths += [str(p.relative_to(REPO)) for p in (REPO/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_RECEIPT.json']
    paths += ['scripts/evaluation/run_fg2_surface_bias_wave1_v2.py','configs/static_ovmap/surface_bias_wave1_v2.json','tests/test_surface_bias_wave1_v2.py','docs/superpowers/plans/2026-10-13-fg2-surface-bias-wave1-v2.md']
    try:
        identities=dict(scientific_source_digest=source_digest(),spec_sha256=digest_file(REPO/'configs/static_ovmap/surface_bias_wave1_v2.json'),artifact_index_sha256=digest_file(REPO/'artifacts/static_ovmap/surface_bias_wave1_v2/artifact_index.json'))
        result=publish_local_protocol(REPO,root,paths,git(REPO,'branch','--show-current'),receipt='artifacts/static_ovmap/surface_bias_wave1_v2/PUBLICATION_RECEIPT.json',handoff='docs/paper/static_ovmap/surface_bias_wave1_v2/HANDOFF.md',release_identity=identities)
        return result
    except (RuntimeError,OSError,ValueError) as exc:
        write(root/'publication/error.json',dict(status='PUBLICATION_BLOCKED',error=str(exc),local_HEAD=git(REPO,'rev-parse','HEAD'),local_commits_preserved=True))
        raise
