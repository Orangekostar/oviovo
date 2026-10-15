"""One bounded C1/C2 transaction; normal research-branch pushes only."""
from pathlib import Path
import subprocess
from .common import (REPO,BASE,verified,write,ConsumptionIndex,digest_file,
    canonical_digest,DependencyError)
from .reporting import DOCS,ARTIFACTS

SCOPES=('src/static_ovmap/structure_confirmation_v1/','configs/static_ovmap/structure_confirmation_v1.json','scripts/evaluation/run_fg2_structure_confirmation_v1.py','tests/static_ovmap/test_structure_confirmation_v1.py','docs/paper/static_ovmap/structure_confirmation_v1/','docs/superpowers/plans/2026-10-15-fg2-structure-confirmation-v1.md','artifacts/static_ovmap/structure_confirmation_v1/')

def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def allowed(path):return any(path.startswith(s) if s.endswith('/') else path==s for s in SCOPES)

def remote_sha(branch):
    lines=git('ls-remote','--heads','origin','refs/heads/'+branch).decode().splitlines()
    if len(lines)!=1:raise DependencyError('Research branch remote SHA missing or ambiguous')
    return lines[0].split()[0]

def push(branch,root,kind):
    record=Path(root)/'publication'/(kind+'_push_attempts.json');previous=verified(record) if record.exists() else dict(attempts=[])
    failures=previous['attempts']
    head=git('rev-parse','HEAD').decode().strip()
    try:
        if remote_sha(branch)==head:return head
    except DependencyError:pass
    for attempt in range(len(failures),2):
        result=subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=REPO,capture_output=True,text=True)
        failures.append(dict(actual_exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr));write(record,dict(status='COMPLETE',attempts=failures))
        if result.returncode==0:
            head=git('rev-parse','HEAD').decode().strip()
            if remote_sha(branch)!=head:raise DependencyError('Pushed remote full SHA differs')
            return head
    raise DependencyError('Normal research push failed after one retry: '+str(failures))

def staged_check():
    names=[p.decode() for p in git('diff','--cached','--name-only','-z').split(b'\0') if p]
    if not names or any(not allowed(p) for p in names):raise ValueError('Staged files escape task scope or are empty')
    sizes=[]
    for name in names:
        size=int(git('cat-file','-s',':'+name).decode());sizes.append(dict(path=name,bytes=size))
        if size>40*2**20:raise ValueError('Git blob exceeds40MiB: '+name)
    if sum(r['bytes'] for r in sizes)>256*2**20:raise ValueError('Task staged bytes exceed256MiB')
    subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True)
    return sizes

def completion_sources():
    paths=sorted((REPO/'src/static_ovmap/structure_confirmation_v1').glob('*.py'))+[REPO/'configs/static_ovmap/structure_confirmation_v1.json',REPO/'scripts/evaluation/run_fg2_structure_confirmation_v1.py',REPO/'tests/static_ovmap/test_structure_confirmation_v1.py']
    return {str(p.relative_to(REPO)):digest_file(p) for p in paths}

def terminal_resume(binding):
    root=Path(binding['output_root']);path=root/'publication/final.json'
    if not path.exists():return None
    final=verified(path)
    if final['status']!='PUSH_VERIFIED' or final['sources']!=completion_sources():return None
    branch=git('branch','--show-current').decode().strip();head=git('rev-parse','HEAD').decode().strip()
    if head!=final['C2'] or remote_sha(branch)!=head:raise DependencyError('Completed publication local/remote SHA changed')
    if git('status','--porcelain=v1','-z','--untracked-files=all','--',*SCOPES):raise DependencyError('Published task files changed after C2')
    # No numerical work or commits when all consumed files still have their bound bytes.
    for memo in final.get('verification_memos',[]):
        index=ConsumptionIndex(memo)
        import json
        for row in json.loads(Path(memo).read_text())['entries']:index.identity(row['identity']['path'],row['identity'])
    return final

def run(binding,*,phase='publish'):
    root=Path(binding['output_root']);cached=terminal_resume(binding)
    if cached:return cached
    report=verified(root/'report.json');observed=verified(root/'execution_observed.json');primary=verified(root/'review/primary_review.json')
    if primary['status']!='COMPLETE':raise DependencyError('Concrete primary artifact review required before publication')
    branch=git('branch','--show-current').decode().strip()
    if branch!=binding['task_specification']['branch']:raise ValueError('Wrong publication branch')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    c1_path=root/'publication/C1.json'
    transaction_path=root/'publication/transaction.json'
    transaction=verified(transaction_path) if transaction_path.exists() else dict(C1=None,C2=None)
    if c1_path.exists():c1=verified(c1_path)['C1']
    else:
        # Explicit files, NUL protocol and per-task size checks; no whole-repository staging.
        if transaction['C1'] is None:
            paths=[]
            for scope in SCOPES:
                target=REPO/scope
                paths.extend(str(p.relative_to(REPO)) for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts) if target.is_dir() else paths.append(scope)
            if any(not (REPO/p).is_file() for p in paths):raise ValueError('Required publication path missing')
            subprocess.run(['git','add','-f','--',*paths],cwd=REPO,check=True);sizes=staged_check()
            write(root/'publication/staged_files.json',dict(status='COMPLETE',files=sizes,NUL_filename_protocol=True,report=report['identity'],scientific_actual_exit_code=observed['actual_exit_code']))
            subprocess.run(['git','commit','-m','Add FG2 structural controls and fixed-family confirmation evidence'],cwd=REPO,check=True);transaction['C1']=git('rev-parse','HEAD').decode().strip();write(transaction_path,dict(transaction,status='C1_COMMITTED'))
        elif git('rev-parse','HEAD').decode().strip()!=transaction['C1']:raise ValueError('Pending C1 local HEAD changed')
        c1=push(branch,root,'C1')
        write(c1_path,dict(status='PUSH_VERIFIED',C1=c1,branch=branch,remote_SHA=c1,report=report['identity'],scientific_actual_exit_code=observed['actual_exit_code'],primary_review=primary['identity']))
    if transaction.get('C2'):
        if git('rev-parse','HEAD').decode().strip()!=transaction['C2']:raise ValueError('Pending C2 local HEAD changed')
        c2=push(branch,root,'C2')
        memos=[str(p) for p in root.rglob('*verifications.json') if p.is_file() and not p.is_symlink()]
        return write(root/'publication/final.json',dict(status='PUSH_VERIFIED',C1=c1,C2=c2,local_HEAD=c2,remote_HEAD=c2,branch=branch,scientific_actual_exit_code=observed['actual_exit_code'],scientific_status=report['status'],sources=completion_sources(),verification_memos=memos,terminal_resume_new_workers=0,terminal_resume_new_heads=0,terminal_resume_new_commits=0))
    if remote_sha(branch)!=c1:raise DependencyError('C1 no longer equals the remote before C2 receipt')
    receipt=dict(status='PUSH_VERIFIED',C1=c1,remote_SHA=c1,branch=branch,scientific_actual_exit_code=observed['actual_exit_code'],report_identity=report['identity'],scope='C1_DATA_AND_CODE_ONLY;C2_RECORDED_EXTERNALLY')
    write(REPO/DOCS/'publication_receipt.json',receipt)
    handoff=REPO/DOCS/'HANDOFF.md';text=handoff.read_text();links='\nC1：`'+c1+'`\n\n[RESULTS](https://github.com/Orangekostar/oviovo/blob/'+c1+'/'+str(DOCS)+'/RESULTS.md) · [HANDOFF](https://github.com/Orangekostar/oviovo/blob/'+c1+'/'+str(DOCS)+'/HANDOFF.md)\n'
    if links not in text:handoff.write_text(text+links)
    subprocess.run(['git','add','--',str(DOCS/'publication_receipt.json'),str(DOCS/'HANDOFF.md')],cwd=REPO,check=True);staged_check()
    changed=[p.decode() for p in git('diff','--cached','--name-only','-z').split(b'\0') if p]
    if set(changed)!={str(DOCS/'publication_receipt.json'),str(DOCS/'HANDOFF.md')}:raise ValueError('C2 changed more than receipt and handoff')
    subprocess.run(['git','commit','-m','Record verified FG2 evidence publication'],cwd=REPO,check=True);transaction['C2']=git('rev-parse','HEAD').decode().strip();write(transaction_path,dict(transaction,status='C2_COMMITTED'));c2=push(branch,root,'C2')
    memos=[str(p) for p in root.rglob('*verifications.json') if p.is_file() and not p.is_symlink()]
    return write(root/'publication/final.json',dict(status='PUSH_VERIFIED',C1=c1,C2=c2,local_HEAD=c2,remote_HEAD=c2,branch=branch,scientific_actual_exit_code=observed['actual_exit_code'],scientific_status=report['status'],sources=completion_sources(),verification_memos=memos,terminal_resume_new_workers=0,terminal_resume_new_heads=0,terminal_resume_new_commits=0))
