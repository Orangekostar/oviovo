"""Two normal research commits; actual completed resume is observed separately."""
import subprocess
from pathlib import Path
from .common import REPO,BASE,DOCS,ARTIFACTS,read,write,verified,digest_file,ConsumptionIndex,DependencyError

SCOPES=('src/static_ovmap/semantic_update_evidence_v1/',
        'scripts/evaluation/run_semantic_update_evidence_v1.py',
        'configs/static_ovmap/semantic_update_evidence_v1.json',
        'tests/static_ovmap/test_semantic_update_data.py',
        'tests/static_ovmap/test_semantic_update_math.py',
        'tests/static_ovmap/test_semantic_update_training.py',
        'tests/static_ovmap/test_semantic_update_publication.py',
        'docs/superpowers/plans/2026-10-16-ovmap-semantic-update-evidence-v1.md',
        str(DOCS)+'/',str(ARTIFACTS)+'/')

def git(*args):return subprocess.check_output(['git',*args],cwd=REPO)
def allowed(name):return any(name.startswith(s) if s.endswith('/') else name==s for s in SCOPES)

def current_sources():
    paths=list((REPO/'src/static_ovmap/semantic_update_evidence_v1').glob('*.py'))
    paths += [REPO/s for s in SCOPES if s.endswith('.py') or s.endswith('.json')]
    return {str(p.relative_to(REPO)):digest_file(p) for p in sorted(set(paths))}

def remote_sha(branch):
    rows=git('ls-remote','--heads','origin','refs/heads/'+branch).decode().splitlines()
    if not rows:return None
    if len(rows)!=1:raise DependencyError('Remote research branch ambiguous')
    return rows[0].split()[0]

def check_files(names):
    if not names or any(not allowed(n) for n in names):raise ValueError('Empty or out-of-scope publication')
    rows=[dict(path=n,bytes=(REPO/n).stat().st_size) for n in names]
    if any(r['bytes']>40*2**20 for r in rows):raise ValueError('Blob above 40MiB')
    if sum(r['bytes'] for r in rows)>512*2**20:raise ValueError('Task above 512MiB')
    return rows

def push(root,branch,kind):
    path=Path(root)/'publication'/(kind+'_attempts.json')
    attempts=verified(path)['attempts'] if path.exists() else []
    sha=git('rev-parse','HEAD').decode().strip()
    if remote_sha(branch)==sha:return sha
    for _ in range(len(attempts),2):
        p=subprocess.run(['git','push','origin','HEAD:refs/heads/'+branch],cwd=REPO,text=True,capture_output=True)
        attempts.append(dict(actual_exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
        write(path,dict(status='EXIT_OBSERVED',attempts=attempts))
        if p.returncode==0 and remote_sha(branch)==sha:return sha
    raise DependencyError('Normal push failed after one retry; see '+str(path))

def terminal_resume(root):
    root=Path(root);path=root/'publication/final.json'
    if not path.exists():return None
    final=verified(path)
    if final['sources']!=current_sources():raise DependencyError('Published task sources changed')
    if git('rev-parse','HEAD').decode().strip()!=final['C2'] or remote_sha(final['branch'])!=final['C2']:
        raise DependencyError('Published local/remote full SHA changed')
    if git('status','--porcelain=v1','-z','--untracked-files=all','--',*SCOPES):raise DependencyError('Published scoped files changed')
    # Only this task's exact direct consumed records; no historical recursive scan.
    memo=root/'input_verifications.json'
    if memo.exists():
        index=ConsumptionIndex(memo)
        for row in read(memo)['entries']:
            index.identity(row['identity']['path'],row['identity'])
    for row in final['result_files']:
        if digest_file(root/row['path'])!=row['sha256']:raise DependencyError('Consumed task result changed: '+row['path'])
    return final

def publish(binding):
    root=Path(binding['output_root']);done=terminal_resume(root)
    if done:return done
    report=verified(root/'report.json');observed=verified(root/'execution_observed.json');review=verified(root/'review/primary_review.json')
    if review['status']!='COMPLETE' or review['report_identity']!=report['identity']:
        raise DependencyError('Concrete primary review of current report required')
    branch=git('branch','--show-current').decode().strip()
    if branch!=binding['task_specification']['branch']:raise ValueError('Wrong publication branch')
    origin=git('remote','get-url','origin').decode().strip()
    if origin not in ('git@github.com:Orangekostar/oviovo.git','https://github.com/Orangekostar/oviovo.git'):raise ValueError('Wrong origin')
    subprocess.run(['git','merge-base','--is-ancestor',BASE,'HEAD'],cwd=REPO,check=True)
    transaction_path=root/'publication/transaction.json'
    tx=verified(transaction_path) if transaction_path.exists() else dict(C1=None,C2=None)
    if not tx['C1']:
        paths=[]
        for scope in SCOPES:
            target=REPO/scope
            paths += [str(p.relative_to(REPO)) for p in target.rglob('*') if p.is_file() and '__pycache__' not in p.parts] if target.is_dir() else [scope]
        sizes=check_files(paths)
        subprocess.run(['git','add','-f','--',*paths],cwd=REPO,check=True)
        staged=[p.decode() for p in git('diff','--cached','--name-only','-z').split(b'\0') if p]
        if set(staged)!=set(paths):raise ValueError('Staged file domain differs from bounded publication')
        check=subprocess.run(['git','diff','--cached','--check'],cwd=REPO,capture_output=True,text=True)
        if check.returncode:
            write(root/'publication/format_check_failure.json',dict(status='FAILED',actual_exit_code=check.returncode,stdout=check.stdout,stderr=check.stderr))
            raise ValueError('Staged formatting check failed; exact diagnostics saved in publication/format_check_failure.json')
        write(root/'publication/staged_files.json',dict(status='COMPLETE',files=sizes,NUL_filename_protocol=True,scientific_actual_exit_code=observed['actual_exit_code']))
        subprocess.run(['git','commit','-m','Add semantic update diagnostics and conditional A-gate evidence'],cwd=REPO,check=True)
        tx['C1']=git('rev-parse','HEAD').decode().strip();write(transaction_path,dict(tx,status='C1_COMMITTED'))
    elif not tx['C2'] and git('rev-parse','HEAD').decode().strip()!=tx['C1']:raise DependencyError('Pending C1 local HEAD changed')
    c1=push(root,branch,'C1') if not tx['C2'] else tx['C1']
    receipt=root/'publication/C1.json'
    if not receipt.exists():write(receipt,dict(status='PUSH_VERIFIED',C1=c1,remote_SHA=c1,scientific_status=report['status']))
    if not tx['C2']:
        docs_receipt=DOCS/'publication_receipt.json';handoff=DOCS/'HANDOFF.md'
        write(REPO/docs_receipt,dict(status='PUSH_VERIFIED',C1=c1,remote_SHA=c1,branch=branch,scientific_actual_exit_code=observed['actual_exit_code'],C2_recorded_externally=True))
        links='\nC1：`'+c1+'`\n\n[RESULTS](https://github.com/Orangekostar/oviovo/blob/'+c1+'/'+str(DOCS)+'/RESULTS.md) · [HANDOFF](https://github.com/Orangekostar/oviovo/blob/'+c1+'/'+str(DOCS)+'/HANDOFF.md)\n'
        with (REPO/handoff).open('a') as f:f.write(links)
        subprocess.run(['git','add','--',str(docs_receipt),str(handoff)],cwd=REPO,check=True)
        staged=[p.decode() for p in git('diff','--cached','--name-only','-z').split(b'\0') if p]
        if set(staged)!={str(docs_receipt),str(handoff)}:raise ValueError('C2 scope escaped receipt/HANDOFF')
        subprocess.run(['git','commit','-m','Record verified semantic update evidence publication'],cwd=REPO,check=True)
        tx['C2']=git('rev-parse','HEAD').decode().strip();write(transaction_path,dict(tx,status='C2_COMMITTED'))
    elif git('rev-parse','HEAD').decode().strip()!=tx['C2']:raise DependencyError('Pending C2 local HEAD changed')
    c2=push(root,branch,'C2')
    files=[root/p for p in ('science.json','report.json','source_binding.json','class_split.json','workload.json',
          'input_verifications.json','execution_observed.json','review/primary_review.json') if (root/p).is_file()]
    if (root/'A').is_dir():files+=sorted(p for p in (root/'A').iterdir() if p.is_file())
    files+=sorted((root/'phases').glob('*.json'))
    return write(root/'publication/final.json',dict(status='PUSH_VERIFIED',C1=c1,C2=c2,branch=branch,
        scientific_status=report['status'],scientific_actual_exit_code=observed['actual_exit_code'],sources=current_sources(),
        result_files=[dict(path=str(p.relative_to(root)),sha256=digest_file(p)) for p in files],
        terminal_resume_status='NOT_YET_OBSERVED'))
