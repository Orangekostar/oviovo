"""Task publication helpers; real Git results, never simulated pushes."""
from pathlib import Path
import subprocess
from .common import REPO,PHASES,verified,write,digest_file,source_digest,canonical_digest
from .binding import load_binding

SCOPE=('configs/static_ovmap/contribution_attribution_v1.json',
       'docs/paper/static_ovmap/contribution_attribution_v1',
       'docs/superpowers/plans/2026-10-14-ovmap-contribution-attribution-v1.md',
       'scripts/evaluation/run_ovmap_contribution_attribution_v1.py',
       'src/static_ovmap/contribution_attribution_v1',
       'tests/static_ovmap/test_contribution_attribution_v1.py',
       'artifacts/static_ovmap/contribution_attribution_v1')

def _git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()

def _staged_paths():
    return [path for path in subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=REPO,text=True).split('\0') if path]

def _network(*args):
    for attempt in range(2):
        result=subprocess.run(['git',*args],cwd=REPO,text=True,capture_output=True)
        if result.returncode==0:return result.stdout.strip()
        reason=(result.stderr or result.stdout).strip()
        if attempt or any(s in reason.lower() for s in ('permission denied','authentication failed','non-fast-forward','rejected','conflict')):raise RuntimeError(reason)
    raise AssertionError('Unreachable network retry')

def _remote(branch):
    lines=_network('ls-remote','origin','refs/heads/'+branch).splitlines()
    return lines[0].split()[0] if lines else None

def _files():
    files=[]
    for relative in SCOPE:
        path=REPO/relative
        files.extend([path] if path.is_file() else sorted(p for p in path.rglob('*') if p.is_file() and '__pycache__' not in p.parts))
    return files

def _snapshot():
    return canonical_digest({str(p.relative_to(REPO)):digest_file(p) for p in _files()})

def _c1_content():
    return canonical_digest({str(p.relative_to(REPO)):digest_file(p) for p in _files()
                             if p.name not in ('PUBLICATION_RECEIPT.json','HANDOFF.md')})

def _final(root,binding,report,observed,c1,c2):
    remote=_remote(binding['task_branch'])
    if remote!=c2:raise RuntimeError('C2 remote SHA differs')
    return write(Path(root)/'publication/final.json',dict(status='PUSH_VERIFIED',publication_status='PUSH_VERIFIED',execution_status=report['execution_status'],evidence_status=report['evidence_status'],
                 C1=c1,C2=c2,local_sha=c2,remote_sha=remote,branch=binding['task_branch'],numeric_source_digest=source_digest(),report_source_sha256=digest_file(Path(__file__).with_name('reporting.py')),
                 task_snapshot_digest=_snapshot(),scientific_exit_code=observed['exit_code'],task_bytes=check_sizes(_files()),submission_evidence_complete=False))

def terminal_resume(root):
    from .resume import check_consumed,phase_unchanged
    from .runner import OUTPUTS
    root=Path(root);final=verified(root/'publication/final.json');binding=load_binding(root)
    if final['publication_status']!='PUSH_VERIFIED':return False
    check_consumed(root)
    if not all(phase_unchanged(root,p,root/OUTPUTS[p]) for p in PHASES):return False
    if final['numeric_source_digest']!=source_digest() or final['report_source_sha256']!=digest_file(Path(__file__).with_name('reporting.py')) or final['task_snapshot_digest']!=_snapshot():return False
    if _git('rev-parse','HEAD')!=final['C2'] or _remote(binding['task_branch'])!=final['C2']:raise RuntimeError('Published C2 local/remote identity changed')
    return True

def run(root):
    root=Path(root);binding=load_binding(root);folder=root/'publication';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'final.json').exists():
        if terminal_resume(root):return verified(folder/'final.json')
        raise ValueError('Published inputs or task files changed; preserve the existing C1/C2 instead of claiming those edits were published')
    report=verified(root/'report.json');observed=verified(root/'execution_observed.json');review=verified(root/'primary_review.json')
    if review['status']!='APPROVED':raise ValueError('Publication requires actual primary review of the result')
    artifact=verified(REPO/'artifacts/static_ovmap/contribution_attribution_v1/artifact_index.json')
    if artifact['numeric_source_digest']!=source_digest() or artifact['report_source_sha256']!=digest_file(Path(__file__).with_name('reporting.py')):
        raise ValueError('Scientific/report source changed after artifact generation')
    if review.get('reviewed_numeric_source_digest')!=source_digest() or review.get('reviewed_artifact_index')!=artifact['identity'] or review.get('reviewed_report_source_sha256')!=artifact['report_source_sha256']:
        raise ValueError('Primary review does not cover the current sources and artifact index')
    from .resume import check_consumed,phase_unchanged
    from .runner import OUTPUTS
    check_consumed(root)
    if report['execution_status']=='COMPLETE' and not all(phase_unchanged(root,p,root/OUTPUTS[p]) for p in PHASES):
        raise ValueError('Complete publication requires unchanged completed scientific phases')
    if _git('branch','--show-current')!=binding['task_branch']:raise ValueError('Task branch changed')
    total=check_sizes(_files());docs=REPO/'docs/paper/static_ovmap/contribution_attribution_v1';receipt_path=docs/'PUBLICATION_RECEIPT.json';journal=folder/'journal.json'
    try:
        old=verified(journal) if journal.exists() else {}
        c1=old.get('C1')
        if c1 and old['C1_content_digest']!=_c1_content():raise ValueError('Scientific/report snapshot changed after C1; preserve publication journal')
        if old.get('C2'):
            c2=old['C2']
            if _git('rev-parse','HEAD')!=c2:raise RuntimeError('Recorded C2 differs from current HEAD')
            remote=_remote(binding['task_branch'])
            if remote not in (c1,c2):raise RuntimeError('Recorded C2 recovery found a changed remote')
            if remote!=c2:_network('push','origin',c2+':refs/heads/'+binding['task_branch'])
            return _final(root,binding,report,observed,c1,c2)
        if c1 is None:
            staged=_staged_paths()
            if staged:raise ValueError('Unexpected pre-staged files: '+str(staged))
            subprocess.run(['git','add','--',*SCOPE],cwd=REPO,check=True)
            changed=_staged_paths()
            if any(not any(p==s or p.startswith(s+'/') for s in SCOPE) for p in changed):raise ValueError('Staged path outside task namespace')
            subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True,capture_output=True,text=True)
            subprocess.run(['git','commit','-m','Add frozen E/R/S contribution attribution and exposed evidence'],cwd=REPO,check=True)
            c1=_git('rev-parse','HEAD');write(journal,dict(status='C1_LOCAL',C1=c1,task_bytes=total,C1_content_digest=_c1_content()))
        remote=_remote(binding['task_branch'])
        if remote!=c1:
            if remote and _git('merge-base',c1,remote)!=remote:raise RuntimeError('Remote research branch has unrelated changes; normal push blocked')
            _network('push','-u','origin',c1+':refs/heads/'+binding['task_branch'])
        if _remote(binding['task_branch'])!=c1:raise RuntimeError('C1 push verification differs')
        receipt=write(receipt_path,dict(status='C1_PUSH_VERIFIED',C1=c1,remote_sha=c1,branch=binding['task_branch'],
                      numeric_source_digest=source_digest(),spec_sha256=binding['task_spec_sha256'],artifact_index=artifact['identity'],scientific_execution_observed=observed,
                      evidence_status=report['evidence_status'],C2_not_proven_by_this_receipt=True))
        link=f'https://github.com/Orangekostar/oviovo/blob/{c1}/docs/paper/static_ovmap/contribution_attribution_v1/'
        handoff=docs/'HANDOFF.md';text=handoff.read_text();marker='\n## C1 publication verified\n'
        if marker not in text:handoff.write_text(text+marker+f'\nC1: {c1}\n\n[RESULTS]({link}RESULTS.md) · [HANDOFF]({link}HANDOFF.md)\n\nC1普通push及完整remote SHA核验已完成。此收据不预言C2；C2真实核验记录在OUT/publication/final.json。\n')
        subprocess.run(['git','add','--',str(receipt_path.relative_to(REPO)),str(handoff.relative_to(REPO))],cwd=REPO,check=True)
        if _staged_paths():
            subprocess.run(['git','diff','--cached','--check'],cwd=REPO,check=True,capture_output=True,text=True)
            subprocess.run(['git','commit','-m','Record verified C1 publication and permanent evidence links'],cwd=REPO,check=True)
        c2=_git('rev-parse','HEAD');write(journal,dict(status='C2_LOCAL',C1=c1,C2=c2,task_bytes=total,C1_content_digest=_c1_content()))
        _network('push','origin',c2+':refs/heads/'+binding['task_branch'])
        return _final(root,binding,report,observed,c1,c2)
    except Exception as exc:
        write(folder/'blocked.json',dict(status='PUBLICATION_BLOCKED',publication_status='PUBLICATION_BLOCKED',reason=str(exc),ordinary_push_only=True))
        raise

def check_sizes(files,file_limit=40*2**20,total_limit=150*2**20):
    total=0
    for path in files:
        path=Path(path)
        if path.is_symlink() or not path.is_file():raise ValueError('Only task-owned regular files can be published: '+str(path))
        size=path.stat().st_size
        if size>file_limit:raise ValueError('Task file budget exceeded: '+str(path))
        total+=size
    if total>total_limit:raise ValueError('Task total publication budget exceeded')
    return total
