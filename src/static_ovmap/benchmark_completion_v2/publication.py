"""Scoped C1/C2 publication, resumable pushes and independent remote-result read."""
from pathlib import Path
import hashlib
import random
import shutil
import subprocess
import sys
from urllib.parse import quote

from .common import (REPO, DOCS, ARTIFACTS, BASE, read, verified, write,
                     digest_file, canonical_digest, StageFailure)

BRANCH = 'research/ovmap-benchmark-completion-v2'


def git(repo, *args, binary=False):
    child = subprocess.run(['git', *args], cwd=repo, capture_output=True)
    if child.returncode:
        raise RuntimeError('git ' + args[0] + ': ' + child.stderr.decode(errors='replace')[-2000:])
    return child.stdout if binary else child.stdout.decode().strip()


def remote_head(repo):
    text = git(repo, 'ls-remote', 'origin', 'refs/heads/' + BRANCH)
    return text.split()[0] if text else None


def inventory(repo, files):
    result = []
    for name in sorted(set(files)):
        path = Path(repo)/name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink() or not path.is_file():
            raise ValueError('Publication requires scoped regular files: ' + name)
        if path.stat().st_size > 40*2**20:
            raise ValueError('Single Git blob exceeds 40 MiB: ' + name)
        result.append(dict(path=name, sha256=digest_file(path), bytes=path.stat().st_size))
    return result


def _commit(repo, files, message):
    unrelated = set(filter(None, git(repo,'diff','--cached','--name-only','-z').split('\0'))) - set(files)
    if unrelated:
        raise ValueError('Unrelated staged files must be preserved: ' + repr(sorted(unrelated)))
    git(repo,'add','--',*files)
    git(repo,'diff','--cached','--check','--',*files)
    changed=set(filter(None,git(repo,'diff','--cached','--name-only','-z').split('\0')))
    if not changed or not changed <= set(files):
        raise ValueError('Scoped publication commit is empty or includes unrelated files')
    git(repo,'commit','-m',message)
    return git(repo,'rev-parse','HEAD')


def _push(repo, out, state, stage, commit):
    # A prior push may have completed before its receipt write was interrupted.
    if remote_head(repo) == commit:
        return dict(remote_SHA=commit, push_attempts=state.get(stage+'_push_attempts',0))
    count=state.get(stage+'_push_attempts',0)
    failure=None
    while count < 2:
        count += 1
        state[stage+'_push_attempts']=count
        write(Path(out)/'publication/transaction.json',state)
        try:
            git(repo,'push','origin','HEAD:refs/heads/'+BRANCH)
            remote=remote_head(repo)
            if remote != commit:
                raise RuntimeError('Remote full SHA differs from local commit')
            return dict(remote_SHA=remote,push_attempts=count)
        except RuntimeError as exc:
            failure=str(exc)
            state[stage+'_push_failure']=failure
            write(Path(out)/'publication/transaction.json',state)
            if any(text in failure for text in ('non-fast-forward','Permission denied','Authentication failed')):
                break
    raise StageFailure('Normal push could not be verified; attempts preserved: '+str(failure),
                       missing=['existing origin push access; no force push is used'])


def _read_remote_result(repo, commit, name):
    remote=git(repo,'remote','get-url','origin')
    if remote.startswith('https://github.com/') or remote.startswith('git@github.com:'):
        repository=remote.split('github.com',1)[1].lstrip('/:').removesuffix('.git')
        endpoint='repos/'+repository+'/contents/'+quote(name,safe='/')+'?ref='+commit
        if shutil.which('gh'):
            child=subprocess.run(['gh','api',endpoint,
                '-H','Accept: application/vnd.github.raw+json'],capture_output=True)
            if child.returncode:
                raise StageFailure('Remote result content could not be read',missing=[name])
            payload=child.stdout
            mode='GITHUB_CONTENTS_RAW_AT_COMMIT'
        else:
            from urllib.request import Request, urlopen
            from urllib.error import URLError
            request=Request('https://api.github.com/'+endpoint, headers={
                'Accept':'application/vnd.github.raw+json',
                'User-Agent':'ovmap-benchmark-completion-v2'})
            try:
                with urlopen(request,timeout=30) as response:
                    payload=response.read((Path(repo)/name).stat().st_size+1)
            except (URLError,OSError) as error:
                raise StageFailure('Remote public result content could not be read',missing=[name]) from error
            mode='GITHUB_CONTENTS_RAW_AT_COMMIT_PUBLIC_HTTP'
    elif Path(remote).is_dir():
        # Integration tests use a distinct temporary bare remote, never the local worktree.
        payload=git(Path(remote),'show',commit+':'+name,binary=True)
        mode='LOCAL_BARE_REMOTE_INTEGRATION_TEST'
    else:
        raise StageFailure('Unsupported remote content reader',missing=['GitHub origin or explicit local bare test remote'])
    sha=hashlib.sha256(payload).hexdigest()
    if sha != digest_file(Path(repo)/name):
        raise ValueError('New result read from remote differs from C1 bytes')
    return dict(path=name,commit=commit,sha256=sha,bytes=len(payload),mode=mode)


def publish_transaction(*, repo, out, files, receipt, handoff, release_identity, result_files):
    repo, out=Path(repo),Path(out)
    transaction=out/'publication/transaction.json'
    final=out/'publication/final.json'
    if git(repo,'branch','--show-current') != BRANCH:
        raise ValueError('Publication must stay on the fixed research branch')
    if final.is_file():
        old=verified(final)
        if old['release_identity'] != release_identity or git(repo,'rev-parse','HEAD') != old['C2']:
            raise ValueError('Published transaction identity/head changed; preserve the existing release')
        if git(repo,'status','--porcelain','--',*files,receipt) or remote_head(repo) != old['C2']:
            raise ValueError('Published scoped files or remote ref changed')
        return old
    if transaction.is_file():
        state=verified(transaction)
        state.pop('identity',None)
        if state['release_identity'] != release_identity:
            raise ValueError('Pending publication scientific identity changed')
    else:
        checked=inventory(repo,files)
        state=dict(status='DONE',stage='C1_PREPARED',parent=git(repo,'rev-parse','HEAD'),
            release_identity=release_identity,files=checked,result_files=list(result_files),receipt=receipt,handoff=handoff)
        if not result_files or not set(result_files) <= set(files):
            raise ValueError('C1 requires a new result artifact for remote-content verification')
        for name in result_files:
            exists=subprocess.run(['git','cat-file','-e',state['parent']+':'+name],cwd=repo,capture_output=True).returncode==0
            if exists:
                raise ValueError('Remote proof must read a newly added result, not an old parent file')
        write(transaction,state)
    for item in state['files']:
        if item['path']==handoff and state['stage'] in ('C1_REMOTE_VERIFIED','C2_LOCAL'):
            continue
        if digest_file(repo/item['path']) != item['sha256']:
            raise ValueError('Pending C1 scientific bytes changed: '+item['path'])
    c1_message='OVMap completion v2 C1 '+release_identity
    c2_message='OVMap completion v2 C2 '+release_identity
    if state['stage']=='C1_PREPARED':
        head=git(repo,'rev-parse','HEAD')
        if head==state['parent']:
            head=_commit(repo,files,c1_message)
        elif git(repo,'log','-1','--format=%s') != c1_message or git(repo,'rev-parse','HEAD^') != state['parent']:
            raise ValueError('C1 preparation head changed')
        state.update(stage='C1_LOCAL',C1=head)
        write(transaction,state)
    if state['stage']=='C1_LOCAL':
        if git(repo,'rev-parse','HEAD') != state['C1']:
            raise ValueError('Pending C1 head changed')
        state['C1_push']=_push(repo,out,state,'C1',state['C1'])
        chosen=random.SystemRandom().choice(state['result_files'])
        state['remote_result_read']=_read_remote_result(repo,state['C1'],chosen)
        state['stage']='C1_REMOTE_VERIFIED'
        write(transaction,state)
    if state['stage']=='C1_REMOTE_VERIFIED':
        write(repo/receipt,dict(status='DONE',C1=state['C1'],remote_SHA=state['C1_push']['remote_SHA'],
            remote_result_read=state['remote_result_read'],files=state['files'],
            release_identity=release_identity,scope='C1_ONLY;C2_RECEIPT_IS_EXTERNAL'))
        marker='\nVerified scientific commit C1: `'+state['C1']+'`.'
        path=repo/handoff
        if marker not in path.read_text():
            path.write_text(path.read_text()+marker+' [Immutable results](https://github.com/Orangekostar/oviovo/blob/'
                +state['C1']+'/'+str(DOCS/'RESULTS.md')+').\n'
                'The C2 full SHA and actual post-publication resume receipt remain at $OUT/publication/final.json.\n')
        head=git(repo,'rev-parse','HEAD')
        if head==state['C1']:
            head=_commit(repo,[receipt,handoff],c2_message)
        elif git(repo,'log','-1','--format=%s') != c2_message or git(repo,'rev-parse','HEAD^') != state['C1']:
            raise ValueError('C2 preparation head changed')
        changed=set(git(repo,'diff-tree','--no-commit-id','--name-only','-r',head).splitlines())
        if changed != {receipt,handoff}:
            raise ValueError('C2 changed files beyond C1 receipt and handoff')
        state.update(stage='C2_LOCAL',C2=head)
        write(transaction,state)
    if git(repo,'rev-parse','HEAD') != state['C2']:
        raise ValueError('Pending C2 head changed')
    second=_push(repo,out,state,'C2',state['C2'])
    return write(final,dict(status='DONE',C1=state['C1'],C2=state['C2'],local_HEAD=state['C2'],
        remote_HEAD=second['remote_SHA'],branch=BRANCH,release_identity=release_identity,
        remote_result_read=state['remote_result_read'],C1_receipt=receipt,
        C1_push_attempts=state.get('C1_push_attempts',0),C2_push_attempts=state.get('C2_push_attempts',0),
        actual_post_publication_resume=None,publication_is_not_scientific_completion=True))


def release_overflow(binding, index):
    """Use only existing same-repository gh access; otherwise keep every local part."""
    root=Path(binding['output_root'])
    parts={part['sha256']:part for row in index['files'] for part in row['parts']
           if part['distribution']!='GIT'}
    if not parts:
        return index
    tag='ovmap-completion-v2-'+index['identity'][:16]
    available=shutil.which('gh') and subprocess.run(['gh','auth','status'],cwd=REPO,capture_output=True).returncode==0
    if available:
        try:
            exists=subprocess.run(['gh','release','view',tag],cwd=REPO,capture_output=True).returncode==0
            if not exists:
                subprocess.run(['gh','release','create',tag,'--target',BASE,'--title',tag,
                    '--notes','Derived experiment shards. Protocol, roles and coverage are in the research branch.'],
                    cwd=REPO,check=True,capture_output=True)
            listing=subprocess.run(['gh','release','view',tag,'--json','assets'],cwd=REPO,
                                   check=True,capture_output=True,text=True)
            import json
            assets={row['name']:row for row in json.loads(listing.stdout)['assets']}
            for part in parts.values():
                path=root/'publication/release_assets'/part['path']
                if path.name not in assets:
                    subprocess.run(['gh','release','upload',tag,str(path)],cwd=REPO,check=True,capture_output=True)
            listing=subprocess.run(['gh','release','view',tag,'--json','assets'],cwd=REPO,
                                   check=True,capture_output=True,text=True)
            assets={row['name']:row for row in json.loads(listing.stdout)['assets']}
            for row in index['files']:
                for part in row['parts']:
                    if part['distribution']!='GIT':
                        asset=assets[Path(part['path']).name]
                        if asset['size']!=part['bytes']:
                            raise ValueError('Release asset byte count differs')
                        part.update(distribution='GITHUB_RELEASE',url=asset['url'])
        except (subprocess.CalledProcessError,ValueError,KeyError) as exc:
            write(root/'publication/release_error.json',dict(status='BLOCKED_DEPENDENCY',
                reason=type(exc).__name__,local_parts_preserved=True,tag=tag))
    from .publication_export import finalize_index
    return finalize_index(binding,index)


def publish(binding, args):
    root=Path(binding['output_root'])
    pending=root/'publication/transaction.json'
    if pending.is_file():
        state=verified(pending)
        return publish_transaction(repo=REPO,out=root,files=[row['path'] for row in state['files']],
            receipt=state['receipt'],handoff=state['handoff'],release_identity=state['release_identity'],
            result_files=state['result_files'])
    primary=root/'review/primary_final.json'
    if not primary.is_file() or not verified(primary).get('requirements_review_complete'):
        raise StageFailure('Publication is waiting for the primary requirement-by-requirement review',
                           missing=[str(primary)])
    if any(row.get('runnable') for row in verified(root/'remaining_jobs.json')['jobs']):
        raise StageFailure('Finite scientific jobs are still runnable; keep collecting their results before C1',
                           status='BLOCKED_RESOURCE',missing=[str(root/'remaining_jobs.json')])
    from .lifecycle import producer_identity
    primary_record=verified(primary)
    if (primary_record.get('reviewed_producer_identity') != producer_identity()
            or primary_record.get('result_store_identity') != verified(root/'result_store.json')['identity']):
        raise ValueError('Primary review does not bind the current source and numerical result store')
    from .publication_export import export, _is_remote_result_source
    index=release_overflow(binding,export(binding))
    files=[]
    for folder in ('src/static_ovmap/benchmark_completion_v2',str(DOCS),str(ARTIFACTS)):
        files += [str(p.relative_to(REPO)) for p in (REPO/folder).rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.name!='PUBLICATION_RECEIPT.json']
    files += ['configs/static_ovmap/benchmark_completion_v2.json',
        'scripts/evaluation/run_ovmap_benchmark_completion_v2.py',
        'scripts/evaluation/render_ovmap_benchmark_completion_supplement.py',
        'scripts/evaluation/monitor_ovmap_benchmark_capacity.py',
        'scripts/evaluation/test_ovmap_benchmark_capacity_guard.py',
        'scripts/evaluation/run_ovmap_parallel_scenes.py',
        'scripts/evaluation/test_ovmap_parallel_scenes.py',
        'docs/superpowers/plans/2026-10-23-ovmap-parallel-scenes.md',
        'docs/superpowers/plans/2026-10-22-ovmap-benchmark-completion-v2.md']
    files += [str(p.relative_to(REPO)) for p in (REPO/'tests/static_ovmap').glob('test_benchmark_completion_*.py')]
    identity=canonical_digest(dict(artifacts=index['identity'],source=inventory(REPO,files),
                                   primary_review=verified(primary)['identity']))
    result_files=sorted({part['path'] for row in index['files']
        if _is_remote_result_source(row['source']) and len(row['parts']) == 1
        for part in row['parts'] if part['distribution'] == 'GIT'})
    if not result_files:
        raise StageFailure('Publication requires a complete numerical result in Git before C1',
                           missing=[str(ARTIFACTS/'artifact_index.json')])
    return publish_transaction(repo=REPO,out=root,files=files,
        receipt=str(DOCS/'PUBLICATION_RECEIPT.json'),handoff=str(DOCS/'HANDOFF.md'),
        release_identity=identity,result_files=result_files)


def finish(binding, args):
    """Only after the phase receipt exists; never close while scientific jobs run."""
    root=Path(binding['output_root'])
    final=root/'publication/final.json'
    if not final.is_file():
        return None
    record=verified(final)
    if record.get('actual_post_publication_resume') is not None:
        return record
    remaining=verified(root/'remaining_jobs.json')['jobs']
    if any(row.get('runnable') for row in remaining):
        return record
    from .lifecycle import close
    coverage=verified(root/'review/coverage.json')
    flags=dict(coverage['completion_flags'],publication_complete=True)
    receipts=[root/name for name in ('protocol_lock.json','model_lock.json','simple_recipe_lock.json',
        'calibration/summary.json','legacy/prediction_lock.json','legacy/score.json',
        'tables/receipt.json','review/coverage.json','review/primary_final.json','remaining_jobs.json')
        if (root/name).is_file()]
    close(args,flags=flags,receipts=receipts,remaining_jobs=remaining)
    before={str(p):(digest_file(p),p.stat().st_mtime_ns) for p in [*receipts,root/'terminal.json']}
    command=[sys.executable,str(REPO/'scripts/evaluation/run_ovmap_benchmark_completion_v2.py'),
             '--output-root',str(root),'--resume']
    for key in ('semantic_root','structure_root','path_map','asset_config','spec','gpu','max_gpu_hours'):
        if getattr(args,key,None) is not None:
            command.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
    child=subprocess.run(command,cwd=REPO,capture_output=True,text=True)
    after={str(p):(digest_file(p),p.stat().st_mtime_ns) for p in [*receipts,root/'terminal.json']}
    expected=verified(root/'terminal.json')['exit_code']
    passed=child.returncode==expected and before==after and 'new_workers=0 scientific_imports=0 writes=0' in child.stdout
    record['actual_post_publication_resume']=dict(status='DONE' if passed else 'FAILED_RUNTIME',
        actual_exit_code=child.returncode,expected_exit_code=expected,scientific_hashes_and_mtimes_equal=before==after,
        stdout=child.stdout,stderr=child.stderr,command=command,synthetic=False)
    write(final,record)
    if not passed:
        raise RuntimeError('Actual post-publication terminal resume failed; receipt preserved externally')
    return record
