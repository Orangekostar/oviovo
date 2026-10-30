"""Execute explicit frozen baseline leaves and retain actual exits for every stage."""
import argparse
import fcntl
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2];sys.path[:0]=[str(REPO/'src'),str(REPO)]
from static_ovmap.benchmark_completion_v2.common import verified,write,digest_file,canonical_digest


def output_descriptor(path):
    path=Path(path)
    return dict(path=str(path),sha256=digest_file(path),bytes=path.stat().st_size)


def run(config_path):
    config=verified(config_path);folder=Path(config['runtime']);folder.mkdir(parents=True,exist_ok=True)
    assert digest_file(config['supervisor'])==config['supervisor_sha256']
    completed=[]
    with (folder/'owner.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for job in config['jobs']:
            launch=verified(job['launch']);actual_path=Path(launch['local'])/'actual_exit.json'
            if not actual_path.exists():
                code=subprocess.run([sys.executable,config['supervisor'],'--config',job['launch']],check=False).returncode
                assert code==0,job['name']+' failed; preserve actual exit and review before retry'
            actual=verified(actual_path);synchronized=verified(actual_path.with_name('synchronized.json'))
            assert actual['actual_child_exit']==0 and synchronized['status']=='DONE'
            assert actual['command']==launch['command'] and actual['source_sha256']==launch['source_sha256']
            assert digest_file(actual['log']['path'])==actual['log']['sha256']
            key=canonical_digest(dict(job=job,launch=launch['identity'],actual=actual['identity']))
            receipt_path=folder/'stages'/(job['name']+'.json')
            if receipt_path.exists():
                row=verified(receipt_path);assert row['input_identity']==key
                for desc in row['outputs']:
                    assert Path(desc['path']).stat().st_size==desc['bytes'] and digest_file(desc['path'])==desc['sha256']
            else:
                outputs=[output_descriptor(p) for p in job['outputs']]
                row=write(receipt_path,dict(status='DONE',input_identity=key,method=job['method'],scene=job['scene'],stage=job['stage'],
                    outputs=outputs,actual_exit=actual['identity'],launch=launch['identity'],source_sha256=launch['source_sha256'],GT_labels_read=False))
            completed.append(dict(name=job['name'],receipt=row['identity']))
            write(folder/'progress.json',dict(status='RUNNING',config=config['identity'],completed=completed,
                completed_stages=len(completed),expected_stages=len(config['jobs'])))
            print('CONTROLLED_QUEUE_STAGE_DONE',job['name'],len(completed),len(config['jobs']),flush=True)
        write(folder/'complete.json',dict(status='DONE',config=config['identity'],completed=completed,
            completed_stages=len(completed),expected_stages=len(config['jobs']),all_actual_child_exits_zero=True))
        print('CONTROLLED_QUEUE_CLOSED',len(completed),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args();run(args.config)

if __name__=='__main__':main()
