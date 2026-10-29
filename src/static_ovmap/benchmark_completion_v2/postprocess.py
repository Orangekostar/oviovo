"""Durable single monitor: aggregate after streaming exits, preserving partials."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import REPO, read, write, digest_file

PHASES=('benchmark_score','statistics','baselines','tables','review')


def alive(marker):
    path=Path('/proc')/str(marker['pid'])/'cmdline'
    try:
        command=[item.decode() for item in path.read_bytes().split(b'\0') if item]
    except FileNotFoundError:
        return False
    return command==marker['command']


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--controller',required=True)
    args=parser.parse_args(argv)
    root=Path(args.output_root).resolve()
    folder=root/'bootstrap/postprocess'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('POSTPROCESS_ALREADY_MONITORED',flush=True)
            return 0
        marker=read(root/'bootstrap/streaming_standard_process.json')
        write(folder/'process.json',dict(process_state='WAITING_FOR_EXISTING_STREAM',pid=os.getpid(),
            stream=marker,source_sha256=digest_file(__file__),no_GPU_or_scientific_imports=True))
        while alive(marker):
            time.sleep(15)
        stream_exit=root/'bootstrap/streaming_standard_exit.json'
        if not stream_exit.is_file() or stream_exit.stat().st_mtime < marker['started_unix']:
            write(folder/'exit.json',dict(status='BLOCKED_RESOURCE',
                reason='STREAM_EXIT_RECEIPT_MISSING;NO_DUPLICATE_DRIVER_WAS_LAUNCHED'))
            return 20
        completed={}
        for phase in PHASES:
            command=[args.controller,str(REPO/'scripts/evaluation/run_ovmap_benchmark_completion_v2.py'),
                     '--output-root',str(root),'--resume','--phase',phase]
            started=time.monotonic()
            with (folder/(phase+'.log')).open('w') as stream:
                child=subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'))
            completed[phase]=dict(actual_child_exit=child.returncode,seconds=time.monotonic()-started,command=command)
            write(folder/(phase+'_exit.json'),completed[phase])
        code=0 if all(row['actual_child_exit']==0 for row in completed.values()) else 20
        write(folder/'exit.json',dict(status='DONE',actual_coordinator_exit=code,phases=completed,
            stream_actual_exit=read(stream_exit)['actual_exit_code'],
            only_phase_invocations_complete=True,scientific_completion_requires_separate_primary_review=True,
            publication_not_attempted=True))
        return code


if __name__=='__main__':
    sys.exit(main())
