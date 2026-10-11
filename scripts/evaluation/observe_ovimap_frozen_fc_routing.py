#!/usr/bin/env python3
"""Observe one real all --resume subprocess independently of tool-session lifetime."""
from pathlib import Path
import datetime
import fcntl
import os
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from static_ovmap.frozen_fc_routing.runner import parser
from static_ovmap.frozen_fc_routing.common import write,verified

def main():
    args=parser().parse_args()
    if args.phase!='all' or not args.resume or args._worker:raise ValueError('Observer requires real all --resume')
    root=Path(args.storage_root or args.output_root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.all-observer.lock').open('a+') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempt=str(time.time_ns());log=root/'logs'/('all_'+attempt+'.log');log.parent.mkdir(parents=True,exist_ok=True)
        argv=[sys.executable,'-u',str(ROOT/'scripts/evaluation/run_ovimap_frozen_fc_routing.py'),*sys.argv[1:]]
        started=time.perf_counter();utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with log.open('w') as stream:
            process=subprocess.Popen(argv,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
            write(root/'observer_running.json',dict(status='RUNNING',observer_pid=os.getpid(),pid=process.pid,argv=argv,
                  started_UTC=utc,stdout=str(log),attempt=attempt,observer_survives_tool_session=True))
            code=process.wait()
        terminal=verified(root/'execution_last.json') if (root/'execution_last.json').exists() else {}
        result=verified(root/'result_store.json') if (root/'result_store.json').exists() else {}
        observed=dict(status='OBSERVED_SUCCESS' if code==0 else 'OBSERVED_FAILURE',argv=argv,exit_code=code,
            elapsed_seconds=time.perf_counter()-started,started_UTC=utc,stdout=str(log),attempt=attempt,
            terminal_identity=terminal.get('identity'),result_store_identity=result.get('identity'),
            scientific_status=result.get('status'),observer_survives_tool_session=True)
        write(root/'observations'/(attempt+'.json'),observed);write(root/'execution_observed.json',observed)
        write(root/'observer_running.json',dict(status='FINISHED',observer_pid=os.getpid(),pid=process.pid,exit_code=code,
            stdout=str(log),attempt=attempt))
        return code

if __name__=='__main__':raise SystemExit(main())
