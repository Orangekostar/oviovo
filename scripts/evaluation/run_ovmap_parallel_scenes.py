"""Independent finite scene scheduler; frozen scientific modules stay unchanged."""
import argparse
import contextlib
import copy
import ctypes
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
sys.path.insert(0, str(REPO))
from static_ovmap.benchmark_completion_v2.common import (
    canonical_digest, digest_file, verified, write, StageFailure)

BENCHMARK = 'scannet200_val'
METHODS = {v+'_seed'+str(s) for v in ('U00','U10','U01','U11') for s in (17,29)}


@contextlib.contextmanager
def file_lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        # Poll nonblocking acquisition: the output NFS mount did not reliably
        # wake all three real blocking flock waiters in the regression fixture.
        while True:
            try:
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if not blocking:
                    raise
                time.sleep(.1)
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def process_info(pid):
    try:
        row = Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
        return dict(state=row[0], start=row[19])
    except FileNotFoundError:
        return None


def waiting_on_lock(pid, path):
    stat = Path(path).stat()
    target = f'{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}:{stat.st_ino}'
    for line in Path('/proc/locks').read_text().splitlines():
        if '->' not in line:
            continue
        parts = line.replace('->','').split()
        if len(parts) >= 6 and parts[1] == 'FLOCK' and parts[4] == str(pid) and parts[5] == target:
            return True
    return False


def pinned_signal(pid, start, sig):
    if process_info(pid) is None or process_info(pid)['start'] != start:
        raise ValueError('Pinned controller identity changed')
    libc = ctypes.CDLL(None,use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int,ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    fd = libc.pidfd_open(pid,0)
    if fd < 0:
        raise OSError(ctypes.get_errno(),'pidfd_open failed')
    try:
        if process_info(pid) is None or process_info(pid)['start'] != start:
            raise ValueError('Controller identity changed after pidfd_open')
        if libc.pidfd_send_signal(fd,sig,None,0) < 0:
            raise OSError(ctypes.get_errno(),'pidfd_send_signal failed')
    finally:
        os.close(fd)


def pause_at_gate(pid, start, path):
    row = process_info(pid)
    if not row or row['start'] != start:
        raise ValueError('Wrong original PID/start')
    children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    if children or not waiting_on_lock(pid,path):
        raise ValueError('Original controller is not child-free at the GPU flock gate')
    pinned_signal(pid,start,signal.SIGSTOP)
    for _ in range(100):
        row = process_info(pid)
        if row and row['start'] == start and row['state'] in ('T','t'):
            return dict(pid=pid,start=start,state=row['state'],signal='SIGSTOP',
                        children_signalled=False,reason='USER_AUTHORIZED_MULTI_GPU_HANDOVER')
        time.sleep(.02)
    raise RuntimeError('Controller pause was not observed')


def check_sources(root):
    baseline = verified(root/'bootstrap/before_ovi_scenes38_to312_entry.json')
    for name, row in baseline['files'].items():
        path = Path(name)
        if digest_file(path) != row['sha256'] or path.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Frozen source changed: '+name)
    return len(baseline['files'])


def required_gpu(root, scene):
    devices = set()
    for path in (root/'jobs/benchmark'/BENCHMARK/scene).glob('*/receipt.json'):
        row = verified(path)
        device = row.get('env',{}).get('CUDA_VISIBLE_DEVICES')
        if device not in (None,''):
            devices.add(int(device))
    if len(devices)>1:
        raise ValueError('Existing job devices disagree: '+scene)
    return next(iter(devices),None)


def choose_scene(root, pending, gpu, assignments):
    for scene in pending:
        device = assignments.get(scene,required_gpu(root,scene))
        if device is None or device == gpu:
            return scene
    return None


def validate_descriptor(row):
    path = Path(row['path'])
    if path.stat().st_size != row['bytes'] or digest_file(path) != row['sha256']:
        raise ValueError('Completed scene bytes changed: '+str(path))


def completed_scene(root, scene):
    pred_path = root/'benchmark/predictions'/BENCHMARK/scene/'receipt.json'
    score_path = root/'benchmark/evaluation'/BENCHMARK/scene/'receipt.json'
    if not score_path.exists():
        return None
    if not pred_path.exists():
        raise ValueError('Score exists without its frozen prediction: '+scene)
    pred, score = verified(pred_path), verified(score_path)
    gt = verified(root/'benchmark/GT'/BENCHMARK/scene/'receipt.json')
    if (pred['status']!='DONE' or score['status']!='DONE' or pred['completed_rows']!=8
            or score['logical_rows']!=8 or set(pred['methods'])!=METHODS or set(score['rows'])!=METHODS
            or pred['scene']!=scene or score['scene']!=scene
            or pred['benchmark']!=BENCHMARK or score['benchmark']!=BENCHMARK
            or score['prediction_lock']!=pred['identity'] or gt['prediction_freeze']!=pred['identity']
            or score['GT']!=gt['identity']):
        raise ValueError('Completed scene chain differs: '+scene)
    validate_descriptor(gt['GT'])
    for method, descriptor in pred['methods'].items():
        manifest = verified(descriptor['path'])
        row = score['rows'][method]
        source = verified(pred['methods'][row.get('alias_of',method)]['path'])
        if (manifest['identity']!=descriptor['identity'] or row['prediction']!=source['identity']
                or source['prediction_identity']!=manifest['prediction_identity']
                or row['prediction_identity']!=manifest['prediction_identity']
                or row['GT']!=gt['identity']
                or verified(row['matches']['path'])['identity']!=row['matches']['identity']):
            raise ValueError('Completed method chain differs: '+scene+'/'+method)
        validate_descriptor(manifest['arrays'])
    return dict(status='DONE',scene=scene,prediction=pred['identity'],score=score['identity'])


def cache_safe_execute(execute, root):
    def wrapped(binding, job_name, *args, **kwargs):
        stage = job_name.rsplit('/',1)[-1]
        lock = (file_lock(root/'bootstrap/parallel_scenes'/('cache_'+stage+'.lock'))
                if stage in ('F_MV1000','native_query') else contextlib.nullcontext())
        with lock:
            return execute(binding,job_name,*args,**kwargs)
    return wrapped


def execute_science(root, scene, gpu):
    from static_ovmap.benchmark_completion_v2 import runtime
    from static_ovmap.benchmark_completion_v2.benchmark import predict_scene
    from static_ovmap.benchmark_completion_v2.capacity import check_scene, observe_scene
    from static_ovmap.benchmark_completion_v2.semantic_refresh import prepare_standard_score, check_alias
    from static_ovmap.benchmark_completion_v2.standard_score import score_scene
    from static_ovmap.benchmark_completion_v2.score_compat import install
    binding = copy.deepcopy(verified(root/'binding.json'))
    binding['gpu'] = str(gpu)
    if binding['asset_config'].get('gpu_lock'):
        raise ValueError('A shared custom GPU lock requires an explicit device mapping')
    protocol = verified(root/'protocol_lock.json')
    check_alias(binding,protocol)
    install()
    original = runtime.execute
    runtime.execute = cache_safe_execute(original,root)
    try:
        check_scene(binding,BENCHMARK,scene)
        path = root/'benchmark/predictions'/BENCHMARK/scene/'receipt.json'
        pred = verified(path) if path.exists() else None
        if pred is None or pred['status']!='DONE' or pred['completed_rows']!=8:
            predict_scene(binding,BENCHMARK,scene)
        prepare_standard_score(binding,BENCHMARK,scene)
        score_scene(binding,BENCHMARK,scene,
            formal_connection=scene==protocol['scopes'][BENCHMARK]['scenes'][0])
        observe_scene(binding,BENCHMARK,scene)
    finally:
        runtime.execute = original
    return completed_scene(root,scene)


def work_scene(root, scene, gpu, check_frozen=True):
    with file_lock(root/'bootstrap/parallel_scenes/scene_locks'/(scene+'.lock'),False):
        if check_frozen:
            check_sources(root)
        done = completed_scene(root,scene)
        if done:
            return dict(done,reused_complete_scene=True,physical_gpu=gpu)
        needed = required_gpu(root,scene)
        if needed is not None and needed != gpu:
            raise ValueError('Partial scene must retain its actual execution device')
        return dict(execute_science(root,scene,gpu),reused_complete_scene=False,physical_gpu=gpu)


def enough_space(free, reserve, estimates):
    return free >= reserve+2*sum(estimates)


def child_result(child, path):
    code = child.poll()
    if code is None:
        raise ValueError('Child still running')
    if not path.exists():
        return dict(status='FAILED_RUNTIME',actual_exit_code=code,reason='Missing worker receipt')
    row = verified(path)
    if code != 0 or row['status']!='DONE':
        return dict(row,actual_exit_code=code,status=row['status'] if code==20 else 'FAILED_RUNTIME')
    return dict(row,actual_exit_code=code)


def handover(root, pid, start):
    folder = root/'bootstrap/parallel_scenes'
    prior = folder/'handover.json'
    row = process_info(pid)
    if prior.exists():
        old = verified(prior)
        if old['pid']!=pid or old['start']!=start or not row or row['start']!=start or row['state'] not in ('T','t'):
            raise ValueError('Previous handover no longer pins a paused controller')
        return old
    lock = Path('/mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-0.lock')
    write(folder/'handover_wait.json',dict(status='WAITING_FOR_SAFE_BOUNDARY',pid=pid,start=start,
          started_unix=time.time(),old_current_child_allowed_to_finish=True))
    while True:
        row = process_info(pid)
        if not row or row['start']!=start or row['state'] in ('Z','T','t'):
            raise ValueError('Original controller unexpectedly exited or paused before handover')
        with file_lock(lock):
            while True:
                children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
                # N/Q holds the device lock inside its child. Do not trap a
                # child that was launched just before the gate acquired it.
                nq = any(b'benchmark_completion_v2.nq_worker' in Path(f'/proc/{p}/cmdline').read_bytes()
                         for p in children if Path(f'/proc/{p}/cmdline').exists())
                if nq:
                    break
                if not children and waiting_on_lock(pid,lock):
                    action = pause_at_gate(pid,start,lock)
                    return write(prior,dict(action,status='DONE',observed_unix=time.time(),
                        old_controller_resume_forbidden_while_parallel_active=True))
                row = process_info(pid)
                if not row or row['start']!=start or row['state'] in ('Z','T','t'):
                    raise ValueError('Original controller identity/state changed at gate')
                time.sleep(.2)
        time.sleep(.5)


def coordinator(root, gpus, pid, start):
    folder = root/'bootstrap/parallel_scenes'
    with file_lock(folder/'coordinator.lock',False):
        check_sources(root)
        for path in (folder/'workers').glob('*/process.json'):
            old = verified(path)
            live = process_info(old['pid'])
            if live and live['start']==old['start'] and live['state']!='Z':
                raise ValueError('An existing scene worker is still live: '+str(path))
        binding = verified(root/'binding.json')
        protocol = verified(root/'protocol_lock.json')
        scenes = protocol['scopes'][BENCHMARK]['scenes']
        if len(scenes)!=312 or len(set(scenes))!=312 or gpus!=[0,2]:
            raise ValueError('This reviewed launch is finite ScanNet312 on physical GPU0 and GPU2')
        own = process_info(os.getpid())
        write(folder/'entry.json',dict(status='RUNNING',pid=os.getpid(),start=own['start'],
            gpus=gpus,protocol=protocol['identity'],model=verified(root/'model_lock.json')['identity'],
            scientific_binding=binding['identity'],source_sha256=digest_file(__file__),
            user_authorization='Use other GPU cards to accelerate tasks, 2026-10-23',
            scientific_settings_changed=False,max_scene_workers=2,max_own_CPU_scorers=2,
            existing_OVI_CPU_scorers=1,started_unix=time.time()))
        handover(root,pid,start)
        completed, blocked, pending = [], [], []
        for scene in scenes:
            row = completed_scene(root,scene)
            if row:completed.append(dict(row,reused_complete_scene=True))
            else:pending.append(scene)
        assignment_path = folder/'assignments.json'
        assignments = verified(assignment_path)['scenes'] if assignment_path.exists() else {}
        for scene in pending:
            required = required_gpu(root,scene)
            if required is not None:
                if required not in gpus or assignments.get(scene,required)!=required:
                    raise ValueError('Existing partial assignment cannot be adopted: '+scene)
                assignments[scene] = required
        write(assignment_path,dict(scenes=assignments,gpus=gpus))
        reserve = int(binding['specification']['runtime']['minimal_disk_reserve_GiB']*2**30)
        active = {}
        resource_stop = False
        def progress(status='RUNNING'):
            return write(folder/'progress.json',dict(status=status,completed=completed,blocked=blocked,
                pending=pending,active={str(g):dict(scene=v['scene'],pid=v['child'].pid,
                    start=v['start'],estimated_bytes=v['estimate']) for g,v in active.items()},
                resource_dispatch_stopped=resource_stop,expected_scenes=312,updated_unix=time.time()))
        while pending or active:
            if digest_file(__file__) != verified(folder/'entry.json')['source_sha256']:
                raise ValueError('Independent scheduler source changed while active')
            for gpu, job in list(active.items()):
                if job['child'].poll() is None:continue
                result = child_result(job['child'],job['receipt'])
                result.update(scene=job['scene'],physical_gpu=gpu)
                (completed if result['status']=='DONE' else blocked).append(result)
                job['log'].close()
                del active[gpu]
            free = shutil.disk_usage(root).free
            if not enough_space(free,reserve,[j['estimate'] for j in active.values()]):
                resource_stop = True
            for gpu in gpus:
                if gpu in active or resource_stop:continue
                scene = choose_scene(root,pending,gpu,assignments)
                if scene is None:continue
                previous_worker = folder/'workers'/scene/'receipt.json'
                if previous_worker.exists():
                    blocked.append(dict(verified(previous_worker),scene=scene,
                        reason='Previous worker failure requires explicit review; no automatic retry'))
                    pending.remove(scene)
                    continue
                check_sources(root)
                from static_ovmap.benchmark_completion_v2.capacity import estimate_scene
                try:
                    estimate = estimate_scene(binding,BENCHMARK,scene)['estimated_bytes']
                except StageFailure as exc:
                    pending.remove(scene)
                    blocked.append(dict(scene=scene,status=exc.status,reason=str(exc),missing=exc.missing))
                    continue
                if not enough_space(shutil.disk_usage(root).free,reserve,
                                    [j['estimate'] for j in active.values()]+[estimate]):
                    resource_stop = True
                    break
                assignments[scene] = gpu
                write(assignment_path,dict(scenes=assignments,gpus=gpus))
                out = folder/'workers'/scene
                out.mkdir(parents=True,exist_ok=True)
                receipt = out/'receipt.json'
                if receipt.exists():
                    raise ValueError('A previous worker result requires explicit review before rerun: '+scene)
                log = (out/'worker.log').open('a')
                argv = [sys.executable,'-u',str(Path(__file__).resolve()),'--output-root',str(root),
                        '--worker-scene',scene,'--worker-gpu',str(gpu)]
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
                    OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
                child = subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
                pin = process_info(child.pid)
                active[gpu] = dict(child=child,scene=scene,estimate=estimate,receipt=receipt,
                    log=log,start=pin['start'] if pin else None)
                write(out/'process.json',dict(pid=child.pid,start=active[gpu]['start'],argv=argv,
                    physical_gpu=gpu,env={k:env[k] for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS')},
                    started_unix=time.time(),source_sha256=digest_file(__file__)))
                pending.remove(scene)
            progress()
            if resource_stop and not active:break
            time.sleep(5)
        status = 'BLOCKED_RESOURCE' if resource_stop else 'FAILED_RUNTIME' if blocked else 'DONE'
        result = progress(status)
        write(folder/'scan_result.json',dict(result,actual_finite_scan_scenes_finished=not pending,
            replica_status='BLOCKED_DEPENDENCY_SEE_EXISTING_PROTOCOL_AND_REPLICA_RECEIPTS',
            completed_unix=time.time()))
        if len(completed)==312 and not blocked and not pending:
            row = process_info(pid)
            if not row or row['start']!=start or row['state'] not in ('T','t'):
                raise ValueError('Old controller must remain pinned and paused until retirement')
            pinned_signal(pid,start,signal.SIGTERM)
            pinned_signal(pid,start,signal.SIGCONT)
            write(folder/'old_controller_retirement.json',dict(pid=pid,start=start,
                signals=['SIGTERM','SIGCONT'],reason='ALL_312_SCAN_SCENES_FINISHED_BY_PARALLEL_ENTRY',
                actual_old_exit_code='WAIT_FOR_ORIGINAL_SUPERVISOR_RECEIPT',observed_unix=time.time()))
        return 0 if status=='DONE' else 20


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output-root',required=True,type=Path)
    parser.add_argument('--old-pid',type=int)
    parser.add_argument('--old-start')
    parser.add_argument('--gpus',default='0,2')
    parser.add_argument('--worker-scene')
    parser.add_argument('--worker-gpu',type=int)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if args.worker_scene:
        receipt = root/'bootstrap/parallel_scenes/workers'/args.worker_scene/'receipt.json'
        try:
            protocol = verified(root/'protocol_lock.json')
            assignment = verified(root/'bootstrap/parallel_scenes/assignments.json')
            if (args.worker_scene not in protocol['scopes'][BENCHMARK]['scenes']
                    or assignment['scenes'].get(args.worker_scene)!=args.worker_gpu
                    or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.worker_gpu)):
                raise ValueError('Worker actual device/scope differs from its immutable assignment')
            row = work_scene(root,args.worker_scene,args.worker_gpu)
            write(receipt,dict(row,finished_unix=time.time(),source_sha256=digest_file(__file__)))
            return 0
        except Exception as exc:
            write(receipt,dict(status=exc.status if isinstance(exc,StageFailure) else 'FAILED_RUNTIME',
                scene=args.worker_scene,physical_gpu=args.worker_gpu,reason=repr(exc),
                traceback=traceback.format_exc(),finished_unix=time.time()))
            return 20 if isinstance(exc,StageFailure) else 1
    if not args.old_pid or not args.old_start:
        parser.error('Pinned original PID/start are required for reviewed handover')
    return coordinator(root,[int(x) for x in args.gpus.split(',')],args.old_pid,args.old_start)


if __name__=='__main__':
    raise SystemExit(main())
