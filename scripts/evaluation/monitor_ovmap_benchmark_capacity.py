"""Finite live disk checks beside an existing controller; no scientific workers."""
import argparse
import ctypes
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from static_ovmap.benchmark_completion_v2.common import digest_file, verified, write


@dataclass(frozen=True)
class Config:
    root: Path
    controller_pid: int
    controller_start: str
    protocol_identity: str
    first: int
    last: int
    poll_seconds: float = 5.


def state_dir(config):
    return config.root/'bootstrap'/f'capacity_guard{config.first}_to{config.last}'


def process_info(pid):
    try:
        values = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    return dict(state=values[0], start=values[19])


def controller_live(config):
    row = process_info(config.controller_pid)
    return row is not None and row['start'] == config.controller_start and row['state'] != 'Z'


def optional_record(path):
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    return verified(path)


def check_sources(config):
    baseline = verified(config.root/'bootstrap/before_ovi_scenes38_to312_entry.json')
    for name, row in baseline['files'].items():
        path = Path(name)
        if digest_file(path) != row['sha256'] or path.stat().st_mtime_ns != row['mtime_ns']:
            raise ValueError('Protected input changed: '+name)
    return len(baseline['files'])


def configuration(config):
    protocol = verified(config.root/'protocol_lock.json')
    if protocol['identity'] != config.protocol_identity:
        raise ValueError('Protocol identity changed')
    scope = protocol['scopes']['scannet200_val']
    scenes = scope['scenes']
    if (scope['status'] != 'DONE' or len(scenes) != len(set(scenes))
            or not 1 <= config.first <= config.last <= len(scenes)
            or not 0 < config.poll_seconds <= 30 or config.controller_pid <= 1):
        raise ValueError('Invalid finite capacity scope')
    binding = verified(config.root/'binding.json')
    if Path(binding['output_root']).resolve() != config.root.resolve():
        raise ValueError('Output root differs from binding')
    reserve = int(binding['specification']['runtime']['minimal_disk_reserve_GiB']*2**30)
    if reserve < 20*2**30:
        raise ValueError('Disk reserve cannot be below the contract minimum')
    check_sources(config)
    return scenes[config.first-1:config.last], reserve


def known_required_free(config, reserve):
    estimates = [verified(path)['estimate']['estimated_bytes']
                 for path in (config.root/'resources/capacity').glob('*/*.json')]
    return reserve+2*max(estimates, default=0)


def inspect_scene(config, scene):
    """Read an already committed source receipt and the sensor header only."""
    receipt_path = config.root/'assets/datasets/scannet/receipts'/f'{scene}.json'
    assets = optional_record(receipt_path)
    if assets is None or assets['status'] != 'DONE':
        return None
    if assets['scene'] != scene or assets['GT_files'] or assets['GT_labels_opened']:
        raise ValueError('Capacity input must be the original label-free scene assets')
    from static_ovmap.module_validation.scannet_frames import sensor_header
    sensor = Path(assets['files']['.sens']['path'])
    mesh = Path(assets['files']['_vh_clean_2.ply']['path'])
    stamps = {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
              for path in (sensor, mesh)}
    for suffix, path in (('.sens', sensor), ('_vh_clean_2.ply', mesh)):
        declared = assets['files'][suffix].get('bytes')
        if declared is not None and declared != path.stat().st_size:
            raise ValueError('Committed scene asset byte count changed')
    with sensor.open('rb') as stream:
        header = sensor_header(stream)
        consumed = stream.tell()
        if consumed > 4500:
            raise ValueError('Sensor-header read exceeded its fixed bound')
        stream.seek(0)
        header_sha = hashlib.sha256(stream.read(consumed)).hexdigest()
    if stamps != {str(path): [path.stat().st_size, path.stat().st_mtime_ns]
                  for path in (sensor, mesh)}:
        raise ValueError('Scene assets changed during header observation')
    frames = min(200, header['source_frame_count'])
    rgb = header['color_width']*header['color_height']
    depth = header['depth_width']*header['depth_height']
    if min(frames, rgb, depth) <= 0:
        raise ValueError('Scene dimensions must be positive')
    planned = frames*(3*rgb+2*depth+128*max(rgb, depth))+16*mesh.stat().st_size
    previous = [verified(path)['task_owned_bytes']
                for path in (config.root/'resources/storage').glob('*/*.json')]
    estimated = max(planned, max(previous, default=0))
    binding = verified(config.root/'binding.json')
    reserve = int(binding['specification']['runtime']['minimal_disk_reserve_GiB']*2**30)
    return dict(scene=scene, source_receipt=str(receipt_path), source_identity=assets['identity'],
        input_stamps=stamps, sensor_header_bytes_read=consumed, sensor_header_sha256=header_sha,
        pixel_plan_bytes=planned, estimated_bytes=estimated, required_free_bytes=reserve+2*estimated,
        source_frame_upper_bound=frames, estimate_is_measured_consumption=False,
        mesh_payload_bytes_read=0, GT_labels_read=False, image_payloads_decoded=False,
        full_sensor_SHA_independently_checked=False)


def stop_controller(config):
    """Use a kernel process handle; a reused PID can never receive SIGSTOP."""
    if not controller_live(config):
        raise ValueError('Controller identity is absent or changed before pause')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.pidfd_open.argtypes = [ctypes.c_int, ctypes.c_uint]
    libc.pidfd_open.restype = ctypes.c_int
    libc.pidfd_send_signal.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    libc.pidfd_send_signal.restype = ctypes.c_int
    descriptor = libc.pidfd_open(config.controller_pid, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), 'pidfd_open failed')
    try:
        if not controller_live(config):
            raise ValueError('Controller identity changed after opening the process handle')
        if process_info(config.controller_pid)['state'] in ('T', 't'):
            return dict(signal_sent=False, already_stopped=True, pid=config.controller_pid,
                        start=config.controller_start, stop_observed=True, children_signalled=False)
        sys.audit('ovmap.capacity_guard.pause', config.controller_pid, config.controller_start)
        if libc.pidfd_send_signal(descriptor, signal.SIGSTOP, None, 0) < 0:
            raise OSError(ctypes.get_errno(), 'pidfd_send_signal failed')
        action = dict(signal_sent=True, already_stopped=False, pid=config.controller_pid,
                      start=config.controller_start, signal='SIGSTOP', children_signalled=False)
        for _ in range(20):
            row = process_info(config.controller_pid)
            if row and row['start'] == config.controller_start and row['state'] in ('T', 't'):
                return dict(action, stop_observed=True)
            time.sleep(0.05)
        return dict(action, stop_observed=False)
    finally:
        os.close(descriptor)


def preflight(config):
    scenes, reserve = configuration(config)
    if not controller_live(config):
        raise ValueError('Original controller is not live with its pinned identity')
    observation = inspect_scene(config, scenes[0])
    required = max(known_required_free(config, reserve),
                   observation['required_free_bytes'] if observation else reserve)
    free = shutil.disk_usage(config.root).free
    return dict(status='DONE' if free >= required else 'BLOCKED_RESOURCE',
                observation=observation, free_bytes=free, required_free_bytes=required,
                signal_sent=False, preflight_only=True)


def run(config):
    scenes, reserve = configuration(config)
    folder = state_dir(config)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/'owner.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = folder/'progress.json'
        if os.path.lexists(progress):
            raise ValueError('A prior capacity run exists; inspect it before any new launch')
        observations, completed = {}, {}
        samples = folder/'samples.jsonl'
        source_sha = digest_file(__file__)
        sample_count = 0
        minimum_free = None
        stop_action = None

        def save(status, **extra):
            return write(progress, dict(status=status, controller_pid=config.controller_pid,
                controller_start=config.controller_start, protocol=config.protocol_identity,
                first=config.first, last=config.last, completed=list(completed.values()),
                observed_scenes=list(observations), sample_count=sample_count,
                minimum_observed_free_bytes=minimum_free, source_sha256=source_sha,
                no_new_workers_or_training=True, historical_prechecks_backfilled=False,
                full_objective_complete=False, **extra))

        try:
            while True:
                if digest_file(__file__) != source_sha:
                    raise ValueError('The running guard source changed')
                pending = []
                for scene in scenes:
                    if scene in completed:
                        continue
                    scored = optional_record(config.root/'benchmark/evaluation/scannet200_val'/scene/'receipt.json')
                    if scored is not None:
                        if (scored['status'] != 'DONE' or scored['scene'] != scene
                                or scored['benchmark'] != 'scannet200_val' or scored['logical_rows'] != 8):
                            raise ValueError('Existing scene score has an invalid status or identity')
                        completed[scene] = dict(scene=scene, score=scored['identity'],
                            observed_during_run=scene in observations)
                    else:
                        pending.append(scene)
                if not pending:
                    save('DONE', signal_sent=False, sample_log_sha256=digest_file(samples) if samples.exists() else None)
                    return 0
                if not controller_live(config):
                    save('BLOCKED_DEPENDENCY', signal_sent=False, reason='PINNED_CONTROLLER_NO_LONGER_LIVE')
                    return 20
                required = max([known_required_free(config, reserve)]+
                               [row['required_free_bytes'] for row in observations.values()])
                for scene in pending:
                    if scene not in observations:
                        observation = inspect_scene(config, scene)
                        if observation is None:
                            continue
                        check_sources(config)
                        observation['status'] = 'DONE'
                        observation['observed_unix_seconds'] = time.time()
                        observation['first_job_marker_present_at_observation'] = any(
                            (config.root/'jobs/benchmark/scannet200_val'/scene/job/'process.json').exists()
                            for job in ('cropformer', 'native_map', 'F_MV1000', 'native_query'))
                        observations[scene] = write(folder/'observations'/f'{scene}.json', observation)
                    observation = observations[scene]
                    if verified(observation['source_receipt'])['identity'] != observation['source_identity']:
                        raise ValueError('Observed scene asset receipt changed')
                    if any([Path(name).stat().st_size, Path(name).stat().st_mtime_ns] != stamp
                           for name, stamp in observation['input_stamps'].items()):
                        raise ValueError('Observed scene asset metadata changed')
                    required = max(required, observation['required_free_bytes'])
                free = shutil.disk_usage(config.root).free
                minimum_free = free if minimum_free is None else min(minimum_free, free)
                sample_count += 1
                with samples.open('a') as stream:
                    stream.write(json.dumps(dict(unix_seconds=time.time(), free_bytes=free,
                        required_free_bytes=required, pending_scenes=len(pending)))+'\n')
                    stream.flush()
                if free < required:
                    check_sources(config)
                    action = stop_action = stop_controller(config)
                    save('BLOCKED_RESOURCE', free_bytes=free, required_free_bytes=required,
                         stop_action=action, signal_sent=action['signal_sent'],
                         reason='DISK_RESERVE_BELOW_EXISTING_FORMULA',
                         current_child_may_complete=True, automatic_resume=False,
                         sample_log_sha256=digest_file(samples))
                    return 20
                save('RUNNING', free_bytes=free, required_free_bytes=required,
                     pending_scenes=len(pending), signal_sent=False)
                time.sleep(config.poll_seconds)
        except Exception as error:
            save('FAILED_RUNTIME', error_type=type(error).__name__, error=str(error),
                 signal_sent=stop_action['signal_sent'] if stop_action else False,
                 stop_action=stop_action)
            traceback.print_exc()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--controller-start', required=True)
    parser.add_argument('--expected-protocol', required=True)
    parser.add_argument('--first', type=int, default=44)
    parser.add_argument('--last', type=int, default=312)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = Config(args.output_root, args.controller_pid, args.controller_start,
                    args.expected_protocol, args.first, args.last)
    if args.preflight:
        report = preflight(config)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return 0 if report['status'] == 'DONE' else 20
    return run(config)


if __name__ == '__main__':
    raise SystemExit(main())
