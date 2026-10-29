"""Isolated capacity-guard checks; never signal an existing project process."""
import dataclasses
import contextlib
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from static_ovmap.benchmark_completion_v2.common import digest_file, verified, write


def load_guard(test):
    path = Path(__file__).with_name('monitor_ovmap_benchmark_capacity.py')
    test.assertTrue(path.is_file(), 'The independent live capacity guard is missing')
    spec = importlib.util.spec_from_file_location('capacity_guard_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def proc(pid):
    values = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
    return values[0], values[19]


class CapacityGuardTests(unittest.TestCase):
    def setUp(self):
        self.guard = load_guard(self)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.parent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        self.addCleanup(self.clean_parent)
        protocol = write(self.root/'protocol_lock.json', dict(scopes={
            'scannet200_val': dict(status='DONE', scenes=['first', 'second'])}))
        write(self.root/'binding.json', dict(output_root=str(self.root), specification={
            'runtime': dict(minimal_disk_reserve_GiB=20)}))
        frozen = self.root/'frozen.txt'
        frozen.write_text('protected input')
        write(self.root/'bootstrap/before_ovi_scenes38_to312_entry.json', dict(files={
            str(frozen): dict(sha256=digest_file(frozen), mtime_ns=frozen.stat().st_mtime_ns)}))
        self.config = self.guard.Config(self.root, self.parent.pid, proc(self.parent.pid)[1],
                                        protocol['identity'], 1, 2, 0.01)
        self.disk_patch = patch.object(self.guard.shutil, 'disk_usage', return_value=SimpleNamespace(free=2**40))
        self.disk = self.disk_patch.start()
        self.addCleanup(self.disk_patch.stop)

    def clean_parent(self):
        if self.parent.poll() is None:
            self.parent.send_signal(signal.SIGCONT)
            self.parent.terminate()
            self.parent.wait(timeout=5)

    def assets(self, scene):
        folder = self.root/'inputs'/scene
        folder.mkdir(parents=True, exist_ok=True)
        sensor = folder/'data.sens'
        name = b'fixture'
        sensor.write_bytes(struct.pack('<IQ', 4, len(name))+name+struct.pack('<64f', *([1.]*64))+
                           struct.pack('<iiIIIIfQ', 2, 1, 8, 6, 4, 3, 1000., 2)+b'unread payload')
        mesh = folder/'mesh.ply'
        mesh.write_bytes(b'mesh')
        return write(self.root/'assets/datasets/scannet/receipts'/f'{scene}.json', dict(
            status='DONE', scene=scene, GT_files=False, GT_labels_opened=False,
            files={'.sens': dict(path=str(sensor), bytes=sensor.stat().st_size, sha256=digest_file(sensor)),
                   '_vh_clean_2.ply': dict(path=str(mesh), bytes=4, sha256=digest_file(mesh))}))

    def score(self, scene, **values):
        return write(self.root/'benchmark/evaluation/scannet200_val'/scene/'receipt.json', dict(
            status='DONE', scene=scene, benchmark='scannet200_val', logical_rows=8, **values))

    def test_header_only_estimate_matches_original_capacity_implementation(self):
        assets = self.assets('first')
        observation = self.guard.inspect_scene(self.config, 'first')
        self.assertEqual(observation['pixel_plan_bytes'], 12688)
        self.assertEqual(observation['required_free_bytes'], 21474861856)
        self.assertLess(observation['sensor_header_bytes_read'], Path(assets['files']['.sens']['path']).stat().st_size)
        from static_ovmap.benchmark_completion_v2.capacity import estimate_scene
        binding = verified(self.root/'binding.json')
        with patch('static_ovmap.benchmark_completion_v2.acquisition.scannet_scene', return_value=assets):
            original = estimate_scene(binding, 'scannet200_val', 'first')
        self.assertEqual(observation['estimated_bytes'], original['estimated_bytes'])
        self.assertEqual(observation['pixel_plan_bytes'], original['pixel_plan_bytes'])

    def test_missing_asset_receipt_waits_without_creating_it(self):
        self.assertIsNone(self.guard.inspect_scene(self.config, 'first'))
        self.assertFalse((self.root/'assets').exists())

    def test_broken_asset_receipt_is_not_treated_as_pending(self):
        path = self.root/'assets/datasets/scannet/receipts/first.json'
        path.parent.mkdir(parents=True)
        path.symlink_to(self.root/'missing.json')
        with self.assertRaises(FileNotFoundError):
            self.guard.inspect_scene(self.config, 'first')

    def test_corrupt_or_missing_referenced_asset_is_rejected(self):
        row = self.assets('first')
        Path(row['files']['.sens']['path']).unlink()
        with self.assertRaises(FileNotFoundError):
            self.guard.inspect_scene(self.config, 'first')

    def test_preflight_with_low_space_never_stops_or_writes(self):
        self.assets('first')
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.disk.return_value = SimpleNamespace(free=0)
        report = self.guard.preflight(self.config)
        self.assertEqual(report['status'], 'BLOCKED_RESOURCE')
        self.assertFalse(report['signal_sent'])
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_wrong_process_start_never_receives_stop(self):
        wrong = dataclasses.replace(self.config, controller_start='0')
        with self.assertRaises(ValueError):
            self.guard.stop_controller(wrong)
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_pidfd_stop_leaves_current_child_running(self):
        code = ('import subprocess,sys,time\n'
                'child=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"])\n'
                'print(child.pid,flush=True)\ntime.sleep(120)\n')
        parent = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                  text=True, start_new_session=True)
        child = int(parent.stdout.readline())
        try:
            config = dataclasses.replace(self.config, controller_pid=parent.pid, controller_start=proc(parent.pid)[1])
            action = self.guard.stop_controller(config)
            self.assertTrue(action['signal_sent'])
            self.assertIn(proc(parent.pid)[0], ('T', 't'))
            self.assertNotIn(proc(child)[0], ('T', 't', 'Z'))
        finally:
            os.killpg(parent.pid, signal.SIGCONT)
            os.killpg(parent.pid, signal.SIGTERM)
            parent.wait(timeout=5)
            parent.stdout.close()

    def test_low_space_run_pauses_pinned_controller_and_records_block(self):
        self.assets('first')
        self.disk.return_value = SimpleNamespace(free=0)
        self.assertEqual(self.guard.run(self.config), 20)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['status'], 'BLOCKED_RESOURCE')
        self.assertTrue(result['stop_action']['signal_sent'])
        self.assertIn(proc(self.parent.pid)[0], ('T', 't'))
        self.assertEqual((self.root/'frozen.txt').read_text(), 'protected input')

    def test_finite_run_records_each_scene_and_finishes_without_stop(self):
        for scene in ['first', 'second']:
            self.assets(scene)
        pending = iter(['first', 'second'])
        def advance(_):
            self.score(next(pending))
        with patch.object(self.guard.time, 'sleep', side_effect=advance):
            self.assertEqual(self.guard.run(self.config), 0)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['status'], 'DONE')
        self.assertEqual([r['scene'] for r in result['completed']], ['first', 'second'])
        self.assertEqual(len(list((self.guard.state_dir(self.config)/'observations').glob('*.json'))), 2)
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_existing_run_is_not_restarted(self):
        folder = self.guard.state_dir(self.config)
        folder.mkdir(parents=True)
        write(folder/'progress.json', dict(status='RUNNING'))
        before = (folder/'progress.json').read_bytes()
        with self.assertRaises(ValueError):
            self.guard.run(self.config)
        self.assertEqual((folder/'progress.json').read_bytes(), before)
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_exclusive_lock_refuses_a_second_owner(self):
        folder = self.guard.state_dir(self.config)
        folder.mkdir(parents=True)
        with (folder/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.guard.run(self.config)

    def test_missing_controller_stops_guard_without_signalling(self):
        self.parent.terminate()
        self.parent.wait(timeout=5)
        self.assertEqual(self.guard.run(self.config), 20)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['status'], 'BLOCKED_DEPENDENCY')
        self.assertFalse(result['signal_sent'])

    def test_last_completed_score_can_finish_after_controller_exit(self):
        for scene in ['first', 'second']:
            self.score(scene)
        self.parent.terminate()
        self.parent.wait(timeout=5)
        self.assertEqual(self.guard.run(self.config), 0)
        self.assertEqual(verified(self.guard.state_dir(self.config)/'progress.json')['status'], 'DONE')

    def test_changed_protected_source_is_rejected(self):
        (self.root/'frozen.txt').write_text('changed')
        with self.assertRaises(ValueError):
            self.guard.preflight(self.config)
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_preflight_keeps_known_buffer_while_assets_are_pending(self):
        write(self.root/'resources/capacity/scannet200_val/previous.json', dict(
            status='DONE', estimate=dict(estimated_bytes=32*2**30)))
        self.disk.return_value = SimpleNamespace(free=80*2**30)
        result = self.guard.preflight(self.config)
        self.assertEqual(result['status'], 'BLOCKED_RESOURCE')
        self.assertEqual(result['required_free_bytes'], 84*2**30)
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_run_keeps_known_buffer_while_assets_are_pending(self):
        write(self.root/'resources/capacity/scannet200_val/previous.json', dict(
            status='DONE', estimate=dict(estimated_bytes=32*2**30)))
        self.disk.return_value = SimpleNamespace(free=80*2**30)
        original_sleep = self.guard.time.sleep
        def interrupted(seconds):
            if seconds == self.config.poll_seconds:
                raise RuntimeError('The guard ignored the known insufficient buffer')
            original_sleep(seconds)
        with contextlib.redirect_stderr(io.StringIO()), patch.object(self.guard.time, 'sleep', side_effect=interrupted):
            self.assertEqual(self.guard.run(self.config), 20)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['required_free_bytes'], 84*2**30)
        self.assertIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_pause_action_survives_a_subsequent_receipt_write_failure(self):
        self.assets('first')
        self.disk.return_value = SimpleNamespace(free=0)
        original_write = self.guard.write
        failed = False
        def fail_once(path, value):
            nonlocal failed
            if value['status'] == 'BLOCKED_RESOURCE' and not failed:
                failed = True
                raise OSError('isolated receipt-write failure after pause')
            return original_write(path, value)
        with contextlib.redirect_stderr(io.StringIO()), patch.object(self.guard, 'write', side_effect=fail_once):
            self.assertEqual(self.guard.run(self.config), 1)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['status'], 'FAILED_RUNTIME')
        self.assertTrue(result['signal_sent'])
        self.assertTrue(result['stop_action']['signal_sent'])
        self.assertIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_sent_pause_is_retained_when_stopped_state_is_not_yet_observable(self):
        self.assets('first')
        self.disk.return_value = SimpleNamespace(free=0)
        delayed_state = dict(state='D', start=self.config.controller_start)
        with patch.object(self.guard, 'process_info', return_value=delayed_state), \
                patch.object(self.guard.time, 'sleep', return_value=None), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.guard.run(self.config), 20)
        result = verified(self.guard.state_dir(self.config)/'progress.json')
        self.assertEqual(result['status'], 'BLOCKED_RESOURCE')
        self.assertTrue(result['signal_sent'])
        self.assertFalse(result['stop_action']['stop_observed'])
        self.assertIn(proc(self.parent.pid)[0], ('T', 't'))

    def test_broken_existing_progress_is_never_overwritten(self):
        folder = self.guard.state_dir(self.config)
        folder.mkdir(parents=True)
        progress = folder/'progress.json'
        progress.symlink_to(self.root/'missing-progress.json')
        with contextlib.redirect_stderr(io.StringIO()), \
                patch.object(self.guard.time, 'sleep', side_effect=RuntimeError('unexpected resumed loop')), \
                self.assertRaises(ValueError):
            self.guard.run(self.config)
        self.assertTrue(progress.is_symlink())
        self.assertNotIn(proc(self.parent.pid)[0], ('T', 't'))


if __name__ == '__main__':
    unittest.main()
