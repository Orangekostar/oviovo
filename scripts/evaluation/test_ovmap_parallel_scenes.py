"""Isolated multi-GPU scheduling checks; never touch existing research jobs."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'src'))
from static_ovmap.benchmark_completion_v2.common import write, verified, digest_file


class ParallelTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name('run_ovmap_parallel_scenes.py')
        self.assertTrue(path.exists(), 'Independent parallel scene entry is missing')
        spec = importlib.util.spec_from_file_location('parallel_under_test', path)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def job(self, scene, gpu):
        return write(self.root/'jobs/benchmark/scannet200_val'/scene/'cropformer/receipt.json',
                     dict(env={'CUDA_VISIBLE_DEVICES': str(gpu)}))

    def fixture(self):
        root = self.root
        array = root/'mask.npz'
        array.write_bytes(b'actual immutable fixture bytes')
        manifest = write(root/'manifest.json', dict(prediction_identity='same_mask_labels',arrays=dict(path=str(array),
            sha256=digest_file(array), bytes=array.stat().st_size)))
        methods = {v+'_seed'+str(s): dict(path=str(root/'manifest.json'), identity=manifest['identity'])
                   for v in ('U00','U10','U01','U11') for s in (17,29)}
        pred = write(root/'benchmark/predictions/scannet200_val/a/receipt.json',
            dict(status='DONE', scene='a', benchmark='scannet200_val', methods=methods, completed_rows=8))
        gt = write(root/'benchmark/GT/scannet200_val/a/receipt.json', dict(prediction_freeze=pred['identity'],
            GT=dict(path=str(array), sha256=digest_file(array), bytes=array.stat().st_size)))
        match = write(root/'matches.json.gz', dict(status='DONE', matches={}))
        rows = {k:dict(prediction=manifest['identity'],prediction_identity='same_mask_labels', GT=gt['identity'],
            matches=dict(path=str(root/'matches.json.gz'), identity=match['identity'])) for k in methods}
        write(root/'benchmark/evaluation/scannet200_val/a/receipt.json', dict(status='DONE',
            scene='a', benchmark='scannet200_val', logical_rows=8, rows=rows,
            prediction_lock=pred['identity'], GT=gt['identity']))
        return array

    def test_partial_scene_keeps_original_gpu(self):
        self.job('a',0)
        self.assertEqual(self.m.required_gpu(self.root,'a'),0)

    def test_conflicting_existing_devices_rejected(self):
        self.job('a',0)
        write(self.root/'jobs/benchmark/scannet200_val/a/F_MV1000/receipt.json',
              dict(env={'CUDA_VISIBLE_DEVICES':'2'}))
        with self.assertRaises(ValueError): self.m.required_gpu(self.root,'a')

    def test_unassigned_scene_can_use_second_device(self):
        self.assertIsNone(self.m.required_gpu(self.root,'new'))

    def test_pending_queue_respects_existing_assignment(self):
        self.job('a',0)
        self.assertEqual(self.m.choose_scene(self.root,['a','b'],2,{}),'b')
        self.assertEqual(self.m.choose_scene(self.root,['a','b'],0,{'a':2}),'b')

    def test_empty_queue_ends(self):
        self.assertIsNone(self.m.choose_scene(self.root,[],2,{}))

    def test_completed_receipts_skip_visual_work(self):
        self.fixture()
        with patch.object(self.m,'execute_science',side_effect=AssertionError('must reuse')):
            row=self.m.work_scene(self.root,'a',2,check_frozen=False)
        self.assertEqual(row['status'],'DONE')
        self.assertTrue(row['reused_complete_scene'])

    def test_changed_mask_bytes_rejected_before_reuse(self):
        array=self.fixture();array.write_bytes(b'changed')
        with self.assertRaises(ValueError): self.m.completed_scene(self.root,'a')

    def test_score_without_prediction_is_rejected(self):
        self.fixture()
        (self.root/'benchmark/predictions/scannet200_val/a/receipt.json').unlink()
        with self.assertRaises(ValueError): self.m.completed_scene(self.root,'a')

    def test_official_score_alias_reuses_equal_predictions_across_manifests(self):
        self.fixture()
        pred_path=self.root/'benchmark/predictions/scannet200_val/a/receipt.json'
        pred=verified(pred_path)
        original=verified(self.root/'manifest.json')
        original=write(self.root/'manifest.json',dict(original,prediction_identity='same_mask_labels'))
        other=write(self.root/'other.json',dict(original,seed=29))
        for method in pred['methods']:
            pred['methods'][method]=dict(path=str(self.root/'manifest.json'),identity=original['identity'])
        pred['methods']['U11_seed29']=dict(path=str(self.root/'other.json'),identity=other['identity'])
        pred=write(pred_path,pred)
        gt_path=self.root/'benchmark/GT/scannet200_val/a/receipt.json'
        gt=write(gt_path,dict(verified(gt_path),prediction_freeze=pred['identity']))
        score_path=self.root/'benchmark/evaluation/scannet200_val/a/receipt.json'
        score=verified(score_path)
        for row in score['rows'].values():
            row.update(prediction=original['identity'],prediction_identity='same_mask_labels',GT=gt['identity'])
        score['rows']['U11_seed29']['alias_of']='U00_seed17'
        write(score_path,dict(score,prediction_lock=pred['identity'],GT=gt['identity']))
        self.assertEqual(self.m.completed_scene(self.root,'a')['status'],'DONE')
        other=write(self.root/'other.json',dict(other,prediction_identity='different_mask_labels'))
        pred['methods']['U11_seed29']['identity']=other['identity']
        pred=write(pred_path,pred)
        gt=write(gt_path,dict(gt,prediction_freeze=pred['identity']))
        score=verified(score_path)
        for row in score['rows'].values():row['GT']=gt['identity']
        write(score_path,dict(score,prediction_lock=pred['identity'],GT=gt['identity']))
        with self.assertRaises(ValueError):self.m.completed_scene(self.root,'a')

    def test_missing_scene_is_pending(self):
        self.assertIsNone(self.m.completed_scene(self.root,'a'))

    def test_budget_accounts_for_both_inflight_scenes(self):
        self.assertTrue(self.m.enough_space(220,20,[30,70]))
        self.assertFalse(self.m.enough_space(219,20,[30,70]))

    def test_scene_lock_is_exclusive_in_another_process(self):
        with self.m.file_lock(self.root/'scene.lock'):
            code='import fcntl,sys; f=open(sys.argv[1],"a"); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'
            child=subprocess.run([sys.executable,'-c',code,str(self.root/'scene.lock')],capture_output=True)
        self.assertNotEqual(child.returncode,0)

    def test_same_cache_stage_serializes_real_writes(self):
        path=Path(__file__).with_name('run_ovmap_parallel_scenes.py')
        code=('import importlib.util,sys,time; from pathlib import Path; '
              's=importlib.util.spec_from_file_location("p",sys.argv[1]); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); '
              'r=Path(sys.argv[2]); '
              '\nwith m.file_lock(r/"fc.lock"):\n'
              ' p=r/"counter"; n=int(p.read_text()) if p.exists() else 0; time.sleep(.1); p.write_text(str(n+1))\n')
        children=[subprocess.Popen([sys.executable,'-c',code,str(path),str(self.root)],
            env=dict(os.environ,PYTHONPATH=str(REPO/'src'))) for _ in range(3)]
        try:
            self.assertEqual([p.wait(timeout=30) for p in children],[0,0,0])
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=5)
        self.assertEqual((self.root/'counter').read_text(),'3')

    def test_stage_adapter_preserves_actual_env_and_arguments(self):
        calls=[]
        def execute(*a,**kw): calls.append((a,kw));return 'receipt'
        wrapped=self.m.cache_safe_execute(execute,self.root)
        self.assertEqual(wrapped({'gpu':2},'benchmark/x/a/F_MV1000',['real'],env={'CUDA_VISIBLE_DEVICES':'2'}),'receipt')
        self.assertEqual(calls,[(({'gpu':2},'benchmark/x/a/F_MV1000',['real']),{'env':{'CUDA_VISIBLE_DEVICES':'2'}})])

    def test_frozen_source_change_rejected(self):
        p=self.root/'source';p.write_text('frozen')
        write(self.root/'bootstrap/before_ovi_scenes38_to312_entry.json',dict(files={str(p):
            dict(sha256=digest_file(p),mtime_ns=p.stat().st_mtime_ns)}))
        self.assertEqual(self.m.check_sources(self.root),1)
        p.write_text('modified')
        with self.assertRaises(ValueError):self.m.check_sources(self.root)

    def test_handover_stops_only_at_actual_gpu_flock(self):
        lock=self.root/'gpu.lock';ready=self.root/'ready'
        with self.m.file_lock(lock):
            code=('import fcntl,sys,time; from pathlib import Path; '
                  'f=open(sys.argv[1],"a"); Path(sys.argv[2]).write_text("ready"); '
                  'fcntl.flock(f,fcntl.LOCK_EX); time.sleep(60)')
            child=subprocess.Popen([sys.executable,'-c',code,str(lock),str(ready)])
            try:
                for _ in range(100):
                    if ready.exists() and self.m.waiting_on_lock(child.pid,lock):break
                    time.sleep(.02)
                start=self.m.process_info(child.pid)['start']
                action=self.m.pause_at_gate(child.pid,start,lock)
                self.assertEqual(action['state'],'T')
                self.assertEqual(self.m.process_info(child.pid)['state'],'T')
            finally:
                child.terminate();child.send_signal(signal.SIGCONT);child.wait(timeout=5)

    def test_wrong_pid_start_never_signalled(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            with self.assertRaises(ValueError):self.m.pause_at_gate(child.pid,'wrong',self.root/'no-lock')
            self.assertNotEqual(self.m.process_info(child.pid)['state'],'T')
        finally:child.terminate();child.wait(timeout=5)

    def test_real_child_exit_is_not_assumed_zero(self):
        child=subprocess.Popen([sys.executable,'-c','raise SystemExit(7)'])
        child.wait(timeout=5)
        row=self.m.child_result(child,self.root/'missing.json')
        self.assertEqual(row['actual_exit_code'],7)
        self.assertEqual(row['status'],'FAILED_RUNTIME')

    def test_zero_exit_without_worker_receipt_is_failure(self):
        child=subprocess.Popen([sys.executable,'-c','pass']);child.wait(timeout=5)
        self.assertEqual(self.m.child_result(child,self.root/'missing.json')['status'],'FAILED_RUNTIME')

    def test_finite_coordinator_launches_two_devices_and_records_real_exits(self):
        scenes=['cached'+str(i) for i in range(310)]+['new0','new2']
        write(self.root/'binding.json',dict(output_root=str(self.root),specification={
            'runtime':dict(minimal_disk_reserve_GiB=20)}))
        write(self.root/'protocol_lock.json',dict(scopes={'scannet200_val':dict(scenes=scenes)}))
        write(self.root/'model_lock.json',dict(status='DONE'))
        old=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        start=self.m.process_info(old.pid)['start']
        self.m.pinned_signal(old.pid,start,signal.SIGSTOP)
        real_popen=subprocess.Popen
        launches=[]
        def launch(argv,**kw):
            gpu=kw['env']['CUDA_VISIBLE_DEVICES'];scene=argv[argv.index('--worker-scene')+1]
            launches.append((scene,gpu))
            target=self.root/'bootstrap/parallel_scenes/workers'/scene/'receipt.json'
            code=('import sys,time; from pathlib import Path; '
                  'sys.path.insert(0,sys.argv[1]); '
                  'from static_ovmap.benchmark_completion_v2.common import write; '
                  'time.sleep(.1); write(Path(sys.argv[2]),dict(status="DONE",scene=sys.argv[3],physical_gpu=int(sys.argv[4])))')
            return real_popen([sys.executable,'-c',code,str(REPO/'src'),str(target),scene,gpu],**kw)
        real_sleep=time.sleep
        try:
            with patch.object(self.m,'check_sources',return_value=93), \
                 patch.object(self.m,'handover',return_value={'status':'DONE'}), \
                 patch.object(self.m,'completed_scene',side_effect=lambda r,s:dict(status='DONE',scene=s) if s.startswith('cached') else None), \
                 patch('static_ovmap.benchmark_completion_v2.capacity.estimate_scene',return_value={'estimated_bytes':10}), \
                 patch.object(self.m.shutil,'disk_usage',return_value=SimpleNamespace(free=2**40)), \
                 patch.object(self.m.subprocess,'Popen',side_effect=launch), \
                 patch.object(self.m.time,'sleep',side_effect=lambda t:real_sleep(min(t,.01))):
                result=self.m.coordinator(self.root,[0,2],old.pid,start)
            self.assertEqual(result,0)
            self.assertEqual(launches,[('new0','0'),('new2','2')])
            progress=verified(self.root/'bootstrap/parallel_scenes/progress.json')
            self.assertEqual(len(progress['completed']),312)
            self.assertEqual([r['actual_exit_code'] for r in progress['completed'][-2:]],[0,0])
            self.assertEqual(old.wait(timeout=5),-signal.SIGTERM)
        finally:
            if old.poll() is None:old.terminate();old.send_signal(signal.SIGCONT);old.wait(timeout=5)

    def test_live_orphan_worker_prevents_duplicate_coordinator(self):
        child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])
        try:
            write(self.root/'bootstrap/parallel_scenes/workers/a/process.json',dict(
                pid=child.pid,start=self.m.process_info(child.pid)['start']))
            with patch.object(self.m,'check_sources',return_value=93):
                with self.assertRaisesRegex(ValueError,'still live'):
                    self.m.coordinator(self.root,[0,2],1,'bad')
        finally:child.terminate();child.wait(timeout=5)

    def test_standalone_entry_exposes_both_inherited_import_namespaces(self):
        script=Path(__file__).with_name('run_ovmap_parallel_scenes.py')
        code=('import runpy,sys; runpy.run_path(sys.argv[1],run_name="import_only"); '
              'from src.static_ovmap.composition_study.io import read_json; '
              'from static_ovmap.benchmark_completion_v2.native_inputs import build')
        child=subprocess.run([sys.executable,'-B','-I','-c',code,str(script)],cwd=self.root,
            capture_output=True,text=True)
        self.assertEqual(child.returncode,0,child.stderr)


if __name__=='__main__':unittest.main()
