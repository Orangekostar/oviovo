"""Importer contract tests. All generated fixtures stay in TemporaryDirectory.

The successful path deliberately simulates sealed upstream records; it is a
software test, never evidence of a real 312-scene scientific result.
"""
import hashlib
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
from static_ovmap.benchmark_completion_v2.common import (
    StageFailure, canonical_digest, digest_file, verified, write)
from static_ovmap.benchmark_completion_v2.tables import external_cell


def candidate():
    return importlib.import_module('static_ovmap.benchmark_completion_v2.open3dis_score')


def integration_candidate(name):
    return importlib.import_module('static_ovmap.benchmark_completion_v2.' + name)


def descriptor(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest_file(path))


def change(path, **updates):
    return write(path, dict(verified(path), **updates))


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.module = candidate()
        self.temp = tempfile.TemporaryDirectory(prefix='TEST_ONLY_open3dis_import_')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'formal'
        self.scores = self.base / 'scores'
        self.root.mkdir()
        self.scores.mkdir()
        self.binding = dict(output_root=str(self.root), asset_config=dict(external=dict(Open3DIS=dict(
            common_scoring_root=str(self.scores)))))

    def fixture(self):
        # Independent tiny fixtures exercise the full cardinality contract
        # without requiring local datasets, actual class labels or GT values.
        scope = dict(scenes=[f'scene{i:04d}_00' for i in range(312)], expected_scenes=312,
            instance_classes=198, instance_ids=list(range(1,199)),
            instance_names=[f'TEST_CLASS_{i}' for i in range(198)],
            target_point_order='TEST_ONLY_ORIGINAL_ORDER')
        self.scenes = scope['scenes']
        source = self.base / 'source'
        source.mkdir()
        runner = source / 'open3dis_scoring.py'
        evaluator = source / 'eval_semantic_instance.py'
        runner.write_text('# TEST FIXTURE ONLY\n')
        evaluator.write_text('# TEST EVALUATOR FIXTURE ONLY\n')
        protocol = write(source / 'protocol_lock.json', dict(scopes=dict(scannet200_val=scope),
            official_sources=dict(evaluator_scan=descriptor(evaluator)), TEST_FIXTURE_ONLY=True))
        write(self.scores / 'protocol_lock.json', protocol)
        write(self.root / 'protocol_lock.json', protocol)
        metrics = dict(apall=.42, ap50=.5, ap25=.6)
        per_class = {name: dict(ap=.42, **{'ap50%':.5, 'ap25%':.6}) for name in scope['instance_names']}
        prediction = dict(path='/TEST_ONLY/no_prediction_values_read.pth', bytes=1, sha256='a'*64)
        self.freeze = write(self.scores / 'predictions.json', dict(status='DONE',
            benchmark='scannet200_val', scenes=self.scenes, protocol=protocol['identity'],
            predictions={scene: prediction for scene in self.scenes},
            all_author_predictions_frozen_before_this_job_GT_access=True))
        g3 = write(source / 'receipt.json', dict(status='DONE', synthetic=False,
            all_class_IoU_arrays_equal=True, original_protocol=protocol['identity'],
            scene=self.scenes[0], benchmark='scannet200_val', full_GT_denominator=True,
            archive_confidence_field_used=False, points=1, masks=1,
            common_metrics=metrics, author_metrics=metrics, tolerance=1e-12,
            source_files=[dict(path=prediction['path'], sha256=prediction['sha256'])]))
        sources = [descriptor(p) for p in [runner,source/'protocol_lock.json',source/'receipt.json',evaluator]]
        self.config = write(self.scores / 'configuration.json', dict(status='REVIEWED_FOR_ONE_LAUNCH',
            protocol=protocol['identity'], prediction_freeze=self.freeze['identity'],
            G3=g3['identity'], expected_scenes=self.scenes, sources=sources,
            confidence='AUTHOR_EVAL_CONSTANT_1', variant='ISBNET_GSAM_REFINED_SECOND_3D_PASS',
            author_pin='bc2a3b3ee7161f5754ad7a7fe77202e9a66adc8e', one_CPU_scorer=True,
            new_GPU_calls=0, new_training_updates=0, production_sources_modified=False))
        archives=[]
        for row in sources:
            path=self.scores/'archived_sources'/Path(row['path']).name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(Path(row['path']).read_bytes())
            archives.append(dict(original=row, archived=descriptor(path)))
        write(self.scores/'consumed_source_snapshot.json',dict(status='DONE',
            configuration=self.config['identity'],files=archives))
        sources_by_scene={}
        for scene in self.scenes:
            model=write(self.root/'benchmark/predictions/scannet200_val'/scene/'receipt.json', dict(
                all_available_predictions_frozen_before_new_GT_scoring=True,status='DONE'))
            gt=write(self.root/'benchmark/GT/scannet200_val'/scene/'receipt.json',dict(status='DONE',
                scene=scene,benchmark='scannet200_val', full_GT_denominator=True,GT_filtered_by_prediction=False,
                prediction_freeze=model['identity'],target_points=1,
                GT=dict(path='/TEST_ONLY/no_GT_values_read.txt',bytes=1,sha256='b'*64)))
            match=write(self.scores/'scenes'/scene/'matches.json.gz',dict(status='DONE',matches={scene:dict(gt={},pred={})}))
            self_score=write(self.scores/'scenes'/scene/'receipt.json',dict(status='DONE',scene=scene,
                method='Open3DIS',benchmark='scannet200_val',configuration=self.config['identity'],
                input_identity=canonical_digest(dict(configuration=self.config['identity'],scene=scene,
                    prediction=prediction,GT=gt['identity'],confidence=self.config['confidence'])),
                prediction=prediction,prediction_freeze=self.freeze['identity'],GT_receipt=gt['identity'],GT=gt['GT'],
                points=1,masks=1,original_GT_instances=0,metrics=metrics,per_class=per_class,
                metric_unit='FRACTION',confidence=self.config['confidence'],full_GT_denominator=True,
                matches_identity=match['identity'],matches_file=descriptor(self.scores/'scenes'/scene/'matches.json.gz'),
                synthetic=False,archive_confidence_field_used=False))
            sources_by_scene[scene]=dict(score=self_score['identity'],matches=match['identity'])
        self.pool=write(self.scores/'pool.json',dict(status='DONE',method='Open3DIS',benchmark='scannet200_val',
            configuration=self.config['identity'],protocol=protocol['identity'],source_scenes=sources_by_scene,
            variant=self.config['variant'],official_pin=self.config['author_pin'],confidence=self.config['confidence'],
            full_scene_coverage=True,completed_scenes=312,expected_scenes=312,ordered_scenes=self.scenes,
            locally_scored_author_predictions=True,locally_inferred=False,G3_real_first_scene=g3['identity'],
            full_GT_denominator=True,metric_unit='FRACTION',synthetic=False,metrics=metrics,per_class=per_class,
            new_GPU_calls=0,new_training_updates=0,
            aggregation='OFFICIAL_EVALUATE_MATCHES_ALL_ORDERED_SCENES;NO_SCENE_AP_AVERAGING'))
        role=self.scores/'roles/open3dis_queue'
        launch=write(role/'launch.json',dict(status='RUNNING',child=dict(pid=1,start='1',state='R'),
            command=['python','-u',str(runner),'queue'],source_sha256=sources[0]['sha256'],
            supervisor_sha256='c'*64,env=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')))
        (role/'runtime.log').write_text('TEST FIXTURE ONLY\n')
        actual=write(role/'actual_exit.json',dict(status='EXITED',actual_child_exit=0,
            local_first_actual_exit=True,child=launch['child'],command=launch['command'],launch=launch['identity'],
            source_sha256=launch['source_sha256'],supervisor_sha256=launch['supervisor_sha256'],
            log=descriptor(role/'runtime.log')))
        write(role/'synchronized.json',dict(status='DONE',actual_exit=actual['identity'],source_unchanged=True,
            source_after_sha256=launch['source_sha256'],errors=[]))

    def run_pool(self):
        return self.module.pool(self.binding)

    def assert_unpublished(self):
        self.assertFalse((self.root/'external').exists())

    def test_unconfigured_is_noop(self):
        self.assertIsNone(self.module.pool(dict(output_root=str(self.root))))
        self.assert_unpublished()

    def test_declared_pending_job_is_blocked(self):
        with self.assertRaises(StageFailure): self.run_pool()
        self.assert_unpublished()

    def test_complete_pool_import_table_selection_and_repeat_are_stable(self):
        self.fixture()
        result=self.run_pool()
        self.assertTrue(result['full_scene_coverage'])
        self.assertFalse(result['locally_inferred'])
        self.assertTrue(result['locally_scored_author_predictions'])
        self.assertEqual(result['metrics'],self.pool['metrics'])
        self.assertEqual(result['completed_scenes'],312)
        archived=verified(self.root/'external/Open3DIS/pools/scannet200_val_sources.json')
        for item in archived['files']:
            raw=item['utf8'].encode('utf-8')
            self.assertEqual(len(raw),item['original']['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(),item['original']['sha256'])
        results={'external:Open3DIS:scannet200_val':result,
                 'author:Open3DIS:scannet200_val':dict(metrics=dict(apall=.9))}
        self.assertEqual(external_cell(results,'Open3DIS','scannet200_val','apall')['value'],.42)
        self.assertNotIn('annotation',external_cell(results,'Open3DIS','scannet200_val','apall'))
        files={str(p):(digest_file(p),p.stat().st_mtime_ns) for p in (self.root/'external').rglob('*') if p.is_file()}
        self.assertEqual(self.run_pool(),result)
        self.assertEqual(files,{str(p):(digest_file(p),p.stat().st_mtime_ns) for p in (self.root/'external').rglob('*') if p.is_file()})

    def test_complete_flag_cannot_hide_missing_scene(self):
        self.fixture()
        source=dict(self.pool['source_scenes']);source.pop(self.scenes[-1])
        change(self.scores/'pool.json',source_scenes=source)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_nonzero_actual_exit_is_rejected(self):
        self.fixture()
        change(self.scores/'roles/open3dis_queue/actual_exit.json',actual_child_exit=1)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_exit_from_other_launch_is_rejected(self):
        self.fixture()
        change(self.scores/'roles/open3dis_queue/actual_exit.json',launch='d'*64)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_changed_point_order_is_rejected(self):
        self.fixture()
        protocol=verified(self.root/'protocol_lock.json')
        protocol['scopes']['scannet200_val']['target_point_order']='another point order'
        write(self.root/'protocol_lock.json',protocol)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_changed_class_order_is_rejected(self):
        self.fixture()
        protocol=verified(self.root/'protocol_lock.json')
        protocol['scopes']['scannet200_val']['instance_ids'].reverse()
        write(self.root/'protocol_lock.json',protocol)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_changed_evaluator_is_rejected(self):
        self.fixture()
        protocol=verified(self.root/'protocol_lock.json')
        protocol['official_sources']['evaluator_scan']['sha256']='e'*64
        write(self.root/'protocol_lock.json',protocol)
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_unrelated_global_change_is_explicitly_carried(self):
        self.fixture()
        current=change(self.root/'protocol_lock.json',unrelated_replica_asset_metadata='updated')
        result=self.run_pool()
        self.assertEqual(result['protocol'],current['identity'])
        self.assertTrue(result['protocol_carry']['global_identity_changed'])
        self.assertTrue(result['protocol_carry']['exact_scannet_scope_equal'])

    def test_tampered_match_file_is_rejected_before_publication(self):
        self.fixture()
        with (self.scores/'scenes'/self.scenes[-1]/'matches.json.gz').open('ab') as stream: stream.write(b'changed')
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_historical_source_archive_survives_source_upgrade(self):
        self.fixture()
        (self.base/'source/open3dis_scoring.py').write_text('# upgraded current source\n')
        self.assertTrue(self.run_pool()['full_scene_coverage'])

    def test_corrupt_source_snapshot_is_rejected(self):
        self.fixture()
        (self.scores/'archived_sources/open3dis_scoring.py').write_text('# changed archive\n')
        with self.assertRaises(ValueError): self.run_pool()
        self.assert_unpublished()

    def test_baseline_recipe_and_table_use_same_formal_pool(self):
        self.fixture()
        baseline = integration_candidate('baselines')
        table = integration_candidate('tables')
        with mock.patch.object(baseline,'PINS',{'Open3DIS':('TEST ONLY',self.config['author_pin'])}), \
             mock.patch('static_ovmap.benchmark_completion_v2.author_results.build',return_value={'results':{}}), \
             mock.patch('static_ovmap.benchmark_completion_v2.external_pooling.ovi',return_value={'benchmarks':{}}):
            method=baseline.run(self.binding)['methods']['Open3DIS']
            recipe=method['scopes']['scannet200_val']
            results={}
            table.add_open3dis_result(self.binding,results)
        row=results['external:Open3DIS:scannet200_val']
        self.assertEqual(recipe['local_pool']['identity'],row['identity'])
        self.assertEqual(recipe['status'],'DONE')
        self.assertEqual(recipe['coverage'],'LOCAL_FULL')
        self.assertEqual(recipe['missing'],[])
        self.assertFalse(recipe['locally_inferred'])
        self.assertTrue(recipe['locally_scored_author_predictions'])
        self.assertTrue(method['full_method_not_run'])
        self.assertEqual(len(recipe['observed_local_scenes']),312)
        self.assertEqual(row['source']['identity'],row['identity'])
        self.assertEqual(row['source']['path'],'external/Open3DIS/pools/scannet200_val.json')
        self.assertEqual(table.external_cell(results,'Open3DIS','scannet200_val','apall')['value'],.42)


    def test_complete_import_enters_redacted_export_and_repeat_preserves_files(self):
        from static_ovmap.benchmark_completion_v2.publication_export import candidates, decode_record, export
        self.fixture()
        pool = self.run_pool()
        public = self.base / 'public'
        public.mkdir()
        write(self.root/'result_store.json', dict(status='DONE', results={
            'external:Open3DIS:scannet200_val':pool}, TEST_FIXTURE_ONLY=True))
        raw = self.root/'external/Open3DIS/GT/scannet200_val/raw.txt'
        raw.parent.mkdir(parents=True)
        raw.write_text('TEST_ONLY_NO_GT_VALUES')
        selected = {str(p.relative_to(self.root)) for p in candidates(self.root)}
        expected = {'external/Open3DIS/pools/scannet200_val'+suffix+'.json'
                    for suffix in ('','_import','_sources')}
        for scene in self.scenes:
            expected.update({'external/Open3DIS/evaluation/scannet200_val/'+scene+'/'+name
                             for name in ('receipt.json','Open3DIS_matches.json.gz')})
        self.assertEqual(len(expected),627)
        self.assertTrue(expected <= selected)
        self.assertNotIn(str(raw.relative_to(self.root)),selected)
        originals = {p:(digest_file(p),p.stat().st_mtime_ns)
                     for p in (self.root/'external/Open3DIS').rglob('*') if p.is_file()}
        def do_export():
            with mock.patch('static_ovmap.benchmark_completion_v2.readout_replay.prepare',return_value={}), \
                 mock.patch('static_ovmap.benchmark_completion_v2.score_artifacts.prepare',return_value={}), \
                 mock.patch('static_ovmap.benchmark_completion_v2.legacy_score_artifacts.prepare',return_value={}):
                return export(self.binding,repo=public,git_limit=64*1024*1024)
        index = do_export()
        rows = {r['source']:r for r in index['files']}
        self.assertTrue(expected <= set(rows))
        for name in expected:
            row = rows[name]
            self.assertEqual({p['distribution'] for p in row['parts']},{'GIT'})
            payload = b''.join((public/p['path']).read_bytes() for p in row['parts'])
            self.assertEqual(hashlib.sha256(payload).hexdigest(),row['payload_sha256'])
            decoded = decode_record(payload)
            self.assertEqual(decoded['original_identity'],verified(self.root/name)['identity'])
            self.assertEqual(decoded['original_sha256'],digest_file(self.root/name))
            self.assertNotIn(str(self.root),str(decoded['record']))
        scores = {p:(digest_file(p),p.stat().st_mtime_ns) for p in public.rglob('*') if p.is_file()}
        self.assertEqual(do_export()['identity'],index['identity'])
        self.assertEqual(scores,{p:(digest_file(p),p.stat().st_mtime_ns) for p in public.rglob('*') if p.is_file()})
        self.assertEqual(originals,{p:(digest_file(p),p.stat().st_mtime_ns)
                                   for p in (self.root/'external/Open3DIS').rglob('*') if p.is_file()})


if __name__=='__main__':
    unittest.main()
