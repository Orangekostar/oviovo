#!/usr/bin/env python3
"""Actual CPU tests of reference arithmetic and the authored experiment contract.
These tests do not claim parent FC/model/evaluator or benchmark execution.
"""
from pathlib import Path
import json, unittest, ast, re
import numpy as np
import reference_ops as r
ROOT=Path(__file__).resolve().parent
SPEC=json.loads((ROOT/'02_冻结实验配置.json').read_text(encoding='utf-8'))

class Algebra(unittest.TestCase):
    def setUp(self):
        self.b=np.array([[1.,0.,0.]],np.float32)
        self.z=np.array([[0.,1.,0.]],np.float32)
    def test_cap_angle_and_norm(self):
        q,f=r.cap_vectors(self.b,self.z,10)
        self.assertAlmostEqual(float(np.linalg.norm(q)),1.,places=6)
        self.assertAlmostEqual(float(np.degrees(np.arccos(q[0,0]))),10.,places=4)
        self.assertFalse(f[0])
    def test_cap_no_change_within_bound(self):
        z=np.array([[np.cos(.05),np.sin(.05),0]],np.float32)
        q,_=r.cap_vectors(self.b,z,10);np.testing.assert_array_equal(q,z)
    def test_cap_180_identity(self):
        q,f=r.cap_vectors(self.b,self.z,180);np.testing.assert_array_equal(q,self.z)
    def test_cap_antipodal(self):
        q,f=r.cap_vectors(self.b,-self.b,10);np.testing.assert_array_equal(q,self.b);self.assertTrue(f[0])
    def test_nmix_endpoints(self):
        for a,expected in ((0,self.b),(1,self.z)):
            q,_=r.normalized_mix(self.b,self.z,a);np.testing.assert_array_equal(q,expected)
    def test_nmix_midpoint(self):
        q,_=r.normalized_mix(self.b,self.z,.5)
        np.testing.assert_allclose(q,[[2**-.5,2**-.5,0]],atol=1e-7)
    def test_nmix_zero_fallback(self):
        q,f=r.normalized_mix(self.b,-self.b,.5);np.testing.assert_array_equal(q,self.b);self.assertTrue(f[0])
    def test_offset_preserves_intragroup_order(self):
        s=np.array([[.2,.5,.3,.1]],np.float32);m=np.array([1,1,0,0],bool)
        a=r.offset_scores(s,m,.025)
        np.testing.assert_array_equal(np.argsort(s[:,m]),np.argsort(a[:,m]));np.testing.assert_array_equal(a[:,~m],s[:,~m])
    def test_normalization_offset_is_real_control(self):
        # Positive score rescaling is not harmless when a fixed group offset is applied.
        s=np.array([[.7,.6]],np.float32);m=np.array([1,0],bool)
        self.assertNotEqual(int(r.offset_scores(s,m,.15).argmax()),int(r.offset_scores(s*2,m,.15).argmax()))
    def test_mf_off_has_no_hidden_nq(self):
        scores={'N':np.array([0.,1.]),'Q':np.array([0.,1.]),'F':None}
        self.assertEqual(r.global_label(scores,np.array([7,9]),{'N':1,'Q':1,'F':1},False),0)
        self.assertEqual(r.global_label(scores,np.array([7,9]),{'N':1,'Q':1,'F':1},True),9)
    def test_grouped_complete_and_missing(self):
        s={'N':np.array([1.,0.]),'Q':np.array([2.,0.]),'F':np.array([0.,1.])}
        t=dict(N=1.,Q=1.,F=1.)
        want=.25*r.stable_softmax(s['N'])+.25*r.stable_softmax(s['Q'])+.5*r.stable_softmax(s['F'])
        np.testing.assert_allclose(r.grouped(s,t),want,atol=1e-14)
        np.testing.assert_allclose(r.grouped(dict(N=None,Q=None,F=s['F']),t),r.stable_softmax(s['F']))

class MetricsSelection(unittest.TestCase):
    def test_abstentions_remain(self):
        m=r.recognition(np.array([1,1,2]),np.array([1,0,2]),[1],[2])
        self.assertEqual(m['originals'],3);self.assertEqual(m['abstentions'],1);self.assertEqual(m['B_macro'],.5)
    def test_macro_micro_differ(self):
        m=r.recognition(np.array([1,1,1,2,3]),np.array([1,1,1,0,3]),[1,2],[3])
        self.assertEqual(m['B_macro'],.5);self.assertEqual(m['B_micro'],.75)
    def test_empty_novel_null(self):
        m=r.recognition(np.array([1]),np.array([1]),[1],[2]);self.assertIsNone(m['H_macro'])
    def test_actual_zero_not_missing(self):
        m=r.recognition(np.array([1,2]),np.array([0,0]),[1],[2]);self.assertEqual(m['H_macro'],0)
    def test_mean_h_not_h_mean(self):
        mean=(r.harmonic(1,.2)+r.harmonic(.2,1))/2
        self.assertNotAlmostEqual(mean,r.harmonic(.6,.6))
    def test_argmax_axis_order_tie(self):
        self.assertEqual(r.predict(np.array([[.3,.3]]),np.array([90,2]))[0],90)
    def test_defined_false_not_scored_as_class(self):
        self.assertEqual(r.predict(np.array([[.9,.1]]),np.array([1,2]),np.array([False]))[0],0)
    def test_oof_reuses_seed17_choice(self):
        y=np.array([1,2,1,2]);f=np.array(['a','a','b','b'])
        rr=[dict(candidate_id='A',operations=0),dict(candidate_id='B',operations=1)]
        p17=np.array([y,[2,1,2,1]]);p29=np.array([[2,1,2,1],y])
        out,folds=r.oof(p17,p29,y,f,rr,[1],[2])
        self.assertTrue(all(v['recipe']=='A' for v in folds));np.testing.assert_array_equal(out[1],p29[0])
    def test_tie_rules(self):
        m=dict(B_macro=.4,N_macro=.6,H_macro=.48)
        rows=[dict(recipe=dict(candidate_id='Z',operations=0),metrics=m),dict(recipe=dict(candidate_id='A',operations=1),metrics=m)]
        self.assertEqual(r.best(rows)['recipe']['candidate_id'],'Z')
    def test_missing_defined_rejected_for_selection(self):
        with self.assertRaises(ValueError):r.best([dict(recipe={},metrics={'H_macro':None})])

class Contract(unittest.TestCase):
    def test_candidate_counts(self):
        rr=r.recipes(SPEC['calibration']);self.assertEqual(len(rr),165)
        self.assertEqual(len({x['candidate_id'] for x in rr}),165)
        self.assertEqual(165*4*2+11,SPEC['calibration']['total_logical_rows'])
        self.assertEqual(sum(x['kind']=='CAP' for x in rr),55)
    def test_scope_counts(self):
        self.assertEqual(26*4*2,SPEC['legacy']['maximum_logical_scene_rows'])
        self.assertEqual((312+8)*4*2,SPEC['standard']['max_own_scene_rows_if_320_scenes'])
        self.assertEqual((312+8)*4,SPEC['baselines']['max_local_scene_method_jobs_if_320_scenes'])
        self.assertEqual(2*2*1000,SPEC['statistics']['family_draw_sets'])
    def test_unique_acyclic_stages(self):
        self.assertEqual(len(SPEC['phases']),len(set(SPEC['phases'])))
        self.assertEqual(set(r.check_dag(SPEC['dependencies'])),set(SPEC['phases']))
    def test_profile_and_no_old_gate(self):
        self.assertEqual(SPEC['model']['step'],1000);self.assertEqual(SPEC['model']['profile'],'matched1000')
        self.assertFalse(SPEC['cache']['reuse_old_gate']);self.assertFalse(SPEC['cache']['parent_all_allowed'])
        self.assertEqual(SPEC['model']['training_updates'],0)
    def test_scientific_table_contract(self):
        tabs=SPEC['reporting']['main_tables'];self.assertEqual(set(tabs),{'T1','T2','T3'})
        self.assertIn('GLOBAL',tabs['T2']);self.assertFalse(SPEC['reporting']['main_seed_columns'])
    def test_single_heading_numbers(self):
        text=(ROOT/'00_给Codex的最终执行指令.md').read_text(encoding='utf-8')
        nums=re.findall(r'^## (\d+)\.',text,flags=re.M)
        self.assertEqual(nums,[str(i) for i in range(1,18)])
    def test_reference_sources_unique(self):
        sources=json.loads((ROOT/'sources.json').read_text(encoding='utf-8'))
        self.assertEqual(len(sources),len({s['id'] for s in sources}))
        self.assertTrue(any(s['id']=='E25' and 'models.py' in s['path'] for s in sources))
    def test_local_files_referenced_exist(self):
        for name in ['00_给Codex的最终执行指令.md','01_源码依据与任务绑定.md','02_冻结实验配置.json',
                     '03_算子数据与评分合同.md','04_Benchmark与外部方法执行.md','reference_ops.py']:
            self.assertTrue((ROOT/name).is_file(),name)
    def test_reference_syntax(self):
        ast.parse((ROOT/'reference_ops.py').read_text());ast.parse(Path(__file__).read_text())
    def test_statuses_not_assumed_complete(self):
        required={'software_complete','standard_coverage_complete','external_comparisons_complete','publication_complete','claim_supported'}
        self.assertTrue(required <= set(SPEC['completion']['separate']))
        self.assertIsNone(SPEC['runtime']['total_gpu_hour_cap'])
        self.assertFalse(SPEC['completion']['tables_with_gaps_means_all_experiments_complete'])

if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    record={'scope':'LOCAL_CPU_REFERENCE_AND_DOCUMENT_CONTRACT_ONLY',
            'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors),
            'failures':[str(x[0]) for x in result.failures], 'errors':[str(x[0]) for x in result.errors],
            'server_or_benchmark_executed':False,'real_model_weights_loaded':False,
            'GPU_used':False,'official_evaluator_executed':False,'github_push_performed':False}
    (ROOT/'06_实际CPU检查结果.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raise SystemExit(0 if result.wasSuccessful() else 1)
