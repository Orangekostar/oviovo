import copy
import unittest
import numpy as np
from kernels import (Evidence, canonical_sites, quadrature, grouped, softmax,
                     support_weights, updated, choose_second, query_scores,
                     screen_gate, final_gate, METRICS)

class Contracts(unittest.TestCase):
    def test_01_actual_group_availability(self):
        s={'N':[3.,0.],'Q':[0.,2.],'F':[1.,0.]};t=dict(N=1.,Q=1.,F=1.)
        expected=.25*softmax(s['N'])+.25*softmax(s['Q'])+.5*softmax(s['F'])
        np.testing.assert_allclose(grouped(s,t),expected,atol=1e-15)
        np.testing.assert_allclose(grouped({'F':s['F']},t),softmax(s['F']))
        self.assertIsNone(grouped({},t))

    def test_02_vocab_counterexample_and_score_invariance(self):
        s={'N':np.array([3.,0.,10.]),'F':np.array([0.,2.,-10.])};t={'N':1.,'F':1.}
        short={k:v[:2] for k,v in s.items()}
        self.assertGreater(grouped(short,t)[0],grouped(short,t)[1])
        self.assertLess(grouped(s,t)[0],grouped(s,t)[1])
        np.testing.assert_allclose(grouped(short,t,probability=False),grouped(s,t,probability=False)[:2])

    def test_03_two_view_pose_mean_is_redundant(self):
        rows=np.array([[.8,.1,.1],[.2,.4,.4]])
        a=rows.mean(0);b=np.mean([rows.mean(0)],0);c=np.mean([rows[0],rows[1]],0)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(a,c)

    def test_04_surface_duplicate_invariance(self):
        x=np.array([[0.,0,0],[1,0,0],[0,1,0]])
        k,w,a=canonical_sites(x,[[0,1,2]],[7,7,7])
        k2,w2,a2=canonical_sites(np.tile(x,(2,1)),[[0,1,2],[5,4,3]],[7]*6)
        self.assertEqual(k,k2);np.testing.assert_allclose(w,w2)
        self.assertEqual(a2['duplicates'],1);self.assertAlmostEqual(w.sum(),.5)
        _,wa,aa=canonical_sites(np.tile(x,(2,1)),[[0,1,2],[3,4,5]],[7,7,7,8,8,8])
        self.assertEqual(aa['ambiguous'],1);self.assertEqual(len(wa),0)

    def test_05_quadrature_area_and_order(self):
        x=np.column_stack((np.arange(20),np.zeros(20),np.zeros(20)));a=np.arange(1,21,dtype=float)
        y,w=quadrature(x,a,5);yr,wr=quadrature(x[::-1],a[::-1],5)
        self.assertLessEqual(len(y),5);self.assertAlmostEqual(w.sum(),a.sum())
        np.testing.assert_array_equal(y,yr);np.testing.assert_array_equal(w,wr)

    def test_06_support_mass_and_nonredundancy(self):
        A=np.ones(3);O=np.array([[1,1,0],[0,1,0]],bool)
        w=support_weights(A,O,'SUPPORT');wa=support_weights(A,O,'AREA')
        self.assertAlmostEqual(w.sum(),2.);np.testing.assert_allclose(w,[1.5,.5])
        self.assertFalse(np.allclose(w/w.sum(),wa/wa.sum()))
        same=np.array([[1,1,0],[1,1,0]],bool)
        ws=support_weights(A,same,'SUPPORT');np.testing.assert_allclose(ws/ws.sum(),[.5,.5])

    def test_07_exact_duplicate_and_permutation(self):
        r1=Evidence('a',np.array([2.,0.]),np.array([1,1,0],bool))
        r2=Evidence('b',np.array([0.,2.]),np.array([0,1,1],bool))
        p=np.array([.6,.4]);A=np.ones(3)
        for kind in ('MEAN','AREA','SUPPORT'):
            expected=updated(p,[r1,r2],A,kind)
            np.testing.assert_array_equal(expected,updated(p,[r2,r1,r1],A,kind))
        bad=Evidence('a',np.array([0.,0.]),r1.footprint)
        with self.assertRaises(ValueError):updated(p,[r1,bad],A,'MEAN')

    def test_08_withdrawal_and_exact_prior(self):
        r=Evidence('a',np.array([2.,0.]),np.array([1,1],bool));p=np.array([.31,.69])
        np.testing.assert_array_equal(updated(p,[],np.ones(2),'SUPPORT'),p)
        expected=.75*p+.25*softmax(r.cosine)
        np.testing.assert_allclose(updated(p,[r],np.ones(2),'SUPPORT'),expected)

    def test_09_disagreement_has_effect_without_future_scores(self):
        O=np.array([[1,1,1],[1,1,0],[0,1,1]],bool);A=np.ones(3)
        ids=[1,2,3];pixels=np.array([100,120,110]);theta=np.array([0,30,30])
        self.assertEqual(choose_second('VERIFY',ids,O,A,pixels,theta),2)
        self.assertEqual(choose_second('DISAGREEMENT',ids,O,A,pixels,theta,interest=[0,0,1]),3)
        self.assertEqual(choose_second('DISAGREEMENT',ids,O,A,pixels,theta,interest=[0,0,0]),2)

    def test_10_query_controls_and_ties(self):
        O=np.ones((3,2),bool);A=np.ones(2);ids=[10,9,4];pixels=np.array([100,110,110]);theta=np.array([0,0,0])
        self.assertEqual(choose_second('AREA',ids,O,A,pixels,theta),4)
        self.assertEqual(choose_second('VERIFY',ids,O,A,pixels,theta),4)
        self.assertEqual(choose_second('COVERAGE',ids,O,A,pixels,theta,bins=[{1},{1,2},{1}]),9)

    def test_11_final_gate_all_five_and_strict_D2(self):
        g={k:{m:.20 for m in METRICS} for k in ('replica8','scannet_cf18')};d=copy.deepcopy(g);c=copy.deepcopy(g)
        self.assertFalse(final_gate(c,g,d));c['scannet_cf18']['apall']+=.001
        self.assertTrue(final_gate(c,g,d))
        for m in METRICS:
            bad=copy.deepcopy(c);bad['replica8'][m]-=.00001;self.assertFalse(final_gate(bad,g,d))
        bad=copy.deepcopy(c);bad['scannet_cf18']['macc']=None;self.assertFalse(final_gate(bad,g,d))

    def test_12_screen_signal_not_upgrade(self):
        g={k:{m:.20 for m in METRICS} for k in ('replica_probe2','cf_probe2')};b=copy.deepcopy(g);c=copy.deepcopy(g)
        c['replica_probe2']['apall']-=.0004;c['cf_probe2']['apall']+=.0016
        self.assertTrue(screen_gate(c,b,g,changed=True,corrections_net=1,gt50_net=0))
        self.assertFalse(screen_gate(c,b,g,changed=False,corrections_net=1,gt50_net=0))
        self.assertFalse(screen_gate(c,b,g,changed=True,corrections_net=0,gt50_net=0))
        c['replica_probe2']['apall']-=.002
        self.assertFalse(screen_gate(c,b,g,changed=True,corrections_net=2,gt50_net=0))

if __name__=='__main__':unittest.main(verbosity=2)
