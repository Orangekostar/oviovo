"""Ten small synthetic checks; no claim about real CUDA/data integration."""
import copy
import unittest
import numpy as np
from scipy.special import softmax
from decision_reference import source_components, mean_view_scores, update_probability, decide, target_flags


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.s = {'N':[.6,.1,.2], 'Q':[.2,.4,.1], 'F':[.1,.2,.5]}
        self.t = {'N':.2,'Q':.25,'F':.3}
        self.a = np.array([.1,.7,.15])
        self.c = np.array([.25,.3,.2])

    def test_01_grouped_source_weights(self):
        p,q,f,w = source_components(self.s,self.t)
        expected=.25*softmax(np.array(self.s['N'])/.2)+.25*softmax(np.array(self.s['Q'])/.25)+.5*softmax(np.array(self.s['F'])/.3)
        np.testing.assert_allclose(p,expected,atol=1e-15)
        self.assertEqual(w,.5)
        self.assertAlmostEqual(p.sum(),1.)

    def test_02_missing_source_rules(self):
        p,q,f,w=source_components({'F':self.s['F']},self.t)
        self.assertIsNone(q)
        self.assertEqual(w,1.)
        for method in ('SU05_F_BLEND','SU07_GLOBAL_BLEND'):
            x=update_probability(method,{'F':self.s['F']},self.t,self.a,self.c)
            np.testing.assert_allclose(x,.5*p+.5*softmax(self.a/.3))
        with self.assertRaises(ValueError):
            update_probability('SU04_F_REPLACE',{'N':self.s['N']},self.t,self.a,self.c)

    def test_03_class_id_not_index(self):
        s={'F':[0.,0.,0.]}
        result=decide('SU04_F_REPLACE',[10,100,900],10,s,{'F':1.},common_domain=True,
                      a=[0.,1.,5.],c=[0.,0.,0.])
        self.assertEqual(result.class_id,900)
        with self.assertRaises(ValueError):
            decide('SU04_F_REPLACE',[10,10,900],10,s,{'F':1.},common_domain=True,a=self.a,c=self.c)

    def test_04_replacement_and_identity(self):
        p,q,f,w=source_components(self.s,self.t)
        r=update_probability('SU04_F_REPLACE',self.s,self.t,self.a,self.c)
        np.testing.assert_allclose(r,.5*q+.5*softmax(self.a/.3))
        for name in ('SU04_F_REPLACE','SU05_F_BLEND'):
            same=update_probability(name,self.s,self.t,self.s['F'],self.c)
            np.testing.assert_allclose(same,p,atol=1e-15)

    def test_05_group_blend_coefficients(self):
        r=update_probability('SU05_F_BLEND',self.s,self.t,self.a,self.c)
        expected=sum([.25*softmax(np.array(self.s[n])/self.t[n]) for n in ('N','Q','F')])+.25*softmax(self.a/.3)
        np.testing.assert_allclose(r,expected,atol=1e-15)

    def test_06_equal_new_mass_distinct_old_weights(self):
        p,q,f,w=source_components(self.s,self.t)
        r5=update_probability('SU05_F_BLEND',self.s,self.t,self.a,self.c)
        r7=update_probability('SU07_GLOBAL_BLEND',self.s,self.t,self.a,self.c)
        np.testing.assert_allclose(r7,.75*p+.25*softmax(self.a/.3),atol=1e-15)
        np.testing.assert_allclose(r5-r7,.125*(q-f),atol=1e-15)
        self.assertGreater(np.max(np.abs(r5-r7)),1e-5)

    def test_07_coarse_and_paired_identity(self):
        r5=update_probability('SU05_F_BLEND',self.s,self.t,self.a,self.a)
        r6=update_probability('SU06_F_COARSE',self.s,self.t,self.a,self.a)
        np.testing.assert_array_equal(r5,r6)
        p,*_=source_components(self.s,self.t)
        r8=update_probability('SU08_PAIRED_DELTA',self.s,self.t,self.a,self.a)
        np.testing.assert_array_equal(r8,p)
        shifted=update_probability('SU08_PAIRED_DELTA',self.s,self.t,[.2,.2,.2],[0.,0.,0.])
        np.testing.assert_array_equal(shifted,p)

    def test_08_mean_cosines_before_softmax(self):
        views=np.array([[1.,0.,0.],[0.,.2,0.]])
        avg=mean_view_scores(views)
        np.testing.assert_array_equal(avg,views.mean(0))
        self.assertGreater(np.max(np.abs(softmax(avg/.3)-np.mean(softmax(views/.3,axis=1),axis=0))),1e-3)
        with self.assertRaises(ValueError):
            mean_view_scores(np.ones((3,3)))

    def test_09_matched_keep_and_replay(self):
        for method in ('SU02_HARD_MATCHED','SU03_STABLE_MATCHED','SU05_F_BLEND'):
            result=decide(method,[1,2,3],1,self.s,self.t,common_domain=False,a=self.a,c=self.c,
                          parent_hard_class=2,parent_stable_class=3)
            self.assertEqual(result.class_id,1)
            result=decide(method,[1,2,3],1,self.s,self.t,common_domain=True,protected=True,
                          a=self.a,c=self.c,parent_hard_class=2,parent_stable_class=3)
            self.assertEqual(result.class_id,1)
        r=decide('SU02_HARD_MATCHED',[1,2,3],1,self.s,self.t,common_domain=True,
                 parent_hard_class=2)
        self.assertEqual(r.class_id,2)

    def test_10_unrounded_all_metric_gate(self):
        g={d:{m:.2 for m in ('apall','ap50','ap25','miou','macc')} for d in ('replica8','scannet_cf18')}
        d=copy.deepcopy(g);d['scannet_cf18']['apall']=.201
        x=copy.deepcopy(g);x['scannet_cf18']['apall']=.202
        self.assertTrue(target_flags(x,g,d)['target_met'])
        x['replica8']['ap25']-=1e-5
        self.assertFalse(target_flags(x,g,d)['target_met'])
        x['replica8']['ap25']=None
        with self.assertRaises(ValueError):target_flags(x,g,d)

if __name__=='__main__':
    unittest.main(verbosity=2)
