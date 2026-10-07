"""Bounded mathematical checks; not tests of the user's real FC/AnyUp pipeline."""
import copy
import unittest
import numpy as np
from reference_kernels import occupancy_trigger, contrast_decision, geometry_factor, reweight_attention, target_flags, choose_method, METRICS

class ContractTests(unittest.TestCase):
    def test_trigger_area_and_hard_support(self):
        self.assertTrue(occupancy_trigger(np.full((3,3),.1), 0)[0])
        self.assertFalse(occupancy_trigger(np.ones((3,3)), 9)[0])
        with self.assertRaises(ValueError): occupancy_trigger(np.zeros(3), 0)

    def test_nontriggered_and_missing_controls(self):
        s=np.array([.31,.305,.1])
        a=contrast_decision(s,np.array([[.9,.1,.1]]),.1,triggered=False)
        self.assertTrue(a['accepted']); np.testing.assert_array_equal(a['scores'],s)
        b=contrast_decision(s,None,.1,triggered=True)
        c=contrast_decision(s,np.empty((0,3)),.1,triggered=True)
        np.testing.assert_array_equal(b['scores'],c['scores'])
        self.assertFalse(b['accepted']);self.assertTrue(b['feature_available'])

    def test_zero_strength_and_vocabulary_ties(self):
        s=[.5,.5,.1]
        a=contrast_decision(s,np.array([[.9,.1,.1]]),0,triggered=True,strength=0)
        self.assertEqual(a['proposed_class_index'],0)
        self.assertFalse(a['accepted']);np.testing.assert_array_equal(a['scores'],s)

    def test_control_can_harm_same_class_neighbor(self):
        # Do not pretend soft spatial contrast has a no-regression theorem.
        a=contrast_decision([.5,.49],np.array([[.7,.1]]),0,triggered=True)
        self.assertEqual(a['proposed_class_index'],1)

    def test_geometry_neutrality_and_window_support(self):
        a=np.array([[.2,.8,0],[0,.5,.5]])
        np.testing.assert_allclose(reweight_attention(a,np.ones_like(a)),a)
        changed=reweight_attention(a,np.array([[1,.05,.2],[.2,.05,1]]))
        np.testing.assert_allclose(changed.sum(1),1)
        np.testing.assert_array_equal(changed==0,a==0)
        with self.assertRaises(ValueError): reweight_attention(np.zeros((1,3)),np.ones((1,3)))

    def test_unknown_owner_depth_are_neutral(self):
        g=geometry_factor([0,0],[1,1],[np.nan,2.],[np.nan,np.nan])
        np.testing.assert_allclose(g,1)
        full=geometry_factor([1,.3],[0,.2],[2.],[2.,3.])
        nod=geometry_factor([1,.3],[0,.2],[2.],[2.,3.],use_depth=False)
        self.assertEqual(full[0,0],nod[0,0]);self.assertLess(full[0,1],nod[0,1])
        self.assertGreater(full.min(),0)

    def base_records(self):
        b1={'replica8':dict(zip(METRICS,[.1239,.2606,.3975,.3027,.3840])),
            'scannet_cf18':dict(zip(METRICS,[.0823,.1783,.2476,.1902,.2849]))}
        b0=copy.deepcopy(b1);b0['scannet_cf18'].update(apall=.0831,ap50=.1796)
        b0['replica8'].update(apall=.1174,ap50=.2450,miou=.2969)
        return b1,b0

    def test_all_five_replica_metrics_are_protected(self):
        b1,b0=self.base_records();c=copy.deepcopy(b1)
        c['scannet_cf18'].update(apall=.0842,ap50=.1800)
        self.assertTrue(target_flags(c,b1,b0)['material_target_met'])
        c['replica8']['macc']-=.0001
        self.assertFalse(target_flags(c,b1,b0)['target_met'])

    def test_research_selection_prefers_simple_tie_and_has_fallback(self):
        b1,b0=self.base_records();c=copy.deepcopy(b1)
        c['scannet_cf18'].update(apall=.0842,ap50=.1800)
        rec={'EV01_G1_V2':b1,'EV00_D2':b0,'simple':c,'complex':copy.deepcopy(c)}
        self.assertEqual(choose_method(rec,['simple','complex']),'simple')
        rec['simple']['replica8']['ap25']-=.01;rec['complex']['replica8']['ap25']-=.01
        self.assertEqual(choose_method(rec,['simple','complex']),'EV01_G1_V2')

if __name__=='__main__': unittest.main()
