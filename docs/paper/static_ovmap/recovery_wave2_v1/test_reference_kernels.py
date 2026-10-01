"""Synthetic specification checks only; no production/native/model claims."""
import unittest
import numpy as np
from reference_kernels import (grouped_endpoints,weight_readout,append_recovered,
    association_actions,prepare_actions_without_native_state,crop_priority,conflict_suppress,gain_status)

class ReferenceContractTests(unittest.TestCase):
    def test_weight_endpoints_and_full_group_formula(self):
        s={'N':np.array([.8,.2]),'Q':np.array([.6,.4]),'F':np.array([.1,.9])}
        e,d=grouped_endpoints(s)
        np.testing.assert_array_equal(weight_readout(s,1/3),e)
        np.testing.assert_array_equal(weight_readout(s,.5),d)
        np.testing.assert_allclose(weight_readout(s,.4),.6*(s['N']+s['Q'])/2+.4*s['F'])
    def test_missing_native_does_not_change_with_gamma(self):
        s={'N':np.array([.8,.2]),'Q':None,'F':np.array([.1,.9])}
        for g in (1/3,.4,.45,.5):
            np.testing.assert_allclose(weight_readout(s,g),[.45,.55])
    def test_no_source_is_not_uniform_expert(self):
        self.assertIsNone(weight_readout({'N':None,'Q':None,'F':None},.4))
    def test_recovery_cannot_overwrite_old_owner_or_class(self):
        r=np.array([1,2,2,2,3,3]);o=np.array([1,1,0,0,0,0]);y=np.array([5,5,0,0,0,0])
        a,b=append_recovered(r,o,y,{2:8},minimum=1)
        np.testing.assert_array_equal(a,[1,1,2,2,0,0]);np.testing.assert_array_equal(b,[5,5,8,8,0,0])
        with self.assertRaises(ValueError): append_recovered(r,o,y,{1:8},minimum=1)
    def test_fallback_allocates_nothing(self):
        rows,highest=prepare_actions_without_native_state([('USE_NATIVE',None),('ASSIGN_EXISTING',7)],10)
        self.assertEqual(highest,10);self.assertEqual(rows[0],('USE_NATIVE',None))
    def test_two_local_parts_can_share_an_owner(self):
        I=np.array([[150.],[150.]])
        a1=association_actions(I,[170,170],[1000],'A1')
        a2=association_actions(I,[170,170],[1000],'A2')
        a3=association_actions(I,[170,170],[1000],'A3')
        self.assertEqual(sum(a[0]=='ASSIGN_EXISTING' for a in a1),1)
        self.assertEqual(sum(a[0]=='ASSIGN_EXISTING' for a in a2),2)
        self.assertEqual(a2,a3) # each reverse .15; group reverse .30
    def test_ambiguous_multi_proposal_falls_back(self):
        out=association_actions([[110,100]],[250],[500,500],'A3')
        self.assertEqual(out,[('USE_NATIVE',None)])
    def test_partial_sam_overlap_preserves_entire_crop(self):
        C=np.array([[1,1,1,1,0,0]])
        M=np.array([[[1,0,0,0,1,0]]],bool)
        W=np.array([[0,-1,-1,-1,0,-1]])
        out,_,_=crop_priority(C,M,W)
        np.testing.assert_array_equal(out,C)
    def test_attached_sam_completes_only_background(self):
        C=np.array([[1,1,1,1,0,2]])
        M=np.array([[[1,1,1,1,1,0]]],bool)
        W=np.array([[0,0,0,0,0,-1]])
        out,adds,att=crop_priority(C,M,W)
        np.testing.assert_array_equal(out,[[1,1,1,1,1,2]])
        self.assertEqual(att,{0:1});self.assertEqual(int(adds[0].sum()),1)
    def test_unknown_space_is_not_conflict(self):
        P=np.ones((1,5),bool);A=np.array([[0,0,1,1,1]],bool);O=np.array([[1,1,0,0,0]])
        V=np.array([[1,1,0,0,0]],bool)
        keep,state=conflict_suppress(P,A,O,V,minimum=1)
        np.testing.assert_array_equal(keep,A);self.assertEqual(state,'KEEP')
    def test_conflict_suppresses_only_additions(self):
        P=np.ones((1,5),bool);A=np.array([[0,0,1,1,1]],bool);O=np.array([[1,1,2,2,0]])
        V=np.array([[1,1,0,0,0]],bool)
        keep,state=conflict_suppress(P,A,O,V,minimum=1)
        self.assertFalse(keep.any());self.assertEqual(state,'SUPPRESS_ADDITIONS')
    def test_percentage_point_units(self):
        b={'apall':.10,'ap50':.20,'miou':.30}
        self.assertEqual(gain_status(b,{'apall':.102,'ap50':.20,'miou':.30}),'NET_GAIN_WITH_GUARDRAILS')
        self.assertEqual(gain_status(b,{'apall':.101,'ap50':.20,'miou':.30}),'NO_QUALIFYING_NET_GAIN')

if __name__=='__main__': unittest.main(verbosity=2)
