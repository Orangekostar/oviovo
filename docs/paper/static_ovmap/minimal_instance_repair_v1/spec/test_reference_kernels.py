"""Synthetic specification checks only. No new benchmark evidence is produced."""
import unittest
import numpy as np
from reference_kernels import (
    SPEC, Operation, apply_operations, clique_prefixes, pair_vote,
    pose_banks, rates, representative_source_row, reread_decision,
    target_met, unit_summary, verified_pairs,
)

class SpecificationTests(unittest.TestCase):
    def test_pose_banks_are_disjoint_and_bounded(self):
        frames=[]
        for i in range(80):
            pose=np.eye(4); pose[0,3]=(i//2)*.3
            frames.append((i,pose))
        p,v=pose_banks(frames)
        self.assertEqual(len(p)+len(v),32)
        self.assertFalse(set(p)&set(v))
        self.assertTrue(all(i%2==0 for i in p+v))
        self.assertEqual(pose_banks(list(reversed(frames))),(p,v))

    def test_barycentric_membership_and_tie(self):
        f=np.array([[9,3,7],[1,4,8]])
        out=representative_source_row(f,np.array([0,0,1]),np.array([[.1,.1],[.5,0],[.1,.8]]))
        np.testing.assert_array_equal(out,[9,3,8])
        with self.assertRaises(ValueError):
            representative_source_row(f,np.array([0]),np.array([[.9,.9]]))

    def test_unknown_and_neutral_votes_are_not_positive(self):
        self.assertIsNone(unit_summary([0]*50))
        a=unit_summary([4]*35+[8]*15)
        b=unit_summary([7]*35+[9]*15)
        self.assertEqual(pair_vote(a,b),(0,0))
        vote=pair_vote((4,.9),(7,.9))
        self.assertEqual(vote,(0,1))
        n,s,d=rates([None,(1,0),(0,0),(0,1)])
        self.assertEqual(n,3); self.assertAlmostEqual(s,1/3); self.assertAlmostEqual(d,1/3)
        self.assertEqual(rates([None]),(0,None,None))

    def test_clique_requirement_rejects_transitive_merge(self):
        edges={frozenset(('C:1','C:2')),frozenset(('C:2','C:3'))}
        self.assertEqual(clique_prefixes('C:1',['C:2','C:3'],edges),[('C:1','C:2')])
        edges.add(frozenset(('C:1','C:3')))
        self.assertEqual(clique_prefixes('C:1',['C:2','C:3'],edges)[-1],('C:1','C:2','C:3'))

    def test_verification_does_not_accept_unknown_or_separation(self):
        self.assertTrue(verified_pairs([(2,1.,0.),(2,1.,0.),(2,1.,0.)],.5))
        self.assertFalse(verified_pairs([(1,1.,0.)],.1))
        self.assertFalse(verified_pairs([(2,0.,1.)],.1))
        self.assertFalse(verified_pairs([(0,None,None)],0))

    def test_structural_edit_changes_roles_not_surface_core(self):
        b=np.array([1,1,0,0,0,0,0]); r=np.array([1,1,2,2,3,3,0])
        g=np.array([1,1,2,2,3,3,0]); c=np.array([5,5,6,6,6,6,0])
        units={'I:1':np.array([0,1]),'C:2':np.array([2,3]),'C:3':np.array([4,5])}
        o,s=apply_operations(g,c,b,r,units,[Operation(('I:1','C:2'),1,5)])
        np.testing.assert_array_equal(o,[1,1,1,1,3,3,0]); np.testing.assert_array_equal(s,[5,5,5,5,6,6,0])
        o,s=apply_operations(g,c,b,r,units,[Operation(('C:2','C:3'),4,7)])
        np.testing.assert_array_equal(o,[1,1,4,4,4,4,0]); np.testing.assert_array_equal(s,[5,5,7,7,7,7,0])
        with self.assertRaises(ValueError):
            apply_operations(g,c,b,r,units,[Operation(('I:1','C:2'),1,5),Operation(('C:2','C:3'),4,7)])
        # A toy union can cross an IoU threshold even when neither part can.
        truth={2,3,4,5,6}
        self.assertLess(len({2,3}&truth)/len({2,3}|truth),.5)
        self.assertGreater(len({2,3,4,5}&truth)/len({2,3,4,5}|truth),.5)

    def test_stability_checks_are_stricter_than_simple_relabel(self):
        full=np.array([[.1,.4,.2],[.2,.4,.1]])
        regions=np.vstack((full,[.1,.35,.2]))
        self.assertEqual(reread_decision(0,full,regions,distinct_nonfull=1),(1,True,True))
        self.assertEqual(reread_decision(0,full,full,distinct_nonfull=0),(1,True,False))
        mixed=np.array([[.1,.4,.2],[.35,.3,.2]])
        proposed,simple,stable=reread_decision(0,mixed,np.vstack((mixed,[.1,.4,.2])),distinct_nonfull=1)
        self.assertEqual(proposed,1); self.assertTrue(simple); self.assertFalse(stable)

    def test_five_metric_protection_and_fixed_slate(self):
        import copy
        ref={c:{m:.3 for m in SPEC['evaluation']['metrics']} for c in SPEC['cohorts']}
        d2=copy.deepcopy(ref); d2['scannet_cf18']['apall']=.31
        candidate=copy.deepcopy(ref); candidate['scannet_cf18']['apall']=.32
        self.assertTrue(target_met(candidate,ref,d2))
        candidate['replica8']['ap25']-=.00001
        self.assertFalse(target_met(candidate,ref,d2))
        self.assertEqual(len(SPEC['methods'])*sum(map(len,SPEC['cohorts'].values())),234)
        self.assertEqual(len(SPEC['methods'])*len(SPEC['cohorts']),18)

if __name__=='__main__':
    unittest.main(verbosity=2)
