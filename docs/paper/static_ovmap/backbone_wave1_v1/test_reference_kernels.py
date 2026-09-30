"""Synthetic specification tests only, not native/GPU/scene experiment tests."""
import unittest
import numpy as np
from reference_kernels import (
    simultaneous_fusion, native_order_reference, canonical_fragments,
    directional_assignments, positive_assignment, native_eligibility,
    pinned_forward_indices, choose_development,
)

class ReferenceTests(unittest.TestCase):
    def case(self):
        d=np.ones((15,20), np.int32)
        p=np.zeros_like(d); p.ravel()[:100]=11; p.ravel()[100:180]=22; p.ravel()[180:250]=33
        return d,p

    def test_simultaneous_partition_and_background(self):
        d,p=self.case(); rows=simultaneous_fusion(d,p)
        self.assertEqual(len(rows),4)
        np.testing.assert_array_equal(np.stack([r.mask for r in rows]).sum(0),np.ones_like(d))
        self.assertEqual(sum(r.group_key is None for r in rows),1)

    def test_empty_residual(self):
        d=np.ones((15,20),np.int32); p=np.repeat([1,2,3],100).reshape(d.shape)
        rows=simultaneous_fusion(d,p)
        self.assertEqual(len(rows),3)
        self.assertTrue(all(np.isfinite(r.overlap) for r in rows))

    def test_initial_small_depth_region_is_removed(self):
        self.assertEqual(simultaneous_fusion(np.ones((7,7),int),np.ones((7,7),int)),[])

    def test_small_carved_intersection_is_not_removed(self):
        d=np.ones((15,20),int); p=np.zeros_like(d); p.ravel()[:40]=1
        rows=simultaneous_fusion(d,p)
        self.assertTrue(any(r.group_key and r.mask.sum()==40 for r in rows))

    def test_id_renaming_does_not_change_physical_output(self):
        d,p=self.case(); renamed=np.zeros_like(p)
        for a,b in [(11,3),(22,100),(33,1)]: renamed[p==a]=b
        self.assertEqual(canonical_fragments(simultaneous_fusion(d,p)),canonical_fragments(simultaneous_fusion(d,renamed)))

    def test_native_order_can_change_physical_output(self):
        d,p=self.case()
        self.assertNotEqual(canonical_fragments(native_order_reference(d,p)),canonical_fragments(native_order_reference(d,p,reverse=True)))
        self.assertNotEqual(canonical_fragments(native_order_reference(d,p)),canonical_fragments(simultaneous_fusion(d,p)))

    def test_directional_denominators_have_nontrivial_mutual_filter(self):
        x=directional_assignments([[800,500,500],[700,100,700]],[2700,1600],[2400,1400,1400])
        self.assertEqual(x['forward'],{(0,0),(1,2)})
        self.assertEqual(x['backward'],{(0,1),(1,2)})
        self.assertEqual(x['mutual'],{(1,2)})

    def test_dummy_unmatched_and_empty_support(self):
        x=directional_assignments([[50,20]],[200],[100,100])
        self.assertEqual(x['forward'],set())
        x=directional_assignments(np.zeros((2,0)),[100,100],[])
        self.assertEqual(x['mutual'],set())

    def test_transposing_the_same_scores_is_not_a_new_direction(self):
        b=np.array([[.7,.1,.2],[.2,.8,.1]])
        a=positive_assignment(b,np.ones_like(b,bool))
        reverse={(j,i) for i,j in positive_assignment(b.T,np.ones_like(b.T,bool))}
        self.assertEqual(a,reverse)

    def test_existing_ratio_gate_is_a_real_simple_control(self):
        self.assertTrue(native_eligibility(101,1000,20,.2,enable_ratio=False))
        self.assertFalse(native_eligibility(101,1000,20,.2,enable_ratio=True))
        self.assertTrue(native_eligibility(201,1000,20,.2,enable_ratio=True))

    def test_inclusive_sam_bounds(self):
        self.assertEqual(pinned_forward_indices(2,0,5),[2])
        self.assertEqual(pinned_forward_indices(2,1,5),[2,3])

    def test_selection_bands_use_percentage_points(self):
        common={'ap50':.2,'added_visual_encodings':0,'median_seconds':10.,'changed_blocks':1}
        rows=[dict(common,id='A',apall=.1,miou=.2),dict(common,id='B',apall=.0997,miou=.22),dict(common,id='C',apall=.099,miou=.9)]
        self.assertEqual(choose_development(rows),'B')
        with self.assertRaises(ValueError):
            directional_assignments([[100,100]],[150],[100,100])

if __name__=='__main__':
    unittest.main(verbosity=2)
