"""Twelve synthetic checks, not actual model/evaluator validation."""
import copy
import unittest
import numpy as np
from rules import (interior_points, order_window, transform_points, split_panorama,
                   count_view_votes, arbitrate, labels_for_partition, paired_labels, pilot_target)

class RulesTest(unittest.TestCase):
    def test_01_points_stay_in_largest_component(self):
        mask = np.zeros((16, 24), bool); mask[1:12, 1:12] = True; mask[2:6, 18:22] = True
        points = interior_points(mask)
        self.assertEqual(len(points), 3); self.assertEqual(len(set(points)), 3)
        self.assertTrue(all(mask[y, x] and x < 12 for x, y in points))
        self.assertEqual(points, interior_points(mask))

    def test_02_order_window_and_anchor_index(self):
        order, a, b = order_window([50, 10, 80, 30], 50, 10)
        self.assertEqual(order, [10, 30, 50, 80]); self.assertEqual((a, b), (2, 0))
        with self.assertRaises(ValueError): order_window([1, 1, 2], 1, 2)

    def test_03_anisotropic_coordinate_scaling(self):
        points = np.array([[100., 200.], [0., 0.]])
        q = transform_points(points, (400, 800))
        np.testing.assert_array_equal(q, [[128., 512.], [0., 0.]])
        np.testing.assert_array_equal(points, [[100., 200.], [0., 0.]])

    def test_04_split_before_resize(self):
        x = np.concatenate([np.full((2, 3), k) for k in [0., 4., -4.]], axis=1)
        tiles = split_panorama(x, 3)
        self.assertEqual(tiles.shape, (3, 2, 3))
        for tile, value in zip(tiles, [0., 4., -4.]): self.assertTrue(np.all(tile == value))

    def test_05_views_not_pixels_count(self):
        src = [np.zeros((1, 100), int), np.zeros((1, 1), int)]
        valid = [np.ones_like(s, bool) for s in src]
        masks = [np.ones((1, 100), bool), np.zeros((1, 1), bool)]
        n, k = count_view_votes(src, valid, masks, 1)
        np.testing.assert_array_equal(n, [2]); np.testing.assert_array_equal(k, [1])

    def test_06_invalid_view_is_unknown(self):
        n, k = count_view_votes([np.array([[-1]])], [np.array([[False]])], [np.array([[False]])], 2)
        np.testing.assert_array_equal(n, [0, 0]); np.testing.assert_array_equal(k, [0, 0])
        base = np.array([1, 0]); out = arbitrate(base, [1], {1:(n,k)}, {1:np.ones(2,bool)}, np.ones(2,bool))
        np.testing.assert_array_equal(out, base)

    def test_07_edit_domain_and_protected_core(self):
        base = np.array([1, 3, 3, 0]); n=np.full(4,3);k=np.full(4,3)
        out=arbitrate(base,[1],{1:(n,k)},{1:np.array([1,1,1,0],bool)},np.array([1,1,0,1],bool))
        np.testing.assert_array_equal(out,[1,1,3,0])

    def test_08_exact_competing_tie_keeps_old(self):
        base=np.array([7]);out=arbitrate(base,[1,2],{1:(np.array([3]),np.array([2])),
            2:(np.array([6]),np.array([4]))},{1:np.array([1],bool),2:np.array([1],bool)},np.array([1],bool))
        np.testing.assert_array_equal(out,base)

    def test_09_negative_removal_and_prompt_protection(self):
        base=np.array([1,1,1]); n=np.array([3,3,1]);k=np.array([0,0,0])
        out=arbitrate(base,[1],{1:(n,k)},{1:np.ones(3,bool)},np.ones(3,bool),{0:1})
        np.testing.assert_array_equal(out,[1,0,1])

    def test_10_one_class_per_owner(self):
        out=labels_for_partition(np.array([1,1,2,0,2]),{1:10,2:20},{1:30})
        np.testing.assert_array_equal(out,[30,30,20,0,20])

    def test_11_joint_semantic_availability(self):
        old=[np.array([.3,.1]),np.array([.3,.1])];new=[np.array([.1,.4]),None]
        self.assertEqual(paired_labels(1,[1,2],old,new),(1,1,False))
        new[1]=np.array([.1,.4])
        self.assertEqual(paired_labels(1,[1,2],old,new),(1,2,True))

    def test_12_five_metric_pilot_protection(self):
        base={c:{m:.1 for m in ['apall','ap50','ap25','miou','macc']} for c in ['replica_probe2','cf_probe2']}
        candidate=copy.deepcopy(base);candidate['cf_probe2']['apall']=.12
        self.assertTrue(pilot_target(candidate,base,{'apall':.11,'ap50':.1}))
        candidate['replica_probe2']['macc']=.099
        self.assertFalse(pilot_target(candidate,base,{'apall':.11,'ap50':.1}))

if __name__ == '__main__': unittest.main(verbosity=2)
