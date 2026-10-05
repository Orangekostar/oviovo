import unittest
import numpy as np
from invariants import (View, retain_top3, ideal_candidate_roi, science_key,
                        summarize_times, balanced_schedule, candidate_choice, SCIENCE_FIELDS)

class ContractExamples(unittest.TestCase):
    def test_top3_matches_full_sort(self):
        views=[View(n, i, str(i)) for i,n in enumerate([120,130,300,180,100,450,201])]
        self.assertEqual(retain_top3(views), sorted(views,key=lambda v:v.rank)[:3])

    def test_top3_preserves_digest_ties(self):
        views=[View(100,2,'z'),View(200,1,'a'),View(100,2,'b'),View(100,2,'a')]
        self.assertEqual(retain_top3(views), sorted(views,key=lambda v:v.rank)[:3])

    def test_g1_is_g3_prefix(self):
        out=retain_top3([View(300,4,'x'),View(400,2,'y')])
        self.assertEqual(out[:1], [View(400,2,'y')])

    @staticmethod
    def box(z=(2.,3.)):
        return np.array([[x,y,d] for x in (-.1,.1) for y in(-.1,.1) for d in z])

    def test_roi_contains_integer_projected_samples(self):
        K=np.array([[100.,0,50],[0,100.,50],[0,0,1]])
        roi=ideal_candidate_roi(self.box(),K,(100,100))
        x0,y0,x1,y1=roi
        for x in np.linspace(-.1,.1,5):
            for y in np.linspace(-.1,.1,5):
                u,v=100*x/2.5+50,100*y/2.5+50
                self.assertTrue(x0 <= u < x1 and y0 <= v < y1)

    def test_near_plane_returns_full(self):
        self.assertEqual(ideal_candidate_roi(self.box((-1.,2.)),np.eye(3),(10,10)),'FULL')

    def test_behind_returns_no_rays(self):
        self.assertIsNone(ideal_candidate_roi(self.box((-3.,-2.)),np.eye(3),(10,10)))

    def test_producer_identity_not_semantics(self):
        base={key: key for key in SCIENCE_FIELDS}
        a=dict(base,producer='old',request_id='old-id')
        b=dict(base,producer='new',request_id='new-id')
        self.assertEqual(science_key(a),science_key(b))
        b['mask_digest']='different'
        self.assertNotEqual(science_key(a),science_key(b))

    def test_complete_time_grid_uses_means(self):
        rows=[{'scene':s,'repeat':r,'seconds':t} for s,r,t in [('a',1,4),('a',2,6),('b',1,20),('b',2,22)]]
        self.assertEqual(summarize_times(rows,['a','b'])['cohort_mean'],13.)
        with self.assertRaises(ValueError):summarize_times(rows[:-1],['a','b'])

    def test_new_repetition_keys(self):
        scenes=[f's{i}' for i in range(8)]
        arms=['U2','G1R','G1F','G3F']
        plan=balanced_schedule(scenes,arms)
        self.assertEqual(len(plan),64)
        self.assertEqual(len(set(plan)),64)
        self.assertEqual(plan[32],('s7','G3F',2))

    def test_simplicity_tie_and_reference(self):
        order=['R0','R1','R2','R3']
        self.assertEqual(candidate_choice({'R0':45,'R1':42,'R2':41.5,'R3':41.4},order),'R1')
        self.assertEqual(candidate_choice({'R0':45,'R1':44.5},order),'R0')

if __name__=='__main__':unittest.main(verbosity=2)
