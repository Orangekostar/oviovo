import unittest
import numpy as np
import torch
from torch import nn
from reference_kernels import *

torch.set_num_threads(1)

class ContractReferences(unittest.TestCase):
    def test_family_separation_and_scan_suffix(self):
        scans=[f'scene{i:04d}_{j:02d}' for i in range(60) for j in (2,0,1)]
        split=family_split(scans,{'scene0011','scene0050'})
        self.assertEqual([len(split[k]) for k in ('train','dev','holdout')],[24,4,4])
        used=[family_id(s) for names in split.values() for s in names]
        self.assertEqual(len(used),len(set(used)))
        self.assertNotIn('scene0011',used);self.assertTrue(all(s.endswith('_00') for names in split.values() for s in names))
    def test_too_few_families_is_not_a_smoke_training_result(self):
        with self.assertRaises(ValueError):family_split([f'scene{i:04d}_00' for i in range(10)],[])
    def test_class_split_fixed_and_disjoint(self):
        base,novel=split_classes(range(1,201));self.assertEqual((len(base),len(novel)),(160,40))
        self.assertFalse(set(base)&set(novel));self.assertEqual((base,novel),split_classes(range(1,201)))
    def test_frozen_head_input_gradients_and_no_parameter_change(self):
        torch.manual_seed(1);phi=nn.Sequential(nn.Linear(5,7),nn.GELU(),nn.Linear(7,4)).double().requires_grad_(False)
        before={k:v.clone() for k,v in phi.state_dict().items()};logits=nn.Parameter(torch.zeros(6,dtype=torch.float64))
        raw=torch.randn(6,5,dtype=torch.float64);z=frozen_projection((logits.softmax(0)[:,None]*raw).sum(0),phi)
        (-z[0]).backward();self.assertGreater(float(logits.grad.norm()),1e-10)
        self.assertTrue(all(p.grad is None for p in phi.parameters()))
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in phi.state_dict().items()))
    def test_nonlinear_pool_and_project_do_not_commute(self):
        phi=nn.ReLU();x=torch.tensor([[-2.,3.],[1.,-1.]])
        self.assertFalse(torch.allclose(phi(x.mean(0)),phi(x).mean(0)))
    def test_padding_not_an_observation(self):
        x=torch.tensor([[1.,2.],[3.,4.],[10000.,10000.]])
        self.assertTrue(torch.equal(masked_mean(x,torch.tensor([1,1,0],dtype=torch.bool)),torch.tensor([2.,3.])))
    def test_group_permutation_invariance(self):
        torch.manual_seed(8);model=GroupReference().double();x=torch.randn(8,5,dtype=torch.float64)
        groups=torch.tensor([0,1,0,1,2,2,3,3]);valid=torch.tensor([1,1,1,1,1,0,1,1],dtype=torch.bool)
        perm=torch.tensor([7,0,3,6,4,1,5,2]);self.assertTrue(torch.allclose(model(x,groups,valid),model(x[perm],groups[perm],valid[perm]),atol=1e-12,rtol=1e-12))
    def test_view_and_surface_grouping_are_not_just_renaming(self):
        torch.manual_seed(6);model=GroupReference().double();x=torch.randn(8,5,dtype=torch.float64);valid=torch.ones(8,dtype=torch.bool)
        view=torch.tensor([0,0,0,0,1,1,1,1]);site=torch.tensor([0,1,2,3,0,1,2,3])
        self.assertGreater(float((model(x,view,valid)-model(x,site,valid)).abs().max().detach()),1e-5)
    def test_unknown_membership_does_not_become_negative(self):
        x=torch.tensor([2.,-2.,20.],requires_grad=True);loss=masked_bce(x,torch.tensor([1,0,-1]));loss.backward()
        self.assertEqual(float(x.grad[2]),0.);self.assertGreater(float(x.grad[:2].norm()),0.)
    def test_camera_depth_to_color_transform(self):
        e=np.eye(4);e[0,3]=.2;t=np.eye(4);t[2,3]=3
        world,uv=register_one_depth_point(np.array([0.,0.]),2.,np.eye(3),np.eye(3),e,t)
        np.testing.assert_allclose(world,[0,0,5]);np.testing.assert_allclose(uv,[.1,0])
    def test_feature_pixel_centers(self):
        grid=normalized_feature_grid(np.array([[0.,0.],[1.,1.]]),(2,2),(4,4),(4,4))
        np.testing.assert_allclose(grid,[[-.5,-.5],[.5,.5]],atol=1e-12)
    def test_replace_f_missing_sources_and_whole_gate(self):
        from scipy.special import softmax
        new=np.array([2.,-1.]);p=replace_f({'N':None,'Q':None},{'F':1.},new)
        np.testing.assert_allclose(p,softmax(new));p=replace_f({'N':new,'Q':-new},{'N':1.,'Q':1.,'F':1.},new)
        np.testing.assert_allclose(p,.25*softmax(new)+.25*softmax(-new)+.5*softmax(new))
        base={k:.1 for k in ('apall','ap50','ap25','miou','macc')};good={**base,'apall':.12}
        self.assertTrue(upgrade_gate(base,good,base,base,base))
        bad={**base,'ap25':.09};self.assertFalse(upgrade_gate(bad,good,base,base,base))

if __name__=='__main__':unittest.main(verbosity=2)
