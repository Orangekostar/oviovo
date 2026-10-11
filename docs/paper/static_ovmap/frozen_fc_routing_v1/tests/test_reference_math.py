import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference_math import (FGReference, unit, route, correct, matched_mass, match_delta_norm,
                            membership_loss, qualifies, path_counts)
torch.set_num_threads(1)


def fixture():
    torch.manual_seed(123)
    phi=nn.Sequential(nn.Linear(5,6),nn.Tanh()).requires_grad_(False)
    raw=torch.randn(3,4,5)
    x=dict(fc_vectors=unit(torch.randn(3,6)),view_valid=torch.ones(3,dtype=torch.bool),
           local_raw=raw,local_unit=unit(phi(raw)).detach(),metadata=torch.randn(3,4,8),
           local_valid=torch.tensor([[1,1,0,1],[1,0,1,1],[1,1,1,0]],dtype=torch.bool),
           frame_ids=torch.tensor([10,20,30]))
    return x,phi


def head(group='SURFACE',routing=True):
    torch.manual_seed(1026)
    return FGReference(group,routing,dim=6,hidden=12)


class ReferenceTests(unittest.TestCase):
    def test_01_matched_shapes_and_parameter_count(self):
        states=[]
        for grouping,routing in [('VIEW',False),('SURFACE',False),('SURFACE',False),('SURFACE',True)]:
            torch.manual_seed(1026); m=FGReference(grouping,routing)
            self.assertEqual(sum(p.numel() for p in m.parameters()),216194)
            states.append(m.state_dict())
            self.assertFalse(any('ma.' in k or k.startswith('output.') for k in states[-1]))
        for st in states[1:]:
            self.assertTrue(all(torch.equal(st[k],states[0][k]) for k in st))

    def test_02_step0_and_zero_rho_identity(self):
        x,phi=fixture(); m=head()
        b=unit(x['fc_vectors'].mean(0))
        r=m(x,phi,training_forward=True,completed_steps=0)
        self.assertTrue(torch.equal(r['embedding'],b))
        r['embedding'].sum().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
        r=m(x,phi,rho_override=0.)
        self.assertTrue(torch.equal(r['embedding'],b))
        self.assertTrue(torch.equal(r['null'],torch.ones(4)))

    def test_03_missing_local_not_fake_evidence(self):
        x,phi=fixture();x['local_valid'].zero_();x['local_raw'].fill_(float('nan'))
        r=head()(x,phi)
        self.assertEqual(r['fallback'],'NO_LOCAL_SUPPORT_FC_FALLBACK')
        self.assertTrue(torch.equal(r['embedding'],unit(x['fc_vectors'].mean(0))))
        self.assertEqual(r['group_count'],0)

    def test_04_frozen_phi_input_gradient(self):
        x,phi=fixture();m=head()
        r=m(x,phi); loss=(r['embedding']*torch.arange(1.,7.)).sum();loss.backward()
        self.assertGreater(float(m.quality[0].weight.grad.norm()),0)
        self.assertGreater(float(m.queries.grad.norm()),0)
        self.assertGreater(float(m.membership[-1].weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in phi.parameters()))

    def test_05_consistent_view_site_permutation(self):
        x,phi=fixture();m=head();r=m(x,phi)['embedding']
        vp=torch.tensor([2,0,1]);sp=torch.tensor([3,1,0,2]);y={}
        for k,v in x.items():
            y[k]=v[vp][:,sp] if k in ('local_raw','local_unit','metadata','local_valid') else v[vp]
        torch.testing.assert_close(r,m(y,phi)['embedding'],atol=2e-6,rtol=2e-6)

    def test_06_grouping_is_not_renamed_average(self):
        x,phi=fixture();a=head('VIEW',False);b=head('SURFACE',False)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertFalse(torch.allclose(a(x,phi)['embedding'],b(x,phi)['embedding'],atol=1e-7,rtol=1e-7))

    def test_07_route_monotonicity_group_count_normalization(self):
        e=torch.zeros(4,3);rho=torch.ones(3)
        a,n=route(e,rho);torch.testing.assert_close(a.sum(-1),torch.full((4,),.5))
        aa,nn=route(torch.zeros(4,30),torch.ones(30));torch.testing.assert_close(aa.sum(-1),a.sum(-1))
        lower=route(e,torch.tensor([.5,1.,1.]))[0]
        self.assertTrue(bool((lower[:,0]<a[:,0]).all()))
        self.assertTrue(bool((lower[:,1:]>=a[:,1:]).all()))

    def test_08_matched_mass_not_norm_then_norm_control(self):
        e=torch.zeros(1,2);a,_=route(e,torch.tensor([.1,.9]));am=matched_mass(e,a)
        torch.testing.assert_close(a.sum(-1),am.sum(-1))
        base=torch.tensor([1.,0.]);values=torch.tensor([[-1.,1.],[-1.,-1.]])
        _,d=correct(base,values,a,.5);_,dm=correct(base,values,am,.5)
        self.assertGreater(abs(float(d.norm()-dm.norm())),1e-3)
        _,audit=match_delta_norm(base,d,dm);self.assertEqual(audit['status'],'MATCHED')
        torch.testing.assert_close(audit['delta'].norm(),d.norm())
        z,audit=match_delta_norm(base,d,torch.zeros_like(d));self.assertIsNone(z)
        self.assertEqual(audit['status'],'UNDEFINED_ZERO_CONTROL')

    def test_09_neutral_rho_and_convex_nonzero_bound(self):
        torch.manual_seed(4);e=8*torch.rand(4,6)-4;a,_=route(e,torch.full((6,),.3))
        torch.testing.assert_close(a,matched_mass(e,a))
        b=torch.tensor([1.,0.]);u=torch.tensor([[-1.,0.]]).repeat(6,1)
        a,_=route(torch.full((4,6),4.),torch.ones(6));_,d=correct(b,u-b,a,.5)
        self.assertGreaterEqual(float((b+d).norm()),1/(1+torch.exp(torch.tensor(4.))).item()-1e-6)

    def test_10_membership_ignores_unknown_and_inputs_exclude_GT(self):
        x,phi=fixture();r=head()(x,phi)
        y=torch.full((3,4),-1);loss,n=membership_loss(r,y)
        self.assertEqual(n,0);self.assertEqual(float(loss.detach()),0)
        y[0,0]=1;y[0,1]=0;loss,n=membership_loss(r,y)
        self.assertEqual(n,2);self.assertTrue(torch.isfinite(loss))
        with self.assertRaises(ValueError):head()({**x,'class_id':12},phi)

    def test_11_gate_and_state_paths(self):
        ref=dict(A=.37288510,M=.50612374,C=.42058081)
        row=dict(A=ref['A']+.005,M=ref['M'],C=ref['C'],step=250,net_corrections=1)
        self.assertTrue(qualifies(row,ref))
        self.assertFalse(qualifies({**row,'step':0},ref))
        self.assertFalse(qualifies({**row,'net_corrections':0},ref))
        self.assertFalse(qualifies({**row,'A':ref['A']},ref))
        self.assertEqual(path_counts(None,False),(8000,0,0))
        self.assertEqual(path_counts('FG4_ROUTE',False),(16000,0,0))
        self.assertEqual(path_counts('FG4_ROUTE',True),(16000,286,22))

    def test_12_phi_order_and_eta_schedule(self):
        x,phi=fixture();sample=x['local_raw'][0,:2]
        self.assertFalse(torch.allclose(unit(phi(sample.mean(0))),unit(unit(phi(sample)).mean(0)),atol=1e-5,rtol=1e-5))
        m=head();self.assertEqual(m(x,phi,training_forward=True,completed_steps=100)['eta'],.25)
        self.assertEqual(m(x,phi,training_forward=True,completed_steps=200)['eta'],.5)
        self.assertEqual(m(x,phi)['eta'],.5)

if __name__=='__main__':unittest.main(verbosity=2)
