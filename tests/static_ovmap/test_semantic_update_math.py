import numpy as np
import torch
from static_ovmap.semantic_update_evidence_v1.constraints import simplex,radius,prepare_dictionary,evidence
from static_ovmap.semantic_update_evidence_v1.calibration import angle_cap,recipes,best_candidate
from static_ovmap.semantic_update_evidence_v1.models import make_models
from static_ovmap.learned_object_readout.training import model_digest

def test_projection_gradients_finite_iterations_and_radius():
    D=torch.eye(3,768);a=torch.tensor([.98,.01,.01]);d=prepare_dictionary(D,a)
    q=(a@D+.15*D[1]).requires_grad_();r=evidence(q,d,128);ref=evidence(q,d,256)
    assert torch.linalg.vector_norm(r['preunit']-ref['preunit'])<=.002
    assert r['FW_gap']<=1e-4 and r['steps']==128
    assert torch.all(r['coefficients']>=0) and torch.isclose(r['coefficients'].sum(),torch.tensor(1.))
    assert torch.linalg.vector_norm(r['preunit']-D[0])<=.500001
    r['embedding'][1].backward();assert q.grad.norm()>0 and torch.isfinite(q.grad).all()
    q2=D[0]+2*D[1];p,t=radius(D[0],q2);assert abs(float((p-D[0]).norm())-.5)<1e-6
    assert torch.isclose(simplex(torch.tensor([-1.,2.,3.])).sum(),torch.tensor(1.))

def test_true_antipodal_boundary_and_shared_network():
    b=np.zeros((1,768),np.float32);b[0,0]=1
    z,flag=angle_cap(b,-b,10);assert flag.tolist()==[True] and np.array_equal(z,b)
    assert np.array_equal(angle_cap(b,-b,180)[0],-b)
    models=make_models(17);assert len({model_digest(m) for m in models.values()})==1
    assert len({id(x) for x in models['FREE'].blocks})==4
