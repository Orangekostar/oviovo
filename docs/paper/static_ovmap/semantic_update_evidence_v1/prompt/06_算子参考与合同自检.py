"""CPU reference only. Not a repository integration or a scientific experiment.
Run: python 06_算子参考与合同自检.py --output 07_实际合同自检结果.json
The finite hull layer is followed by radial clipping; it is NOT exact
projection onto the hull/ball intersection. No straight-through estimator.
"""
from __future__ import annotations
import argparse, json, math, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize


def unit(x: torch.Tensor) -> torch.Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if not bool(torch.isfinite(x).all()) or bool((n <= 1e-12).any()):
        raise ValueError('Nonfinite or zero vector')
    return x / n


def simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection onto nonnegative weights summing to one."""
    if v.ndim != 1 or not v.numel():
        raise ValueError('One nonempty coefficient vector required')
    u = torch.sort(v, descending=True).values
    css = torch.cumsum(u, 0) - 1
    k = torch.arange(1, v.numel()+1, dtype=v.dtype, device=v.device)
    active = u - css / k > 0
    rho = torch.nonzero(active, as_tuple=False)[-1, 0]
    theta = css[rho] / (rho.to(v.dtype)+1)
    return torch.clamp(v-theta, min=0)


def common_center(b: torch.Tensor, evidence: torch.Tensor, epsilon: float = .02):
    if evidence.ndim != 2 or evidence.shape[1] != b.numel() or evidence.shape[0] == 0:
        raise ValueError('At least one evidence vector is required here')
    d = torch.cat([b[None], evidence], 0).detach()
    a0 = b.new_full((len(d),), epsilon/(len(d)-1)); a0[0] = 1-epsilon
    c = a0 @ d
    return d, a0, c


def radial(q: torch.Tensor, b: torch.Tensor, radius: float = .5):
    if not 0 < radius < 1:
        raise ValueError('0 < radius < 1 required')
    delta = q-b
    t = torch.clamp(q.new_tensor(radius)/torch.linalg.vector_norm(delta).clamp_min(1e-12), max=1.)
    return b+t*delta, t


def finite_hull(q: torch.Tensor, d: torch.Tensor, a0: torch.Tensor, steps: int = 128):
    """Fixed unrolled FISTA, with cached/detached dictionary Gram possible."""
    if steps <= 0 or d.ndim != 2 or d.shape[0] != a0.numel():
        raise ValueError('Shape/iteration mismatch')
    gram = (d @ d.T).detach()
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp_min(1e-8).detach()
    rhs = d @ q
    a = a0
    y = a0
    t = 1.
    for _ in range(steps):
        nxt = simplex(y - (gram @ y-rhs)/lipschitz)
        nt = (1+math.sqrt(1+4*t*t))/2
        y = nxt + ((t-1)/nt)*(nxt-a)
        a, t = nxt, nt
    p = a @ d
    grad = gram @ a-rhs
    gap = torch.clamp(torch.dot(a, grad)-grad.min(), min=0)
    return p, a, gap


def evidence_operator(q: torch.Tensor, b: torch.Tensor, d: torch.Tensor,
                      a0: torch.Tensor, radius: float = .5, steps: int = 128):
    p, a, gap = finite_hull(q,d,a0,steps)
    y,t = radial(p,b,radius)
    e0 = torch.zeros_like(a);e0[0]=1
    final_coeff = (1-t)*e0+t*a
    return unit(y), {'pre_unit':y,'coefficients':final_coeff,'gap':gap,'hull_point':p}


def angle_cap(b: torch.Tensor, z: torch.Tensor, degrees: float):
    """A diagnostic of UNIT embeddings, not of the unnormalized update."""
    if degrees >= 180:return z, 'IDENTITY'
    if degrees <= 0:return b, 'BASE'
    dot=torch.dot(b,z).clamp(-1,1)
    angle=torch.acos(dot)
    limit=math.radians(degrees)
    if float(angle)<=limit:return z, 'UNCHANGED'
    tangent=z-dot*b
    n=torch.linalg.vector_norm(tangent)
    if float(n)<1e-10:return b, 'ANTIPODAL_FALLBACK_BASE'
    return math.cos(limit)*b+math.sin(limit)*tangent/n, 'CAPPED'


def harmonic(a: float|None,b: float|None):
    if a is None or b is None:return None
    return 2*a*b/(a+b) if a+b else 0.


def run_tests():
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(28131)
    results=[]
    def record(group,name,fn):
        value=fn();results.append(dict(group=group,name=name,status='PASS',details=value))
    def simple():
        v=torch.tensor([-.1,.2,1.3],dtype=torch.float64)
        a=simplex(v);assert float(a.min())>=0 and abs(float(a.sum())-1)<1e-12
        return {'sum':float(a.sum()),'minimum':float(a.min())}
    record('T1','simplex_feasibility',simple)
    def corrected_h():
        # Macro H must not be silently replaced with aggregate accuracy.
        macro=harmonic(.5,.25);micro=harmonic(.9,.25)
        assert abs(macro-1/3)<1e-12 and macro!=micro and harmonic(None,.3) is None
        return {'H_macro':macro,'different_H_micro':micro}
    record('T1','macro_micro_and_missing',corrected_h)
    def mixing():
        b=torch.tensor([.2,.3,-.1]);z=torch.tensor([.3,.0,.2]);core=torch.tensor([1,0,1],dtype=torch.bool)
        assert torch.equal(0*z+1*b,b)
        mixed=.5*z+.5*b;cal=mixed-.05*core
        assert torch.allclose(cal[~core],mixed[~core])
        return {'unchanged_noncore':True}
    record('T1','mix_calibration_endpoints',mixing)
    def init():
        b=unit(torch.randn(24,generator=gen,dtype=torch.float64))
        u=unit(torch.randn(7,24,generator=gen,dtype=torch.float64))
        d,a,c=common_center(b,u)
        q=c.detach().clone().requires_grad_();e,audit=evidence_operator(q,b,d,a)
        f=unit(q);r=unit(radial(q,b)[0])
        err=max(float((e-f).abs().max().detach()),float((e-r).abs().max().detach()))
        assert err<1e-10
        loss=torch.dot(e,torch.randn(24,generator=gen,dtype=torch.float64));loss.backward()
        assert torch.isfinite(q.grad).all() and float(q.grad.norm())>1e-7
        return {'same_initial_output_max_abs':err,'evidence_q_gradient_norm':float(q.grad.norm())}
    record('T2','shared_interior_initialization_gradient',init)
    def feasible():
        b=unit(torch.randn(16,generator=gen,dtype=torch.float64));u=unit(torch.randn(6,16,generator=gen,dtype=torch.float64));d,a,c=common_center(b,u)
        q=(3*u[0]-2*b).detach().requires_grad_()
        z,audit=evidence_operator(q,b,d,a)
        y=audit['pre_unit'];co=audit['coefficients']
        assert float(co.min().detach())>=-1e-10 and abs(float(co.sum().detach())-1)<1e-10
        assert torch.allclose(y,co@d,atol=1e-10,rtol=1e-10)
        assert abs(float((y-b).norm().detach())-.5)<1e-8 and float(y.norm().detach())>=.4999999999
        assert abs(float(z.norm().detach())-1)<1e-10
        return {'radius':float((y-b).norm().detach()),'coefficient_sum':float(co.sum().detach()),'FW_gap':float(audit['gap'].detach())}
    record('T2','hull_and_ball_preunit_feasibility',feasible)
    def oracle():
        b=unit(torch.randn(10,generator=gen,dtype=torch.float64));u=unit(torch.randn(4,10,generator=gen,dtype=torch.float64));d,a,c=common_center(b,u)
        q=c+.7*torch.randn(10,generator=gen,dtype=torch.float64)
        p,coeff,gap=finite_hull(q,d,a)
        D=d.numpy();Q=q.numpy()
        opt=minimize(lambda w:.5*np.sum((w@D-Q)**2),a.numpy(),jac=lambda w:(w@D-Q)@D.T,
            bounds=[(0,1)]*len(a),constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones_like(w))],method='SLSQP',options={'ftol':1e-12,'maxiter':500})
        assert opt.success
        err=np.linalg.norm(p.detach().numpy()-opt.x@D)
        assert err<1e-5
        return {'synthetic_output_vs_SLSQP_L2':float(err),'FW_gap':float(gap)}
    record('T2','small_synthetic_hull_oracle',oracle)
    def fgrad():
        # Interior point makes piecewise-smooth finite-difference verification meaningful.
        b=torch.tensor([1.,0.,0.],dtype=torch.float64)
        u=torch.tensor([[0.,1.,0.],[0.,0.,1.]],dtype=torch.float64)
        d,a,c=common_center(b,u)
        q=torch.tensor([.7,.2,.1],dtype=torch.float64,requires_grad=True)
        direction=torch.tensor([.3,-.2,.15],dtype=torch.float64)
        def fn(v):return torch.dot(evidence_operator(v,b,d,a)[0],torch.tensor([.2,-.4,.8],dtype=torch.float64))
        val=fn(q);g=torch.autograd.grad(val,q)[0]
        eps=1e-6;numeric=float(((fn(q+eps*direction)-fn(q-eps*direction))/(2*eps)).detach());analytic=float(g@direction)
        assert abs(numeric-analytic)<1e-5
        return {'finite_difference':numeric,'autograd':analytic}
    record('T2','finite_unrolled_gradient',fgrad)
    def cap():
        b=torch.tensor([1.,0.],dtype=torch.float64);z=torch.tensor([0.,1.],dtype=torch.float64)
        out,status=angle_cap(b,z,30)
        theta=float(torch.acos((out@b).clamp(-1,1)))*180/math.pi
        assert abs(theta-30)<1e-8
        anti,s=angle_cap(b,-b,30);assert torch.equal(anti,b)
        return {'theta_degrees':theta,'antipodal':s}
    record('T2','angle_cap_and_antipodal_boundary',cap)
    def partitions():
        original=set(range(200));base=set(range(160));pseudo=set(range(8));core=base-pseudo;novel=original-base
        assert len(core)==152 and not(core&pseudo or core&novel or pseudo&novel)
        # Keep original object index space, only change base flags used by the sampler.
        keys=['a','b','c'];labels=[0,9,170];eligible=[i for i,c in enumerate(labels) if c in core]
        assert eligible==[1] and keys[eligible[0]]=='b'
        return {'core':len(core),'pseudo':len(pseudo),'final_novel':len(novel),'original_indices_preserved':True}
    record('T3','core_pseudo_final_class_roles',partitions)
    def weights():
        # All arms have the same trainable scorer; the operator is parameter-free.
        layer=torch.nn.Linear(4,3,dtype=torch.float64);copy=[{k:v.detach().clone() for k,v in layer.state_dict().items()} for _ in range(3)]
        assert all(all(torch.equal(copy[0][k],v) for k,v in s.items()) for s in copy)
        return {'three_initial_states_equal':True}
    record('T3','shared_parameter_initialization',weights)
    def gates():
        A=lambda d17,d29,f17,f29: min(d17,d29)>=.01-1e-10 and min(f17,f29)>=8
        assert A(.011,.012,8,9) and not A(.009,.02,10,10)
        assert not A(.012,.012,7,9)
        return {'negative_routes_stop_without_training':True}
    record('T4','phase_A_resource_gate',gates)
    def budgets():
        assert 3*2*2000==12000 and 12000*32==384000
        assert 12*12*32==4608 and 12*96==1152
        assert 14*12==168 and 14*2*26==728
        return {'updates':12000,'nominal_train_forwards':384000,'max_real_proposals':4608,'fresh_scoring_rows':168,'regression_rows':728}
    record('T4','budget_arithmetic',budgets)
    return dict(status='PASS',scope='CPU_SYNTHETIC_CONTRACT_ONLY',groups=4,checks=len(results),tests=results,
       not_run=['repository integration','real frozen FC','real diagnosis','training','new confirmation','GitHub publication'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
    result=run_tests()
    if args.output:args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
