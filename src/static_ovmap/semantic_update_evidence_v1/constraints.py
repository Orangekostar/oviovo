"""Finite differentiable output operators; no exact intersection claim."""
import math
import torch

def simplex(x):
    u,_=torch.sort(x,descending=True)
    j=torch.arange(1,x.numel()+1,device=x.device,dtype=x.dtype)
    excess=u.cumsum(0)-1
    rho=torch.nonzero(u-excess/j>0,as_tuple=False)[-1,0]
    return torch.clamp(x-excess[rho]/j[rho],min=0)

def radius(b,q,r=.5):
    t=torch.clamp(r/torch.clamp(torch.linalg.vector_norm(q-b),min=1e-12),max=1.)
    return b+t*(q-b),t

def prepare_dictionary(D,a_ref):
    if D.ndim!=2 or not torch.isfinite(D).all():raise ValueError('Finite dictionary required')
    if not torch.allclose(D.norm(dim=-1),torch.ones(len(D),device=D.device),atol=1e-5,rtol=0):
        raise ValueError('Dictionary rows must be unit')
    if a_ref.shape!=(len(D),) or bool((a_ref<0).any()) or not torch.isclose(a_ref.sum(),a_ref.new_tensor(1.),atol=1e-6):
        raise ValueError('Invalid reference coefficients')
    with torch.no_grad():
        D=D.detach();K=D@D.T;L=torch.linalg.eigvalsh(K)[-1].clamp(min=1e-8)
    return dict(D=D,K=K,L=L,a_ref=a_ref.detach())

def evidence(q,dictionary,steps=128):
    D,K,L,a=dictionary['D'],dictionary['K'],dictionary['L'],dictionary['a_ref']
    rhs=D@q;y=a;t=1.
    for _ in range(steps):
        nxt=simplex(y-(K@y-rhs)/L);tn=(1+math.sqrt(1+4*t*t))/2
        y=nxt+((t-1)/tn)*(nxt-a);a=nxt;t=tn
    p=a@D;out,scale=radius(D[0],p)
    e0=torch.zeros_like(a);e0[0]=1
    coefficients=(1-scale)*e0+scale*a;g=K@a-rhs
    return dict(preunit=out,embedding=torch.nn.functional.normalize(out,dim=0),
                coefficients=coefficients,hull_coefficients=a,
                FW_gap=torch.clamp(a@g-g.min(),min=0),steps=steps,
                exact_intersection_projection=False)
