"""Algebra-only reference. No FG2 assets, training, datasets, or map evaluation.
Run: python reference_branch_compose.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def unit(x: np.ndarray) -> np.ndarray:
    x=np.asarray(x,dtype=np.float64)
    n=np.linalg.norm(x)
    if not np.isfinite(x).all() or n<=1e-12:
        raise ValueError("finite nonzero vector required")
    return x/n

def compose(base: np.ndarray, groups: np.ndarray, alpha: np.ndarray, eta: float=.5) -> np.ndarray:
    """Use the supplied base in BOTH the base term and every local residual."""
    base=np.asarray(base,dtype=np.float64)
    groups=np.asarray(groups,dtype=np.float64)
    alpha=np.asarray(alpha,dtype=np.float64)
    if base.ndim!=1 or groups.ndim!=2 or groups.shape[1]!=base.size:
        raise ValueError("expected base[D], groups[J,D]")
    if alpha.ndim!=2 or alpha.shape[1]!=len(groups) or alpha.shape[0]<1:
        raise ValueError("expected alpha[Q,J]")
    if not 0<=eta<=.5 or not np.isfinite(groups).all() or not np.isfinite(alpha).all():
        raise ValueError("invalid eta or finite inputs")
    if np.any(alpha<0) or np.any(alpha.sum(1)>1+1e-12):
        raise ValueError("subprobability route required")
    if len(groups)==0:
        return base.copy()
    return unit(base+eta*(alpha@(groups-base)).mean(0))

def effects(h00:float,h10:float,h01:float,h11:float) -> dict[str,float]:
    b=.5*((h10-h00)+(h11-h01))
    l=.5*((h01-h00)+(h11-h10))
    return dict(base=b,local=l,interaction=h11-h10-h01+h00,total=h11-h00)

def self_test() -> dict:
    rng=np.random.default_rng(17)
    groups=np.stack([unit(x) for x in rng.normal(size=(7,12))])
    alpha=rng.uniform(size=(4,7)); alpha/=alpha.sum(1,keepdims=True)+1
    b=unit(rng.normal(size=12)); b2=unit(rng.normal(size=12))
    z=compose(b,groups,alpha)
    native=unit(b+.5*(alpha@(groups-b)).mean(0))
    np.testing.assert_allclose(z,native,rtol=0,atol=0)
    mass=alpha.sum(1).mean(); g=(alpha@groups).mean(0)
    np.testing.assert_allclose(z,unit((1-.5*mass)*b+.5*g),rtol=1e-12,atol=1e-12)
    correct=compose(b2,groups,alpha)
    wrong=unit(b2+.5*(alpha@(groups-b)).mean(0))
    assert np.linalg.norm(correct-wrong)>1e-5
    np.testing.assert_array_equal(compose(b,np.zeros((0,12)),np.zeros((4,0))),b)
    for _ in range(20):
        e=effects(*rng.normal(size=4))
        assert abs(e['base']+e['local']-e['total'])<1e-12
    np.testing.assert_allclose(compose(b,groups,alpha,0),b,rtol=1e-12,atol=1e-12)
    return {'status':'PASSED','checks':6,'scope':'ALGEBRA_ONLY_NO_REAL_FG2',
            'fg2_forwards':0,'new_training_updates':0,
            'checked':['native_formula','equivalent_decomposition','wrong_base_counterexample',
                       'empty_local','two_factor_sum','zero_eta']}

if __name__=='__main__':
    result=self_test()
    print(json.dumps(result,ensure_ascii=False,indent=2))
    Path(__file__).with_name('reference_selftest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
