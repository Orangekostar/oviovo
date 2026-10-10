"""Local NumPy API adapter for the unchanged released scorer in the FC environment."""
import numpy as np

class ReleasedNumpy:
    def __getattr__(self,name):return getattr(np,name)
    @staticmethod
    def in1d(ar1,ar2,assume_unique=False,invert=False,*,kind=None):
        return np.isin(np.asarray(ar1).reshape(-1),ar2,assume_unique=assume_unique,invert=invert,kind=kind)

def adapt(namespace):
    if not hasattr(namespace['np'],'in1d'):
        namespace['np']=ReleasedNumpy()
    return namespace
