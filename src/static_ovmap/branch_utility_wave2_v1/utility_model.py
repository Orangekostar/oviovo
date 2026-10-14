"""Fixed1345-parameter utility regressor and TRAIN-only standardization."""
import numpy as np
import torch
from torch import nn

class Normalizer:
    def __init__(self,mean,std):
        self.mean=np.asarray(mean,np.float32);self.std=np.asarray(std,np.float32)
        if self.mean.shape!=(24,) or self.std.shape!=(24,) or not np.isfinite([self.mean,self.std]).all() or (self.std<1e-6).any():raise ValueError('Invalid fixed24 normalizer')
    @classmethod
    def fit(cls,features):
        x=np.asarray(features,np.float64)
        if x.ndim!=2 or x.shape[1]!=24 or not len(x) or not np.isfinite(x).all():raise ValueError('TRAIN action rows required')
        return cls(x.mean(0),np.maximum(x.std(0),1e-6))
    def transform(self,features,without_new=False):
        x=(np.asarray(features,np.float32)-self.mean)/self.std
        if without_new:x=x.copy();x[...,16:]=0
        return x
    def record(self):return dict(mean=self.mean.tolist(),std=self.std.tolist(),fit_role='TRAIN_ONLY',std_ddof=0,std_floor=1e-6,rows='ALL_NONKEEP_TRAIN_ACTION_ROWS;KEEP_FIXED_ZERO_NO_FEATURES')

class UtilityRegressor(nn.Module):
    def __init__(self,seed):
        super().__init__();torch.manual_seed(int(seed)+2009)
        self.layers=nn.Sequential(nn.Linear(24,32),nn.GELU(),nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)
    def forward(self,x):return self.layers(x).squeeze(-1)
