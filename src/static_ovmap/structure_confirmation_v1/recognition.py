"""DEV scoring reuses exact parent arithmetic; G receives no object identifiers."""
import torch
from static_ovmap.frozen_fc_routing.recognition import evaluate as parent_evaluate
from .grouping import method_inputs
from .common import CONDITIONS

def evaluate(head,method,loader,items,fc,budget,*,save_path,reference=None):
    class Adapter(torch.nn.Module):
        def __init__(self):super().__init__();self.reader=head;self.position=0;self.audits=[]
        def forward(self,x,phi,**kwargs):
            obj=selected[self.position//4];condition=CONDITIONS[self.position%4]
            inputs,audit=method_inputs(method,x,obj['key'],condition)
            if audit is not None:self.audits.append(audit)
            budget.consume('main_head',persist=False);self.position+=1
            return self.reader(inputs,phi,**kwargs)
    selected=[o for o in items if o['base']];adapter=Adapter()
    result=parent_evaluate(adapter,loader,selected,fc,save_path=save_path,reference=reference)
    budget.flush()
    if adapter.position!=len(selected)*4:raise ValueError('DEV invocation coverage differs')
    return result
