import copy
import torch
from static_ovmap.semantic_update_evidence_v1.class_roles import draw_schedule
from static_ovmap.semantic_update_evidence_v1.stages import checkpoint_paths,seed29_lock
from static_ovmap.structure_confirmation_v1.checkpoints import save_resume,restore_resume

def test_original_index_sampler_and_checkpoint_contract():
    objects=[dict(key='excluded',class_id=99,supported=True),dict(key='a',class_id=3,supported=True),dict(key='b',class_id=4,supported=True)]
    a=draw_schedule(objects,[3,4],17,updates=2);b=draw_schedule(objects,[3,4],17,updates=2)
    assert (a==b).all() and set(a[:,0])=={1,2}
    assert len(checkpoint_paths())==30 and len(checkpoint_paths(seeds=(17,),steps=(1000,)))==3
    lock=seed29_lock(dict(step=500,recipe='x',seed=17));assert lock['step']==500 and not lock['reselected']

def test_completed_update_resume_next_step_equality(tmp_path):
    torch.manual_seed(3);m=torch.nn.Linear(3,2);o=torch.optim.AdamW(m.parameters());x=torch.randn(4,3)
    def step(model,opt):
        opt.zero_grad();model(x).square().sum().backward();opt.step()
    step(m,o);save_resume(tmp_path,m,o,seed=17,method='FREE',completed=1,input_identity='fixed',validations=[])
    n=copy.deepcopy(m);p=torch.optim.AdamW(n.parameters());s=restore_resume(tmp_path,n,p,input_identity='fixed')
    assert s['step']==1
    step(m,o);step(n,p)
    assert all(torch.equal(v,n.state_dict()[k]) for k,v in m.state_dict().items())
