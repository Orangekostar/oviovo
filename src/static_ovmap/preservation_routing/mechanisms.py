"""Fixed post-lock interventions and denominators; none is a contender."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .common import arrays_record,objects,verified,write
from .models import teacher,absolute_route,unit

def interventions(binding,fc,loader,heads):
    root=Path(binding['output_root']);items=objects(binding,'dev');selected=[]
    for family in sorted({o['family'] for o in items}):
        selected.append(sorted([o for o in items if o['base'] and o['family']==family],key=lambda o:o['key'])[0])
    records={};preserve={}
    with torch.no_grad():
        for obj in selected:
            for cond in range(4):
                x,_,_=loader.load(obj,cond,8);_,membership=loader.targets(obj,cond,8)
                truth=obj['class_id'];t=teacher(x);fc_pred=fc.ids[int((t@fc.text_tensor.T).argmax())]
                for name,head in heads.items():
                    r=head(x,fc.phi);z=r['embedding'];prediction=fc.ids[int((z@fc.text_tensor.T).argmax())]
                    key=obj['key']+'|'+str(cond)+'|'+name
                    entry=dict(FC_prediction=fc_pred,prediction=prediction,truth=truth,teacher_correct=fc_pred==truth,
                               class_correction=fc_pred!=truth and prediction==truth,class_harm=fc_pred==truth and prediction!=truth,
                               no_op=('clean','truncate','append','truncate_append')[cond] in obj['no_op_conditions'])
                    if 'alpha' in r:
                        members=r['members'];valid=r['local_valid'];m=membership[r['view_indices']].flatten()
                        mass=r['alpha'].mean(0);weights=r['quality_weights']
                        negative=weights@(m==0).float();target=weights@(m==1).float();unknown=weights@(m<0).float()
                        interventions={}
                        for tag,rho in [('all_one',torch.ones_like(r['predicted_rho'])),('times_.1',r['predicted_rho']*.1),
                                        ('times_.01',r['predicted_rho']*.01),('all_zero',torch.zeros_like(r['predicted_rho']))]:
                            a,n=absolute_route(r['energies'],rho);delta=.5*head.output((a@r['values']).mean(0))
                            zi=unit(r['base_embedding']+delta)
                            interventions[tag]=dict(local_mass=float(a.sum(-1).mean()),null=float(n.mean()),delta_norm=float(delta.norm()),
                                            prediction=fc.ids[int((zi@fc.text_tensor.T).argmax())],embedding_change=float((zi-z).norm()))
                        rho=r['predicted_rho'].clone();rho[0]*=.1;a,n=absolute_route(r['energies'],rho)
                        interventions['first_group_lowered']=dict(group_index=0,alpha_before=float(r['alpha'][:,0].mean()),alpha_after=float(a[:,0].mean()))
                        entry.update(group_count=r['group_count'],positive_tokens=int((valid&(membership[r['view_indices']]==1)).sum()),
                                     negative_tokens=int((valid&(membership[r['view_indices']]==0)).sum()),
                                     unknown_tokens=int((valid&(membership[r['view_indices']]<0)).sum()),
                                     negative_coefficient_mass=float((mass*negative).sum()),target_coefficient_mass=float((mass*target).sum()),
                                     unknown_coefficient_mass=float((mass*unknown).sum()),
                                     negative_coefficient_value_norm_proxy=float((mass*negative*r['values'].norm(dim=-1)).sum()),
                                     null=float(r['null_weight'].mean()),interventions=interventions)
                        # Consistent complete-token site permutation in nonanchor views, independent of GT.
                        xp={k:v.clone() for k,v in x.items()};g=torch.Generator(device='cpu')
                        g.manual_seed(int(hashlib.sha256(obj['key'].encode()).hexdigest()[:16],16)%(2**63))
                        for view in range(1,len(x['view_valid'])):
                            perm=torch.randperm(x['local_valid'].shape[1],generator=g).to(fc.device)
                            for field in ('local_raw','local_unit','metadata','local_valid'):xp[field][view]=x[field][view][perm]
                        permuted=head(xp,fc.phi)['embedding']
                        entry['wrong_correspondence']=dict(prediction=fc.ids[int((permuted@fc.text_tensor.T).argmax())],
                                                          embedding_change=float((permuted-z).norm()),GT_chosen=False)
                        arr=arrays_record(root/'diagnostics/routing'/(hashlib.sha256(key.encode()).hexdigest()+'.npz'),
                                 {k:r[k].cpu().numpy() for k in ('predicted_rho','effective_rho','quality_weights','alpha','null_weight','delta_raw','delta','members','local_valid')},fc.index)
                        entry['arrays']=arr
                    records[key]=entry
    return write(root/'diagnostics/interventions.json',dict(status='COMPLETE',fixed_DEV_objects=[o['key'] for o in selected],
           records=records,intervention_values_predeclared=True,GT_after_predictor_lock=True,
           coefficient_norm_mass_is_proxy=True,does_not_claim_monotonic_accuracy=True,
           hypotheses_not_full_explanation=True))['identity']
