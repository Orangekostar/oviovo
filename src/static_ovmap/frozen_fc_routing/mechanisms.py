"""Post-lock mass/norm controls and target-sidecar coefficient diagnostics."""
from pathlib import Path
import numpy as np
import torch
from .common import arrays_record,write
from .models import interventions,route
from .diagnostics import score_record,summarize_rows


def calibration(scores,labels):
    p=np.asarray(scores,float);y=np.asarray(labels,int);positive=int(y.sum());negative=len(y)-positive
    if not positive or not negative:return dict(status='UNDEFINED_SINGLE_CLASS',known=len(y),positive=positive,negative=negative,accuracy=None,Brier=None,AUROC=None,bins=[])
    order=np.argsort(p,kind='stable');ranks=np.empty(len(p),float);start=0
    while start<len(p):
        stop=start+1
        while stop<len(p) and p[order[stop]]==p[order[start]]:stop+=1
        ranks[order[start:stop]]=(start+1+stop)/2;start=stop
    auc=(ranks[y==1].sum()-positive*(positive+1)/2)/(positive*negative)
    bins=[]
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        bins.append(dict(lower=i/10,upper=(i+1)/10,n=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,
                         target_frequency=float(y[mask].mean()) if mask.any() else None))
    return dict(status='DEFINED',known=len(y),positive=positive,negative=negative,
                accuracy=float(((p>=.5)==y).mean()),Brier=float(((p-y)**2).mean()),AUROC=float(auc),bins=bins)

class Collector:
    def __init__(self,binding,fc,loader,role,name,*,targets_available=True,base_only=False):
        self.binding,self.fc,self.loader=binding,fc,loader
        self.role,self.name,self.targets_available,self.base_only=role,name,targets_available,base_only
        self.rows=[];self.coefficients=[];self.probabilities=[];self.labels=[];self.cosines=[];self.embeddings=[]

    def __call__(self,obj,condition,x,result,score):
        if self.base_only and not obj['base']:return
        if result is None:return
        targets=None
        if self.targets_available:
            _,target=self.loader.targets(obj,condition,8);targets=target[result['view_indices']]
        row=dict(key=obj['key'],family=obj['family'],condition=condition,seed_method=self.name,role=self.role,
            groups=result['group_count'],local_observations=int(result['local_valid'].sum()),
            fallback=result['fallback'],membership_status='COMPUTED' if targets is not None else 'NOT_COMPUTED',
            purity_is_causal_attribution=False)
        if result['group_count']:
            a=result['alpha'];quality=result['quality_weights'];mean_alpha=a.mean(0)
            local_coeff=mean_alpha@quality
            row.update(local_mass=float(a.sum(-1).mean()),null_mass=float(result['null'].mean()),
                       per_query_local_mass=a.sum(-1).tolist(),per_query_null_mass=result['null'].tolist(),
                       delta_norm=float(result['delta'].norm()),group_norm_proxy=float((result['eta']*mean_alpha*result['values'].norm(dim=-1)).sum()))
            # The same chosen first physical group is halved; no outcome-based selection.
            rho=result['effective_rho'].clone();rho[0]*=.5;half,half_null=route(result['energies'],rho)
            if not bool((half[:,0]<=a[:,0]+1e-7).all()):raise ValueError('Own route increased after membership reduction')
            row.update(halved_group=int(result['group_ids'][0]),own_alpha_before=a[:,0].tolist(),own_alpha_after=half[:,0].tolist(),
                       null_after_half=half_null.tolist(),total_local_mass_after_half=half.sum(-1).tolist())
            if targets is not None:
                valid=result['local_valid'].flatten();y=targets.flatten()
                for label,code in [('positive',1),('negative',0),('unknown',-1)]:
                    mask=valid&(y==code);row[label+'_mass']=float(local_coeff[mask].sum());row[label+'_tokens']=int(mask.sum())
                    share=quality[:,mask].sum(-1)
                    row[label+'_group_norm_proxy']=float((result['eta']*mean_alpha*result['values'].norm(dim=-1)*share).sum())
                if abs(sum(row[k+'_mass'] for k in ('positive','negative','unknown'))-row['local_mass'])>2e-6:raise ValueError('Local coefficient target partition lost mass')
        else:
            row.update(local_mass=0.,null_mass=1.,delta_norm=0.,group_norm_proxy=0.)
            if targets is not None:
                row.update({k+'_mass':0. for k in ('positive','negative','unknown')})
                row.update({k+'_tokens':0 for k in ('positive','negative','unknown')})
        if targets is not None:
            known=result['local_valid']&(targets>=0)
            self.probabilities.extend(result['membership_logits'][known].sigmoid().tolist());self.labels.extend(targets[known].tolist())
        self.coefficients.append(row)
        if self.name.removeprefix('S29_')!='FG4_ROUTE':return
        control=interventions(result);ref_correct=int(self.fc.ids[int(score.argmax())]==obj['class_id'])
        candidates={'FG4_ROUTE':dict(embedding=result['embedding'],delta=result['delta'],paired_defined=True,status='ORIGINAL'),**control}
        norm_defined=control['MATCHED_MASS_AND_DELTA_NORM']['paired_defined']
        for name,c in candidates.items():
            value,sc=score_record(c['embedding'],self.fc,obj,condition,name,role=self.role,seed_method=self.name,
                paired_defined=c['paired_defined'],norm_pair_defined=norm_defined,intervention_status=c['status'],
                correction=int(not ref_correct and self.fc.ids[int((c['embedding']@self.fc.text_tensor.T).argmax())]==obj['class_id']),
                harm=int(ref_correct and self.fc.ids[int((c['embedding']@self.fc.text_tensor.T).argmax())]!=obj['class_id']))
            value.update(delta_norm=float(c['delta'].norm()) if c['delta'] is not None else None,
                         local_mass=float(c['alpha'].sum(-1).mean()) if 'alpha' in c else row['local_mass'],
                         null_mass=float(c['null'].mean()) if 'null' in c else row['null_mass'])
            if name=='MATCHED_MASS':
                value['max_mass_error']=float((c['alpha'].sum(-1)-result['alpha'].sum(-1)).abs().max()) if result['group_count'] else 0.
                value['max_null_error']=float((c['null']-result['null']).abs().max()) if result['group_count'] else 0.
                if value['max_mass_error']>2e-6 or value['max_null_error']>2e-6:raise ValueError('Matched-mass changed alpha/null total')
            if name=='MATCHED_MASS_AND_DELTA_NORM' and norm_defined:
                value['norm_error']=abs(value['delta_norm']-row['delta_norm'])
                if value['norm_error']>2e-6:raise ValueError('Matched-norm changed delta norm')
            self.rows.append(value);self.cosines.append(sc.cpu().numpy());self.embeddings.append(c['embedding'].cpu().numpy())

    def finish(self,directory):
        directory=Path(directory);summary=[]
        if self.rows:
            # Ordinary FG4 remains complete. Pairwise norm cells share the explicit validity mask.
            summary=summarize_rows([r for r in self.rows if r['method']!='MATCHED_MASS_AND_DELTA_NORM'])
            pairs=[r for r in self.rows if r['method'] in ('FG4_ROUTE','MATCHED_MASS_AND_DELTA_NORM') and r['norm_pair_defined']]
            summary += [dict(r,paired_scope='SHARED_NORM_DEFINED_ONLY') for r in summarize_rows(pairs)]
            arr=arrays_record(directory/'interventions.npz',dict(embeddings=np.asarray(self.embeddings,np.float32),
                full200_cosines=np.asarray(self.cosines,np.float32),keys=np.asarray([r['key'] for r in self.rows]),
                methods=np.asarray([r['method'] for r in self.rows]),conditions=np.asarray([r['condition'] for r in self.rows]),valid_ids=np.asarray(self.fc.ids)),self.fc.index)
            write(directory/'interventions.json',dict(status='COMPLETE',rows=self.rows,arrays=arr,used_for_selection=False,
                norm_undefined=sum(not r['norm_pair_defined'] for r in self.rows if r['method']=='FG4_ROUTE')))
        # Query mean is computed above; object mean then dataset mean, not token-weighted averaging.
        object_means={}
        names=('local_mass','null_mass','delta_norm','group_norm_proxy','positive_mass','negative_mass','unknown_mass')
        for r in self.coefficients:object_means.setdefault(r['key'],[]).append(r)
        means={k:float(np.mean([np.mean([r[k] for r in rows]) for rows in object_means.values()]))
               if object_means and all(k in r for r in self.coefficients) else None for k in names}
        write(directory/'coefficients.json',dict(status='COMPLETE',rows=self.coefficients,means=means,
            aggregation='QUERY_MEAN_THEN_CONDITION_MEAN_WITHIN_OBJECT_THEN_OBJECT_MEAN',proxy_not_causal=True))
        return write(directory/'receipt.json',dict(status='COMPLETE',method=self.name,role=self.role,summary=summary,
            original_objects=len(object_means),variant_records=len(self.coefficients),mass_means=means,
            membership=calibration(self.probabilities,self.labels) if self.targets_available else {'status':'NOT_COMPUTED'},
            GT_targets_forward=False,used_for_selection=False))
