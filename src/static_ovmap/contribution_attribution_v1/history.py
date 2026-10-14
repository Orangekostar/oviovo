"""Descriptive factorial increments; no natural-causal innovation shares."""
def factorial(p00,p10,p01,p11):
    e_add=p10-p00; e_cond=p11-p01; r_add=p01-p00; r_cond=p11-p10
    return dict(E_add=e_add,E_cond=e_cond,R_add=r_add,R_cond=r_cond,interaction=p11-p10-p01+p00,
                mean_marginal_E=(e_add+e_cond)/2,mean_marginal_R=(r_add+r_cond)/2,total=p11-p00)

def run(root):
    from pathlib import Path
    from .common import verified,write,METRICS,HISTORICAL,ConsumptionIndex,read_document,unit_identity,cached_record,canonical_digest
    from .binding import load_binding
    root=Path(root);binding=load_binding(root);inventory=verified(root/'inventory.json');index=ConsumptionIndex(root/'history/verifications.json')
    key=unit_identity('history',dict(inventory=inventory['identity']),producers=('common','history'),cohorts=binding['cohorts'])
    old=cached_record(root/'history.json',key)
    if old:return old
    effects=[];per_scene=[];controls=[];lineage=[];missing=[]
    for cohort,scenes in binding['cohorts'].items():
        pools=inventory['historical_pools'][cohort]
        if not all(a in pools for a in ('A0','A1','A2','A3')):
            missing.append(dict(cohort=cohort,reason='MISSING_MACHINE_PRECISION_FACTORIAL_POOL'));continue
        for alias,pooled in pools.items():
            if pooled['scene_order']!=scenes:raise ValueError('Historical pool scene order differs')
            controls.append(dict(cohort=cohort,method=alias,source_method=HISTORICAL[alias],metrics=pooled['metrics'],pool_identity=pooled['identity'],measurement_kind='REUSED_EXACT',comparability='F_ONLY_PLUS_SAME_G1_RECOVERY' if alias=='A5' else 'DEFINED_PARENT_RECIPE'))
        for metric in METRICS:
            values=[pools[a]['metrics'][metric] for a in ('A0','A1','A2','A3')]
            effects.append(dict(cohort=cohort,metric=metric,scope='COMPLETE_ORDERED_POOL',values=values,effects=factorial(*values) if all(v is not None for v in values) else None))
        for scene in scenes:
            rows={r['method']:r for r in inventory['historical_rows'] if r['scene']==scene}
            if not all(HISTORICAL[a] in rows for a in ('A0','A1','A2','A3')):
                missing.append(dict(scene=scene,reason='MISSING_HISTORICAL_FACTORIAL_SCENE'));continue
            for metric in METRICS:
                values=[rows[HISTORICAL[a]]['metrics'][metric] for a in ('A0','A1','A2','A3')]
                per_scene.append(dict(cohort=cohort,scene=scene,metric=metric,values=values,effects=factorial(*values) if all(v is not None for v in values) else None))
            area_lock=read_document(Path(binding['area_root'])/'predictions'/scene/'receipt.json',index)
            fg_lock=read_document(Path(binding['fg_parent_root'])/'predictions'/scene/'receipt.json',index)
            apath=area_lock['predictions']['CT_A3_ER'];bpath=fg_lock['methods']['REF_G1']['manifest']
            a=read_document(apath,index,sealed=False);b=read_document(bpath,index,sealed=False)
            for path,item in ((apath,a),(bpath,b)):
                index.identity(Path(path).parent/item['arrays']['path'],item['arrays'])
            equal=all(a[k]==b[k] for k in ('geometry','owner_ids','semantic_labels','instance_ranks','prediction_key'))
            lineage.append(dict(scene=scene,A3_manifest=apath,FG_REF_G1_manifest=bpath,exact_content_identity=equal,geometry_identity=a['geometry'],owner_partition_digest=a['owner_ids']['sha256'],semantic_digest=a['semantic_labels']['sha256'],rank_digest=canonical_digest(a['instance_ranks']),status='SAME_DOMAIN' if equal else 'DIFFERENT_DOMAIN_NO_GAIN_WATERFALL'))
    result=write(root/'history.json',dict(status='PARTIAL_DEPENDENCY' if missing else 'COMPLETE',input_identity=key,ER_effects=effects,scene_effects=per_scene,controls=controls,lineage=lineage,missing=missing,A5_is_F_plus_R=True,N_only_not_assumed_native_equal=True,mean_marginals_close_total=True,interpretation='DEFINED_SYSTEM_DESCRIPTIVE_MARGINALS_NOT_NATURAL_CAUSAL_SHARES',new_head_forwards=0,new_rgb_encodings=0))
    index.write_memo(root/'history/verifications.json');return result
