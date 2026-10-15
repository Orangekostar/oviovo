"""Evidence checks for the primary agent's final contract review."""
from pathlib import Path
import subprocess
import numpy as np
from .common import REPO,verified,write,ConsumptionIndex,digest_file,CHECKPOINTS,METHODS

def group_correspondence(binding):
    """Count retained physical pairs from saved mappings without another forward."""
    from collections import defaultdict
    from itertools import combinations
    root=Path(binding['output_root']);rows=[];locks={}
    for domain,relative in [('REAL','confirmation/real_prediction_lock.json'),('GT','confirmation/GT/prediction_lock.json')]:
        lock=verified(root/relative);locks[domain]=lock['identity'];totals=defaultdict(lambda:defaultdict(int))
        for pointer in lock['records']:
            record=verified(pointer['path']);seen={}
            for coverage in record['coverage']:
                audit=coverage.get('G_audit')
                if audit is None:continue
                prefix=lock['columns'][coverage['column']]['prefix'];key=(coverage['condition'],prefix)
                if key in seen:
                    if seen[key]!=audit:raise ValueError('G mapping differs by model seed/step')
                    continue
                seen[key]=audit;counts=totals[key];counts['object_conditions']+=1
                groups=defaultdict(list)
                for pairs in audit['source_slots']:
                    for destination,source in pairs:groups[destination].append(source)
                counts['valid_tokens']+=audit['valid_tokens'];counts['same_slot_tokens']+=audit['retained_same_slot_tokens']
                for sources in groups.values():
                    counts['groups']+=1
                    if len(sources)<2:counts['singleton_groups']+=1;continue
                    counts['multiview_groups']+=1;counts['fully_coherent_multiview_groups']+=int(len(set(sources))==1)
                    for left,right in combinations(sources,2):
                        counts['cross_view_pairs']+=1;counts['same_physical_site_pairs']+=int(left==right)
        for (condition,prefix),counts in sorted(totals.items()):
            for name in ('groups','singleton_groups','multiview_groups','fully_coherent_multiview_groups','cross_view_pairs','same_physical_site_pairs'):
                counts.setdefault(name,0)
            rows.append(dict(domain=domain,condition=condition,prefix=prefix,**counts,
                retained_physical_pair_fraction=counts['same_physical_site_pairs']/counts['cross_view_pairs'] if counts['cross_view_pairs'] else None,
                fully_coherent_multiview_group_fraction=counts['fully_coherent_multiview_groups']/counts['multiview_groups'] if counts['multiview_groups'] else None))
    return write(root/'engineering/G_correspondence.json',dict(status='COMPLETE',rows=rows,prediction_locks=locks,
        definition='Within each resulting G group, compare canonical source site indices across every pair of valid views; singleton groups have no pair denominator',
        model_seed_and_step_duplicates_counted_once=True,new_head_calls=0,labels_read=False,
        source=digest_file(Path(__file__))))

def check(binding):
    root=Path(binding['output_root']);index=ConsumptionIndex(root/'review/verifications.json');checks=[]
    def require(name,condition,evidence):
        if not condition:raise ValueError('Final evidence audit failed: '+name)
        checks.append(dict(name=name,status='PASS',evidence=evidence))
    role=verified(root/'role_lock.json');require('Fixed CONF12 roles',len(role['roles']['CONF'])==12 and role['confirmation_plan_blob']==binding['task_specification']['confirmation']['plan_git_blob'],'role_lock.json')
    training=verified(root/'training/receipt.json');require('Sixteen mandatory fresh arms',len(training['arms'])==16 and training['nominal_updates']==32000 and training['frozen_before']==training['frozen_after'],'training/receipt.json')
    initial={};actual_updates=0;state_files=[]
    for seed in (17,29):
        for method in METHODS:
            receipt=verified(root/f'training/seed{seed}/{method}/receipt.json');require('Five permanent actual states '+method+' '+str(seed),[r['step'] for r in receipt['validations']]==list(CHECKPOINTS) and receipt['completed_steps']==2000,str(Path(f'training/seed{seed}/{method}/receipt.json')))
            actual_updates+=receipt['completed_steps']
            if method!='MV_ATTN':initial.setdefault(seed,set()).add(receipt['initial_model_digest'])
            for r in receipt['validations']:
                index.identity(r['model']['path'],r['model']);dev=verified(r['path']);require('Exact DEV original/condition support '+method+' '+str(seed)+' '+str(r['step']),dev['original_objects']==69 and dev['variant_records']==276,r['path'])
                index.identity(dev['arrays']['path'],dev['arrays'])
                with np.load(dev['arrays']['path'],allow_pickle=False) as a:require('Full200 DEV scores '+method+' '+str(seed)+' '+str(r['step']),a['full200_cosines'].shape==(69,4,200),dev['arrays'])
                state_files.append(r['model'])
    require('FG initialization paired within both seeds',all(len(v)==1 for v in initial.values()) and actual_updates==32000,'sixteen training receipts')
    selection=verified(root/'model_selection_lock.json')
    from .selection import choose_step
    for method in METHODS:
        winner=choose_step(selection['curves'][method+':seed17'])['step']
        require('Seed29 does not reselect '+method,all(selection['arms'][method+':seed'+str(seed)]['selected']['step']==winner for seed in (17,29)),'model_selection_lock.json')
    real=verified(root/'confirmation/real_score.json');gt=verified(root/'confirmation/GT/score.json');maps=verified(root/'map/score.json');stats=verified(root/'statistics/receipt.json');cost=verified(root/'cost/receipt.json');budget=verified(root/'workload.json')
    require('CONF420/35 official sparse scores',real['scene_rows_n']==420 and real['pools_n']==35 and all(r['trace_parity']['ap_exact'] and r['trace_parity']['pr_and_fn_exact'] for r in real['pools']),'confirmation/real_score.json')
    require('Original GT denominator and true heldout',gt['full_original_denominator'] and gt['heldout_CE160'] is None and gt['competition_classes']==200,'confirmation/GT/score.json')
    require('Whole-map572/44 coverage',maps['scene_method_coverage']==572 and maps['full_cohort_pool_coverage']==44,'map/score.json')
    require('Exactly4000 paired family draw sets',stats['draw_sets']==4000 and not stats['old_attribution_bootstrap_rerun'] and all(r['two_seeds_not_concatenated'] and r['same_draws_GT_and_real'] for r in stats['pairs']),'statistics/receipt.json')
    require('Cost3648 calls with nested head timings',cost['head_calls']==3648 and cost['head_time_nested_in_total'] and cost['new_RGB_encodings']==0,'cost/receipt.json')
    require('All actual resource counters inside frozen limits',all(budget['counts'][k]<=limit for k,limit in budget['limits'].items()),'workload.json')
    require('No missing nominal training/DEV calls',budget['counts']['main_head']>=1024000+22080 and budget['counts']['optimizer_updates']>=32000,'workload.json')
    require('All old source files unchanged',not subprocess.check_output(['git','diff','--name-only',binding['task_source_base'],'--','src/static_ovmap/frozen_fc_routing','src/static_ovmap/learned_object_readout','src/static_ovmap/preservation_routing','src/static_ovmap/contribution_attribution_v1'],cwd=REPO),'git diff against sealed BASE for inherited numerical modules')
    tests=verified(root/'engineering/focused_tests.json');require('Four required focused groups',all(v=='PASS' for v in tests['groups'].values()),'engineering/focused_tests.json')
    gradient=verified(root/'engineering/real_FC_gradient.json');require('Real FC phi input derivative with frozen backbone',gradient['ordinary_tensor'] and gradient['phi_input_gradient_max']>0 and not gradient['frozen_FC_parameter_gradients'] and not gradient['membership_parameter_gradients'],'engineering/real_FC_gradient.json')
    correspondence=group_correspondence(binding);require('Measured physical-pair residue without extra forwards',bool(correspondence['rows']) and correspondence['new_head_calls']==0,'engineering/G_correspondence.json')
    index.write_memo(root/'review/verifications.json')
    return write(root/'review/automated_evidence.json',dict(status='COMPLETE',checks=checks,permanent_model_files=state_files,source_binding=binding['identity'],scope='Concrete automated evidence checks; primary source/diff/visual review remains separate',primary_review_performed=False))
