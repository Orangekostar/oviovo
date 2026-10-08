"""Prespecified arbitration; held views cannot change the common proposal order."""

from itertools import combinations

import numpy as np

from .proposals import group_record, pair_key


def verification_score(group, rates, *, minimum_views=2, minimum_same=.60,
                       maximum_separate=.20, minimum_score=.05, separation=2., penalty=.10):
    pairs = [rates.get(pair_key(a,b),{'n':0,'same':None,'separate':None})
             for a,b in combinations(group['units'],2)]
    if any(p['n'] < minimum_views for p in pairs):
        return {'status':'KEEP_INSUFFICIENT_VERIFICATION','minimum_views':min(p['n'] for p in pairs)}
    same = float(np.mean([p['same'] for p in pairs]))
    separate = float(np.mean([p['separate'] for p in pairs]))
    score = same-separation*separate-penalty*group['edit_cost']
    passed = same >= minimum_same and separate <= maximum_separate and score >= minimum_score
    return {'status':'VERIFICATION_ACCEPTED' if passed else 'KEEP_VERIFICATION_REJECTED',
            'same':same,'separate':separate,'score':score,'minimum_views':min(p['n'] for p in pairs)}


def select_groups(library, verification_rates, *, verified, max_operations=8, **thresholds):
    selected,ledger,used = [],[],set()
    for group in library:
        result = verification_score(group,verification_rates,**thresholds) if verified else {'status':'DIRECT_PROPOSAL'}
        if result['status'].startswith('KEEP_'):
            status = result['status']
        elif not used.isdisjoint(group['units']):
            status = 'KEEP_CONFLICT'
        elif len(selected) == max_operations:
            status = 'KEEP_OPERATION_CAP'
        else:
            status = 'SELECTED_PROVISIONAL'
            selected.append(group)
            used.update(group['units'])
        ledger.append({'digest':group['digest'],'units':group['units'],'status':status,
                       'verification':result if verified else None})
    return selected,ledger


def _greedy(proposals, cap):
    return select_groups(proposals,{},verified=False,max_operations=cap)


def structural_decisions(unit_set, directed_neighbors, proposal, verification, library, support, repair):
    units = unit_set.units
    nearest,evidence = [],[]
    for seed in unit_set.seeds:
        hosts = [row for row in directed_neighbors[seed] if units[row['unit']].kind == 'I']
        if hosts:
            host = min(hosts,key=lambda r:(r['distance'],r['unit']))
            # Nearest attachment consumes no 2D votes; zero area does not remove it.
            names = sorted([seed,host['unit']])
            from static_ovmap.module_validation.contracts import canonical_digest
            total = sum(units[n].area for n in names)
            nearest.append({'units':names,'seed':seed,'host':units[host['unit']].owner,
                'output_owner':units[host['unit']].owner,'fresh_owner':None,'distance':host['distance'],
                'digest':canonical_digest({'units':names,'supports':[units[n].support_hash for n in names]}),
                'edit_cost':units[seed].area/total if total else None,'proposal_score':None})
        eligible = []
        for host in hosts:
            rates = proposal[pair_key(seed,host['unit'])]
            if (rates['n'] < repair['minimum_proposal_views_per_pair'] or rates['same'] < repair['minimum_same_rate']
                    or rates['separate'] > repair['maximum_separate_rate']):
                continue
            record = group_record([seed,host['unit']],units,proposal,
                                  separation=repair['separation_penalty'],penalty=repair['edit_penalty'])
            if record is not None and record['proposal_score'] >= repair['minimum_repair_score']:
                record.update(seed=seed,distance=host['distance'],output_owner=record['host'],fresh_owner=None)
                eligible.append(record)
        if eligible:
            evidence.append(min(eligible,key=lambda r:(-r['proposal_score'],r['distance'],r['units'])))
    nearest.sort(key=lambda r:(r['distance'],r['seed'],f"I:{r['host']}"))
    evidence.sort(key=lambda r:(-r['proposal_score'],r['edit_cost'],r['digest']))
    result = {}
    for name,groups in [('IR02_NEAREST_ATTACH',nearest),('IR03_EVIDENCE_ATTACH',evidence)]:
        selected,ledger = _greedy(groups,support['max_applied_operations'])
        result[name] = {'selected':selected,'ledger':ledger,'proposal_count':len(groups)}
    thresholds = {'minimum_views':repair['minimum_verification_views_per_pair'],
        'minimum_same':repair['minimum_same_rate'],'maximum_separate':repair['maximum_separate_rate'],
        'minimum_score':repair['minimum_repair_score'],'separation':repair['separation_penalty'],'penalty':repair['edit_penalty']}
    for name,verified in [('IR04_DIRECT_GROUP',False),('IR05_VERIFIED_REPAIR',True)]:
        selected,ledger = select_groups(library,verification,verified=verified,
                                       max_operations=support['max_applied_operations'],**thresholds)
        result[name] = {'selected':selected,'ledger':ledger,'proposal_count':len(library)}
    return result
