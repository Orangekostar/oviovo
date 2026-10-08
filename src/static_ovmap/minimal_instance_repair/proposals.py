"""Exact spatial universe and shared positive-consensus clique-prefix library."""

from itertools import combinations
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.recovery_wave2.binding import read

from .binding import load_scene, seal
from .observations import pair_rates


def pair_key(left, right):
    return tuple(sorted((left,right)))


class SupportDistances:
    def __init__(self, xyz, units):
        self.xyz, self.units = np.asarray(xyz),units
        self.bounds = {name:(self.xyz[u.rows].min(0),self.xyz[u.rows].max(0)) for name,u in units.items()}
        self.trees, self.cache = {},{}

    def lower_bound(self, left, right):
        alo,ahi = self.bounds[left]
        blo,bhi = self.bounds[right]
        return float(np.linalg.norm(np.maximum(0,np.maximum(alo-bhi,blo-ahi))))

    def exact(self, left, right):
        key = pair_key(left,right)
        if key not in self.cache:
            a,b = sorted((left,right),key=lambda name:(len(self.units[name].rows),name))
            if b not in self.trees:
                self.trees[b] = cKDTree(self.xyz[self.units[b].rows])
            rows = self.units[a].rows
            best = np.inf
            for start in range(0,len(rows),32768):
                distance,_ = self.trees[b].query(self.xyz[rows[start:start+32768]],k=1,workers=1)
                best = min(best,float(np.min(distance)))
            self.cache[key] = best
        return self.cache[key]


def spatial_neighbors(inputs, settings):
    units, seeds = inputs.units.units,inputs.units.seeds
    metric = SupportDistances(inputs.xyz,units)
    hosts = [name for name,u in units.items() if u.kind == 'I']
    residuals = [name for name,u in units.items() if u.kind == 'C']
    neighbors, edges = {},{}
    for seed in seeds:
        unit = units[seed]
        old = inputs.d2.owner_ids[inputs.raw == unit.owner]
        names,counts = np.unique(old,return_counts=True)
        overlap = dict(zip(map(int,names),map(int,counts),strict=True))
        host_rows, residual_rows = [],[]
        for host in hosts:
            fraction = overlap.get(units[host].owner,0)/unit.raw_rows
            if metric.lower_bound(seed,host) > settings['neighbor_distance_m'] and fraction < settings['raw_overlap_candidate_fraction']:
                continue
            distance = metric.exact(seed,host)
            if distance <= settings['neighbor_distance_m'] or fraction >= settings['raw_overlap_candidate_fraction']:
                host_rows.append({'unit':host,'distance':distance,'raw_overlap_fraction':fraction})
        for other in residuals:
            if other == seed or metric.lower_bound(seed,other) > settings['neighbor_distance_m']:
                continue
            distance = metric.exact(seed,other)
            if distance <= settings['neighbor_distance_m']:
                residual_rows.append({'unit':other,'distance':distance,'raw_overlap_fraction':0.})
        host_rows.sort(key=lambda r:(r['distance'],r['unit']))
        residual_rows.sort(key=lambda r:(r['distance'],r['unit']))
        selected = (host_rows[:settings['max_incumbent_neighbors']]
                    +residual_rows[:settings['max_residual_neighbors']])
        neighbors[seed] = selected
        for row in selected:
            edges[pair_key(seed,row['unit'])] = row['distance']
    # Undirected union, without expanding a non-seed's own neighborhood.
    shared = {seed:[] for seed in seeds}
    for (left,right),distance in edges.items():
        for a,b in ((left,right),(right,left)):
            if a in shared:
                shared[a].append({'unit':b,'distance':distance})
    for rows in shared.values():
        rows.sort(key=lambda r:(r['distance'],r['unit']))
    return neighbors,shared,edges


def group_record(names, units, rates, *, separation=2., penalty=.10):
    names = tuple(sorted(names))
    pairs = [rates[pair_key(a,b)] for a,b in combinations(names,2)]
    total = sum(units[name].area for name in names)
    if total <= 0:
        return None
    incumbents = [units[name] for name in names if units[name].kind == 'I']
    if len(incumbents) > 1:
        raise ValueError('a repair hypothesis cannot merge two original incumbents')
    cost = (sum(units[name].area for name in names if units[name].kind == 'C')/total if incumbents
            else 1-max(units[name].area for name in names)/total)
    same = float(np.mean([p['same'] for p in pairs]))
    separate = float(np.mean([p['separate'] for p in pairs]))
    digest = canonical_digest({'units':list(names),'supports':[units[name].support_hash for name in names]})
    return {'units':list(names),'digest':digest,'host':incumbents[0].owner if incumbents else None,
            'support_hash':canonical_digest([units[name].support_hash for name in names]),
            'physical_area':total,'edit_cost':cost,'proposal_same':same,'proposal_separate':separate,
            'proposal_min_views':min(p['n'] for p in pairs),
            'proposal_score':same-separation*separate-penalty*cost}


def build_library(units, seeds, neighbors, proposal_rates, *, max_parent_id,
                  minimum_views=2, minimum_same=.60, max_units=4, cap=96,
                  separation=2., penalty=.10):
    positive = {key for key,row in proposal_rates.items()
                if row['n'] >= minimum_views and row['same'] >= minimum_same}
    groups = {}
    def emit(names):
        record = group_record(names,units,proposal_rates,separation=separation,penalty=penalty)
        if record is not None:
            groups[tuple(sorted(names))] = record
    for seed in seeds:
        eligible = [r for r in neighbors.get(seed,[]) if pair_key(seed,r['unit']) in positive]
        eligible.sort(key=lambda r:(-proposal_rates[pair_key(seed,r['unit'])]['same'],r['distance'],r['unit']))
        current = [seed]
        for row in eligible:
            name = row['unit']
            if len(current) == max_units:
                break
            if (sum(units[n].kind=='I' for n in current)+int(units[name].kind=='I') > 1
                    or any(pair_key(n,name) not in positive for n in current)):
                continue
            current.append(name)
            emit(current)
        for row in eligible:
            if units[row['unit']].kind == 'I':
                emit([seed,row['unit']])
    ordered = sorted(groups.values(),key=lambda r:(-r['proposal_score'],r['edit_cost'],r['digest']))[:cap]
    lex = {row['digest']:i for i,row in enumerate(sorted(ordered,key=lambda r:r['digest']))}
    for row in ordered:
        row['library_lex_index'] = lex[row['digest']]
        row['fresh_owner'] = max_parent_id+1+row['library_lex_index'] if row['host'] is None else None
        row['output_owner'] = row['host'] if row['host'] is not None else row['fresh_owner']
    return ordered


def bank_rates(observations, edges, bank, *, separation_dominance=.80):
    frames = [frame for frame in observations['frames'] if frame['bank']==bank and frame['panoptic_available']]
    return {key:pair_rates([(f['units'][key[0]],f['units'][key[1]]) for f in frames],
                            separation_dominance=separation_dominance) for key in edges}


def propose_scene(binding, scene):
    from .verification import structural_decisions
    inputs = load_scene(binding,scene)
    root = Path(binding['output_root'])
    observation = read(root/'observations'/scene/'receipt.json')
    support,repair = binding['specification']['support'],binding['specification']['repair']
    key = canonical_digest({'observer':observation['identity'],'supports':inputs.units.identity,
        'support':support,'repair':repair,'producer':inputs.index.identity(__file__),
        'verification':inputs.index.identity(Path(__file__).with_name('verification.py'))})
    path = root/'proposals'/scene/'receipt.json'
    if path.exists():
        old = read(path)
        if old['input_identity'] != key:
            raise ValueError('completed proposal inputs changed; invalidate only this scene descendants')
        return old
    begin = time.perf_counter()
    directed,neighbors,edges = spatial_neighbors(inputs,support)
    proposal = bank_rates(observation,edges,'proposal',separation_dominance=binding['specification']['observer']['separation_dominance'])
    verification = bank_rates(observation,edges,'verification',separation_dominance=binding['specification']['observer']['separation_dominance'])
    max_id = int(max(inputs.raw.max(initial=0),inputs.d2.owner_ids.max(initial=0),inputs.g1.owner_ids.max(initial=0)))
    library = build_library(inputs.units.units,inputs.units.seeds,neighbors,proposal,max_parent_id=max_id,
        minimum_views=repair['minimum_proposal_views_per_pair'],minimum_same=repair['minimum_same_rate'],
        max_units=support['max_units_per_hypothesis'],cap=support['max_group_hypotheses'],
        separation=repair['separation_penalty'],penalty=repair['edit_penalty'])
    decisions = structural_decisions(inputs.units,directed,proposal,verification,library,support,repair)
    result = seal({'status':'PROVISIONAL_STRUCTURE_LOCKED','scene':scene,'input_identity':key,
        'support_identity':inputs.units.identity,'observation_identity':observation['identity'],
        'directed_neighbors':directed,'undirected_neighbors':neighbors,
        'pairs':[{'units':list(pair),'distance':edges[pair],'proposal':proposal[pair],'verification':verification[pair]} for pair in sorted(edges)],
        'library':library,'decisions':decisions,'class_scores_read':False,'blocked':observation['blocked'],
        'seed_count':len(inputs.units.seeds),'candidate_edges':len(edges),
        'positive_edges':sum(r['n']>=repair['minimum_proposal_views_per_pair'] and r['same']>=repair['minimum_same_rate'] for r in proposal.values()),
        'proposal_unknown_pairs':sum(r['n']==0 for r in proposal.values()),
        'verification_unknown_pairs':sum(r['n']==0 for r in verification.values()),
        'elapsed_seconds':time.perf_counter()-begin,'new_neural_inference':0})
    atomic_write_json(path,result)
    inputs.index.write_memo(root/'inputs'/scene/'verifications.json')
    print('PROPOSED',scene,'edges',len(edges),'library',len(library),
          'operations',{k:len(v['selected']) for k,v in decisions.items()},flush=True)
    return result


def propose_study(binding):
    return [propose_scene(binding,scene) for names in binding['cohorts'].values() for scene in names]
