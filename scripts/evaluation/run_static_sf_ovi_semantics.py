#!/usr/bin/env python3
"""CPU-only cached SF-to-OVI class transfer; no GT or image/model access."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
from src.static_ovmap.cache_io import sha256_file
from src.static_ovmap.cached_semantic_transfer import METHODS,condition_receiver_sets,digest,payload_key,native_suggestions,aligned_quality,pair_intersections,donor_choice,support_gate,assert_fixed


def read(p):return json.loads(Path(p).read_text())


def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


def identity(p):
    p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha256_file(p)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=read(args.config);start=time.perf_counter()
    assert cfg['sf_support_gate']['support_threshold']==2/3 and cfg['reverse_correspondence']['min_directional_coverage']==.5
    out=Path(cfg['new_output_root'])/'predictions';out.mkdir(parents=True,exist_ok=False);write(out/'frozen_config.json',cfg)
    old=read(ROOT/cfg['parent_configs'][0]);od=read(ROOT/cfg['parent_configs'][1]);prior=Path(cfg['recorded_roots_to_verify']['previous_od_root'])
    oldbinding=read(prior/'predictions/input_binding.json');oldlarge=read(ROOT/'artifacts/static_ovmap/object_decoupling_v1/large_artifacts.json')
    inventory={};textpath=Path(od['runtime_assets_to_resolve']['native_text_cache']);textmeta=read(textpath.with_suffix('.json'))
    inventory[str(textpath)]=identity(textpath)
    if inventory[str(textpath)]['sha256']!=textmeta['output_sha256']:raise ValueError('text order identity mismatch')
    valid_ids=np.load(textpath)['valid_ids'].tolist()
    if valid_ids!=textmeta['valid_ids']:raise ValueError('text class order differs')
    source_identity={str(p.relative_to(ROOT)):identity(p) for p in [Path(__file__),ROOT/'src/static_ovmap/cached_semantic_transfer.py']}
    allrows=[];funnels=[];stages=[];coordinate=None
    for run in cfg['runs']:
        stage=time.perf_counter();pool=Path(cfg['recorded_roots_to_verify']['frozen_prediction_root'])/run;dest=out/run;dest.mkdir()
        for name in ['AT_O_AREA.json','AT_U00.json','candidates.json','masks.npy','coord.npy']:
            p=pool/name;inventory[str(p)]=identity(p)
            expected=oldbinding['inventory'].get(str(p))
            if expected and expected['sha256']!=inventory[str(p)]['sha256']:raise ValueError('frozen bank identity changed')
        baseline=read(pool/'AT_O_AREA.json');records=read(pool/'candidates.json');masks=np.load(pool/'masks.npy',mmap_mode='r');coords=np.load(pool/'coord.npy',mmap_mode='r')
        owners=np.load(baseline['owner_path']);inventory[baseline['owner_path']]=identity(baseline['owner_path'])
        if coordinate is not None and not np.array_equal(coordinate,coords):raise ValueError('two-run coordinate order mismatch')
        coordinate=coords;receivers=baseline['kept'];donors=[i for i,r in enumerate(records) if r['source']=='SpaCeFormer']
        for i in receivers:
            if records[i]['source']!='OVI' or records[i]['canonical_index']!=i or not np.array_equal(masks[i],owners==i+1):raise ValueError('OVI canonical mask/owner mismatch')
        t0ref=next(r for r in old['runs'] if r['id']==run);t0path=Path(t0ref['t0']);inventory[str(t0path)]=identity(t0path);inventory[t0ref['receipt']]=identity(t0ref['receipt'])
        confirmation=read(ROOT/'artifacts/static_ovmap/object_decoupling_v1/final_asset_confirmation.json')
        if inventory[str(t0path)]['sha256']!=next(a['sha256'] for a in confirmation['assets'] if a['path']==str(t0path)):raise ValueError('T0 changed')
        t0=np.load(t0path);quality=aligned_quality(records,t0)
        if not np.array_equal(coords,t0['coord']):raise ValueError('T0 coordinates changed')
        t0masks=t0['masks']
        for i in donors:
            if not np.array_equal(masks[i],t0masks[records[i]['original_t0_row']]):raise ValueError('SF canonical/query mask alignment failed')
        sempath=prior/'predictions'/run/'semantics.json';native={};native_blocker=None
        if sempath.exists():
            inventory[str(sempath)]=identity(sempath)
            if inventory[str(sempath)]['sha256']!=next(a['sha256'] for a in oldlarge['artifacts'] if a['path']==str(sempath)):raise ValueError('native score records changed')
            saved=read(sempath)
            for target in saved['targets']:
                for view in target['views']:
                    if view['feature_space_id']!=textmeta['feature_space_id']:raise ValueError('native feature space differs')
            native=native_suggestions(saved['targets'],owners,receivers,valid_ids)
        else:native_blocker='missing saved native score records'
        binding_seconds=time.perf_counter()-stage;stage=time.perf_counter()
        table=pair_intersections(owners,masks,receivers,donors,cfg['computation']['point_chunk'])
        np.savez(dest/'reverse_correspondence.npz',intersections=table,receivers=receivers,donors=donors)
        areas=np.array([r['source_area'] for r in records]);hashes={i:digest(masks[i]) for i in donors+receivers}
        donor_registry=[{**records[i],'mask_hash':hashes[i],'quality':quality[i]} for i in donors];write(dest/'donor_registry.json',donor_registry)
        labels={m:list(baseline['labels']) for m in METHODS};rows=[]
        for rindex,i in enumerate(receivers):
            alternatives=[{'canonical_index':j,'query_id':records[j]['spaceformer_query_id'],'original_t0_row':records[j]['original_t0_row'],
                 'class_id':records[j]['original_class_id'],'intersection':int(table[rindex,col]),'area':int(areas[j]),'quality':quality[j],'mask_hash':hashes[j]} for col,j in enumerate(donors)]
            selected,eligible,ties=donor_choice(int(areas[i]),alternatives);original=baseline['labels'][i];gate=support_gate(original,selected,eligible)
            n=native.get(i);sa=n['proposed_label'] if n and n['proposed_label'] is not None else original
            sb=selected['class_id'] if selected else original;sc=sb if gate['accepted'] else original
            for method,label in zip(METHODS,[sa,sb,sc]):labels[method][i]=label
            row={'run':run,'receiver':i,'native_owner_id':records[i]['native_owner_id'],'old_class':original,
                 'source_mask_hash':hashes[i],'source_area':int(areas[i]),'baseline_rank_string':baseline['rank_scores_serialized'][i],
                 'cached_E1_target_present':n is not None,'exact_native_mask_match':n is not None,'native':n,
                 'eligible_donors':eligible,'selected_donor':selected,'geometric_ties':ties,'gate':gate,
                 'final_labels':{m:labels[m][i] for m in METHODS},'changed':{m:labels[m][i]!=original for m in METHODS}}
            rows.append(row)
        allrows.extend(rows);write(dest/'receiver_decisions.json',rows)
        routing=Counter(r['selected_donor']['canonical_index'] for r in rows if r['selected_donor'])
        write(dest/'one_to_many.json',{str(k):v for k,v in routing.items() if v>1})
        binding={'run':run,'source_inputs':inventory.copy(),'valid_ids':valid_ids,'config':cfg,'code':source_identity,'receivers':rows}
        key=payload_key(binding)
        for method in METHODS:
            blocked=native_blocker if method==METHODS[0] else 'missing T0 quality fields' if method==METHODS[2] and any(v is None for v in quality.values()) else None
            receiver_sets=condition_receiver_sets(method,rows)
            doc={**{k:baseline[k] for k in ['kept','rank_scores','rank_scores_serialized','owner_path','ovi_readout_id']},
                 'labels':labels[method],'run':run,'condition':method,'status':'BLOCKED' if blocked else 'COMPLETE_PREDICTION','blocker':blocked,
                 'owner_sha256':inventory[baseline['owner_path']]['sha256'],'input_key':key,'method_key':payload_key({'binding':key,'method':method,'labels':labels[method]}),
                 'new_inference':0,**receiver_sets}
            assert_fixed(baseline,doc);write(dest/(method+'.json'),doc)
            available=[r for r in rows if (r['native'] and r['native']['proposed_label'] is not None) if method==METHODS[0]] if method==METHODS[0] else [r for r in rows if r['selected_donor']]
            proposed=[r for r in available if (r['native']['proposed_label'] if method==METHODS[0] else r['selected_donor']['class_id'])!=r['old_class']]
            funnel={'run':run,'method':method,'receivers':len(rows),'cohort':len(native) if method==METHODS[0] else len(available),
              'valid_suggestions':len(available),'same_label_suggestions':len(available)-len(proposed),'proposed_changes':len(proposed),
              'adopted_changes':len(doc['changed_receivers']),'abstentions':len(rows)-len(available),
              'gate_reasons':dict(Counter(r['gate']['reason'] for r in rows)) if method==METHODS[2] else {},
              'prior_gate_reasons':dict(Counter(r['native']['prior_gate_reason'] for r in rows if r['native'])) if method==METHODS[0] else {},
              'donor_count_distribution':dict(Counter(len(r['eligible_donors']) for r in rows)),
              'single_donor_changed_passes':sum(len(r['eligible_donors'])==1 and r['changed'][method] for r in rows) if method==METHODS[2] else None,
              'duplicate_groups_collapsed':sum(len(r['eligible_donors'])-len(r['gate']['groups']) for r in rows),
              'support_values':[r['gate']['support'] for r in rows if r['gate']['support'] is not None] if method==METHODS[2] else [],
              'SC_differs_from_SB':labels[METHODS[1]]!=labels[METHODS[2]]}
            funnels.append(funnel);print(run,method,'proposed',len(proposed),'changed',len(doc['changed_receivers']),flush=True)
        stages.append({'run':run,'binding_seconds':binding_seconds,'correspondence_and_decisions_seconds':time.perf_counter()-stage})
        if not (out/'real_cache_smoke.json').exists():write(out/'real_cache_smoke.json',{'receivers':rows[:3],'fixed_owner_and_masks_verified':True,'new_inference':0})
    write(out/'input_binding.json',{'inventory':inventory,'source':source_identity,'config':identity(args.config),'valid_ids':valid_ids,'feature_space_id':textmeta['feature_space_id'],'GT_opened':False})
    write(out/'intervention_funnel.json',funnels)
    write(out/'prediction_manifest.json',{'status':'COMPLETE','conditions':6,'new_inference':0,'new_image_forwards':0,'new_text_forwards':0,
          'raw_logits_read':False,'stages':stages,'seconds':time.perf_counter()-start,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'command':[sys.executable,*sys.argv],'source':source_identity})


if __name__=='__main__':main()
