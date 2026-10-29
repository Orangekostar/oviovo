"""Only explicit real MV_ATTN matched1000 exports; no selected-map registry."""
from pathlib import Path
import numpy as np
from .common import (SEEDS, verified, write, save_arrays, ConsumedInputs,
                     canonical_digest, digest_file, StageFailure, source_identity)


def load_models(binding,*,device='cuda'):
    from static_ovmap.structure_confirmation_v1.export import load_export
    folder=Path(binding['structure_public']);manifest=verified(folder/'weights/manifest.json')
    index=ConsumedInputs(binding['output_root']);index.identity(folder/'weights/manifest.json')
    models={};records={}
    for seed in SEEDS:
        profile=f'MV_ATTN:seed{seed}:matched1000'
        if profile not in manifest['profiles']:
            raise StageFailure('Explicit matched1000 export profile is unavailable',missing=[profile])
        key=manifest['profiles'][profile];row=manifest['weights'][key]
        index.identity(folder/row['path'],row['transport'])
        if row['base']:
            base=manifest['weights'][row['base']];index.identity(folder/base['path'],base['transport'])
        model,state=load_export(folder,key,device=device)
        if (state['method'],state['seed'],state['step'])!=('MV_ATTN',seed,1000):
            raise ValueError('Actual exported state is not the required method/seed/step')
        if (row['method'],row['seed'],row['step'])!=('MV_ATTN',seed,1000):
            raise ValueError('Weight manifest disagrees with actual state')
        model.eval().requires_grad_(False);models[seed]=model
        records[str(seed)]=dict(profile=profile,method=state['method'],seed=state['seed'],
            step=state['step'],source_sha256=row['source_sha256'],source_bytes=row['source_bytes'],
            transport=row['transport'],export_path=str(folder/row['path']),
            architecture_source=digest_file(Path(__file__).parent.parent/'structure_confirmation_v1/models.py'))
    index.save()
    return models,records


def configure_torch():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.mha.set_fastpath_enabled(False)


def lock(binding,*,device='cuda'):
    """Load both true states and prove region-only parity on two real objects."""
    import torch
    from .region_inputs import CachedSparse,region_only
    from static_ovmap.frozen_fc_routing.models import FrozenFCBase
    root=Path(binding['output_root']);inherited=binding['inherited']
    if not inherited:
        raise StageFailure('Local lineage is required for actual readout input parity')
    lr=Path(inherited['lr_parent_root']);features=verified(lr/'features/regression.json')
    model_heads,weights=load_models(binding,device=device)
    loader=CachedSparse(binding,features);checks=[]
    examples=[inherited['cohorts']['replica8'][0],inherited['cohorts']['scannet_cf18'][0]]
    base_module=FrozenFCBase().to(device).eval()
    heads=0
    with torch.no_grad():
        for scene in examples:
            path=lr/'regression'/scene/'manifest.json'
            loader.index.identity(path);scene_meta=verified(path)
            obj=scene_meta['objects'][0]
            inputs,source=loader.load(obj,device=device)
            region=region_only(inputs)
            context_path=loader.resolver.resolve(inherited['scenes'][scene]['context']['path'])
            loader.index.identity(context_path,inherited['scenes'][scene]['context'])
            context=loader.resolver.rewrite(verified(context_path))
            loader.index.identity(context['FC_text']['path'],context['FC_text'])
            with np.load(context['FC_text']['path'],allow_pickle=False) as arrays:
                text=arrays['text_embeddings'];ids=arrays['valid_ids']
            b=base_module(inputs)
            for seed,model in model_heads.items():
                full=model(inputs,None);heads+=1
                reduced=model(region,None);heads+=1
                be=float((full['base']-reduced['base']).abs().max())
                ze=float((full['embedding']-reduced['embedding']).abs().max())
                if not torch.equal(b,full['base']) or be>1e-6 or ze>1e-6:
                    raise ValueError('Real region-only b/z parity failed')
                z=reduced['embedding'].cpu().numpy()
                if int(np.argmax(text@full['embedding'].cpu().numpy()))!=int(np.argmax(text@z)):
                    raise ValueError('Real region-only top1 parity failed')
                key=canonical_digest(dict(object=obj['key'],inputs=source['arrays']['sha256'],
                    weight=weights[str(seed)]['source_sha256'],prefix=8,condition='clean'))
                out=root/'readout_cache'/(key+'.npz')
                save_arrays(out,b=b.cpu().numpy(),z=z,valid_ids=ids)
                record=write(out.with_suffix('.json'),dict(status='DONE',object_key=obj['key'],
                    input_identity=key,weight_sha256=weights[str(seed)]['source_sha256'],seed=seed,
                    source_inputs=source,arrays=dict(path=str(out),bytes=out.stat().st_size,
                    sha256=digest_file(out)),created_by='G2_ACTUAL_REGION_PARITY'))
                checks.append(dict(scene=scene,object_key=obj['key'],seed=seed,full_input=source,
                    b_max_absolute_error=be,z_max_absolute_error=ze,top1_equal=True,
                    original_FC_text_dtype=str(text.dtype),cache_record=str(out.with_suffix('.json')),
                    cache_identity=record['identity']))
    loader.flush()
    value=write(root/'model_lock.json',dict(status='DONE',method='MV_ATTN',profile='matched1000',
        step=1000,seeds=list(SEEDS),prefix=8,condition='clean',weights=weights,
        anchor='static_ovmap.frozen_fc_routing.models.FrozenFCBase; same real input',
        dimension=768,eta=.5,projection_and_preprocessing='ORIGINAL_FROZEN_FC_UNCHANGED',
        original_FC_model_identity=features['FC_model'],FC_execution_config=inherited['assets']['execution_config'],
        calibrator=dict(kind='CAP',angle_degrees=10.,gamma=.025,gamma_domain='F cosine before source temperature'),
        region_only_enabled=True,actual_region_equivalence=checks,actual_head_calls=heads,
        actual_engineering_anchor_calls=2,new_image_encodings=0,new_training_updates=0,
        producer=source_identity(('registry','region_inputs')),
        scope_protocol_locks='Independent benchmark vocabulary/mapping locks; common model lock is not clean-benchmark approval'))
    print('MODEL_LOCK_DONE MV1000 seeds17,29; actual_region_parity4; head_calls='+str(heads),flush=True)
    return value,model_heads
