"""Task-bounded CONF-only FC capture with unchanged preprocessing and operators."""
from pathlib import Path
from types import SimpleNamespace
import shutil
import time
import numpy as np
import torch
from .common import verified,write,arrays_record,unit_key,existing,immutable_record,reserve,canonical_digest

def capture_frame(fc,frame,budget,*,planned_images):
    from PIL import Image
    from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor
    from static_ovmap.module_validation.native_capture import _array_digest
    root=Path(fc.binding['output_root']);fc.index.identity(frame['rgb']['path'],frame['rgb'])
    rgb=np.asarray(Image.open(frame['rgb']['path']).convert('RGB')).copy();image,resized=image_tensor(rgb,fc.device)
    tensor_key=canonical_digest(dict(model=fc.model_key,tensor=_array_digest(image.cpu().numpy())))
    key=canonical_digest(dict(raw=tensor_key,projected=fc.dense_operator,dtype='FP32'))
    path=root/'confirmation/features/grids'/(key+'.json')
    if path.exists():
        row=verified(path);fc.index.identity(row['arrays']['path'],row['arrays'])
        if row['physical_model_identity']!=fc.model_key:raise ValueError('Confirmation feature model changed')
        return row
    begin=time.perf_counter();hit=fc.raw_cache.lookup('dense',tensor_key)
    with torch.no_grad():
        if hit:
            with np.load(hit['arrays']['path'],allow_pickle=False) as a:dense=torch.from_numpy(a['dense'].copy()).to(fc.device)
        else:
            budget.consume('confirmation_FC');dense=fc.operators['extract_features_convnext'](SimpleNamespace(clip_model=fc.model),image)['clip_vis_dense'];fc.images+=1
        projected=fc.project_grid(dense)
        if dense.dtype!=torch.float32 or projected.dtype!=torch.float32 or dense.ndim!=4 or projected.shape[1]!=768:raise ValueError('CONF FC tensor shape/dtype differs')
        if not bool(torch.isfinite(dense).all() and torch.isfinite(projected).all()) or torch.is_inference(dense) or torch.is_inference(projected):raise ValueError('Ordinary finite FP32 cache tensors required')
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        estimate_path=root/'confirmation/features/storage_estimate.json';size=(dense.numel()+projected.numel())*4
        if not estimate_path.exists():
            free=shutil.disk_usage(root).free;estimate=size*planned_images
            write(estimate_path,dict(status='COMPLETE',measured_first_formal_frame=frame['rgb'],raw_shape=list(dense.shape),projected_shape=list(projected.shape),planned_images=planned_images,uncompressed_grid_bytes=estimate,available_bytes=free,reserve_bytes=20*2**30))
            if free<estimate+20*2**30:raise OSError('Formal CONF grid estimate violates storage reserve')
        reserve(root);arrays=arrays_record(path.with_suffix('.npz'),dict(raw=dense.cpu().numpy(),projected=projected.cpu().numpy()),fc.index)
    return write(path,dict(status='COMPLETE',key=key,arrays=arrays,raw_C=int(dense.shape[1]),projected_D=768,raw_hw=list(dense.shape[-2:]),original_hw=list(rgb.shape[:2]),resized_hw=list(resized),padded_hw=list(image.shape[-2:]),input_tensor_key=tensor_key,physical_model_identity=fc.model_key,dense_projection=fc.dense_operator,image=frame['rgb'],physical_image_encodings=int(hit is None),cache_origin='PARENT' if hit else 'ENCODED',extraction_seconds=elapsed,scope='CONFIRMATION_SHARED_ALL_METHODS',L2_normalized_dense=False))

def capture(binding,fc,budget):
    root=Path(binding['output_root']);geometry=verified(root/'confirmation/geometry_lock.json');inputs=verified(root/'confirmation/real_input_lock.json')
    identity=unit_key('confirm_features',dict(geometry=geometry['identity'],real_inputs=inputs['identity'],FC=fc.model_key,projection=fc.dense_operator),producers=('features',))
    path=root/'confirmation/features/manifest.json';old=existing(path,identity)
    if old:return old
    scenes=verified(root/'role_lock.json')['roles']['CONF'];bank=[f for s in scenes for f in verified(root/'confirmation/geometry'/s/'manifest.json')['frames']]
    if len(bank)>1152:raise ValueError('Confirmation frame bank exceeds1152')
    frames={};started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for f in bank:
        frames[f['scene']+':'+str(f['frame_id'])]=capture_frame(fc,f,budget,planned_images=len(bank))
        print('CONF_FC',f['scene'],f['frame_id'],len(frames),'/',len(bank),flush=True)
    fc.index.write_memo(root/'features/verifications.json')
    return immutable_record(path,dict(status='COMPLETE',input_identity=identity,frames=frames,planned_images=len(bank),FC_model=fc.model_key,physical_encodings=sum(r['physical_image_encodings'] for r in {f['key']:f for f in frames.values()}.values()),physical_encodings_this_invocation=fc.images,elapsed_seconds=time.perf_counter()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),all_features_before_annotation_parse=True,raw_C=1536,projected_D=768))
