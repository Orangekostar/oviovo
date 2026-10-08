"""Small isolated-worker I/O; no controller, geometry or annotation imports."""

from collections import Counter
from pathlib import Path
import hashlib
import json
import os
import platform
import tempfile
import time
import traceback

import numpy as np

from static_ovmap.module_validation.contracts import atomic_write_json,canonical_digest


def array_digest(value):
    value=np.asarray(value)
    return canonical_digest({'dtype':value.dtype.str,'shape':value.shape,
        'bytes_sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()})


def sealed(value):
    return {**value,'identity':canonical_digest({k:v for k,v in value.items() if k!='identity'})}


def file_record(path):
    path=Path(path).resolve();h=hashlib.sha256()
    with path.open('rb') as f:
        while data:=f.read(4*1024*1024):h.update(data)
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()}


def save_arrays(path,arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.npz');os.close(fd)
    try:
        np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
    return file_record(path)


def restore_tiles(tiles,query):
    """Bilinear each individual tile, then original H/W; never cross a seam."""
    import torch
    import torch.nn.functional as F
    if len(tiles)!=len(query['frames']):raise RuntimeError('incomplete model frame coverage')
    arrays={};canonical=[]
    for slot,(tile,frame) in enumerate(zip(tiles,query['frames'])):
        if tile.ndim!=2 or not torch.isfinite(tile).all():raise RuntimeError('nonfinite or invalid model logits')
        square=F.interpolate(tile[None,None].float(),size=(1024,1024),mode='bilinear',align_corners=False)
        restored=F.interpolate(square,size=frame['image_size_hw'],mode='bilinear',align_corners=False)
        cmask=(square[0,0]>=0).cpu().numpy();mask=(restored[0,0]>=0).cpu().numpy()
        arrays[f'canonical_mask_{slot}']=cmask;arrays[f'mask_{slot}']=mask;canonical.append(cmask)
    anchor=canonical[query['anchor_slot']];points=np.rint(query['points_xy_canonical']).astype(int).clip(0,1023)
    adherent=bool(all(anchor[y,x] for x,y in points))
    return arrays,{'anchor_adherent':adherent,'positive_prompt_pixels':points.tolist(),
        'empty_raw_frames':[i for i in range(len(tiles)) if not arrays[f'mask_{i}'].any()],
        'original_mask_pixels':[int(arrays[f'mask_{i}'].sum()) for i in range(len(tiles))]}


def run_worker(request_path,loader,infer,model_name):
    """Resident model batches; a failed model call never becomes an empty mask."""
    import torch
    request=json.loads(Path(request_path).read_text());dest=Path(request['receipt'])
    dest.parent.mkdir(parents=True,exist_ok=True);tasks=request['tasks'];counters=Counter();model=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('real CUDA worker unavailable')
        if len(tasks)==0:raise ValueError('no model loading for an empty query batch')
        torch.set_num_threads(4)
        model,load_receipt=loader(request,counters)
        hardware={'GPU':torch.cuda.get_device_name(0),'torch':torch.__version__,
                  'CUDA':torch.version.cuda,'python':platform.python_version(),
                  'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'inference_dtype':'bfloat16_autocast','model_weights_dtype':'float32'}
        rows=[]
        for task in tasks:
            query=task['query'];leaf=Path(task['dest']);leaf.mkdir(parents=True,exist_ok=True)
            start_counts=counters.copy();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.perf_counter()
            try:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    arrays,outcome,decoded=infer(model,query,counters)
                torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                peaks={'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                       'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
                # Hashing and parity/protocol checks are outside the microtiming boundary.
                if callable(decoded):decoded=decoded()
                decoded_ids=[array_digest(image) for image in decoded]
                expected=[f['canonical']['decoded_rgb_sha256'] for f in query['frames']]
                if decoded_ids!=expected:raise RuntimeError('segmentors did not decode the locked canonical JPEG pixels')
                arrays={k:(v.detach().float().cpu().numpy() if torch.is_tensor(v) else v) for k,v in arrays.items()}
                raw=save_arrays(leaf/'raw_masks.npz',arrays)
                row=sealed({'status':'COMPLETE','model':model_name,'scene':query['scene'],'owner':query['owner'],
                    'query_identity':query['identity'],'input_identity':task['input_identity'],
                    'kind':task['kind'],'ordered_frame_ids':query['frame_ids'],'raw_arrays':raw,
                    'outcome':outcome,'decoded_rgb_sha256':decoded_ids,'hardware':hardware,
                    'elapsed_seconds':elapsed,'timing_boundary':'canonical_JPEG_read_through_restored_binary_masks',
                    'model_load_in_timer':False,'cross_query_feature_cache':False,'result_cache':False,
                    'actual_counts':dict(counters-start_counts),**peaks})
                atomic_write_json(leaf/'receipt.json',row);rows.append(row)
                print('SEGMENTED',model_name,query['scene'],query['owner'],'anchor',outcome['anchor_adherent'],
                      'seconds',round(elapsed,3),'peak_GiB',round(peaks['peak_cuda_allocated_bytes']/2**30,3),flush=True)
                del arrays,decoded
            except Exception as exc:
                oom=isinstance(exc,torch.cuda.OutOfMemoryError)
                row=sealed({'status':'CUDA_OOM' if oom else 'EXECUTION_FAILED','model':model_name,
                    'scene':query['scene'],'owner':query['owner'],'query_identity':query['identity'],
                    'input_identity':task['input_identity'],'kind':task['kind'],
                    'elapsed_seconds':time.perf_counter()-begin,'actual_counts':dict(counters-start_counts),
                    'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()})
                atomic_write_json(leaf/'receipt.json',row);rows.append(row)
                atomic_write_json(dest,sealed({'status':row['status'],'model':model_name,'model_load':load_receipt,
                    'hardware':hardware,'rows':rows,'actual_counts':dict(counters)}));raise
        atomic_write_json(dest,sealed({'status':'COMPLETE','model':model_name,'model_load':load_receipt,
            'hardware':hardware,'rows':rows,'actual_counts':dict(counters),'no_unrecorded_warmup_calls':True}))
    except Exception as exc:
        if not dest.exists():atomic_write_json(dest,sealed({'status':'MODEL_LOAD_FAILED','model':model_name,
            'actual_counts':dict(counters),'error_type':type(exc).__name__,'error':str(exc),
            'traceback':traceback.format_exc()}))
        raise
