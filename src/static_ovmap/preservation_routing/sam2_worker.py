"""Pinned eight-image AMG only; runs in the separate, unchanged SAM2 environment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);args=p.parse_args()
    job=json.loads(Path(args.job).read_text());sys.path.insert(0,job['repo'])
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();model=build_sam2(job['config'],job['checkpoint'],device='cuda',apply_postprocessing=False)
    model.eval().requires_grad_(False)
    if any(v.dtype!=torch.float32 for v in model.parameters()):raise ValueError('SAM2 must remain FP32')
    counts=dict(image_encoder_calls=0,mask_decoder_batches=0)
    def image_hook(*unused):counts['image_encoder_calls']+=1
    def mask_hook(*unused):counts['mask_decoder_batches']+=1
    ih=model.image_encoder.register_forward_hook(image_hook);mh=model.sam_mask_decoder.register_forward_hook(mask_hook)
    generator=SAM2AutomaticMaskGenerator(model,**job['settings']);rows={};torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for frame in job['frames']:
            image=np.asarray(Image.open(frame['rgb']['path']).convert('RGB')).copy();begin=time.perf_counter()
            masks=generator.generate(image);retained=[]
            for m in masks:
                mask=np.asarray(m['segmentation'],bool);pixels=int(mask.sum());box=m['bbox']
                if pixels<100 or box[2]<2 or box[3]<2:continue
                digest=hashlib.sha256(mask.tobytes()).hexdigest()
                retained.append(dict(mask=mask,predicted_iou=float(m['predicted_iou']),stability=float(m['stability_score']),
                                     pixels=pixels,digest=digest,bbox=list(map(float,box))))
            retained.sort(key=lambda r:(-r['predicted_iou'],-r['stability'],-r['pixels'],r['digest']))
            retained=retained[:16];key=frame['scene']+':'+str(frame['frame_id'])
            path=Path(job['output_root'])/'masks'/(key.replace(':','_')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,masks=np.stack([r['mask'] for r in retained]) if retained else np.empty((0,*image.shape[:2]),bool))
            rows[key]=dict(generated=len(masks),retained=len(retained),arrays=str(path),
                           metadata=[{k:v for k,v in r.items() if k!='mask'} for r in retained],elapsed_seconds=time.perf_counter()-begin)
            print('SAM2_REAL_PROPOSALS',key,len(masks),len(retained),flush=True)
    torch.cuda.synchronize();ih.remove();mh.remove()
    result=dict(status='COMPLETE',frames=rows,counts=counts,image_count=len(job['frames']),settings=job['settings'],
                no_GT_prompts=True,FP32=True,elapsed_seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
    Path(job['output_root'],'generator_raw.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
