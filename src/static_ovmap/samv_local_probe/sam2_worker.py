"""Pinned SAM2.1 Hiera-L, one independent chronological bidirectional track."""

import argparse
import time

import numpy as np

from .worker_io import restore_tiles,run_worker


def load_model(request,counters):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    cfg=request['specification']['assets']['sam2'];begin=time.perf_counter()
    model=build_sam2_video_predictor(cfg['config'],request['assets']['weights']['sam2_1_hiera_large']['path'],
        device='cuda',apply_postprocessing=False,vos_optimized=False,hydra_overrides_extra=cfg['hydra_overrides_extra'])
    torch.cuda.synchronize()
    if (model.fill_hole_area!=0 or not model.binarize_mask_from_pts_for_mem_enc
            or not model.sam_mask_decoder.dynamic_multimask_via_stability):
        raise RuntimeError('SAM2 baseline postprocessing overrides were not applied')
    def image_hook(module,args):
        counters['SAM2_encoder_calls']+=1;counters['SAM2_encoded_images']+=int(args[0].shape[0])
    model.image_encoder.register_forward_pre_hook(image_hook)
    model.sam_mask_decoder.register_forward_pre_hook(lambda m,a:counters.update(mask_decoder_calls=1))
    return model,{'model':'SAM2.1_HIERA_LARGE','seconds':time.perf_counter()-begin,
        'config':cfg['config'],'hydra_overrides_extra':cfg['hydra_overrides_extra'],
        'strict_checkpoint_coverage':'PINNED_BUILDER_REJECTS_MISSING_OR_UNEXPECTED_KEYS',
        'apply_postprocessing':False,'vos_optimized':False,'CUDA_extension_built':False,
        'load_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}


def infer(model,query,counters):
    import torch
    state=None
    try:
        state=model.init_state(query['canonical_directory'],async_loading_frames=False)
        n=len(query['frames'])
        if state['num_frames']!=n or state['video_height']!=1024 or state['video_width']!=1024:
            raise RuntimeError('SAM2 did not receive exactly the locked canonical JPEG window')
        model.add_new_points_or_box(state,frame_idx=query['anchor_slot'],obj_id=1,
            points=np.asarray(query['points_xy_canonical'],np.float32),labels=np.asarray(query['point_labels'],np.int32),
            normalize_coords=True)
        logits={};coverage={}
        counters['scientific_or_timing_tracks']+=1
        for direction in (False,True):
            visited=[];counters['propagation_passes']+=1
            for slot,obj_ids,scores in model.propagate_in_video(state,start_frame_idx=query['anchor_slot'],reverse=direction):
                if obj_ids!=[1] or scores.shape!=(1,1,1024,1024) or not torch.isfinite(scores).all():
                    raise RuntimeError('SAM2 returned missing object/nonfinite/incorrect frame logits')
                logits[int(slot)]=scores[0,0];visited.append(int(slot));counters['propagated_frame_outputs']+=1
            coverage['reverse' if direction else 'forward']=visited
        if set(logits)!=set(range(n)):raise RuntimeError('SAM2 forward/reverse propagation did not cover every window frame')
        tiles=[logits[i] for i in range(n)];arrays,outcome=restore_tiles(tiles,query)
        arrays['canonical_logits']=torch.stack(tiles)
        # Recover the decoded bytes from the actual upstream normalized image tensor.
        images=state['images']
        def decoded():
            mean=torch.tensor([.485,.456,.406],device=images.device).view(1,3,1,1)
            std=torch.tensor([.229,.224,.225],device=images.device).view(1,3,1,1)
            pixels=((images*std+mean)*255).round().clamp(0,255).to(torch.uint8)
            return list(pixels.permute(0,2,3,1).cpu().numpy())
        outcome.update(frame_coverage=sorted(logits),directional_coverage=coverage,
            same_anchor_both_directions=True,independent_state_per_target=True)
        return arrays,outcome,decoded
    finally:
        if state is not None:model.reset_state(state)
        del state


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    run_worker(parser.parse_args().request,load_model,infer,'SAM2')
