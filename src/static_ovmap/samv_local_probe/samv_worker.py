"""Released SAM-V primitives with strict base/trained tensor coverage."""

from pathlib import Path
import argparse
import sys
import time

import numpy as np

from .worker_io import restore_tiles,run_worker


def load_model(request,counters):
    import torch
    asset=request['assets'];directory=Path(asset['asset_root'])/'SAM-V'
    for path in reversed([directory,directory/'submodules/sam-hq',directory/'submodules/vggt']):
        sys.path.insert(0,str(path))
    import model.sam_vggt_model as upstream
    # The host repository owns a regular `utils` package; the author's utils
    # directory is a namespace package. Load this exact pinned file explicitly.
    import importlib.util
    checkpoint_spec=importlib.util.spec_from_file_location('samv_release_checkpoint',directory/'utils/checkpoint.py')
    checkpoint_module=importlib.util.module_from_spec(checkpoint_spec);checkpoint_spec.loader.exec_module(checkpoint_module)
    DEFAULT_TRAINABLE_PREFIXES=checkpoint_module.DEFAULT_TRAINABLE_PREFIXES
    load_partial_checkpoint=checkpoint_module.load_partial_checkpoint
    weights=asset['weights'];base_receipts=[];original=upstream.sam_model_registry['vit_h']

    def sam_builder(checkpoint):
        begin=time.perf_counter();model=original(checkpoint=None)
        state=torch.load(checkpoint,map_location='cpu')
        result=model.load_state_dict(state,strict=True)
        base_receipts.append({'model':'SAM_VIT_H','tensors':len(state),'missing':result.missing_keys,
            'unexpected':result.unexpected_keys,'all_shapes_checked':True,'seconds':time.perf_counter()-begin})
        del state;return model

    class CheckedSAMV(upstream.SamVGGT):
        def _load_vggt(self,checkpoint,device):
            begin=time.perf_counter();model=upstream.VGGT();raw=torch.load(checkpoint,map_location='cpu')
            state=raw.get('state_dict',raw.get('model',raw))
            state={k.removeprefix('module.'):v for k,v in state.items()}
            result=model.load_state_dict(state,strict=True);del state,raw
            model=model.to(device).eval();torch.cuda.synchronize()
            base_receipts.append({'model':'VGGT_1B','tensors':len(model.state_dict()),'missing':result.missing_keys,
                'unexpected':result.unexpected_keys,'all_shapes_checked':True,'seconds':time.perf_counter()-begin})
            return model

    begin=time.perf_counter();upstream.sam_model_registry['vit_h']=sam_builder
    try:
        model=CheckedSAMV(sam_model_type='vit_h',sam_checkpoint=weights['sam_vit_h_4b8939']['path'],
            vggt_checkpoint=weights['vggt_1b_model']['path'],vggt_img_size=896,sam_encode_chunk=1,
            device='cuda',freeze_sam_encoder=True,freeze_vggt=True)
    finally:upstream.sam_model_registry['vit_h']=original
    base_seconds=time.perf_counter()-begin;begin=time.perf_counter()
    checkpoint=torch.load(weights['sam_v_stage2']['path'],map_location='cpu')
    state=checkpoint['model_state_dict'];expected={k:v for k,v in model.state_dict().items() if k.startswith(DEFAULT_TRAINABLE_PREFIXES)}
    if set(state)!=set(expected) or any(state[k].shape!=expected[k].shape for k in expected):
        raise RuntimeError('stage-2 tensor names/shapes do not cover every fusion and decoder tensor')
    load_partial_checkpoint(model,checkpoint);trained_tensors=len(state);del state,checkpoint,expected
    model.eval();torch.cuda.synchronize();stage2_seconds=time.perf_counter()-begin
    def encoder_hook(module,args):
        counters['SAM_encoder_calls']+=1;counters['SAM_encoded_images']+=int(args[0].shape[0])
    model.sam.image_encoder.register_forward_pre_hook(encoder_hook)
    model.vggt.aggregator.register_forward_pre_hook(lambda m,a:counters.update(VGGT_joint_groups=1,VGGT_frame_slots=int(a[0].shape[1])))
    model.sam.mask_decoder.register_forward_pre_hook(lambda m,a:counters.update(mask_decoder_calls=1))
    return model,{'base_weights':base_receipts,'base_constructor_seconds':base_seconds,
        'stage2_seconds':stage2_seconds,'stage2_tensors':trained_tensors,
        'author_partial_loader':'utils.checkpoint.load_partial_checkpoint',
        'frozen_encoder_key_coverage':'STRICT_COMPLETE','trained_tensor_coverage':'EXACT_NAMES_AND_SHAPES',
        'sam_encode_chunk':1,'vggt_image_size':896,'load_peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}


def infer(model,query,counters):
    import torch
    from PIL import Image
    decoded=[np.asarray(Image.open(frame['canonical']['file']['path']).convert('RGB')) for frame in query['frames']]
    if any(image.shape!=(1024,1024,3) for image in decoded):raise RuntimeError('incorrect canonical SAM-V image size')
    images=torch.from_numpy(np.stack(decoded)).permute(0,3,1,2).contiguous().float().unsqueeze(0).cuda()
    points=torch.tensor(query['points_xy_canonical'],dtype=torch.float32,device='cuda')
    labels=torch.tensor(query['point_labels'],dtype=torch.int64,device='cuda')
    frames=torch.full((len(points),),query['anchor_slot'],dtype=torch.int64,device='cuda')
    counters['scientific_or_timing_joint_calls']+=1
    output=model.forward(sam_pre=images,point_coords_list=[points],point_labels_list=[labels],
        point_frame_indices_list=[frames],multimask_output=False,visualize=False)
    logits=output['low_res_logits'];n=len(decoded)
    if logits.ndim!=4 or logits.shape[:2]!=(1,1) or logits.shape[-1]%n:raise RuntimeError('incorrect single-mask panorama shape')
    h,w=logits.shape[-2:];tiles=logits[0,0].reshape(h,n,w//n).permute(1,0,2)
    arrays,outcome=restore_tiles(tiles,query)
    arrays['low_res_logits']=logits
    outcome.update(low_res_shape=list(logits.shape),panorama_split_before_interpolation=True,
        frame_coverage=list(range(n)),multimask_output=False,visualize=False)
    return arrays,outcome,decoded


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    run_worker(parser.parse_args().request,load_model,infer,'SAMV')
