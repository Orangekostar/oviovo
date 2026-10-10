"""Actual frozen-FC backward, real-shape port parity and discarded 20-step fit."""
import ast
import hashlib
from pathlib import Path
import time

import numpy as np

from .common import canonical_digest, verified, write


def frozen_digest(fc):
    import torch
    digest = hashlib.sha256()
    for name, tensor in fc.model.state_dict().items():
        digest.update(name.encode())
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
        digest.update(memoryview(value).cast('B'))
    digest.update(memoryview(np.ascontiguousarray(fc.text)).cast('B'))
    return digest.hexdigest()


def upstream_head(root):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from timm.layers import DropPath
    from einops import rearrange, repeat
    namespace = dict(torch=torch, nn=nn, F=F, cp=torch.utils.checkpoint,
                     DropPath=DropPath, rearrange=rearrange, repeat=repeat)
    for path, names in [(root/'fcclip/modeling/meta_arch/convnext.py', {'ConvNextBlock', 'LayerNorm'}),
                        (root/'mask_adapter/modeling/meta_arch/mask_adapter_head.py', {'MASKAdapterHead', 'LayerNorm2d'})]:
        nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name in names]
        for node in nodes:
            node.decorator_list = []
            node.body = [n for n in node.body if not isinstance(n, ast.FunctionDef) or n.name != 'from_config']
            for method in node.body:
                if isinstance(method, ast.FunctionDef):
                    method.decorator_list = []
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['MASKAdapterHead']


def engineer(binding):
    import torch
    from static_ovmap.backbone_wave1.runtime import exclusive_lock
    from .features import FrozenFC, ObjectLoader, capture_dataset, execution_config
    from .losses import base_loss
    from .model import ReadoutHead
    root = Path(binding['output_root']); path = root / 'engineering/receipt.json'
    split = verified(root / 'split_manifest.json')
    first = verified(root / 'data/generated' / split['roles']['train'][0] / 'manifest.json')
    objects = [o for o in first['objects'] if o['base']][:2]
    if len(objects) != 2:
        raise ValueError('Engineering fit requires exactly two available TRAIN base objects')
    identity = canonical_digest(dict(binding=binding['identity'], objects=[o['inputs'] for o in objects], updates=20))
    if path.exists():
        previous = verified(path)
        if previous['input_identity'] != identity:
            raise ValueError('Engineering inputs changed')
        return previous
    features = capture_dataset(binding, engineering_objects=objects)
    with exclusive_lock(execution_config(binding)['gpu_lock']):
        started = time.perf_counter(); fc = FrozenFC(binding)
        loader = ObjectLoader(fc, features)
        before = frozen_digest(fc)
        inputs, _, _ = loader.load(objects[0], 0, 2)
        # Real frozen projection and actual FC grids, full specified MA shape.
        torch.manual_seed(17)
        proposed = ReadoutHead('SURFACE').to(fc.device)
        result = proposed(inputs, fc.phi, completed_steps=200, training_forward=True)
        label = fc.base_targets[objects[0]['class_id']]
        ce = torch.nn.functional.cross_entropy((result['embedding'] @ fc.base_text.T / .07)[None],
                                              torch.tensor([label], device=fc.device))
        ce.backward()
        gradient_names = ('ma.final.weight', 'embed.0.weight', 'quality.0.weight', 'queries')
        gradients = {}
        for name in gradient_names:
            grad = dict(proposed.named_parameters())[name].grad
            if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) <= 0:
                raise ValueError('Real frozen-phi backward failed for '+name)
            gradients[name] = float(grad.norm())
        if any(p.grad is not None or p.requires_grad for p in fc.model.parameters()):
            raise ValueError('Frozen FC acquired trainable parameters/gradients')
        # Compare upstream numerical head outputs and all gradients on real U/mask.
        original = upstream_head(root/'external/upstream_sources/MaskAdapter')
        # CUDA weight-gradient reductions may differ at sub-nanoscopic FP32
        # scale between identical calls. Compare the numerical port bitwise on
        # CPU using these actual FC features; real frozen-phi backward stays GPU.
        reference = original('convnext_large_d_320', 16, 256, False, 4)
        port = ReadoutHead('NONE').ma
        port.load_state_dict(reference.state_dict(), strict=True)
        x = inputs['projected'][:1].detach().cpu().clone().requires_grad_(True)
        xp = x.detach().clone().requires_grad_(True)
        mask = inputs['masks'][:1].detach().cpu()
        a, b = reference(x, mask), port(xp, mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward(); b.square().mean().backward()
        torch.testing.assert_close(x.grad, xp.grad, rtol=0, atol=0)
        for (name, p), (other, q) in zip(reference.named_parameters(), port.named_parameters()):
            if name != other:
                raise ValueError('MA numerical parameter names differ')
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        del proposed, reference, port, a, b, x, xp, result
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(17)
        head = ReadoutHead('NONE').to(fc.device)
        from .training import optimizer_for
        optimizer = optimizer_for(head)
        initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
        # Fixed inputs avoid moving targets or choosing favorable corruption after outcomes.
        examples = []
        for obj in objects:
            clean = loader.load(obj, 0, 2)[0]
            corrupt = loader.load(obj, 1, 2)[0]
            examples.append((clean, corrupt, fc.base_targets[obj['class_id']]))
        def objective():
            terms = [base_loss(head(c, fc.phi), head(d, fc.phi), y, fc.base_text)[0] for c, d, y in examples]
            return torch.stack(terms).mean()
        with torch.no_grad(): initial_loss = float(objective())
        values = []
        for step in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not torch.isfinite(loss):
                raise ValueError('Discarded engineering fit returned nonfinite loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1)
            optimizer.step(); values.append(float(loss.detach()))
        with torch.no_grad(): final_loss = float(objective())
        movement = sum(float((head.state_dict()[k]-v).abs().sum()) for k, v in initial.items())
        after = frozen_digest(fc)
        if final_loss >= initial_loss or movement <= 0 or before != after:
            raise ValueError('Engineering fit did not lower loss/move only the new head')
        torch.cuda.synchronize()
        record = write(path, dict(status='COMPLETE', input_identity=identity, original_TRAIN_objects=[o['key'] for o in objects],
                 optimizer_updates=20, discarded=True, scientific_updates=0, initial_loss=initial_loss,
                 final_loss=final_loss, step_losses=values, parameter_absolute_movement=movement,
                 real_gradient_norms=gradients, gradient_eta=.5, frozen_before=before, frozen_after=after,
                 actual_raw_C=features['raw_C'], actual_projected_D=768, actual_MA_upstream_parity='CPU_BITWISE_REAL_FC_OUTPUT_INPUT_AND_PARAMETER_GRADIENTS',
                 real_frozen_phi_backward_device=fc.device,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                 elapsed_seconds=time.perf_counter()-started, features=features['identity']))
        fc.index.write_memo(root / 'features/verifications.json')
        print('ENGINEERING_COMPLETE', initial_loss, '->', final_loss, flush=True)
    return record
