"""Pinned GLA-CLIP classifier-free bridge; model dependencies load only on demand."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import types

from .cache_io import sha256_file
from .dense_features import window_boxes


class GLABackend:
    def __init__(self, gla_root, dino_root, anyup_root, weights, device='cuda:0'):
        import torch
        from torchvision import transforms

        self.torch, self.device = torch, device
        gla_root, dino_root, anyup_root, weights = map(Path, (gla_root, dino_root, anyup_root, weights))
        if 'open_clip' in sys.modules and not Path(sys.modules['open_clip'].__file__).is_relative_to(gla_root):
            raise ValueError('another open_clip implementation was already imported')
        sys.path.insert(0, str(gla_root))
        import open_clip
        from cfg import ModelConfig
        from myutils import UnNormalize
        self.clip = open_clip.load_openai_model(str(weights/'ViT-B-16.pt'), precision='fp16', device=device).eval()
        self.roi_clip = torch.jit.load(str(weights/'ViT-B-16.pt'), map_location=device).eval()
        self.vfm = torch.hub.load(str(dino_root), 'dino_vitb8', source='local', pretrained=False)
        self.vfm.load_state_dict(torch.load(weights/'dino_vitbase8_pretrain.pth', map_location='cpu', weights_only=True), strict=True)
        self.vfm = self.vfm.half().eval().to(device)
        self.vfm_model = 'dino'
        self.model_cfg = ModelConfig(CLIP_type='ProxyCLIP', model_type='ViT-B/16', device=device,
            token_norm=True, KV_token_extension=True, proxy_sim=True, mini_iters=2, initial_crit_pos=.6)
        self.model_cfg.dynamic_beta, self.model_cfg.dynamic_gamma = True, True
        self.model_cfg.beta_alpha, self.model_cfg.gamma_alpha = .3, 30.
        self.beta, self.gamma = self.model_cfg.beta, self.model_cfg.gamma
        self.unnorm = UnNormalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])
        self.norm = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        self.clip_norm = transforms.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])
        self.roi_preprocess = transforms.Compose([transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(), self.clip_norm])
        self.to_tensor = transforms.ToTensor()
        source = gla_root/'gla_clip_segmentor.py'
        cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'GLA_CLIPSegmentation')
        function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward_feature')
        namespace = {'torch': torch, 'F': torch.nn.functional}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
        self.original_forward_feature = types.MethodType(namespace['forward_feature'], self)
        sys.path.insert(0, str(anyup_root))
        from anyup.model import AnyUp
        self.upsampler = AnyUp(use_natten=False)
        self.upsampler.load_state_dict(torch.load(weights/'anyup_paper.pth', map_location='cpu', weights_only=True), strict=True)
        self.upsampler = self.upsampler.eval().to(device)
        self.tokenizer = open_clip.tokenize
        from prompts.imagenet_template import openai_imagenet_template
        self.templates = openai_imagenet_template
        files = [source, gla_root/'open_clip/transformer.py', gla_root/'open_clip/model.py',
                 gla_root/'open_clip/tokenizer.py', gla_root/'open_clip/bpe_simple_vocab_16e6.txt.gz',
                 gla_root/'cfg.py', gla_root/'myutils.py',
                 gla_root/'prompts/imagenet_template.py', Path(__file__), Path(__file__).with_name('dense_features.py'),
                 dino_root/'vision_transformer.py', anyup_root/'anyup/model.py',
                 anyup_root/'anyup/layers/attention/chunked_attention.py',
                 weights/'ViT-B-16.pt', weights/'dino_vitbase8_pretrain.pth', weights/'anyup_paper.pth']
        self.identity = {'sources_and_weights': {str(f): sha256_file(f) for f in files},
            'dense_preprocess': 'PIL_bilinear_short336_nearest8_grid_CLIP_norm_windows224_stride112',
            'dense_layer': 'GLA_ProxyCLIP_classification_preceding_normalized_window_tokens',
            'text': 'OpenAI_CLIP_B16_ImageNet_template_ensemble', 'anyup': 'paper_non_NATTEN_q_chunk4096',
            'native_SigLIP_compatibility': False}
        self.space = 'sha256:'+hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()
        self.roi_space = 'sha256:'+hashlib.sha256((self.space+'original_CLIP_global_ROI').encode()).hexdigest()

    def dense(self, image):
        torch = self.torch
        from PIL import Image
        scale = 336/min(image.height, image.width)
        height = max(224, round(image.height*scale/8)*8)
        width = max(224, round(image.width*scale/8)*8)
        resized = image.resize((width, height), Image.Resampling.BILINEAR)
        tensor = self.clip_norm(self.to_tensor(resized)).to(self.device)
        boxes = window_boxes(height, width)
        crops = torch.stack([tensor[:, y1:y2, x1:x2] for x1, y1, x2, y2 in boxes])
        self.model_cfg.h_grids = len({box[1] for box in boxes})
        self.model_cfg.w_grids = len({box[0] for box in boxes})
        self.model_cfg.token_h = self.model_cfg.token_w = 14
        hooks = self.vfm.blocks[-1].attn.qkv._forward_hooks
        existing = set(hooks)
        try:
            with torch.inference_mode():
                features = self.original_forward_feature(crops, None)
        finally:
            for key in set(hooks)-existing:
                del hooks[key]
        if not torch.isfinite(features).all():
            raise ValueError('nonfinite released GLA features')
        features = torch.nn.functional.normalize(features.float(), dim=-1)
        tokens = features.permute(0, 2, 1).reshape(len(boxes), 512, 28, 28)
        result = torch.zeros((1, 512, height//8, width//8), device=self.device)
        counts = torch.zeros((1, 1, height//8, width//8), device=self.device)
        for patch, (x1, y1, x2, y2) in zip(tokens, boxes):
            result[0, :, y1//8:y2//8, x1//8:x2//8] += patch
            counts[0, :, y1//8:y2//8, x1//8:x2//8] += 1
        if torch.any(counts == 0):
            raise ValueError('uncovered dense feature grid')
        return result/counts, {'resized_shape': [height, width], 'window_count': len(boxes),
            'output_shape': list(result.shape), 'qkv_hook_count_after': len(hooks)}

    def upsample(self, image, low_resolution, method):
        torch = self.torch
        with torch.inference_mode():
            if method == 'bilinear':
                return torch.nn.functional.interpolate(low_resolution, size=(image.height, image.width),
                                                       mode='bilinear', align_corners=False)
            if method != 'anyup':
                raise ValueError('explicit bilinear or official anyup method required')
            rgb = self.norm(self.to_tensor(image)).unsqueeze(0).to(self.device)
            return self.upsampler(rgb, low_resolution, output_size=(image.height, image.width), q_chunk_size=4096)

    def roi(self, image, bbox):
        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError('empty native exclusive bbox')
        crop = self.roi_preprocess(image.crop((x1, y1, x2, y2))).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            return self.roi_clip.encode_image(crop).float()[0]

    def text_features(self, labels):
        result = []
        with self.torch.inference_mode():
            for label in labels:
                queries = self.tokenizer([template(label) for template in self.templates]).to(self.device)
                features = self.clip.encode_text(queries).float()
                features = self.torch.nn.functional.normalize(features, dim=-1).mean(dim=0)
                result.append(self.torch.nn.functional.normalize(features, dim=0))
        return self.torch.stack(result)
