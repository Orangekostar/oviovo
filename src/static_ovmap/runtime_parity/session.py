"""Original FP32 neural calls, with measured stages and optional deferred copies."""

from types import SimpleNamespace

import numpy as np

from static_ovmap.cvpr_compact.area_fallback import PROTOCOL, region_vector as fallback_vector
from static_ovmap.cvpr_compact.region_worker import FCSession, validate_request_outcomes
from static_ovmap.module_validation.contracts import canonical_digest
from static_ovmap.module_validation.native_capture import _array_digest
from static_ovmap.recovery_wave2.recovery_sources import fc_image_identity


class RuntimeSession(FCSession):
    def encode_exact(self, requests, loaders, stages, *, implementation, arm):
        import torch
        from static_ovmap.a7_evidence_upgrade.region_adapter import image_tensor, signed_mask, region_vector

        if self.cache is not None or self.model is None:
            raise ValueError("cold encoding requires a resident original model and no persistent cache")
        deferred = implementation == "R3_RUNTIME_EXACT" and arm != "U2"
        groups = {}
        for rid, request in requests.items():
            groups.setdefault(fc_image_identity(request,self.model_key),[]).append(rid)
        stats = {"physical_image_encodings":0,"encoder_batch_calls":0,"physical_region_poolings":0,
            "area_fallback_poolings":0,"dense_cache_hits":0,"region_cache_hits":0,"legacy_region_cache_hits":0,
            "persistent_feature_cache_enabled":False,"required_dense_receipts":{},"required_region_receipts":{},"requests":{}}
        self.last_stats = stats
        features = {}
        with torch.inference_mode(),stages.span("recognition"):
            for image_key,rids in groups.items():
                dense = image = values = vector = signed = None
                pooled,pending,aliases = {},{},{}
                try:
                    with stages.span("rgb_preprocess"):
                        values = {rid:loaders[rid](rid) for rid in rids}
                        image,size = image_tensor(values[rids[0]]["image"],self.device)
                        tensor_array = image.cpu().numpy()
                        tensor_key = canonical_digest({"model":self.model_key,"tensor":_array_digest(tensor_array)})
                        stages.counters["identity_roundtrip_bytes"] += tensor_array.nbytes
                        del tensor_array
                    stats["physical_image_encodings"] += 1
                    stats["encoder_batch_calls"] += 1
                    with stages.cuda_span("encoder",device=self.device):
                        dense = self.operators["extract_features_convnext"](SimpleNamespace(clip_model=self.model),image)["clip_vis_dense"]
                    if dense.dtype != torch.float32 or dense.ndim != 4 or not torch.isfinite(dense).all():
                        raise ValueError("FC dense operator returned invalid original precision/shape/values")
                    for rid in rids:
                        value,mask = values[rid],values[rid]["target"]
                        if not np.array_equal(value["image"],values[rids[0]]["image"]):
                            raise ValueError("one physical image digest produced different decoded RGB")
                        if mask.dtype != np.bool_ or mask.shape != value["image"].shape[:2]:
                            raise ValueError("FC target mask must remain unchanged full-resolution boolean evidence")
                        pooling = "original_signed_mask_pooling" if arm == "U2" else PROTOCOL
                        key = canonical_digest({"image":tensor_key,"mask":_array_digest(mask),"region":pooling})
                        row = {"request_id":rid,"frame_id":requests[rid]["frame_id"],"status":"PENDING",
                            "image_content_key":image_key,"input_tensor_key":tensor_key,"feature_content_key":key,
                            "target_mask_sha256":_array_digest(mask),"physical_cache_hit":False}
                        stats["requests"][rid] = row
                        if key in pooled:
                            previous = pooled[key]
                            original = stats["requests"][previous]
                            row.update({k:original[k] for k in ("fallback","original_support","dense_hw","area_support","area_mass","dense_mask_support") if k in original})
                            row.update(status="COMPLETE",within_run_region_alias=previous)
                            if deferred:
                                aliases[rid] = previous
                            else:
                                features[rid] = features[previous]
                            continue
                        try:
                            with stages.cuda_span("region_pool_head",device=self.device):
                                signed,support,_ = signed_mask(mask,size,image.shape[-2:],dense.shape[-2:],self.device)
                                if arm == "U2":
                                    if not int(support.sum()):
                                        raise ValueError("EMPTY_DENSE_MASK_SUPPORT")
                                    stats["physical_region_poolings"] += 1
                                    vector,dense_support = region_vector(self.model,self.operators,dense,signed)
                                    audit = {"dense_mask_support":dense_support,"fallback":False,"original_support":dense_support}
                                else:
                                    stats["physical_region_poolings"] += 1
                                    vector,audit = fallback_vector(self.model,self.operators,dense,signed)
                            if deferred:
                                pending[rid] = vector
                            else:
                                features[rid] = vector.cpu().numpy().copy()
                                stages.counters["vector_transfer_calls"] += 1
                                stages.counters["vector_transfer_bytes"] += features[rid].nbytes
                            stats["area_fallback_poolings"] += int(audit["fallback"])
                            pooled[key] = rid
                            row.update(status="COMPLETE",**audit)
                        except ValueError as exc:
                            if arm != "U2" or str(exc) not in {"EMPTY_DENSE_MASK_SUPPORT","INVALID_REGION_FEATURE"}:
                                raise
                            row.update(status="UNAVAILABLE_TECHNICAL_FAILURE",reason=str(exc),fallback=False)
                    if pending:
                        ordered = list(pending)
                        transferred = torch.stack([pending[rid] for rid in ordered]).cpu().numpy()
                        for rid,array in zip(ordered,transferred,strict=True):
                            features[rid] = array.copy()
                        for rid,previous in aliases.items():
                            features[rid] = features[previous]
                        stages.counters["vector_transfer_calls"] += 1
                        stages.counters["vector_transfer_bytes"] += transferred.nbytes
                        del transferred
                except BaseException as exc:
                    for rid in rids:
                        if rid not in features:
                            stats["requests"].setdefault(rid,{"request_id":rid})
                            stats["requests"][rid].update(status="UNAVAILABLE_TECHNICAL_FAILURE",reason=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    dense = image = values = vector = signed = None
                    pending.clear()
                    pooled.clear()
                    aliases.clear()
                    if implementation != "R3_RUNTIME_EXACT" and self.device == "cuda":
                        torch.cuda.empty_cache()
        validate_request_outcomes(requests,features,stats,require_success=False)
        stages.counters.update(logical_requests=len(requests),FC_image_inputs=stats["physical_image_encodings"],
            encoder_batch_calls=stats["encoder_batch_calls"],normal_pool_head_calls=stats["physical_region_poolings"]-stats["area_fallback_poolings"],
            fallback_pool_head_calls=stats["area_fallback_poolings"],feature_cache_hits=0,view_cache_hits=0,result_cache_hits=0)
        return features,stats
