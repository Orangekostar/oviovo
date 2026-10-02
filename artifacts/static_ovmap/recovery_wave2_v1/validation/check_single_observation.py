"""Compare actual U1 observations with artificial duplicated original inputs."""

from pathlib import Path
import argparse
import pickle

import numpy as np
import torch

from static_ovmap.module_validation.contracts import atomic_write_json, canonical_digest
from static_ovmap.module_validation.scannet_study import native_readout
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read
from static_ovmap.recovery_wave2.recovery_sources import single_native_classifier


parser = argparse.ArgumentParser()
parser.add_argument("--output-root", default="/mnt/shared/ww/ovimap-recovery-wave2-v1/attempt_001")
root = Path(parser.parse_args().output_root)
binding = read(root / "resolved_inputs.json")
index = ConsumptionIndex(root / "validation/input_verifications.json")
rows = []
for scene, data in binding["scenes"].items():
    source_path = root / "recovery" / scene / "BB00_NATIVE/cached_sources.json"
    sources = read(source_path)["sources"]["U1"]
    index.identity(source_path)
    native = read(Path(data["native_query_root"]) / "native_query_receipt.json")
    index.identity(native["native_features"]["path"], native["native_features"])
    with Path(native["native_features"]["path"]).open("rb") as stream:
        saved = pickle.load(stream)
    text_identity = data["inherited"]["models"]["native"]["text"]
    index.identity(text_identity["path"], text_identity)
    with np.load(text_identity["path"], allow_pickle=False) as arrays:
        text, canonical, ids = arrays["text_embeddings"], arrays["canonical_embeddings"], arrays["valid_ids"].tolist()
    for owner, source in sources.items():
        if not source["available"]:
            continue
        original = saved[int(owner)]
        assert len(original["frame_id"]) == 1
        duplicate = {"frame_id": original["frame_id"] * 2,
            "feat": np.repeat(original["feat"], 2, axis=0),
            "vis_area": np.repeat(original["vis_area"], 2)}
        inherited = native_readout({int(owner): duplicate}, text, canonical, tuple(ids))[int(owner)]
        feature = torch.as_tensor(inherited["feature"], dtype=torch.float32)
        query = torch.nn.functional.cosine_similarity(feature, torch.as_tensor(text), dim=-1).unsqueeze(0)
        reference = torch.nn.functional.cosine_similarity(feature, torch.as_tensor(canonical), dim=-1).unsqueeze(1)
        inherited_scores = torch.min(torch.exp(query) / (torch.exp(query) + torch.exp(reference)), dim=0).values.numpy()
        actual = single_native_classifier(np.asarray(original["feat"])[0], text, canonical, ids)
        difference = float(np.max(np.abs(np.asarray(actual["scores"]) - inherited_scores)))
        assert difference <= 1e-6
        assert actual["label"] == inherited["class_id"] == source["label"]
        np.testing.assert_array_equal(actual["scores"], source["scores"])
        rows.append({"scene": scene, "raw_owner": int(owner), "request_ids": source["used_request_ids"],
            "actual_single_class": actual["label"], "original_duplicated_class": inherited["class_id"],
            "maximum_absolute_score_difference": difference, "precision": "float32"})
assert rows
receipt = {"status": "PASS", "scope": "ALL_ACTUAL_AVAILABLE_U1_ROWS_IN_FIXED_12_SCENES",
    "artificial_duplicate_is_not_a_new_observation": True, "new_neural_inputs": 0,
    "maximum_absolute_score_difference": max(row["maximum_absolute_score_difference"] for row in rows),
    "objects": rows, "consumed_inputs": index.entries()}
receipt["identity"] = canonical_digest(receipt)
atomic_write_json(root / "validation/single_observation.json", receipt)
print(receipt["status"], len(rows), "real U1 rows", receipt["maximum_absolute_score_difference"])
