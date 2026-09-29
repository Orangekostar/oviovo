"""Count the exact audited region architecture without image inference."""

import argparse
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex


def count_regions(root):
    import open_clip

    root, index = Path(root), InputIndex()
    model = open_clip.create_model("convnext_large_d_320", pretrained=None, device="cpu")
    model.eval().requires_grad_(False)
    shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
    branches = {}
    for branch in ("FC_FROZEN", "OVR"):
        path = root / "e03/smoke/scene0056_00" / branch / "weight_audit.json"
        index.identity(path)
        audit = read_json(path)
        if {r["module_key"]: r["shape"] for r in audit["mapping"]} != shapes or not audit["strict"] or audit["random_active_parameters"]:
            raise ValueError("counted architecture differs from the strictly loaded measured branch")
        branches[branch] = {"loaded_state_elements": sum(r["elements"] for r in audit["mapping"]),
                            "loaded_key_count": audit["loaded_key_count"],
                            "OVR_vs_original_trunk_changed_elements": audit["active_trunk_changed_elements"]}
    result = {"status": "AUDITED_ARCHITECTURE_PARAMETER_COUNT", "architecture": "convnext_large_d_320",
              "parameters_per_branch": sum(p.numel() for p in model.parameters()),
              "visual_parameters_per_branch": sum(p.numel() for p in model.visual.parameters()),
              "trainable_parameters_in_this_study": 0, "new_image_inference": False,
              "branches": branches, "inputs": index.entries(),
              "scope": "Architecture enumeration checked against real strict weight-load mappings; this enumeration model was not used for inference."}
    write_once(root / "environments/region_parameters.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    print(count_regions(parser.parse_args().root), flush=True)
