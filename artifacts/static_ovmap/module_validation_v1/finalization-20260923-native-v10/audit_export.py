import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

root = Path("artifacts/static_ovmap/module_validation_v1/release-20260923-native-v10")


def read(path):
    return json.loads(path.read_text())


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


manifest = read(root / "export_manifest.json")
compaction = manifest["lossless_compaction"]
assert (
    sha(Path(compaction["script"]["path"]).read_bytes())
    == compaction["script"]["sha256"]
)
input_manifest_path = Path(__file__).parent / "intermediate_export_manifest.json.gz"
input_manifest = gzip.decompress(input_manifest_path.read_bytes())
assert sha(input_manifest) == compaction["input_manifest"]["sha256"]
assert len(input_manifest) == compaction["input_manifest"]["bytes"]
assert json.loads(input_manifest)["total_bytes"] == compaction["input_total_bytes"]
actual = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
assert actual == set(manifest["files"]) | {"export_manifest.json"}
transformations = Counter()
for name, expected in manifest["files"].items():
    raw = (root / name).read_bytes()
    assert sha(raw) == expected, name
    source = manifest["source_files"].get(name)
    if source:
        original = Path(source["source"]["path"]).read_bytes()
        assert sha(original) == source["source"]["sha256"], name
        assert len(original) == source["source"]["bytes"], name
        decoded = gzip.decompress(raw) if source.get("encoding") == "gzip" else raw
        if source.get("encoding") == "gzip":
            assert (
                sha(decoded) == source["decoded_sha256"]
                and len(decoded) == source["decoded_bytes"]
            ), name
        transform = source["transformation"]
        transformations[transform] += 1
        if transform == "exact_copy":
            assert decoded == original, name
        elif transform == "request_ledger_without_class_similarity_vector":
            original_value = json.loads(original)
            if "mapping" in original_value:
                original_value["mapping"].pop("similarities", None)
            assert json.loads(decoded) == original_value, name
        elif transform == "partition_counts_without_leaf_arrays":
            before, after = json.loads(original), json.loads(decoded)
            assert {
                k: v for k, v in before.items() if k not in ("groups", "hypotheses")
            } == {k: v for k, v in after.items() if k not in ("groups", "hypotheses")}
            for a, b in zip(before["groups"], after["groups"], strict=True):
                leaves = a.pop("leaf_ids")
                assert b == {**a, "leaf_count": len(leaves)}
            assert set(before["hypotheses"]) == set(after["hypotheses"])
            for group in before["hypotheses"]:
                for a, b in zip(
                    before["hypotheses"][group], after["hypotheses"][group], strict=True
                ):
                    leaves, components = a.pop("leaf_ids"), a.pop("components")
                    assert b == {
                        **a,
                        "leaf_count": len(leaves),
                        "component_count": len(set(components)),
                        "component_leaf_counts": dict(Counter(map(str, components))),
                    }
        else:
            raise AssertionError(transform)
    else:
        assert raw == (Path(manifest["source_attempt"]) / name).read_bytes(), name
total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
assert total == manifest["total_bytes"] <= 100 * 1024 * 1024
heads = {
    str(p.relative_to(root)): p.stat().st_size for p in root.rglob("checkpoint.pt")
}
assert (
    len(heads) == 4
    and sum(heads.values()) == 131528
    and max(heads.values()) <= 10 * 1024 * 1024
)
assert not any(
    p.suffix in (".npz", ".ply", ".png", ".safetensors", ".so")
    for p in root.rglob("*")
    if p.is_file()
)
print(
    json.dumps(
        {
            "status": "PASS",
            "file_count": len(actual),
            "total_bytes": total,
            "heads": heads,
            "transformations": dict(transformations),
            "all_hashes_and_source_transformations_verified": True,
            "compaction_provenance_verified": True,
            "export_manifest_sha256": sha((root / "export_manifest.json").read_bytes()),
        }
    )
)
