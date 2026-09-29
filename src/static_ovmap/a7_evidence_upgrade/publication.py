"""Publish compact real evidence without copying external models or old releases."""

import gzip
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.m2_reviewer_study.binding import InputIndex

ROOT = Path(__file__).resolve().parents[3]
BRANCH = "research/ovimap-a7-evidence-upgrade-wave1"
BUNDLE = Path("artifacts/static_ovmap/a7_evidence_upgrade_wave1/attempt_001")
SECRET = re.compile(rb"(?:-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|hf_[A-Za-z0-9]{30,}|https?://[^\s/:]+:[^\s/@]+@)")


def bundle(binding):
    root, index = Path(binding["output_root"]), InputIndex()
    receipt = read_json(root / "report/receipt.json")
    for item in receipt["documents"]:
        index.identity(item["path"], item)
    review = read_json(root / "review/requirements.json")
    if not review.get("items") or any(v["status"] not in {"VERIFIED", "VERIFIED_EXTERNAL_BLOCK"} or not v.get("evidence") for v in review["items"]):
        raise ValueError("publication requires the primary agent's complete evidence-backed requirement audit")
    selected = []
    for name in ("binding.json", "workers.json", "composition.json", "nomination.json", "transfer_lock.json"):
        selected.append((root / name, Path(name), False))
    for folder in ("rows", "pooled", "calibration", "locked", "e04", "environments", "phases", "audit", "review", "report", "comparisons", "costs"):
        for path in sorted((root / folder).rglob("*.json")):
            if ".before_" in path.name:
                continue
            relative = path.relative_to(root)
            compress = folder in {"report", "audit", "review"} and path.stat().st_size > 512 * 1024
            selected.append((path, Path(str(relative) + ".gz") if compress else relative, compress))
    for folder in ("source_contracts", "diagnostics"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and (path.suffix == ".json" or path.name.endswith(".json.gz")):
                relative = path.relative_to(root)
                compress = path.suffix == ".json"
                selected.append((path, Path(str(relative) + ".gz") if compress else relative, compress))
    spec = read_json(binding["spec"])
    data = read_json(root / "report/data.json")
    for scene, cost in data["scene_costs"].items():
        for worker, value in cost["physical_completed_worker_invocations"].items():
            selected.append((Path(value["receipt"]["path"]), Path("worker_receipts") / scene / (worker + ".json.gz"), True))
    for smoke in data["smokes"]:
        path = Path(smoke["receipt"])
        selected.append((path, Path("smokes") / path.relative_to(root), False))
        audit = path.parent / "weight_audit.json"
        if audit.exists():
            selected.append((audit, Path("smokes") / audit.relative_to(root), False))
    for scene in binding["scenes"]:
        for variant in spec["source_variants"]:
            path = root / variant["family"].lower() / scene / (variant["id"] + ".json")
            selected.append((path, Path("sources") / scene / (variant["id"] + ".json.gz"), True))
    for name in ("sam2", "so400m", "fc_frozen", "ovrcoat"):
        path = root.parent / "assets" / name / "download_receipt.json"
        selected.append((path, Path("assets") / name / path.name, False))
    for name in ("sam2", "ovrcoat"):
        path = root.parent / "assets" / name / "code/LICENSE"
        selected.append((path, Path("assets") / name / "LICENSE", False))
    for path in sorted((root.parent / "assets/model_cards").glob("*")):
        if path.is_file():
            selected.append((path, Path("assets/model_cards") / path.name, False))
    for scene, bound in binding["scenes"].items():
        selected.append((Path(bound["config"]), Path("configs") / (scene + ".json"), False))
    destination, entries, external = ROOT / BUNDLE, [], {}
    for source, relative, compress in selected:
        body = source.read_bytes()
        if source.name.endswith(".gz"):
            inspect = gzip.decompress(body)
        else:
            inspect = body
        if SECRET.search(inspect):
            raise ValueError(f"possible credential in publication input: {source}; not copied")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = gzip.compress(body, mtime=0) if compress else body
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f"existing publication bytes differ: {target}")
        if not target.exists():
            if compress:
                target.write_bytes(payload)
            else:
                shutil.copyfile(source, target)
        entries.append({"source": str(source), "target": str(relative), "source_bytes": len(body),
                        "source_sha256": hashlib.sha256(body).hexdigest(), "published": index.identity(target),
                        "lossless_gzip": compress})
    # Preserve actual external file identities already verified by the final
    # matrix/report audits, without recursively copying the historical release.
    for file in (root / "audit/matrix.json", root / "report/data.json"):
        for item in read_json(file)["inputs"]:
            external[item["path"]] = item
    capture_files = []
    for bound in binding["scenes"].values():
        manifest = read_json(bound["static_manifest"]["path"])
        capture_identity = manifest["capture"]
        index.identity(capture_identity["path"], capture_identity)
        capture_path = Path(capture_identity["path"])
        external[str(capture_path)] = capture_identity
        for frame in read_json(capture_path)["frames"]:
            for name in ("rgb", "depth", "global_owner", "panoptic"):
                path = capture_path.parent / frame[name + "_path"]
                capture_files.append({"path": str(path), "bytes": path.stat().st_size,
                                      "sha256": frame[name + "_sha256"], "hash_source": str(capture_path)})
    external_manifest = {"external_bytes_uploaded": False, "files": [external[k] for k in sorted(external)],
                         "restore": "Restore these exact shared-storage files and original binding inputs; do not download new datasets.",
                         "rebuild": "python scripts/evaluation/run_ovimap_a7_evidence_upgrade.py --phase all --split all --resume --gpus 0,1,2",
                         "large_feature_mask_prediction_arrays": "Referenced by source contracts and actual prediction/evaluator manifests; not included as uploaded bytes."}
    external_manifest["original_RGBD_and_masks"] = capture_files
    external_manifest["capture_hash_scope"] = "Hashes come from byte-verified original capture manifests; publication does not rerun the historical dataset audit."
    external_source = root / "publication/external_files.json"
    write_once(external_source, external_manifest)
    external_target = destination / "external_files.json.gz"
    payload = gzip.compress(external_source.read_bytes(), mtime=0)
    if external_target.exists() and external_target.read_bytes() != payload:
        raise ValueError("published external manifest differs")
    if not external_target.exists():
        external_target.write_bytes(payload)
    result = {"status": "COMPACT_EVIDENCE_BUNDLED", "binding": binding["identity"], "entries": entries,
              "principal_metrics_selection_copied_byte_for_byte": True,
              "external_manifest": index.identity(external_target),
              "external_manifest_decompressed_source": index.identity(external_source)}
    write_once(destination / "publication_manifest.json", result)
    size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if size > 40 * 1024**2:
        raise ValueError(f"new tracked evidence exceeds 40 MiB: {size}; do not commit")
    return {"status": result["status"], "bytes": size, "files": len(entries), "path": str(destination)}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_bundle(binding):
    root, destination, index = Path(binding["output_root"]), ROOT / BUNDLE, InputIndex()
    manifest = read_json(destination / "publication_manifest.json")
    targets = set()
    for entry in manifest["entries"]:
        target, source = destination / entry["target"], Path(entry["source"])
        if entry["target"] in targets:
            raise ValueError("duplicate publication target")
        targets.add(entry["target"])
        index.identity(target, entry["published"])
        body = target.read_bytes()
        if entry["lossless_gzip"]:
            body = gzip.decompress(body)
        if body != source.read_bytes() or hashlib.sha256(body).hexdigest() != entry["source_sha256"]:
            raise ValueError("published evidence differs from actual source bytes")
    for folder in ("rows", "pooled", "calibration", "locked"):
        expected = {str(p.relative_to(root)) for p in (root / folder).rglob("*.json")}
        actual = {p for p in targets if p.startswith(folder + "/")}
        if expected != actual:
            raise ValueError("principal result/decision matrix incomplete in bundle")
    external = Path(manifest["external_manifest"]["path"])
    index.identity(external, manifest["external_manifest"])
    source = Path(manifest["external_manifest_decompressed_source"]["path"])
    if gzip.decompress(external.read_bytes()) != source.read_bytes():
        raise ValueError("external-file manifest compression was not lossless")
    size = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if size > 40 * 1024**2:
        raise ValueError("verified bundle exceeds publication budget")
    result = {"status": "BUNDLE_BYTES_AND_COVERAGE_VERIFIED", "bytes": size, "files": len(targets),
              "scene_rows": sum(p.startswith("rows/") for p in targets),
              "pool_rows": sum(p.startswith("pooled/") for p in targets),
              "manifest": index.identity(destination / "publication_manifest.json"),
              "all_copied_or_decompressed_bytes_equal_original": True}
    write_once(root / "publication/bundle_review.json", result)
    return result


def publish(binding):
    if git("branch", "--show-current") != BRANCH:
        raise ValueError("publication must remain on the prescribed research branch")
    result = bundle(binding)
    verify_bundle(binding)
    scope = ["src/static_ovmap/a7_evidence_upgrade", "tests/a7_evidence_upgrade",
             "scripts/evaluation/run_ovimap_a7_evidence_upgrade.py", str(BUNDLE),
             "docs/paper/static_ovmap/a7_evidence_upgrade_wave1",
             *["docs/paper/static_ovmap/A7_WAVE1_" + name + ".md" for name in ("RESULTS", "HANDOFF", "SELECTION")]]
    staged = git("diff", "--cached", "--name-only").splitlines()
    if any(not any(p == allowed or p.startswith(allowed + "/") for allowed in scope) for p in staged):
        raise ValueError("unrelated staged files preserved; cannot include them in publication commit")
    subprocess.run(["git", "add", "--", *scope], cwd=ROOT, check=True)
    # Official cards are intentionally byte-preserved, including upstream
    # trailing spaces; continue checking every authored file and other artifact.
    card_exclusions = [":(exclude)" + str(BUNDLE / "assets/model_cards" / name)
                       for name in ("fc_frozen.md", "so400m.md")]
    subprocess.run(["git", "diff", "--cached", "--check", "--", ".", *card_exclusions], cwd=ROOT, check=True)
    if git("diff", "--cached", "--name-only"):
        subprocess.run(["git", "commit", "-m", "Publish measured A7 wave-1 evidence and reproducible reports"], cwd=ROOT, check=True)
    head = git("rev-parse", "HEAD")
    subprocess.run(["git", "push", "origin", "HEAD:refs/heads/" + BRANCH], cwd=ROOT, check=True)
    remote = git("ls-remote", "origin", "refs/heads/" + BRANCH).split()
    if len(remote) != 2 or remote[0] != head:
        raise ValueError("remote full SHA does not equal local HEAD")
    receipt = {"status": "PUSH_VERIFIED", "local_full_sha": head, "remote_full_sha": remote[0],
               "branch": BRANCH, "bundle": result, "external_model_and_dataset_bytes_uploaded": False}
    write_once(Path(binding["output_root"]) / "publication" / (head + ".json"), receipt)
    return receipt
