"""Scoped byte-preserving reviewer evidence package and ordinary branch push."""

import gzip
import subprocess
from pathlib import Path

from src.static_ovmap.composition_study.io import read_json, write_once
from src.static_ovmap.module_validation.contracts import atomic_write_json

from .binding import ROOT, InputIndex
from .reporting import assert_report_ready

BRANCH = "research/ovimap-m2-reviewer-evidence-v1"
DOCS = ("M2_REVIEWER_RESULTS.md", "M2_REVIEWER_HANDOFF.md", "M2_CLAIM_LEDGER.md")


def copy_artifact(source, destination, relative, *, compress=False, index=None):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact requires a contained relative path")
    index = index or InputIndex()
    source = Path(source)
    content = source.read_bytes()
    if compress:
        relative = relative.with_name(relative.name + ".gz")
        content = gzip.compress(content, mtime=0)
    target = Path(destination) / relative
    if target.resolve().is_relative_to(Path(destination).resolve()) is False:
        raise ValueError("artifact relative path escapes destination")
    if target.exists() and target.read_bytes() != content:
        raise ValueError(f"immutable artifact changed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(content)
    return {"relative": str(relative), "source": index.identity(source),
            "artifact": index.identity(target), "gzip_encoded": compress}


def verify_bundle(destination, rows, *, limit=100 * 1024 * 1024):
    index, seen, total = InputIndex(), set(), 0
    for row in rows:
        relative = Path(row["relative"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("invalid or duplicate artifact relative path")
        seen.add(str(relative))
        path = Path(destination) / relative
        if not path.resolve().is_relative_to(Path(destination).resolve()):
            raise ValueError("artifact relative path escapes destination")
        total += index.identity(path, row["artifact"])["bytes"]
    if total >= limit:
        raise ValueError(f"artifact bundle exceeds byte limit: {total} >= {limit}")
    return total


def collect_identities(value, external):
    """Collect only explicit absolute file identities, never guessed paths."""
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys() and Path(value["path"]).is_absolute():
            row = {key: value[key] for key in ("path", "bytes", "sha256")}
            if row["path"] in external and external[row["path"]] != row:
                raise ValueError(f"conflicting consumed dependency identities: {row['path']}")
            external[row["path"]] = row
        for item in value.values():
            collect_identities(item, external)
    elif isinstance(value, list):
        for item in value:
            collect_identities(item, external)


def export_artifacts(binding):
    root = Path(binding["output_root"])
    if not (root / "final_report.json").is_file():
        raise ValueError("publication prerequisite missing: run the complete report phase first")
    pointer = read_json(root / "final_report.json")
    report_root = Path(pointer["path"])
    report = read_json(report_root / "report.json")
    assert_report_ready(report["states"])
    if report["status"] != "FINAL_EVIDENCE_READY" or pointer["identity"] != report["identity"] or report["binding"] != binding["identity"]:
        raise ValueError("publication requires this binding's final measured report")
    if set(report.get("matrix_audit", {})) != {"core", "query_ScanNet", "query_Replica"}:
        raise ValueError("publication requires exact measured-matrix audit")
    destination = ROOT / "artifacts/static_ovmap/m2_reviewer_study_v1" / root.name
    index, rows, large_ledgers = InputIndex(), [], []

    def copy(path, relative=None, compress=False):
        rows.append(copy_artifact(path, destination, relative or Path(path).relative_to(root),
                                  compress=compress, index=index))

    for path in sorted(report_root.iterdir()):
        if path.is_file():
            copy(path, path.name, compress=path.name == "report.json")
    for name in ("source_binding.json", "nomination.json", "final_report.json"):
        copy(root / name)
    for directory in ("calibration", "rows", "pooled", "source_audits", "legacy_parity"):
        for path in sorted((root / directory).rglob("*.json")):
            copy(path)  # Principal metric and selection JSON bytes stay exact.
    for path in sorted((root / "query_controls").glob("*.json")):
        copy(path)
    for path in sorted((root / "query_controls/pooled").rglob("*.json")):
        copy(path)
    for scene in binding["scenes"]:
        copy(root / "predictions" / scene / "locked.json")
        copy(binding["scenes"][scene]["config"], Path("configs") / scene / "resolved_config.json")
        for path in sorted((root / "diagnostics" / scene).iterdir()):
            if path.name == "released_attribution.json.gz":
                # Full per-threshold matching traces exceed the compact release
                # budget. Keep exact external identities; owner ledgers and
                # compact AP25/AP50 event summaries are still uploaded.
                large_ledgers.append(index.identity(path))
                continue
            if path.suffix in (".json", ".gz"):
                copy(path, compress=path.suffix == ".json" and path.stat().st_size > 256 * 1024)
        for pattern in ("*_B*/receipt.json", "*_B*/decisions.json", "locked_B*.json", "probabilities_B*.json.gz"):
            for path in sorted((root / "query_controls" / scene).glob(pattern)):
                copy(path, compress=path.suffix == ".json" and path.stat().st_size > 256 * 1024)
    for directory in ("robustness/results", "robustness/text", "robustness/aggregates", "fresh", "validation"):
        for path in sorted((root / directory).rglob("*")):
            if path.is_file() and path.suffix in (".json", ".gz"):
                copy(path, compress=path.suffix == ".json" and path.stat().st_size > 256 * 1024)
    copy(report["cost_ledger"], "cost_ledger.json", compress=True)
    for name in ("folds.json", "final_temperatures.json", "examples.json"):
        copy(Path(binding["composition_root"]) / "calibration" / name, Path("original_calibration") / name)
    # Reuse already-attested dependency hashes; do not crawl the old release or
    # repeatedly hash model weights. Newly packaged bytes are checked below.
    external = {}
    collect_identities(binding["inputs"] + report["inputs"] + large_ledgers, external)
    # Include evaluator contexts and output references actually consumed by
    # these rows. This is a bounded traversal of this study's matrix, not a scan
    # of historical artifacts or a reread of large surfaces/model weights.
    evaluation_paths = set()
    for path in sorted((root / "rows").rglob("*.json")):
        evaluation_paths.add(read_json(path)["evaluation_receipt"])
    for path in sorted(evaluation_paths):
        receipt = read_json(path)
        copy(path, Path("evaluation_receipts") / Path(path).parent.name / "receipt.json", compress=True)
        collect_identities(receipt, external)
    for directory in ("robustness/aggregates", "robustness/text"):
        for path in sorted((root / directory).rglob("receipt.json")):
            collect_identities(read_json(path), external)
    for scene in binding["scenes"]:
        for path in (root / "query_controls" / scene).glob("*_B*/receipt.json"):
            receipt = read_json(path)
            collect_identities(receipt["inputs"] + receipt["outputs"], external)
    write_once(destination / "external_artifacts.json", {"entries": sorted(external.values(), key=lambda r: r["path"]),
               "identity_provenance": "bound inputs and actual execution receipts; not a claim of uploading external bytes",
               "rebuild": f"/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_m2_reviewer_study.py --phase all --output-root {root} --resume"})
    rows.append({"relative": "external_artifacts.json", "artifact": index.identity(destination / "external_artifacts.json")})
    size = verify_bundle(destination, rows)
    write_once(destination / "manifest.json", {"status": "MEASURED_BUNDLE_VERIFIED", "binding": binding["identity"],
               "report": report["identity"], "bytes_without_manifest": size, "files": rows})
    actual = sum(p.stat().st_size for p in destination.rglob("*") if p.is_file())
    if actual >= 100 * 1024 * 1024:
        raise ValueError("full artifact bundle exceeds100MiB limit")
    for name in DOCS:
        copy_artifact(report_root / name, ROOT / "docs/paper/static_ovmap", name)
    return destination


def publish(binding):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()

    if git("branch", "--show-current") != BRANCH:
        raise ValueError("publication requires the prescribed task branch")
    destination = export_artifacts(binding)
    allowed = ["src/static_ovmap/m2_reviewer_study", "tests/m2_reviewer_study",
               "scripts/evaluation/run_ovimap_m2_reviewer_study.py",
               "docs/paper/static_ovmap/m2_reviewer_study_v1",
               "docs/superpowers/plans/2026-09-28-m2-reviewer-evidence.md",
               str(destination.relative_to(ROOT)), *["docs/paper/static_ovmap/" + name for name in DOCS]]
    if any(not any(p == a or p.startswith(a + "/") for a in allowed)
           for p in git("diff", "--cached", "--name-only").splitlines()):
        raise ValueError("unrelated staged files must not enter the publication commit")
    git("add", "--", *allowed)
    # csv.writer uses valid RFC-style CRLF endings. Recognize that convention
    # during whitespace validation without rewriting attested artifact bytes.
    git("-c", "core.whitespace=trailing-space,space-before-tab,cr-at-eol", "diff", "--cached", "--check")
    if git("diff", "--cached", "--name-only"):
        git("commit", "-m", "Publish measured M2 reviewer evidence and reproducible workflow")
    head, remote = git("rev-parse", "HEAD"), git("remote", "get-url", "origin")
    subprocess.run(["git", "push", "origin", f"HEAD:refs/heads/{BRANCH}"], cwd=ROOT, check=True)
    advertised = git("ls-remote", "origin", f"refs/heads/{BRANCH}").split()
    if not advertised or advertised[0] != head:
        raise ValueError("remote full SHA differs from local HEAD")
    receipt = {"status": "PUSH_VERIFIED", "branch": BRANCH, "remote": remote, "local_sha": head,
               "remote_sha": advertised[0], "bundle": str(destination), "binding": binding["identity"]}
    atomic_write_json(Path(binding["output_root"]) / "publication_receipt.json", receipt)
    return receipt
