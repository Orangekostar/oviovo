"""Controller phases; only the inherited FC interpreter executes CUDA workers."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.cvpr_compact.projected_views import _verified_identity
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import read

from .binding import REPO, bind_reference, file_identity
from .workflow import call_plan, code_inventory, run_measurements, select_pilot


def source_inventory():
    files = [*sorted((REPO/"src/static_ovmap/runtime_parity").glob("*.py")),
        REPO/"scripts/evaluation/run_ovimap_runtime_parity.py",REPO/"configs/static_ovmap/runtime_parity_v1.json"]
    return {str(path.relative_to(REPO)):file_identity(path)["sha256"] for path in files}


def run_worker(reference, spec_path, root, phase, selected="R0_REFERENCE"):
    summary = root/phase/"summary.json"
    if summary.is_file():
        existing = read(summary)
        _verified_identity(existing)
        if (existing["planned_calls"]!=len(call_plan(read(spec_path),phase,selected))
                or any(row["implementation_sources"]!=code_inventory() for row in existing["measurements"])):
            raise ValueError("completed phase changed sources or declared workload")
        return existing
    env = {**os.environ,"PYTHONPATH":str(REPO/"src"),"CUDA_VISIBLE_DEVICES":reference["gpu"],
        "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    command = [reference["fc"]["python"],str(REPO/"scripts/evaluation/run_ovimap_runtime_parity.py"),
        "--spec",str(spec_path),"--source-worktree",reference["source_worktree"],"--output-root",str(root),
        "--worker",phase,"--selected",selected,"--resume"]
    subprocess.run(command,cwd=REPO,env=env,check=True)
    return read(summary)


def freeze_implementation(reference, spec, root):
    selection,verification,pilot = (read(root/name) for name in ("pilot_selection.json","verification/summary.json","pilot/summary.json"))
    for value in (selection,verification,pilot):
        _verified_identity(value)
    if verification["status"]!="ALL_26_PASS" or verification["selected"]!=selection["selected"]:
        raise ValueError("freeze requires the selected implementation's full 26-scene parity")
    if any(row["implementation_sources"]!=code_inventory() for row in pilot["measurements"]):
        raise ValueError("measured pilot implementation changed before freeze")
    path = root/"implementation_freeze.json"
    if path.exists():
        value = read(path)
        _verified_identity(value)
        if value["all_task_sources"]!=source_inventory():
            raise ValueError("frozen task sources changed; a crash fix requires a new attempt identity")
        return value
    value = seal({"status":"FROZEN_BEFORE_FINAL","selected":selection["selected"],
        "reference_binding_identity":reference["identity"],"reference_commit":reference["reference_commit"],
        "implementation_sources":code_inventory(),"all_task_sources":source_inventory(),"hardware":pilot["hardware"],
        "calls":call_plan(spec,"final",selection["selected"]),"selection_identity":selection["identity"],
        "parity_identity":verification["identity"],"timer_definition":spec["timing"],
        "parent_metric_identity":reference["imported_scientific_identity"],
        "pre_freeze_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()})
    atomic_write_json(path,value)
    compact = REPO/"artifacts/static_ovmap/runtime_parity_v1"
    for name in ("implementation_freeze.json","pilot_selection.json"):
        shutil.copy2(root/name,compact/name)
    shutil.copy2(root/"verification/summary.json",compact/"verification_summary.json")
    shutil.copy2(root/"pilot/summary.json",compact/"pilot_summary.json")
    subprocess.run(["git","add","src/static_ovmap/runtime_parity","scripts/evaluation/run_ovimap_runtime_parity.py",
        "configs/static_ovmap/runtime_parity_v1.json","tests/evaluation/test_runtime_parity.py",
        "docs/superpowers/plans/2026-10-05-runtime-parity.md","artifacts/static_ovmap/runtime_parity_v1"],cwd=REPO,check=True)
    subprocess.run(["git","commit","-m","research: freeze verified runtime implementation before final cold benchmark"],cwd=REPO,check=True)
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    atomic_write_json(root/"freeze_commit.json",seal({"freeze_identity":value["identity"],"commit":sha}))
    return value


def publication_audit(reference, spec, root):
    final,pilot,parity,freeze,report = (read(root/name) for name in ("final/summary.json","pilot/summary.json",
        "verification/summary.json","implementation_freeze.json","tables/runtime_summary.json"))
    for value in (final,pilot,parity,freeze,report):
        _verified_identity(value)
    if freeze["all_task_sources"]!=source_inventory():
        raise ValueError("frozen implementation sources changed")
    expected = 48 if freeze["selected"]=="R0_REFERENCE" else 64
    if final["calls"]!=expected or pilot["calls"]!=16 or report["normal_cold_calls"]>80 or parity["projection_passes"]>52:
        raise ValueError("required full measured workload or budgets are not satisfied")
    if parity["status"]!="ALL_26_PASS" or final["status"]!="COMPLETE":
        raise ValueError("scientific parity is incomplete")
    for row in final["measurements"]:
        _verified_identity(row)
        if row["hardware"]!=freeze["hardware"] or row["implementation_sources"]!=freeze["implementation_sources"]:
            raise ValueError("final hardware/source identity is inconsistent")
        if abs(sum(row["stage_seconds"].values())-row["seconds"])>1e-7:
            raise ValueError("exclusive stage totals do not reproduce the measured total")
        if any(row["work_counters"].get(name)!=0 for name in ("feature_cache_hits","view_cache_hits","result_cache_hits")):
            raise ValueError("cold recovery consumed persistent evidence")
    source = Path(reference["source_worktree"])
    head = subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    gitdir = Path(subprocess.check_output(["git","rev-parse","--absolute-git-dir"],cwd=source,text=True).strip())
    index_sha = hashlib.sha256((gitdir/"index").read_bytes()).hexdigest()
    snapshot = reference["source_snapshot"]
    if head!=reference["source_head"] or index_sha!=snapshot["source_index_sha256"]:
        raise ValueError("original source HEAD or index changed")
    compact = REPO/"artifacts/static_ovmap/runtime_parity_v1"
    size = sum(p.stat().st_size for p in compact.rglob("*") if p.is_file())
    if size>25*1024**2:
        raise ValueError("compact artifacts exceed budget")
    required = [REPO/"docs/paper/static_ovmap"/("RUNTIME_PARITY_"+name+".md") for name in ("RESULTS","HANDOFF","SELECTION","CLAIMS")]
    required += [root/"tables"/name for name in ("tables_main.json","cell_provenance.json","table_layout_preview.pdf",
        "table3_preview.pdf","supplementary_preview.pdf","per_call_timing.csv","per_scene_timing.csv","costs.json")]
    if any(not p.is_file() or not p.stat().st_size for p in required):
        raise ValueError("required publication deliverables are missing")
    audit = seal({"status":"PASS","reference_binding_identity":reference["identity"],"freeze_identity":freeze["identity"],
        "final_identity":final["identity"],"full_workload_calls":expected,"pilot_calls":16,"CPU_projection_passes":parity["projection_passes"],
        "source_HEAD_preserved":True,"source_index_preserved":True,"artifact_bytes":size,
        "deliverables":[file_identity(p) for p in required],"requirement_review":"see requirement_audit.md"})
    atomic_write_json(root/"validation/completion_audit.json",audit)
    atomic_write_json(compact/"completion_audit.json",audit)
    return audit


def publish(reference, spec, root):
    publication_audit(reference,spec,root)
    subprocess.run(["git","add","docs/paper/static_ovmap/RUNTIME_PARITY_RESULTS.md","docs/paper/static_ovmap/RUNTIME_PARITY_HANDOFF.md",
        "docs/paper/static_ovmap/RUNTIME_PARITY_SELECTION.md","docs/paper/static_ovmap/RUNTIME_PARITY_CLAIMS.md",
        "docs/superpowers/plans/2026-10-05-runtime-parity.md","artifacts/static_ovmap/runtime_parity_v1"],cwd=REPO,check=True)
    changed = subprocess.run(["git","diff","--cached","--quiet"],cwd=REPO).returncode
    if changed:
        subprocess.run(["git","commit","-m","research: publish measured runtime parity tables and evidence"],cwd=REPO,check=True)
    sha = subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    result = subprocess.run(["git","push","-u","origin",spec["branch"]],cwd=REPO)
    if result.returncode:
        atomic_write_json(root/"publication/final.json",seal({"status":"PUSH_FAILED","local_sha":sha,
            "retry_command":"git push -u origin "+spec["branch"]}))
        raise RuntimeError("ordinary GitHub push failed")
    remote = subprocess.check_output(["git","ls-remote","origin","refs/heads/"+spec["branch"]],cwd=REPO,text=True).split()[0]
    if remote!=sha:
        raise RuntimeError("remote branch SHA does not equal the local full SHA")
    receipt = seal({"status":"PUSH_VERIFIED","local_sha":sha,"remote_sha":remote,"branch":spec["branch"],
        "reference_binding_identity":reference["identity"]})
    atomic_write_json(root/"publication/final.json",receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec",type=Path,default=REPO/"configs/static_ovmap/runtime_parity_v1.json")
    parser.add_argument("--source-worktree",type=Path,default=Path('/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/worktree'))
    parser.add_argument("--v2-results-root",type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--path-map",action="append",default=[])
    parser.add_argument("--phase",choices=("bind","profile","screen","verify","freeze","benchmark","tables","publish","all"),default="all")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--worker",choices=("pilot","final"),help=argparse.SUPPRESS)
    parser.add_argument("--selected",default="R0_REFERENCE",help=argparse.SUPPRESS)
    args = parser.parse_args()
    spec_path = args.spec if args.spec.is_absolute() else REPO/args.spec
    spec = read(spec_path)
    root = (args.output_root or Path(spec["output_root"])).resolve()
    mapping = {}
    for pair in args.path_map:
        old,sep,new = pair.partition("=")
        if not sep or not Path(old).is_absolute() or not Path(new).is_absolute() or (old in mapping and mapping[old]!=new):
            parser.error("--path-map requires nonconflicting absolute OLD=NEW roots")
        mapping[old] = new
    if args.worker:
        run_measurements(read(root/"reference_binding.json"),spec,root,args.worker,selected=args.selected)
        return
    reference = bind_reference(spec_path,args.source_worktree,root,v2_results_root=args.v2_results_root,gpu=args.gpu,path_map=mapping)
    phases = ("bind","profile","screen","verify","freeze","benchmark","tables","publish") if args.phase=="all" else (args.phase,)
    for phase in phases:
        print("PHASE",phase,flush=True)
        if phase=="bind":
            continue
        if phase in ("profile","screen"):
            pilot = run_worker(reference,spec_path,root,"pilot")
            selection = select_pilot(pilot["measurements"],spec["pilot_scenes"])
            atomic_write_json(root/"pilot_selection.json",selection)
        elif phase=="verify":
            from .verification import verify_all
            verify_all(reference,spec,root,read(root/"pilot_selection.json")["selected"])
        elif phase=="freeze":
            freeze_implementation(reference,spec,root)
        elif phase=="benchmark":
            freeze = freeze_implementation(reference,spec,root)
            run_worker(reference,spec_path,root,"final",freeze["selected"])
        elif phase=="tables":
            from .reporting import build_report
            build_report(reference,spec,root)
        elif phase=="publish":
            publish(reference,spec,root)
