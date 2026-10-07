"""Commit immutable scientific operators before any new full-cohort GT scores."""

from pathlib import Path
import subprocess

from static_ovmap.cvpr_compact.area_fallback_experiment import seal
from static_ovmap.module_validation.contracts import atomic_write_json
from static_ovmap.recovery_wave2.binding import ConsumptionIndex, read


OPERATORS = ("binding.py", "preparation.py", "evidence.py", "anyup_adapter.py", "acquisition.py",
             "outputs.py", "verifier.py", "evaluation.py", "selection.py")


def freeze(binding, root):
    root = Path(root)
    if (root/"implementation_freeze.json").exists():
        from .evaluation import require_freeze
        return require_freeze(root)
    repo = Path(__file__).resolve().parents[3]
    git_check = ["git", "-c", "core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol"]
    # The supplied matrix has original CRLF bytes; preserve the exact task package.
    subprocess.run([*git_check, "diff", "--check"], cwd=repo, check=True)
    # Only this isolated task's source/spec/test files; previous worktrees/indexes
    # are never staged, and shared model or licensed dataset bytes remain outside Git.
    paths = ["src/static_ovmap/evidence_exploration", "scripts/evaluation/run_ovimap_evidence_exploration.py",
        "tests/evaluation/test_evidence_exploration.py", "configs/static_ovmap/evidence_exploration_v1.json",
        "docs/paper/static_ovmap/evidence_exploration_v1", "docs/superpowers/plans/2026-10-07-evidence-exploration.md"]
    subprocess.run(["git", "add", "--", *paths], cwd=repo, check=True)
    subprocess.run([*git_check, "diff", "--cached", "--check"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "Freeze fixed evidence exploration operators, assets and protocol"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    index = ConsumptionIndex()
    asset = read(root/"assets.json")
    result = seal({"status": "IMPLEMENTATION_CONSTANTS_ASSETS_FROZEN_COMMITTED", "commit": commit,
        "binding_identity": binding["identity"], "constant_specification": binding["specification"],
        "operator_sources": [index.identity(Path(__file__).with_name(name)) for name in OPERATORS]
            + [index.identity(binding["spec"]["path"], binding["spec"])],
        "assets": [index.identity(root/"assets.json"), asset["checkpoint"], *asset["sources"].values()],
        "GT_scores_used_to_choose_constants": False, "new_full_cohort_scoring_started": False})
    atomic_write_json(root/"implementation_freeze.json", result)
    print("IMPLEMENTATION_FROZEN", commit, flush=True)
    return result
