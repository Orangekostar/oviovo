# M2 reviewer study handoff

Status: Measured study complete; scientific support is mixed. Branch: `research/ovimap-m2-reviewer-evidence-v1`.
Base: `8fee8294c1a3e83feeae28782f6ef4700f08d35c`.
External output root: `/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001`.
Report source: `/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001/reports/4357cf36788db730187506848266122131aad3b51dd643e43dc2dbc9b96e55ea`. Final remote SHA belongs in the external publication
receipt; no recursive self-reference or unverified push claim is made here.

## Rebuild/resume

Run from the study worktree with the existing native environment:

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_m2_reviewer_study.py --phase all --resume
```

Query leaves use separate processes and per-job/per-GPU locks. Existing completed
sources and outputs are verified rather than rerun. Native and S2 text workers
use their original separate Python environments. No model/data downloads or
environment reinstall are required. Do not concurrently duplicate a live job;
inspect processes and locks before resuming. Publication requires final stage
readiness, bundle integrity, a normal branch push and exact full remote SHA match.

## Boundaries

N0 deployment and historical source artifacts are unchanged. Raw images, model
weights, surfaces and large arrays stay in shared storage. The small release
contains actual tables/scalars/ledgers plus precise external references, not a
claim that the large referenced bytes were uploaded. The original supplied
specification and SHA manifest are preserved in docs/paper/static_ovmap/
m2_reviewer_study_v1/. Fresh data is unavailable; the real success path is
implemented but unexecuted on a genuinely new scene. Missing initial timing and
failed-attempt overhead remain disclosed.
