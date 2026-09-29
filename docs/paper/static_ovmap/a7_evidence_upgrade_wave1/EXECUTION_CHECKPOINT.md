# Execution checkpoint — A7 wave1

Status: IMPLEMENTATION_IN_PROGRESS; no new measured scene rows yet.

Worktree: /home/ww/crove/ovimap-a7-evidence-upgrade-wave1
Branch: research/ovimap-a7-evidence-upgrade-wave1
Base:1074eb746808ca6882b30ccd827167474bb7eb8f
Authorized root: /mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001

- Read the full execution directive (including separately rereading sections6–10
  that were truncated in the first tool response), protocol and SOURCES.md.
  Preserved all ten original ZIP files with their nested input_reference paths.
  Dedicated worktree created at the exact published base; no historical edits.
- /home/ww/AGENTS.md applies. Root owns scientific and algorithm decisions.
  Only fully specified mechanical leaves may use Luna mechanical_worker.
- Read ccf-experiment-designer and its evidence-design/result-template resources.
  The supplied protocol takes precedence; no new venue, datasets or extra study
  is inferred. Existing three A40 GPUs were idle at initial inventory; shared
  storage had432GiB free. Recheck before actual workers.
- Added task-local readouts.query_readout for RAW_EQ, UNIT_EQ and protocol GMED.
  Float64, eps1e-4,64 steps,tol1e-8, accept-before-stop, no inner normalization.
  n1 direction/n2 midpoint; final norm<=1e-12 uses genuine original area
  direction; nonfinite/missing/zero input is an input error, not availability.
  Records norm spread, normalized influence weights/ESS and convergence.
- Two compact E01 tests pass; task-local Ruff passes. These are mathematical
  fixtures only, not scene results and not a visual adapter smoke.
- Next: bind actual old source configs and unchanged static manifests; export
  actual retained Q raw features and reconstruct area scores exactly, then run
  E01 CAL source generation/fits/metrics. Inspect allowed model assets and obtain
  only pinned official checkpoints within25GiB; record actual access failures.
  Do not assume SAM3/OVR is unavailable before checking official access.
- Full scope remains E01→E02→C0→E03→E04 CAL, one compatible pair and nomination,
  all executable Replica conditions, two ranks/released pools, four tables,
  three reports, <=40MiB bundle and verified normal push. No completion claim.
