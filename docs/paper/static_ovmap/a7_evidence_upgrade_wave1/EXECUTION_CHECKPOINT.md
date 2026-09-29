# Execution checkpoint — A7 wave1

Status: IMPLEMENTATION_IN_PROGRESS; E01 CAL sources/fits/44 rank rows complete.

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

Real binding/E01 CAL continuation:

- Added binding.py, e01.py, calibration.py and evaluation.py. Actual10-scene
  binding is3036348dd8e035215c1d1543a7ef0f6f9f300e79981c8b8c887321108800b998.
  Original static manifests/config/source identities are bound; base ancestry
  and branch checked. New-model assets are explicitly still unresolved.
- CAL scene0056:99 owners/96 available Q; scene0534:93/83. Recovered actual
  final retained raw six-crop means and area weights; retention membership/order
  and original aggregate score parity exact (maximum error0). No Q replay or
  new image inference. Both scenes have three all-owner source JSONs and NPZs.
- All three variants fitted two opposite-scene scalars and one final CAL scalar
  on22 genuine geometrically matched objects. Final T=.010000492837221748 for
  each, lower bound recorded. Untouched source settings are original A7_REFIT,
  including N0 cosine fold/final temperatures, not old canonical-relative N0.
- Actual evaluation completed44 scene/rank rows and22 CAL pools (five controls
  plus six E01 methods). All reference scene metrics and10 reference pools
  exactly match the parent same-rank records. All whole-map predictions locked
  before metrics; DIRECT unavailable owners use explicit N0 labels. Replica
  evaluation is rejected until the full CAL nomination exists.
- CAL official pooled A7_REFIT APall3.732776%, mIoU22.680712%; all three E01 A7
  APall values identical. RAW_EQ/UNIT_EQ mIoU22.698982%; GMED22.729982%. These are
  development measurements, not candidate nomination or transfer improvement.
  e01/CAL_complete.json records measured completion of this family/split only.
- Official HEAD probes: SAM2 weight HTTP200/898083611 bytes; pinned SO400M
  HTTP200/4544143072 bytes. Google OVR author page HTTP200 HTML does not prove
  checkpoint access; still unresolved. Official SAM3 weight HTTP401; existing
  huggingface_hub authentication absent. Do not bypass its gate with a mirror.
- SAM2 official download completed (session19206 exit0),0.836GiB of25GiB model
  allowance. Final file assets/sam2/sam2.1_hiera_large.pt; SHA256
  2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318.
  assets/sam2/download_receipt.json records actual size/hash/source and missing
  machine-timed download duration. Code clone checked out exact required pin
  2b90b9f5ceec907a1c18123530e92e794ad901a4. No SAM image smoke performed yet.
- Existing native/S2 workers have Torch2.1.1 and must remain unchanged. The
  existing /home/ww/miniconda3/envs/oviovo-ovo-official/bin/python has Torch2.5.1,
  torchvision0.20.1, hydra-core1.3.4, iopath0.1.10: compatible SAM2 candidate.
  No install/version sweep or environment upgrade was performed. Pinned SAM2
  build/predictor imports succeeded with Torch2.5.1+cu124/CUDA12.4; this is
  separate from the mandatory real-image smoke.
- Two readout fixtures and task-local Ruff pass. Real source/evaluation/pool
  checks above provide the additional integration evidence, not more synthetic
  test counts. No active visual experiment or download remains after these
  completed sessions. Next implement E02 recognition crops/SAM2 worker and real
  deterministic CAL smoke; then remaining CAL families, composition/selection,
  every executable Replica family, reports and verified publication.
