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

E02 real CAL continuation (supersedes the earlier no-smoke/no-worker state):

- Added crops.py and sam2_worker.py. One deterministic original CAL request
  passed the real pinned SAM2 BF16 smoke with original-resolution output and
  exact original six-crop pixel parity. Full CAL mask runs completed all280/264
  requests, using142/141 image encodings respectively, with no failed masks.
  Smoke is a separately counted additional image encoding/box decode.
- Added recognition_worker.py. Both GLOBAL and SAM2 now have genuine S2
  sources for both CAL scenes:280/264 successful requests and97/91 available
  owners each. All use original request/view order, bbox, areas, own S2 text,
  per-crop unit vectors, raw six-crop view means and final-only normalization.
  Original-model/processor/code/request/geometry identities are verified before
  reusing actual raw crop vectors0/2/4; every request computed only3 changed
  foreground crops. Foreground survives the actual processor. No extra
  background crops, fourth fusion vote, new geometry or GT-dependent requests.
- Both E02 variants' fits and16 scene/rank rows plus8 pools completed (evaluation
  sessions96770/75001 exit0). Final scalars: GLOBAL.010000492837221748,
  SAM2.010800840632491917. Official CAL A7 GLOBAL: APall3.732776%,
  mIoU22.645275%; A7 SAM2: APall3.743512%, AP5014.230065%, AP2528.265637%,
  mIoU22.457145%, mAcc30.325458%. SAM2 has an AP/mIoU tradeoff on CAL, not an
  established net/transfer gain. No nomination or Replica evaluation yet.
- Four task-local fixtures passed, including actual historical crop pixel
  conventions and an independent norm-sensitive area-aggregation fixture.
  Task-local Ruff passed. Real GPU results above are separate integration
  evidence; no parent test/benchmark sweep was rerun.
- Added capacity_assets.py: official exact-revision SO400M download runs in
  session57562 (asset root assets/so400m). Added ovr_assets.py: official author
  Google download runs in session63986 (assets/ovrcoat). Its real HEAD returned
  HTTP200 application/octet-stream, filename ovrcoat.pth,4633915328 bytes.
  OVR source clone is detached at9fd9450d22852d269d426b521663a127f3983a4b.
  Neither checkpoint is yet claimed smoke-tested. Planned downloaded weights
  SAM2+SO400M+OVR total about9.39GiB, leaving room below25GiB for matched FC.
- Next: finish/check E02 CAL pools; implement dynamic-dimension C0 original
  union six-crop worker with its own text/processor and real CAL smoke; inspect
  official OVR active branch and matched OpenCLIP control; persist SAM3 access
  evidence. Full remaining composition, nomination, all executable Replica
  families, diagnostics, costs, reports and verified branch push remain open.

C0 complete CAL and E03 operator preparation:

- Official SO400M downloaded successfully at exact required revision; receipt
  assets/so400m/download_receipt.json. capacity_worker.py passed a real original
  CAL six-crop smoke (session40980 exit0), own image/text dimension1152. FP32
  model parameter count1136008498. No hardcoded S2 feature dimension, no native
  tokenizer/text substitution, original union masks and six crops retained.
- CAL sources completed in sessions32449/32318 exit0:280/264 genuine requests,
  97/91 owners,1680/1584 physical image crops. Worker wall times126.74/114.53s,
  peak allocated4799970304 bytes. Smoke is separate additional work.
- C0 final CAL temperature.010355136194431317; all8 scene/rank rows and4 pools
  completed (92002 exit0). Official pooled DIRECT APall4.291716%, mIoU21.910611%.
  A7 APall3.730059%, AP5014.091911%, AP2527.998771%, mIoU22.812833%,
  mAcc30.105945%: CAL tradeoff, not nomination or transfer result.
- OVR official checkpoint download completed:4633915328 bytes, SHA256
  a72c560721f5a83addf1264df56632dac56137c0f8f6023b8c638baa616aabfd,
  measured download275.38s. Both backbone and frozen_backbone contain543 keys,
  351772609 elements each. Primary inspection confirms all learned-branch
  key names/shapes exactly match the existing OpenCLIP3.3.0/timm1.0.19 model.
- Matched original FC weights resolved through OpenCLIP's registered official
  LAION HF repository before inference, revision654d0f80ff73c58e7281a3ca7dc425589049e2e1.
  fc_assets.py downloaded only the1.407GB safetensors and small tokenizer/config
  files, total1410764773 bytes,70.92s. It does not use OVR frozen_backbone as a
  substitute for independently obtained original weights.
- Added region_adapter.py: strict full key loading and changed-trunk audit;
  pinned unmodified upstream AST operators for dense trunk, MaskPooling,
  projection and classification; exact14 prompts read from pinned source.
  Explicit RGB PIL bilinear800/max1333 geometry, nearest binary mask, signed
  negative padding, >0 dense support. Full upstream Detectron wrapper is not
  executed; operator reuse/parity is disclosed instead. No segmentation head.
- Five compact task tests passed; task-local Ruff passed. Real two-mask smoke
  processes58063(FC_FROZEN GPU0) and59079(OVR GPU1) both exited0 with
  SMOKE_COMPLETE. Feature dimension768; maximum two-mask feature differences
  .16861537/.13443679; both official pre-scale cosine errors7.45058e-8.
  Strict full loading occurs after OpenCLIP constructor's random-init warning;
  the smoke never infers with those initial random weights. Real learned trunk
  differences and all loaded/excluded key mappings are in weight_audit.json.
- Next: finish E03 real smoke and full static dense workers/CAL evaluations;
  E04 access/targets, one composition and nomination, all executable Replica
  families, reports/publication remain required. No Replica GT evaluation or
  candidate selection has been performed in this wave.

Live E03 / NFS state at the end of this continuation:

- Added region_worker.py for complete static FC/OVR sources: one encoding per
  distinct selected frame/model, own14-prompt prototypes, bit-packed original,
  resized and dense masks, actual dense support, no void/logit-scale in scores,
  original-area aggregation and all-owner availability. Three workers launched:
  session56515 PID1247864(scene0056 FC, GPU0),27831 PID1247860(scene0534 FC,
  GPU1),83582 PID1247856(scene0056 OVR, GPU2). All are authoritatively live in
  stateD/rpc_wait_bit_killable after~4.5min. GPU memory remains0. Do not restart
  them on observation timeout or assume source generation succeeded. The fourth
  run scene0534 OVR is not launched yet; reuse a GPU after its worker terminates.
- Shared mount is192.168.100.102:/mnt/data/shared, NFS4.2 hard/timeo600/retrans2.
  A read-only source-count process(session61856) also waits on this mount.
  Local repository reads/tests continue to work. No remount, kill or duplicate
  experiment has been attempted. This is a verified live I/O wait, not a
  terminal task failure and not a reason to drop E03 or shrink the matrix.
- Official SAM3 metadata probe resolved original repo revision
  3c879f39826c281e95690f02c7821c4de09afae7, gated=manual. Exact-revision
  sam3.pt HEAD returned401/X-Error-Code=GatedRepo (session18955 exit0). Added
  sam3_access.py to persist an explicit credential-aware access receipt; its
  process73747/PID1249764 is also live in NFS wait, so access_probe.json is not
  yet claimed written. Existing token was absent in the earlier probe; actual
  credential-aware result must be inspected when it completes. No gate bypass.
- Added localized.py and one compact E04 fixture. Common-view exclusion,
  zero-detection fallback and exact tie behavior pass. Frozen candidates use
  only original source labels/static views/base A7 actual probabilities; no
  E01–E03 outputs. E04 SHORTLIST needs no SAM3 weights and can remain executable
  if SPATIAL/MIX50 are access-blocked. Candidate manifests/evaluation wiring are
  not yet done; no SAM3 inference or fabricated blocked metric exists.
- Review follow-ups before final audit: preserve complete environment snapshots
  after the successful C0/OVR smokes; provide explicit aggregate-feature and
  original-area source-contract sidecars without rewriting measured source or
  metric JSON bytes; distinguish target-cap vs technical fallback; implement
  remaining phase runner, selection/interaction, costs, diagnostics and reports.
