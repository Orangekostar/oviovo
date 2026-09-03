# CROVE Benchmark Alignment Decision Ledger

Date: 2026-09-03

Base commit: `668aefc49034ef97d090b811b1c3f291ecafbd66`

Rule: every stage distinguishes `CODE_EVIDENCE`, `MEASURED_EVIDENCE`,
`LITERATURE_EVIDENCE`, `HYPOTHESIS`, and `BLOCKED`. Pilot outcomes cannot alter
the B1 suitability scores frozen before those outcomes are viewed.

## B0 — Freeze Repository, Protocol, and Evaluator Identity

**Question:** Which exact public Khronos sources and which exact evaluator
identity support the existing TESSE evidence?

**Evidence before:** `CODE_EVIDENCE` from frozen A6/P5/P6 reports, artifact
manifests, the required base commit, and the master audit prompt.

**Frozen source:** `configs/evaluation/external_benchmark_sources.json`, with
Khronos release `742227a88de8b2ac23ac54d719b321c3af88dc75`, latest public source
`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`, and separately recorded artifact
evidence hashes.

**Frozen protocol:** Distinguish RSS release, historical full-triangular public
evaluator, current post-release public evaluator, and the evaluator actually
used to create P5. Repository ownership alone does not establish equivalence.

**Command:** Inspect the exact official Git history and trees; count
`khronos_eval/` paths at the RSS release; trace the current evaluator loop;
hash the P5 build manifest, importer source/binary, and evaluation status; and
inventory local benchmark metadata without reading any new pilot result.

**Result:** `CODE_EVIDENCE`: the RSS release has zero `khronos_eval/` files.
The recoverable public evaluator lineage is post-release. The P5 manifest binds
the importer source and binary but not `evaluate_pipeline.sh`, its executable
closure, the Khronos commit, or the dirty source patch set. Therefore the exact
artifact evaluator status is `UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY`.

**Deviation from paper:** No score or artifact was changed. Existing P5 results
remain historical evidence but are not relabeled as an exact RSS Table I
reproduction.

**Decision:** Keep TESSE as required evidence and make exact paper-protocol
identity plus aggregation parity the first hard gates before interpreting its
method ranking.

**What this rules in/out:** Rules in B2/B3 source-bound parity and reproduction
work. Rules out assuming that latest official source, RSS release source, and
the locally used evaluator are identical.

**Commit:** `9d827cf8e1f765831dea20731ec6f4459f97e7e8`

**Artifacts:** `configs/evaluation/external_benchmark_sources.json` and
`docs/superpowers/reports/2026-09-03-dynamic-mapping-literature-benchmark-audit.md`.

## B1 — Freeze Literature Evidence and Benchmark Suitability Before Pilots

**Question:** Which benchmark protocols can test current state, causal dynamics,
long-term identity, geometry, and fair open-vocabulary inputs before CROVE's
pilot performance is known?

**Evidence before:** `LITERATURE_EVIDENCE` from 12 source-grounded papers and
their frozen official repositories; no CROVE 3RScan or Flat pilot output was
viewed or aggregated.

**Frozen source:** The B0 source registry and
`docs/superpowers/reports/2026-09-03-dynamic-mapping-literature-benchmark-audit.md`.

**Frozen protocol:** Twelve dimensions scored on 0/1/2 with a rationale and
source ID for every cell. Scores measure suitability, not expected method
performance, and cannot alone promote or remove a headline benchmark.

**Command:** Search only primary proceedings, project, arXiv, and official
repository sources; extract dataset, input, temporal, GT, metric, provenance,
reproducibility, and claim-fit fields; then serialize and validate the pre-result
scorecard.

**Result:** `LITERATURE_EVIDENCE`: suitability totals are TESSE official 16,
TESSE current diagonal 18, 3RScan 19, and Panoptic Flat 16. Their strengths are
complementary: TESSE uniquely covers continuous dynamics, 3RScan provides the
strongest real cross-session identity evidence, and Flat isolates controlled
current-surface maintenance.

**Deviation from paper:** None. No paper number is copied into a local CROVE
result, and no adapted diagnostic is labeled official.

**Decision:** Retain all four protocols as separately labeled evidence routes.
Do not replace TESSE or select a new headline benchmark until B2-B8 establish
identity, parity, attribution, adapter validity, and pilot reproducibility.

**What this rules in/out:** Rules in a split evidence package and strict-prefix
pilots. Rules out benchmark shopping, GT oracle rows as final method results,
and promoting 3RScan or Flat because of favorable later scores.

**Commit:** `9d827cf8e1f765831dea20731ec6f4459f97e7e8`

**Artifacts:** `configs/evaluation/benchmark_suitability_pre_result.json`
(SHA-256 `f4de8e629dacc3a8e13a1123636179fe3c20ec0a145bbcdf1aff87c64a607694`),
the source registry, and the literature audit report.

## B2 — Source-Bound Khronos Aggregation Parity

**Question:** Does the local TESSE summarizer change upstream row identity,
duplicate collapse, time slices, or final Object/Dynamic/Change/Background F1?

**Evidence before:** `CODE_EVIDENCE` from the frozen latest Khronos plotting
source; `MEASURED_EVIDENCE` from immutable A6 and P5 official CSVs. No new method
run or parameter search was performed.

**Frozen source:** Khronos commit
`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`, `utils.py` SHA-256
`a7c1bed4b97f8d9e27361296d00cf741ed67d18ae4d3f8e21e259fc4213d0763`,
and the per-file artifact hashes in the B2 report.

**Frozen protocol:** Compare exact normalized row keys, duplicate behavior, all
four upstream slice modes, and final finite metrics with `rtol=0` and
`atol=1e-12`; preserve a metric as N/A when both paths have no finite state.

**Command:** Run `audit_khronos_metric_parity.py` twice with the source-bound A6
and P5 result directories, the frozen Khronos checkout/commit, and separate
atomic JSON outputs.

**Result:** `MEASURED_EVIDENCE`: both audits return `PARITY`. A6 maximum absolute
delta is `1.3877787807814457e-17`; P5 maximum absolute delta is the same. A6
Dynamic F1 remains N/A on both paths. Each object CSV has 946 raw rows but only
43 unique diagonal `(Name, Query)` states.

**Deviation from paper:** This is a post-release aggregation parity audit, not
an RSS 2024 Table I reproduction. No official CSV or historical metric changed.

**Decision:** Rule out local summarizer arithmetic and duplicate handling as the
cause of A6/P5 scores. Keep the paper-protocol identity gate open because the
available artifact lacks off-diagonal historical states.

**What this rules in/out:** Rules in evaluator/protocol identity and
metric-task mismatch as live causes. Rules out changing local F1 arithmetic to
improve the score.

**Commit:** `20c9dc6`

**Artifacts:**
`docs/superpowers/reports/2026-09-03-khronos-metric-parity.md`; A6 audit JSON
SHA-256 `4ecc68bc54f0fff56becacb4d0900fc8cbe19bc2656cd1f213ddce0282b8a6d1`;
P5 audit JSON SHA-256
`f9d4e9e578ea69ee50df85576f0a9a5e6f533fd5ad57844688846e0257bca8e4`.

## B3 — Khronos Paper-Protocol Identity Gate

**Question:** Can the RSS 2024 Table I Apartment, GT-pose,
simulator-GT-semantics condition be reproduced with source-bound public inputs?

**Evidence before:** `LITERATURE_EVIDENCE` from the RSS paper and release
README; `CODE_EVIDENCE` from exact release/latest Git trees and the B0 artifact
identity audit. B2 aggregation parity is not treated as paper-protocol parity.

**Frozen source:** RSS release
`742227a88de8b2ac23ac54d719b321c3af88dc75`, latest official public source
`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e`, and the content hashes in
`configs/external/khronos_source_manifest.json`.

**Frozen protocol:** Apartment, GT pose, simulator GT semantics, 8 cm, 5 m;
reference F1 values Background `0.912`, Object `0.753`, Dynamic `0.841`, Change
`0.646`; absolute tolerance `0.02`, frozen before any run.

**Command:** Count `khronos_eval/` files in both exact Git trees, hash the
release/latest source files, inventory only bounded TESSE assets, and run the
source-manifest contract tests. Do not execute an unbound post-release
approximation as though it were Table I.

**Result:** `BLOCKED`: the RSS release has zero evaluator files; the public
evaluator/configuration is post-release; the local inventory lacks a
content-addressed original paper rosbag; and A6/P5 do not bind the evaluator
closure or dirty patch set. Reproduction status is `BLOCKED_PAPER_ASSET` and no
observed paper-like metrics were created.

**Deviation from paper:** None was run. Existing B2 A6/P5 aggregation parity is
kept separate from RSS Table I protocol identity.

**Decision:** Do not label current CROVE or post-release evaluator outputs
`PAPER_PROTOCOL_EXACT`, and do not use the Table I values as a local
same-protocol baseline until the source, data, config, and evaluator identities
are content-addressed.

**What this rules in/out:** Rules in the public post-release evaluator as a
future explicitly approximate diagnostic once inputs are bound. Rules out
attributing CROVE-versus-paper score differences to the mapping method alone or
changing official arithmetic to close the gap.

**Commit:** `02214ac`

**Artifacts:** `configs/external/khronos_source_manifest.json` and
`docs/superpowers/reports/2026-09-03-khronos-protocol-reproduction.md`.

## B4 — Non-Interfering Khronos Exact Attribution

**Question:** Can evaluator-native object identities explain the TESSE metric
mass without changing any official Khronos output?

**Evidence before:** `CODE_EVIDENCE` from Khronos commit
`63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e` and `MEASURED_EVIDENCE` from the
frozen P5 Apartment map. B3 remains blocked and is not reinterpreted.

**Frozen source:** The exact public commit above, reviewed patch SHA-256
`4d5bd9033b92758e0062e873f9ed19b45a4c7516cb8945930d30c85a660b38c7`,
and P5 input-map SHA-256
`2e3609cb3caa84adc8779e1b358463eafe63b05deeb8a25a4a6daef6730c69ec`.

**Frozen protocol:** Replay all 43 P5 Apartment states twice with the same map
and evaluator configuration. The patched replay may emit JSONL sidecars only;
all three official CSV files must remain byte-identical, and sidecar TP/FP/FN
mass must equal every corresponding official CSV row.

**Command:** Build the clean public source with the reviewed patch; run
unpatched and patched `exp_pipeline`; then run
`audit_khronos_exact_attribution.py` with the source, patch, input map, both
result directories, and both sidecars bound by SHA-256.

**Result:** `MEASURED_EVIDENCE`, status `REAL_REPLAY_PASS`. All three official
CSV pairs are byte-identical. The exact sidecars contain 118,859 static/change
events and 1,135,450 dynamic events. Every per-row Object, Dynamic, Appeared,
and Disappeared TP/FP/FN total matches the official output.

**Deviation from paper:** This is an adapted post-release P5 diagnostic, not an
RSS Table I reproduction and not an eligible ranking result.

**Decision:** Use the exact sidecars for B5 current-diagonal and failure-source
diagnostics. Preserve the B3 paper-protocol block.

**What this rules in/out:** Rules in object-level attribution of P5 failures.
Rules out the sidecar patch and local aggregation as causes of any official
metric change.

**Commit:** `deaf8ae`

**Artifacts:** `external_patches/khronos_eval_exact_attribution.patch`,
`src/evaluation/khronos_attribution.py`, the audit CLI, and the B4 section of
`configs/external/khronos_source_manifest.json`.

## B5 — TESSE Current Diagonal and Input Fairness

**Question:** Do frozen CROVE variants rank differently on a verified current
slice, and can matched GT-semantics conditions separate frontend from mapper
limitations?

**Evidence before:** `CODE_EVIDENCE` from B2/B4 row and attribution contracts;
`MEASURED_EVIDENCE` from frozen A6, c553 static-anchor, P5, and P6-C outputs.
No new mapper training or tuning was performed.

**Frozen source:** Each method's static/dynamic CSV and map timestamp hashes,
plus B4 exact sidecars for P5. K0/K1/C0/C1 meanings and the 50% strict-majority
closure rule were frozen before these results.

**Frozen protocol:** `TESSE_CURRENT_DIAGONAL` requires verified
`belief_time == robot_time`, causal trajectory timestamps, exact duplicate
agreement, and Khronos-compatible NaN/F1 aggregation. Oracle inputs are always
non-ranking and fail closed without source-database and frame alignment proof.

**Command:** Run `evaluate_tesse_current_slice.py` for all four frozen methods,
with B4 sidecars for P5; run the semantic-oracle builder; then run
`audit_tesse_input_fairness.py` on K0/K1/C0/C1 status manifests.

**Result:** `MEASURED_EVIDENCE`: every artifact has 946 raw rows but only 43
unique current-diagonal states. Current and available post-release official
values and ranks are identical, with zero reversals across six finite pairwise
comparisons. Material mismatch is false on this diagonal-only evidence.
Fairness is `INCONCLUSIVE_MISSING_CONDITION` because K0 and C1 are unavailable;
no missing value was converted to zero.

**Deviation from paper:** This does not reproduce a historical RSS 4D grid.
The current diagonal and common-v2 metrics are explicitly diagnostic.

**Decision:** Do not claim a favorable TESSE task-mismatch reversal or a
frontend/backend diagnosis. Preserve the actual finding that the available
evaluator artifacts cannot test retrospective task alignment.

**What this rules in/out:** Rules in B6/B7 independent causal pilots. Rules out
using the current public loop's duplicate rows as historical evidence and
rules out direct C0-versus-paper comparisons under unmatched semantics.

**Commit:** `PENDING_THIS_COMMIT`

**Artifacts:** `docs/superpowers/reports/2026-09-03-tesse-input-fairness.md`,
`docs/superpowers/reports/2026-09-03-tesse-current-vs-full4d.md`, and the B5
evaluation modules and CLIs.
