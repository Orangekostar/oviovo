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

**Commit:** `PENDING_THIS_COMMIT`

**Artifacts:**
`docs/superpowers/reports/2026-09-03-khronos-metric-parity.md`; A6 audit JSON
SHA-256 `4ecc68bc54f0fff56becacb4d0900fc8cbe19bc2656cd1f213ddce0282b8a6d1`;
P5 audit JSON SHA-256
`f9d4e9e578ea69ee50df85576f0a9a5e6f533fd5ad57844688846e0257bca8e4`.
