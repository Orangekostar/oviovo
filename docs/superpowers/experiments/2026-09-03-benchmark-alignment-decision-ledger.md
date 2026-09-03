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

**Commit:** `PENDING_THIS_COMMIT`

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

**Commit:** `PENDING_THIS_COMMIT`

**Artifacts:** `configs/evaluation/benchmark_suitability_pre_result.json`
(SHA-256 `f4de8e629dacc3a8e13a1123636179fe3c20ec0a145bbcdf1aff87c64a607694`),
the source registry, and the literature audit report.
