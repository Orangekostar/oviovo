# Khronos RSS 2024 Paper-Protocol Reproduction Audit

Date: 2026-09-03

Status: `BLOCKED_PAPER_ASSET`

## Decision

The public evidence is insufficient for a `PAPER_PROTOCOL_EXACT` claim. The RSS
release contains the mapper and TESSE launch/configuration path, but contains no
`khronos_eval` implementation. The available evaluator, Apartment evaluation
configuration, and aggregation utilities are post-release. No paper-like run
was executed because doing so would produce a number without exact evaluator
and data identity.

This blocks interpreting the RSS Table I values as a locally reproduced,
same-protocol baseline. It does not invalidate TESSE-CD, the published paper
values, or the B2 aggregation parity result.

## Frozen Paper Protocol

The primary source is the [Khronos RSS 2024 paper](https://www.roboticsproceedings.org/rss20/p081.html).
The target is Table I, Apartment, ground-truth pose, simulator ground-truth
semantics, voxel resolution 8 cm, and sensing range 5 m.

| Metric | Paper target |
| --- | ---: |
| `background_f1` | 0.912 |
| `object_f1` | 0.753 |
| `dynamic_f1` | 0.841 |
| `change_f1` | 0.646 |

The absolute F1 tolerance was frozen before any reproduction at `0.02`. These
numbers are reproduction targets only; they are not current CROVE measurements.

## Source Identity

| Identity | Commit/status | Evaluator evidence |
| --- | --- | --- |
| RSS release | `742227a88de8b2ac23ac54d719b321c3af88dc75` | 0 files under `khronos_eval/`; evaluator not present |
| Latest official public tree | `63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e` | 76 files under `khronos_eval/`; post-release evaluator |
| Existing CROVE A6/P5 artifact | `UNPROVEN_ARTIFACT_EVALUATOR_IDENTITY` | Importer source/binary are bound, evaluator closure and dirty patch set are not |

The RSS release README links the
[simulated paper datasets](https://drive.google.com/drive/folders/1Miv2uPBYzg_FyqMX5vAWI4MDZ70sPLW-)
and describes the Apartment launch change. It does not provide a source-bound
Table I evaluator or evaluation command. The current public
`khronos_eval/config/pipeline/apartment.yaml` and
`khronos_eval/scripts/evaluate_pipeline.sh` therefore establish a public
approximation path, not paper identity.

## Blocking Evidence

1. `git ls-tree -r --name-only 742227a...` finds no `khronos_eval/` file.
2. The current public evaluator/configuration is newer than the paper release.
3. The bounded local inventory contains Apartment ground truth and a derived
   RGB-D export, but no content-addressed original paper rosbag.
4. Existing CROVE artifact records do not bind the evaluator executable closure
   or the dirty source patch set.

Because these are identity gaps, a numerically close post-release run would
still not satisfy `PAPER_PROTOCOL_EXACT`. Conversely, `BENCHMARK_REPRODUCTION_NO_GO`
is not asserted because no source-complete reproduction was available to run
and fail.

## Relationship to B2

B2 proves that the local summarizer matches the frozen latest-public Khronos
aggregation code on A6 and P5 CSVs to `1e-12`. It does not prove that those CSVs
contain the full paper temporal surface, that their evaluator equals Table I,
or that CROVE received simulator ground-truth semantic observations. Therefore
B2 remains `PARITY` while B3 remains `BLOCKED_PAPER_ASSET`.

## Unblock Gate

Reproduction may proceed only after all of the following are content-addressed:

- the original Apartment paper dataset or an author-published equivalent;
- the exact Table I evaluator and configuration, or author evidence binding the
  post-release implementation to Table I;
- a clean GT-pose, simulator-GT-semantics, 8 cm, 5 m command and output manifest.

Only a run satisfying those conditions and all four `0.02` gates may be labeled
`PAPER_PROTOCOL_EXACT`.

## Reproducible Audit Commands

```bash
git -C <KHRONOS_CHECKOUT> ls-tree -r --name-only \
  742227a88de8b2ac23ac54d719b321c3af88dc75 | \
  awk '/^khronos_eval\// {n++} END {print n+0}'

git -C <KHRONOS_CHECKOUT> ls-tree -r --name-only \
  63faadde6ed92220e78fb2f6ca86dcc54bb5cf9e | \
  awk '/^khronos_eval\// {n++} END {print n+0}'

python -m pytest -q tests/evaluation/test_khronos_source_manifest.py
```

Machine-local paths and unbound runtime outputs are intentionally excluded from
the public manifest.
