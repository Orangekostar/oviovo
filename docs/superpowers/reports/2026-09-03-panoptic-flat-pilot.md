# Panoptic Mapping Flat Pilot

Date: 2026-09-03

Status: `BLOCKED_DATASET_ACCESS`

## Frozen Source and Access Attempt

The official source is `ethz-asl/panoptic_mapping` commit
`3926396d92f6e3255748ced61f5519c9b102570f`. Its Flat download script has
SHA-256 `7c274aac0b8ca36c8cdb9964feea79f7f3792a38c2d25be35385dbba5b46385c`
and names the legacy archive
`2021_Panoptic_Mapping/flat_dataset.zip`.

The current official ASL catalog identifies the Flat dataset as a 392.7 MB
two-trajectory synthetic indoor dataset and routes downloads through DOI
`10.3929/ethz-c-000788335`. On 2026-09-03, the DOI resolved to the ETH Zurich
Research Collection, which returned `Access Restricted` to this compute
provider. Both HTTP and HTTPS requests to the legacy archive timed out after
30 seconds. A bounded scan of the approved local roots found Panoptic Mapping
binaries and launch files, but no Flat RGB-D, labels, change log, or structural
ground truth.

## Protocol Validation

The adapter fixes `run1` before `run2` and assigns a monotonic global frame
index without assuming that timestamps continue across visits. Within each
run it requires unique frame IDs and strictly increasing integer timestamps.
Every selected frame requires paired RGB PNG, 32-bit floating TIFF depth, a
finite 4x4 pose, and the condition-specific segmentation input. Global labels,
the moved/added/removed change log, both structural GT point clouds, and every
consumed frame member are content-hashed.

The conditions remain disjoint:

| Condition | Classification | Input |
| --- | --- | --- |
| `Flat-GT-Panoptic` | `ORACLE_DIAGNOSTIC` | Ground-truth panoptic labels |
| `Flat-Predicted-Panoptic` | `NON_ORACLE` | Detectron panoptic image and labels |
| `CROVE-Open-Vocabulary` | `NON_ORACLE` | RGB-D and pose; frontend output supplied separately |

The current-state evaluator rejects future-frame predictions and reports
current object precision/recall, moved/added/removed recall, stale geometry
false-positive mass, current geometry F-score at 5 cm, free-space recall, and
recovery latency. Aggregation across input conditions is rejected. Synthetic
fixtures validate ordering, pairing, depth type, timestamps, causality,
condition separation, count aggregation, and recovery latency.

## Runtime Gate

- adapter and metric protocol: `PASS`;
- official source/catalog identity: `PASS`;
- Flat archive retrieval: `BLOCKED_DATASET_ACCESS`;
- A6/P5/P6-C runs: not run;
- GT-panoptic oracle and predicted-panoptic metrics: unavailable, not zero;
- signed-visibility/current-ownership mechanism sensitivity: unmeasured.

## Decision

Keep Flat as a preregistered controlled mechanism candidate. It cannot support
a benchmark promotion, a favorable CROVE claim, or a conclusion that TESSE is
task-misaligned until the official archive is available and the frozen
A6/P5/P6-C conditions are run without tuning.
