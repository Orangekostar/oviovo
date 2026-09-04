# OVI-MAP x ReScene4D TESSE Two-Visit Protocol

Date: 2026-09-04

## Protocol Identity

| Item | Frozen value |
| --- | --- |
| Protocol | `TESSE_TWO_VISIT_CURRENT_V1` |
| Dataset | TESSE-CD |
| Status | `FROZEN_BEFORE_METHOD_SCORES` |
| Config | `configs/evaluation/tesse_two_visit_current_v1.json` |
| Config file SHA-256 | `720952cd75e214de36e318051d6aae18bd762109ba0d3a8ccb73a14f7398e650` |
| Canonical content SHA-256 | `2f638911509f098287e2fee2f8902d95495f5aa4cb88392181076c42c8c652de` |
| Candidate pairs | 16 Apartment, 16 Office |
| Method predictions used | false |

The protocol was derived on node1 from the source-bound RGB-D export and
common-v2 evaluator targets. The derivation used no OVI, CROVE, ReScene, or
baseline output. Every candidate binds its exact RGB, depth, pose, timestamp,
camera, and export bytes in one visit input digest.

## Selection Rule

Each event contributes a 256-frame `t0` window ending immediately before the
intervention. A `t1` candidate is a non-overlapping 256-frame window ending at
one of that event's frozen common-v2 causal checkpoints and beginning strictly
after the intervention. Candidates must pass all of the following source-data
gates:

| Gate | Frozen minimum |
| --- | ---: |
| Symmetric trajectory coverage within 3 m | 0.05 |
| Common ray-sampled observable volume at 0.25 m | 0.05 |
| Camera-direction histogram intersection | 0.05 |
| Changed objects in evaluator GT | 1 |
| Changed evaluator voxels | 1 |
| Old-location visible-free coverage | 0.05 |

The 3 m trajectory radius captures opposite-side views of the same indoor
room. A 1 m diagnostic was rejected before method execution because Office
camera centers remained 2.3--2.5 m apart even where source-only view-frustum
volume overlapped. This decision did not inspect method performance.

Eligible candidates are selected lexicographically by old-location visibility,
common observable volume, trajectory overlap, camera-direction coverage, and
finally earliest `t1` end frame. No weighted method score is used.

## Frozen Pairs

| Scene | Role/status | t0 frames | t1 frames | Traj. overlap | Common volume | Camera overlap | Changed obj. | Changed voxels | Old-location visibility |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Apartment | development ready | 766--1021 | 1217--1472 | 0.925781 | 0.675184 | 0.636719 | 1 | 18,522 | 0.995303 |
| Office | `OFFICE_NOT_RUN_HELD_OUT` | 2145--2400 | 2446--2701 | 0.085938 | 0.183561 | 0.316406 | 3 | 4,792 | 0.812813 |

Apartment visit input SHA-256 values are
`4f86896e751ec112aa16505a37b67645813a69681369e8e49f73c0bb2aea4698`
and `e86ba4ddea76bfa33a75a09c960e50eb1a3c2d5b7781b7de1e6df4b0e9bd7d9d`.
Office visit input SHA-256 values are
`4d54c40675b5b99d5d72e35ed883b13c0dc1c20bb34d704cc7dd066cccc8cef8`
and `e3ab4bb15ea8faf85da4cd657bfce2d743085f18a466c020d28ed844108cd06d`.

## Causality and Execution

GT event regions and visible-free targets are evaluator-only protocol-design
evidence. They are absent from each `method_input_manifest`, whose
`runtime_ground_truth_inputs` list is empty. The method receives only the two
selected RGB-D+pose windows and their source bindings.

`run_ovi_two_visit.py` copies and reindexes each window into a separate
Replica-layout root, assigns distinct run IDs, and invokes the existing native
OVI command builder and runner independently. The t1 command is rejected if it
contains a t0 run/state path. `evaluate_ovi_two_visit_current.py` revalidates
the protocol, source, method-config, and output hashes before opening any
prediction.

Office remains unauthorized until an `OFFICE_RELEASE_AUTHORIZED` receipt binds
the frozen adapter, temporal reasoner, composer, B0--B6 matrix, metric contract,
and success gates, and records zero prior Office attempts.

Protocol status: `TESSE_TWO_VISIT_PROTOCOL_FROZEN`
