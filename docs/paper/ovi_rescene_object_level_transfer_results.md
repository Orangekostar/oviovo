# OVI-MAP Backbone x ReScene Object-Level Transfer Results

Date: 2026-09-06

Branch: `research/ovi-rescene-object-level-transfer`

## Scope

This report separates five questions: fixed-resolver transfer, real RGB-D to
OVI reconstruction, association on one immutable OVI candidate pool, dense
instance grouping, and visibility-gated historical recovery. The only completed
real D2 pair is the preregistered development pair
`scene0109_00-scene0109_01`. All numbers below are measured; an empty CSV cell
means null or not applicable, never zero.

The primary endpoint IoU is 0.50 and 0.25 is a sensitivity analysis. Surface
metrics use exact 5 cm floor-voxel intersections. Surface precision counts only
GT-positive and observed known-negative voxels; predictions outside both sets
remain unevaluable.

## Fixed Resolver Transfer

The confidence threshold `0.3`, selected on DEV6 native inputs, was applied
unchanged to the two remaining eligible environments. These environments are
resolver-held-out but not checkpoint-held-out: the ReScene checkpoint used the
same validation split for model selection.

| IoU | Aggregate | TP | FP | GT | Precision | Recall | F1 | Rigid TP / GT |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | micro | 20 | 16 | 64 | 0.5556 | 0.3125 | 0.4000 | 6 / 13 |
| 0.50 | environment macro | null | null | null | 0.3030 | 0.1818 | 0.2273 | null |
| 0.25 | micro | 24 | 12 | 64 | 0.6667 | 0.3750 | 0.4800 | 7 / 13 |
| 0.25 | environment macro | null | null | null | 0.5152 | 0.2646 | 0.3447 | null |

Transfer was heterogeneous. At IoU 0.50, `scene0359` produced 0 TP and 3 FP,
whereas `scene0459` produced 20 TP and 13 FP. The result supports transfer of
the resolver rule to these two development environments, not unseen-dataset
generalization.

The earlier DEV6 F1 increase from 0.0646 to 0.2698 was a resolver effect rather
than a new network result. Of 557 cached query rows, the threshold removed 475
low-confidence rows, including five previous true positives. Among the 82
retained rows, reassignment created 18 new true positives; the final IoU-0.50
count changed from 16 TP / 328 FP to 29 TP / 35 FP. No query mask membership was
changed for a retained query.

## Real D2 Input

Both visits were reconstructed independently from their real 3RScan RGB-D
sequences by native OVI-MAP, then visit 1 was transformed exactly once into the
reference coordinate frame.

| Visit | UUID | Frames | OVI entities | Dense points | Appearance support | OVI stage sum |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | `20c993b7-698f-29c5-847d-c8cb8a685f5a` | 63 | 29 | 820617 | 0.7328 | 211.331 s |
| 1 | `20c993b3-698f-29c5-859c-dca8ddecf220` | 53 | 21 | 788946 | 0.8558 | 177.841 s |

The combined D2 view contains 1,609,563 OVI points. Source vertex order, RGB
and instance XYZ equality, finite normals, and exactly-once alignment all pass
the recorded geometry audit. The mapper used neither native mesh segments nor
GT to construct D2.

The granularities remain explicit: D0 diagnostics use native mesh segments,
D2 association uses OVI entities, and U2/U3 readout produces composite objects.
GT endpoint bindings connect these domains only inside the evaluator; their IDs
are never treated as interchangeable method inputs.

## Shared-Candidate Association

G, F, and R consumed the same 29 visit-0 and 21 visit-1 OVI candidates. The
model supplied 92 temporal queries. At the primary IoU 0.50, every method had
0 TP; representation-conditional recall is null because zero persistent GT
identities had both endpoints represented at that threshold.

At IoU 0.25:

| Method | Pairs | TP | FP | Endpoint fail | GT | Precision | End-to-end recall | Conditional TP / GT | Rigid TP / GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G full | 21 | 1 | 20 | 20 | 13 | 0.0476 | 0.0769 | 1 / 1 | 1 / 3 |
| G supported | 19 | 1 | 18 | 18 | 13 | 0.0526 | 0.0769 | 1 / 1 | 1 / 3 |
| F object | 17 | 0 | 17 | 17 | 13 | 0.0000 | 0.0000 | 0 / 1 | 0 / 3 |
| R object | 1 | 0 | 1 | 1 | 13 | 0.0000 | 0.0000 | 0 / 1 | 0 / 3 |

Only one of 13 persistent identities was represented by valid OVI endpoints.
G recovered that identity, while R did not. Thus the D2 failure is not evidence
that the temporal network alone is the only bottleneck: OVI endpoint/proposal
coverage already caps end-to-end recall at 1/13 for this pair.

## Dense Instance Readout

Grouping preserves all 762,458 current owned points and the exact XYZ multiset.
It changes ownership only.

| Variant | Objects | Grouped source entities | Query-supported objects | Rejected conflicts | F1@0.50 | TP / FP@0.25 | F1@0.25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A_ID | 21 | 0 | 3 | 0 | 0.0000 | 3 / 18 | 0.1714 |
| U0 | 21 | 0 | 0 | 0 | 0.0000 | 3 / 18 | 0.1714 |
| U1 | 21 | 0 | 0 | 272 | 0.0000 | 3 / 18 | 0.1714 |
| U2 | 20 | 2 | 2 | 0 | 0.0000 | 3 / 17 | 0.1765 |
| U3 | 20 | 2 | 2 | 0 | 0.0000 | 3 / 17 | 0.1765 |

U2/U3 merge one supported two-fragment object, reducing FP by one at the 0.25
sensitivity threshold. They do not add a TP, do not improve the primary 0.50
metric, and do not change geometry. This is weak instance-organization evidence,
not a geometric or semantic gain.

| Map component | Measured change |
| --- | --- |
| Dense XYZ under grouping | Unchanged for A_ID/U0/U1/U2/U3 |
| Instance organization | U3 changes 21 atomic objects to 20 composites; F1@0.25 increases by 0.0050 |
| Open-vocabulary semantics | Retained from OVI; no semantic-quality gain was measured |
| Historical geometry | C1 recovers 692 points; C2 recovers zero |

## D0-D1-D2 Diagnosis

The same scene UUID pair was evaluated across native processed geometry (D0),
sensor-supported native geometry (D1), and independently reconstructed OVI
geometry (D2).

| Domain | Sensor support | Raw nonempty queries | Confident queries | Represented persistent GT | Best listed association F1@0.25 | Fixed-pool R F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| D0 native | 1.0000 | 0.8900 | 0.0400 | 5 / 13 | 0.2857 (F) | 0.0000 |
| D1 sensor support | 0.2703 | 0.8700 | 0.0400 | 1 / 13 | 0.1176 (F) | 0.0000 |
| D2 OVI | 0.7931 | 0.9200 | 0.0300 | 1 / 13 | 0.0625 (G supported) | 0.0000 |

Raw query occupancy remains high across domains, while proposal representation
falls from 5/13 in D0 to 1/13 in both D1 and D2. The preregistered rule therefore
selected `NO_ADAPTATION` with reason `d1_sensor_support_degradation`; decoder or
mask-head training was not authorized or run. The evidence points first to
sensor/proposal support and OVI object formation, not a demonstrated decoder
domain gap.

## Historical Recovery and Oracle Diagnosis

C0/C1/C2/O1/O2 share the same OVI geometry, global alignment, recovery policy,
and t1 visibility. O1 uses GT identity only with estimated registration. O2 uses
an explicitly typed official 3RScan object transform; neither oracle substitutes
GT surface geometry for the OVI source.

| Variant | Paired | Correct rigid relation | Accepted reg. | Opportunity | Recovered opportunity | New correct / wrong / unknown | Deleted correct | Surface F1 | Old residue |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 0 | 0 | 0 | 16 | 0 | 0 / 0 / 0 | 0 | 0.442020 | 143 |
| C1 (G) | 21 | 1 | 8 | 16 | 0 | 3 / 0 / 15 | 0 | 0.442289 | 123 |
| C2 (R) | 1 | 0 | 0 | 16 | 0 | 0 / 0 / 0 | 0 | 0.442020 | 143 |
| O1 (GT ID) | 1 | 1 | 1 | 16 | 0 | 0 / 0 / 9 | 0 | 0.441989 | 142 |
| O2 (GT ID+pose) | 1 | 1 | 1 | 16 | 0 | 0 / 0 / 16 | 0 | 0.441989 | 142 |

C1 recovered 692 points and removed 3,628 baseline points. It added three
GT-positive 5 cm voxels, no known-negative voxel, and 15 unevaluable voxels;
surface recall increased from 0.283740 to 0.284012 while precision decreased
from 0.999680 to 0.999040. Its F1 gain over C0 is 0.000268, too small and too
pair-specific to support a general improvement claim.

C2's single relation failed registration. More importantly, GT identity could
form only one of three rigid proposal pairs because two rigid objects lacked
valid predicted endpoints. O1 and O2 recovered only unevaluable historical
surface and zero opportunity voxels. Therefore this experiment does not isolate
registration as the dominant limitation. The strongest observed limit is
predicted endpoint coverage followed by the overlap between transformed OVI
history and evaluable current GT/visibility support.

## Compute Reuse

Native OVI mapping dominated measured compute: 389.172 s across the two visits.
The cached ReScene passes took 0.487 s and 0.162 s independently plus 0.438 s
jointly, with 1,100,020,224 bytes peak allocated GPU memory for the joint pass.
G/F/R, grouping, D0-D1-D2 diagnosis, and C0-C2/O1-O2 reused those arrays; no
additional network forward was run for historical recovery. Recovery-only CPU
runtime was not recorded and is therefore not reported. OVI frontend and mapping
peaks were 22,189 MiB and 3,069 MiB per device, respectively; each visit used one
frontend forward and one native mapping run.

## Claim Boundary

Supported:

- OVI-MAP is a working immutable dense geometry/semantic backbone for the
  object-level temporal experiments.
- The fixed resolver rule transfers heterogeneously to two resolver-held-out
  development environments.
- Dense grouping can improve the IoU-0.25 instance sensitivity result without
  moving geometry, but the observed effect is only 0.0050 F1.
- On the completed D2 pair, proposal coverage and predicted endpoints dominate
  the measured association and completion ceiling.

Not supported:

- unseen or held-out D2_EVAL generalization, Office generalization, or SOTA;
- a primary IoU-0.50 instance gain, semantic-quality gain, or robust geometry gain;
- successful ReScene-driven historical completion;
- a decoder adaptation benefit, because adaptation was not run;
- registration as the sole recovery bottleneck, because O2 was endpoint- and
  evaluation-support-limited.

## Claim-Evidence Ledger

| Claim | Authoritative evidence | Status |
| --- | --- | --- |
| Fixed 0.3 resolver transfers beyond DEV6 selection pairs | `resolver_fixed_threshold_transfer.csv` | Supported within the declared development scope |
| D2 is independent RGB-D to OVI reconstruction | `d2_ovimap_smoke_v1.json`, `d2_pair_view_v1.json` | Supported for one pair |
| G/F/R use the same OVI candidates | `shared_candidate_gfr.csv` | Supported for one pair |
| Grouping preserves XYZ | `instance_readout_v1.json` | Supported for all five readouts on one pair |
| Learned recovery improves current geometry | `completion_oracles.csv` | Not supported; only a 0.000268 C1 F1 delta |
| Decoder adaptation is required | `adaptation_decision_v2.json` | Not supported; gate selected no adaptation |

## Adversarial Self-Review

- Contribution: pass for the auditable OVI-backbone evaluation path; needs new
  experiments for a strong learned temporal-mapping contribution.
- Clarity: pass; native segments, OVI entities, and composite objects are named
  separately throughout.
- Experimental strength: needs new experiments; only one real D2 pair is complete
  and primary association/instance results are zero.
- Evaluation completeness: needs new experiments on preregistered D2_EVAL pairs
  and a stronger OVI proposal/endpoint construction.
- Method soundness: pass for fail-closed identity, visibility, oracle, and null
  contracts; net practical benefit is not yet established.
