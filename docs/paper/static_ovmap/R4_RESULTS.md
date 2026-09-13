# R4 Room0: native merge/split results

Status: G1 and G2 implemented and evaluated on the real 200-frame Room0 development
sequence. Both remain OFF in the effective combination. This is not Replica8/ScanNet18
completion, paper AP parity, or evidence of an overall improvement.

## Paired results

All conditions use identical native coordinates and a freshly computed strict-5cm CPU
1NN projection. Unknown semantic owners remain in geometry. B0 exactly reproduces the
original semantic labels, semantic-filtered instance labels and every released semantic
instance mask/class/confidence manifest. There are 116,029 unmatched evaluation vertices.

| Condition | Canonical AP25 | Canonical AP50 | Canonical AP75 | Released instance mIoU | Released mP50 | Released mR50 |
|---|---:|---:|---:|---:|---:|---:|
| B0 full native owners | .618664 | .500867 | .185445 | .5220 | .6790 | .5978 |
| G1 merge | .595811 | .477704 | .179962 | .5032 | .6974 | .5761 |
| G2 merge + split | .594981 | .502171 | .171459 | .5080 | .7051 | .5978 |

Canonical AP is a separately named diagnostic: full projected-domain masks including
void in prediction area, positive raw GT instance IDs, minimum 100 vertices for both
GT and predictions, greedy one-to-one matching at IoU >= threshold, and confidence-ranked
precision-envelope integration from src/evaluation/baselines/static_metrics.py. Confidence
is native vertex count divided by the largest native object's count, computed without GT.
Ties use ascending native ID. This evaluates 84 GT objects; B0/G1/G2 have 67/63/65 exported
canonical predictions. No-GT cases return null. This is not the legacy mP/mR definition,
not the semantic 48-class evaluator, and not a claim about the paper's AP implementation.

| Condition | Semantic mIoU | Semantic mAcc | Released semantic APall | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B0 | .332857 | .381587 | .154316 | .343233 | .368233 |
| G1 | .332966 | .381713 | .154171 | .341931 | .366931 |
| G2 | .332966 | .381713 | .154171 | .341931 | .366931 |

G1's higher released precision accompanies lower recall and canonical AP; it is not a
gain. G2 slightly exceeds B0 AP50 but degrades AP75 and semantic AP. Neither passes an
overall-benefit gate. Thresholds were fixed before these evaluations; no GT-selected
per-instance undo, layer choice or follow-up threshold sweep was performed.

## Actual changes and failure analysis

The graph has 88 native track nodes, six accepted positive edges and 1,179 independently
supported negative edges. It produces 82 G1 components. The six merged pairs are
(1,17), (2,60), (26,147), (40,63), (47,82), (51,105). Another 28 positive candidates fail
view-count, agreement or native-contact requirements. Component-wide constraints are
implemented and tested, although none of these six accepted positive edges triggers a
transitive rejection in this scene.

GT-only diagnosis flags different dominant objects for (2,60) and (26,147). In particular,
owner 147 separately has best GT IoU .8930; absorbing it into owner 26 loses that object.
The (40,63) merge also damages a previously IoU>.50 prediction. Repeated predicted entity
masks plus contact are therefore insufficient to guarantee a safe merge. Negative edges
are fallible prediction evidence, not object truth; the cached PNGs lack per-mask scores
and amodal completeness. These limitations remain explicit rather than being hidden by
the graph's constraint tests.

G2 constructs 386,567 parent-scoped native surface atoms on a 2 cm evidence grid. This does
not change the mapper's 1 cm voxels or any native vertex coordinate. It matches frame-local
mask IDs through shared visible native atoms, never through numeric mask-ID equality.
An anchor must expose 2–4 credible entities; correspondences need >=80% visible-seed
agreement and >=3 independent poses. Each assigned atom needs >=2 votes with >=80%
agreement, total assigned parent coverage >=70%, and each child >=10% and >=50 atoms.
All ambiguous/unseen atoms retain the parent ID. See split.json for exact decisions.

Exactly one parent (ID 2) passes: 54 independent views support children 148 and 149,
with 28,918 and 19,648 atoms; 11,008 atoms remain with the parent. Assigned fraction is
.815221. GT-only best-IoU diagnosis maps children to distinct GT instances 46 and 93
at IoU .604142 and .686059, versus parent best IoU .430423. The residual has best IoU
.189267. Five other parents fail stable coverage, four fail independent separation and
72 have no credible multi-entity anchor. All decisions are saved, including rejections.

## Semantic, export and cost contracts

G1 pools only queries already selected by eligible native B0 members, preserving their
original vis_area fusion. Single-query/featureless members do not silently activate S2.
This allows up to eight selected queries per original eligible member, so it is explicitly
not a fixed-eight-per-merged-object comparison. Source query IDs and original bank records
remain unchanged. G2 children inherit their G1 parent's prediction without encoder calls.

Internal partitions are int64. G1/G2 retain the complete source mappings and compressed
per-vertex restoration ledgers back to B0. Native binary PLY artifacts contain original
coordinates and triangle indices, uint32 instance_id, and reversible 24-bit RGB identities.
They omit non-coordinate source attributes such as normals; they are partition artifacts,
not appearance meshes. The exporter checks integer ranges and verifies the full written
coordinate/triangle/ID arrays. Projected legacy exports also check or explicitly compact
IDs before narrowing, with inverse mappings saved. No uint8 wraparound is allowed.

Measured CPU costs: graph witnesses 17.59 s; graph including native contact and partition
export 88.38 s; surface-atom evidence and G2 split 46.11 s. The atom observation matrix
occupies 154,626,800 bytes. Native PLY export costs are recorded separately in its receipts.
These measurements exclude reused R3 support (74.55 s), historical frontend and VLM costs;
they are not end-to-end FPS. No new image encoder query or 3D training was added. CPU peak
RSS was not measured and is not inferred from array size.

## Evidence and verification

artifacts/static_ovmap/room0_graph contains exact evaluation/readout/split receipts,
compressed complete graph and frame witnesses, native-export receipts, a results table,
and hashes/locations of local large artifacts. The graph source log corroborates the
historical CropFormer entity checkpoint and confidence threshold .5. Prediction entry
points have no GT arguments; GT is opened only by evaluate_static_ovmap_graph.py.

105 relevant tests pass, including mixed-scene/encoder isolation, arbitrary-mask rejection,
camera independence, transitive cannot-link, immutable member queries, wide-ID restoration,
PLY geometry roundtrip, confidence-integrated AP, mask-ID permutations, insufficient split
evidence and ambiguous surface residuals. The full 9,282,303-vertex native export is also
verified against actual source geometry and partition IDs. Validation command:

```bash
/home/ww/miniconda3/bin/python -m pytest -q tests/evaluation/test_static_*.py tests/evaluation/test_ovimap_native.py tests/evaluation/test_run_ovimap_native.py tests/evaluation/test_finalize_ovimap_native.py tests/evaluation/test_run_ovimap_paper_protocol.py tests/evaluation/test_baseline_static_metrics.py
git diff --check
```

Next: R5 one released dense branch with isolated feature space and measured ablations;
R6 the separate trained 3D branch; complete remaining input/protocol provenance and full
scene tables. The strict historical query replay blocker remains separate. The original
handoff A–I audit and global completion are still outstanding.
