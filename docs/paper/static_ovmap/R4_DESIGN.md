# R4 native instance graph design and execution plan

Scope: implement the full R4 requirement in 03_CODEX_IMPLEMENTATION_HANDOFF.md, including
G1 merge, G2 split, fixed evidence-based hierarchy selection, reversible native export and
class-agnostic-first evaluation. Primary agent implements and reviews the scientific code.
User authorization to choose configurations and proceed autonomously remains in force.

## Evidence contract

The nodes are native predicted instance tracks, including owners without semantic queries.
Inputs are the frozen native mesh, its RGB-to-owner identity, original CropFormer entity
mask PNGs, raw depth, camera poses and the depth-consistent owner images from R3. No GT,
text labels or semantic feature values enter graph generation. Mask IDs are local to a
frame; their numeric equality across frames has no meaning.

Use repeated whole-entity predictions as fallible evidence, not object ground truth. The
local CropFormer export comes from class-agnostic entity masks after a score threshold and
overlap arbitration. Cached PNGs do not retain individual scores or amodal masks. Therefore
we cannot claim perfect whole-object masks or establish cannot-link from mask IDs alone.

For each frame, measure the owner/mask contingency table. A credible mask has >=500 pixels,
>=90% valid raw depth and >=70% coverage by depth-consistent native owners. Each owner must
have >=100 visible pixels. A positive witness requires >=80% of each owner's visible area
in the same credible mask, >=50% joint mask coverage and actual native surface contact
within 5 cm. A negative witness requires distinct credible entity masks, each with >=0.65
bidirectional IoU to its native owner. This is evidence for two separately delineated
whole entities rather than merely disjoint parts. Both signs require >=3 independent camera
poses (5 cm translation OR 5 degrees rotation). Negative evidence takes precedence, and
is checked across every pair of members of the candidate merged components. Positive
edges additionally require >=60% agreement among qualifying co-visible frames.

Contact uses actual native vertex representatives on a 2 cm grid; accepted distance is
between actual surface vertices, not bounding boxes or voxel centroids. Grid subsampling
can miss a contact but cannot invent a closer representative pair. Coordinates never move.

## Layers and readout

G1 takes deterministic descending-support positive edges and rejects any component-wide
cannot-link. Ties use sorted native IDs. Each component's public ID is its minimum native
ID. Original per-vertex IDs and complete member mappings are saved. Unknown geometry stays.
Semantic G1 readout uses the pooled native B0-selected queries of component members and
fixed original vis_area fusion (no extra image queries). Singleton semantics remain exact.
Merged semantic query budgets must be disclosed separately from fixed-K S1 comparisons.

G2 lifts separated credible entity observations to native surface atoms, associate them
by cross-view geometric overlap, and require independently repeated separation before
partitioning a parent. Unobserved/ambiguous atoms retain their parent; no GT-guided layer
choice. The original partition and per-vertex change ledger must restore every native ID.
G2 is not implemented by a no-op alias or connected-component-only heuristic.

## Execution and checks

- [x] Write adversarial tests, observe failure, implement instance_graph.py: IDs alone do
  not imply negative edges; repeated camera does not count; A-C prohibition blocks A-B-C.
- [x] Implement native_export.py with int64 partitions, complete parent mapping, explicit
  compact remapping for uint8/uint16, overflow rejection and exact restoration tests.
- [x] Implement the RGB-D producer and run all 200 Room0 frames; retain raw witnesses,
  thresholds, rejected edges, input hashes and CPU preparation/storage costs.
- [x] Implement hierarchy.py with fixed G1 selection; retain B0 and G1 native arrays and
  source-member semantics. Test featureless owners and no background reassignment.
- [x] Evaluate strict-5cm projected arrays with original released mP/mR and semantic AP;
  add separately named canonical integrated AP using the existing common AP integration,
  fixed area confidence ranking and full projected vertex-domain masks. Report AP75 and
  per-change best-IoU diagnostics. Do not label this paper AP parity.
- [x] Implement and evaluate G2 surface-supported splits with independent-view tests and
  exact restoration; save negative results, keep ineffective stages off in the combination.
- [x] Review raw output, relevant tests, diff and original R4 scope. Commit/push delivery
  is verified separately by git ls-remote after the commit exists. Remaining R5/R6/full-scene
  scope stays in IMPLEMENTATION_PLAN.md.
