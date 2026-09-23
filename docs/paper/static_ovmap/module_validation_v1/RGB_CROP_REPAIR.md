# Narrow RGB crop repair

The first corrected-capture G run stopped on a valid `(1, 2, 3)` RGB crop.
Transformers inferred channels-first from its height and raised
`mean must have 1 elements ... got 3`. A `(3, 5, 3)` crop instead silently used
the wrong channel axis. Both failures were reproduced with the installed
SigLIP processor; an ordinary `(8, 9, 3)` crop was unaffected.

`rgb_siglip.py` supplies the shared S/G/Q image backend with explicit
`input_data_format="channels_last"`. Model weights, resolution, precision,
crop coordinates and crop count are unchanged. The image adapter has its own
source identity, included in all three model-cache keys. The immutable backend
used to generate the already frozen native text cache remains unchanged.

The affected first-scene native/SigLIP2 results and failed G output are retained
under `scannet_study_v1/scenes/scene0547_00/historical_rgb_layout_v1` outside Git.
The interrupted Q run had no completed scene trace. Old G/Q request-cache
namespaces remain external and are not reused by the corrected model identity.
The interrupted runs are engineering overhead; their unfinished attempts do
not have a complete cost ledger. No SELECT or confirmation decision had been
made. Native-v10 captures, N0 results and geometry-request preparation remain
valid.

Validation: the regression failed on both narrow shapes before the fix and
passed afterward. All 19 targeted region, geometry-model, query-model,
semantic-model and frozen-semantic tests passed; Ruff and whitespace checks
passed. Real request
`426e76d72d64d32e33da50a0e45e798dd981d2b4208752c500cea40828fbdf79`
(scene0547_00, frame 640) has six `(1, 2, 3)` crops. CPU preprocessing with both
the bound native and SigLIP2 processors passed with the corrected adapter.
The S/G/Q jobs must complete with the corrected input identity before scientific
results are reported.
