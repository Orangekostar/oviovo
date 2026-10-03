# Office1 FC encoding block

The canonical main controller stopped at the `office1` FC recovery leaf after
its initial attempt and two allowed identical retries. `office0` had completed.
The failed command ledger is
`execution/recovery_run/command_eba58538991c601201b8a9ade35dd83bf3a373cd17f6076242bb299258d61192.json`.
All three worker receipts and attempt logs are preserved.

The common registry contains one candidate, raw owner 86. Its fixed G3 list is
frames 1950, 1940, 1930; G1 is the first entry. Their unchanged projected masks have
294, 286, 271 pixels respectively and satisfy the fixed full-resolution visibility
criteria. None has a genuinely retained archived U2 request.

The primary reproduced the original FC preprocessing and signed-mask support on
CPU using the actual selected masks and actual cached dense tensor shapes.
The 680x1200 images resize to 755x1333, pad to 768x1344, and produce 24x42 dense grids.
Nearest-neighbor mask resize followed by the inherited bilinear signed-mask
downsampling leaves zero positive dense support in all three cases. The failure
is exactly `EMPTY_DENSE_MASK_SUPPORT`; it is unrelated to the model constructor's
pretrained-weight warning. The separate full 18-scene FC audit verifies strict loading
of all 543 original keys and the frozen class order.

The original C4/C5 contracts forbid a 0/1 mask substitution, dilation, another
view or a resolution change. C5 and execution section 11 require an all-failed
semantic worker to remain a technical block. No recovered-zero scientific result
has been exported for this failed G1/G3 leaf.

The CPU evidence is published at
`artifacts/static_ovmap/cvpr_compact_tables_v1/validation/office1_dense_support_root_cause.json`,
identity `c4f17dd2c24eeef879fefebdd3a4920c6ac1d7ad0514f97b2802dfef581da7f6`.
This evidence contains the actual mask, dense-array and failed-command hashes.
Its failure diagnosis does not prove any benchmark metric, cold timing or final
publication requirement complete.

The unchanged committed implementation completed independent warm leaves for
the other 24 scenes, plus office1's Native and U2 branches. Predictions and the
released evaluator completed 164 conditions across 25 fully completed scenes,
using at most three CPU jobs. All six CF18 pools cover the full ordered cohort.
No Replica subset has been pooled. Office1's independent prediction construction
and explicit blocked-condition orchestration remain pending. No cold timing has
been reserved for a retry of the failed warm leaf.

The implementation freeze remains
`fd77ab2203f7e8af2ba9cd9e2ef36d652713ad24`. These operational continuations do not
change its source inventory, scientific inputs, constants, A3 designation or
deployment `N0_UNCHANGED`. A subsequent code change must use a documented freeze
amendment and preserve this failure and every already completed unaffected leaf.
