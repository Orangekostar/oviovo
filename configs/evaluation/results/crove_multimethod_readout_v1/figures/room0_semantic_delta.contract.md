# Room0 semantic comparison contract

Question: does any evaluated full-map semantic readout improve on the fixed S2 reference in room0?

Evidence: every scored room0 DEV row in the collected results, including baselines, controls, negative results and explicitly labelled M3 owner variants. Dynamic B3/H2 rows are outside this static question; no room0 method is removed for appearance or performance. All 27 room0 rows, including both local-background heads, are included. This DEV figure does not establish cross-scene confirmation.

Archetype: single-panel quantitative comparison. Horizontal paired differences against the same room0 S2 score expose absolute gains/losses without mixing tasks or datasets. Family and exact method IDs remain visible; gray denotes baseline rows and blue denotes new readouts, not significance.

Backend/export: Python/matplotlib; 183 × 180 mm; editable SVG and PDF plus a 300-dpi PNG preview. Minimum text size 7 pt. Single-panel alignment is not applicable. Inspect the exported PDF for glyph size and collisions and inspect the preview visually.

Statistics: one fixed DEV scene per method, not an average over seeds or independent scenes. No confidence interval, significance test or error bar is supported by these observations. Delta is 100 × (method mIoU − S2 mIoU), in percentage points. No model or image synthesis is used.

Traceability: exported source CSV preserves exact scores, S2 reference, deltas, method IDs, owner-change flags and result hashes. The manifest records source/result counts and the aggregate input hash. No confirmation result participates.
