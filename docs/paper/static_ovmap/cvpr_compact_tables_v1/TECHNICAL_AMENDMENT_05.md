# Technical amendment 05: stable report reconstruction

The actual final-consumer run under amendment04 completed all 26 post-lock
diagnoses, the three6/6/4-row measured tables and four reports. Scientific
coverage remains168/172 outputs,10/14 full pools and22/24 cold observations.
Primary inspection of the actual single-page9pt PDF passed with no overfull
box or clipped rows/captions. All fixed blocked positions remain explicit.

The subsequent publication audit found an ordering error in report
reconstruction. In-memory effect dictionaries and sorted-key serialized JSON
produced different orders for E-only/R-only and conditional effects. All
metrics, deltas, fixed conditions, table cells, scientific outputs and timings
were identical. A regression reproduces the discrepancy by a sorted-key JSON
round trip. The report renderer now follows the existing explicit effect order.

Reports also record their actual generation-freeze revision and producer hash.
Publication verifies that committed generator and reconstructs the report using
its recorded generation revision, preserving the distinction between scientific,
table, report and later publication revisions.

Only reports.py, publication.py and the focused round-trip assertion change.
The 15 scientific producers, all prior scientific/cold evidence, all 26 scene
diagnoses, all numeric tables and the inspected PDF remain byte-identical.
Original amendment04 reports and their receipts are archived before the report
metadata is regenerated. The effect text is retained in its original order;
only the handoff's generation revision and report-receipt provenance change.
No inference, parameter, model, candidate, mask, score, class, pooling or timing
observation is introduced or repeated. Main outcomes are explicitly exposed.

The actual requirement-by-requirement primary review and normal full-SHA GitHub
publication remain pending. Neither this implementation fix nor PDF QA completes
the blocked scientific requirements or the active objective.
