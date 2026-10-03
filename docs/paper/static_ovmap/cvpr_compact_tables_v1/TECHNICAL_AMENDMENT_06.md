# Technical amendment 06: complete analysis scope and lossless release

The actual amendment05 publication audit passed, but the original separate-gzip
release contained215,945,478 bytes and failed the50MiB gate. An actual lossless
packaging probe retained all1,687 original file names and720,425,839 payload
bytes. Deterministic TAR/XZ with a64MiB dictionary occupied57,840,444 bytes;
a256MiB dictionary occupied36,306,492 bytes. The probes made no inference and
did not alter the failed release or any scientific artifact.

Publication now collects exact original files in task-local staging, compresses
the detailed evidence together, and verifies every member's original byte count
and SHA256 by streaming the archive back. Small tables, metrics, PDF, freeze and
validation summaries remain directly readable. The archive contains full actual
scores, decisions, matching traces, class metrics, paid/failed costs and cold
observations, rather than only absolute paths. Standard TAR/XZ extraction restores
the original file names and bytes. Both the manifest and primary audit count
towards the50MiB target. Before removing the older generated representation,
publication preserves each replaced file by content hash in shared storage.

Primary report review also identified omitted fixed methods in Replica Pareto
analysis: the six A0-A5 conditions were compared but U2/G3 were absent. Analysis
now uses all fixed methods belonging to each cohort: eight Replica methods and
six CF18 methods. The fixed A3 primary, prescribed A3 pairwise deltas,2x2 effects,
guardrails and missing-value rules are unchanged. The observed available Replica
frontier is U2; the complete fixed-method frontier remains unavailable because
four required full pools are blocked. This does not compare U2 against an
unmeasured A3. The report explicitly names its comparison scope.

Only reports.py, publication.py and the existing focused assertions change.
All15 scientific producers,168 predictions/scoring records,10 ordered pools,
all26 diagnoses, numeric tables, inspected PDF and22 reserved cold observations
remain exact. Original amendment05 reports and the failed release are archived
before report regeneration. No mask, feature, numerical parameter, method,
candidate, dataset, model, fit or physical timing call is changed or introduced.
Main outcomes remain explicitly exposed. The primary review must record every
archived member in addition to every physical release/implementation/report file.

Scientific coverage remains168/172 outputs,10/14 pools and22/24 cold leaves.
Full scientific completion and the327-requirement primary publication audit
remain distinct. No global COMPLETE or objective-completion claim follows.
