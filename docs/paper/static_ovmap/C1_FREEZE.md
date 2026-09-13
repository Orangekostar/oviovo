# Frozen C1: Room0 development v1

C1 retains only S1a: quality/coverage selection of at most eight retained observations,
native visible-area fusion, and the original minimum-two-query eligibility rule.
It uses measured enrichment and the original SigLIP space, geometry and instance IDs.
Quality weighting, single-query fallback, merge, split, dense replacement/refinement,
extra 3D pretraining and optional query-conditioned review are disabled.

This decision uses Room0 development evidence from R3–R6 and S1. There is no demonstrated
multi-component gain to combine. C1 is equivalent to S1a, not a new independent improvement
and not the arithmetic sum of separate gains. The remaining Replica scenes have not been
used to select this configuration. Historical query scope remains retained native top-10;
discarded query ownership/area is unresolved.

The real Room0 C1 execution produced 71 eligible instances in 0.107 s warm CPU readout.
Every observation record (including class, score, selected query IDs and margins) equals
both the concurrent S1a run and the previously evaluated S1a output. The equality audit in
`artifacts/static_ovmap/room0_c1/equivalence_audit.json` binds source and metric references.
Metrics are explicitly reused under exact output/geometry equivalence, not described as a
second evaluator execution. Thus C1 has S1a's mIoU 0.362175 and semantic AP 0.186723 on Room0.
Historical frontend/VLM cost remains unknown; this is not end-to-end throughput.

Run the existing readout command from `room0_c1/input_binding.json`, whose recorded argv
includes `--conditions B0 S1a C1`. The C1 entry requires `--enrichment`; it is opt-in and
does not alter the default condition list. Full argument vectors bind actual files, not
placeholder assets. `configs/evaluation/ovimap_static_conditions_v1.json` lists all 14
requested condition routes and development statuses. G and D generation share intermediate
work but save separate condition outputs. Q0/Q1 are explicitly optional NOT_RUN with no
implemented review backend or invented budget. This registry is not a universal launcher.

Replica8 evaluation, the extra seven-scene diagnostic, B1 version-drift evaluation and the
final A–I audit remain pending. ScanNet18 inputs are locally unbound as documented by the
scoped inventory; no missing results are imputed from Room0.
