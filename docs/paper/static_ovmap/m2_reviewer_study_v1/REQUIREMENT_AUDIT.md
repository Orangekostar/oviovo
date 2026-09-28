# Reviewer study completion audit

Status: **Pre-push execution, scientific and artifact review complete**.
Post-commit publication is certified only by the external
`publication_receipt.json` with `PUSH_VERIFIED` and equal full local/remote SHAs;
this tracked document intentionally cannot contain its own final commit SHA.
This audit follows sections0–17 of the supplied execution prompt. It does not
turn a passing fixture, a draft report or a file count into completed evidence.

External evidence root: `/mnt/shared/ww/ovimap-m2-reviewer-evidence-v1/attempt_001`.
Paths below are relative to that root unless identified as source files.

| Requirement | Authoritative implementation/evidence | Current determination |
| --- | --- | --- |
| 0–2: independent study, immutable history, original task | Task branch based on8fee8294; preserved task MD/JSON and supplied-file SHA manifest; `source_binding.json` | Scoped Git diff contains only this study; all seven supplied SHA manifest entries verified unchanged |
| 3: two CAL scenes, eight exposed Replica scenes, pre-transfer nomination | `nomination.json`, `calibration/new_folds.json`, `calibration/new_final.json`; core pipeline fits/evaluates CAL before nomination and transfer | A2_NS2 and S2 selected; historical CAL query exposure disclosed |
| 4: genuine scores and availability, immutable native geometry | `source_audits/*.json`, `legacy_parity/*.json`, bound source/config/native array identities; scores.py | Completed real reconstruction and historical control parity |
| 5: five controls, eight variants, A3 abstention and hard ties | fusion.py/core.py; `predictions/*/locked.json`; `rows/` |260 exact scene/method/rank rows validated; no substituted or duplicate rows |
| 6: CAL-only fits, original M2 temperatures, equal-source scene-balanced NLL | calibration.py, original/final/fold JSONs and query temperature fits | Code review confirms range[.01,2], maxiter64, xatol1e-4, five-object/two-class support and no Replica fitting |
| 7: frozen versus official rank, runtime overlaps, actual dataset pools | evaluation.py; `validation/official_export/receipt.json`, `validation/ScanNet_pool.json`, `validation/Replica_pool.json` | Exact real masks/labels/serialized ranks; exact released AP and PR/FN trace parity;52 core pools with ordered scene evaluation identities checked |
| 8: full probabilities, populations, object events and uncertainty | `diagnostics/*/{objects.json.gz,summary.json,mechanisms.json,released_attribution.json.gz}`, report paired comparisons | Ten scenes recorded; strict unique geometry correspondence separated from actual released AP events; office1 losses disclosed |
| 9: matched causal B200, separate seeds, CAL-only curve gate | query_jobs.py/query_replay.py/budget_controls.py; `query_controls/stage_complete.json`, `curve_gate.json` |50 jobs,300 scene rows and60 pools complete;9,993 logical attempts (office4 COMBINE uses193 of200 allowance); CAL gate false, no budget curves |
| 10: required versus physical costs, union, memory/timing | `costs/b7912aa06ffcf8e25bd879eb4c582c860b73418b0c1dfbb8a525bddca38acdea/ledger.json` | Four layers recorded:5,565 new visual forwards/33,390 crops/1,141.964619 inference seconds;0.073547 known scalar-fit seconds;280.250499 evaluation seconds. Missing old timing/memory and failed-attempt overhead remain explicit |
| 11: frozen aggregates and own-model text-only robustness | `robustness/aggregates`, `robustness/text`, `robustness/results`; robustness.py | Eight scenes, six methods, four vocabularies; no new image forwards; singleton softmax limitation and distractor outcomes retained |
| 12: genuine unexposed families or exact absence; real success code | `fresh/status.json`, referenced metadata audit; fresh_execution.py | Explicit insufficient-local-unexposed-families status; all14 authorized local families exposed. Export/capture/S2/query/evaluation code exists but real fresh success is **not tested**, not claimed |
| 13: all nine runnable phases and dependency-aware workflow | CLI and workflow.py | All phases callable; dependency fixture confirms failed query does not suppress independent diagnostics/robustness/fresh. Full `all` not rerun concurrently with live jobs |
| 14: ten focused fixture groups and real parity | `tests/m2_reviewer_study`, real validation receipts above |29 focused tests passed in0.66s; no broad historical regression; changed-path lint passed |
| 15: tablesA–F, three MD reports, separate contribution decisions | `reports/4357cf36788db730187506848266122131aad3b51dd643e43dc2dbc9b96e55ea/` | FINAL_EVIDENCE_READY; exact core/query matrices checked;11 separate claim decisions reviewed. Query performance mixed across CAL, Replica and individual seeds |
| 16: compact real bundle, original metric bytes, external references, verified push | publication.py; `artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/manifest.json`; external publication receipt | Actual package90.008811MiB,1,387 manifested files,560 principal scene rows and24,352 external identities; copied bytes verified. Final normal push/full-SHA proof belongs in external receipt |
| 17: final concrete self-check | This ledger plus exact-matrix audit and final source/claim review | Execution, CAL-only permissions, rank/aggregation labels, probability/availability ledgers, causal query replicas, freshness and package reviewed. Overall delivery requires successful external publication receipt |

Scientific interpretation already constrained by measured core results:
official-current-class Replica pools improve M2 APall slightly but reduce AP50
and mIoU versus N0. Shared sharpening closely reproduces per-source fitting;
cosine replacement changes effective weighting. None of these results imply
universal three-source necessity, online latency superiority, arbitrary-text
retrieval or independently fresh generalization. The final matched-query
comparison is separate: Replica official pooled CAL fusion GAIN uAP8.9100% /
mIoU26.7916%, COMBINE8.3784% /24.9238%; random17/23/41 uAP8.3440% /8.4257% /
8.9717%. GAIN improves over COMBINE and the seed mean on this transfer set but
does not dominate every seed/metric or the CAL controls. No seed was selected.

Full per-threshold released matching traces remain at their exact hashed shared
storage paths because they exceed the compact package allowance. Compressed
per-owner ledgers and compact AP25/AP50 event summaries are in Git; the package
does not claim to upload the externally referenced large traces or raw data.
