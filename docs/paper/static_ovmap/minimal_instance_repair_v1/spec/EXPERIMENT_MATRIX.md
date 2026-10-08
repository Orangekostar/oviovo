# Fixed experiment matrix

Every listed method is evaluated on the same Replica-8 and CF18 cohorts: 26 scenes.
There are nine method identities, 234 scene-method records and 18 full pools. The
52 baseline records may be imported with content identities; unchanged new arms
may use proven scoring aliases, never invented measurements.

| ID | Change relative to G1 v2 | Controlled question |
|---|---|---|
| IR00_D2 | Parent D2 without recovery | Strong CF18 AP control |
| IR01_G1 | Parent G1 v2, R2 execution | Current complete-method reference |
| IR02_NEAREST_ATTACH | Attach eligible residual to nearest incumbent, no learned scores | Is proximity sufficient? |
| IR03_EVIDENCE_ATTACH | Same target universe; proposal-bank same/separate evidence | Does object-role evidence outperform proximity? |
| IR04_DIRECT_GROUP | Apply bounded positive-consensus groups from common library | Are direct multi-view unions sufficient? |
| IR05_VERIFIED_REPAIR | Same library/classifications; disjoint-view validation and edit penalty | Is verification worth retaining? |
| IR06_ANYUP_REREAD | Same fixed incumbent subset; full-region ordinary AnyUp, aggregate margin | Is direct stronger recognition sufficient? |
| IR07_BOUNDARY_STABLE | Same objects/views/AnyUp; full/core/frontend-intersection agreement | Does boundary stability protect correct decisions? |
| IR08_COMBINATION | IR05 structure + IR07 decisions on original incumbent core identity | Are structural repair and semantic rereading complementary? |

Prespecified primary comparisons: IR03−IR02; IR05−IR04; IR07−IR06;
IR08−IR05; IR08−IR07; every arm−IR01 and every arm−IR00.
The last two comparisons are complete-method differences, not proof of independent
component causality. Group output cannot be evaluated with another arm's masks.

No split/merge parameter grid, no Uni3D/SenseFuse, no VLA/LIBERO, no SAM retraining,
no new category templates, and no dataset-specific choice of method are in scope.
