# OVI-MAP x ReScene B7 Handoff

Date: 2026-09-04

1. **Is B3 bit-/metric-equivalent?** The frozen B3 artifact was not modified.
   B7-G is metric-equivalent to B3 on every reported Apartment metric and has
   the same point count; its map bytes differ because 1,792 points were
   replaced by registered points.
2. **Did B7 change visibility authority?** No. The final compatibility adapter
   calls the unchanged B3 signed-visibility engine with the same thresholds.
3. **What registration is used?** Deterministic 2 cm voxel averaging, a 20k
   lexicographic cap, proper PCA-axis initializations, trimmed point-to-point
   ICP and Kabsch without scale. It is reproducible and dependency-light.
4. **How is a transform validated?** Finite SE(3), homogeneous bottom row,
   orthogonal determinant-one rotation, bidirectional inlier/overlap and
   median/p90 residual gates, centroid/extent/motion gates, exact input hashes.
5. **Which relations may transfer history?** Only 1:1 `persistent_static` and
   `persistent_moved` relations with matching known labels in the frozen
   recoverable-label set and an accepted registration.
6. **Which relations fail closed?** Appeared, removed candidate, uncertain,
   split/merge, nonrigid/ineligible labels, low support, degenerate geometry,
   mismatched semantics and failed registration all retain exact B3 behavior.
7. **Can visible-free history be revived?** No. All 326 visible-free candidate
   points were rejected; the recovered visible-free count is zero.
8. **What did B7-G improve over B3?** No evaluated metric. It demonstrated that
   5 registrations and 1,792 safe replacements can be made without regression,
   but newly covered GT surface count is zero.
9. **What is the cost?** Method runtime is 174.198 s, peak RSS 4.254 GB and GPU
   memory zero. BG F@5 cm and surface precision did not regress.
10. **Is the upstream ReScene checkpoint still missing?** Yes. The clean pinned
    checkout at `fb2fe42...` contains no official final checkpoint.
11. **Was ReScene trained?** No. No random initialization, training, or GPU run
    was used for ranking.
12. **Does ReScene beat B4 identity?** Unknown; no ranking-eligible ReScene
    identity output exists.
13. **How large is the OVI-to-ReScene domain gap?** N/A. It was not measured
    because the learned identity path remained blocked.
14. **Does learned B7 beat B7-G?** N/A. Learned B7 was stopped by the B7-G
    no-go gate.
15. **Why are TESSE Object/Change/identity N/A?** The frozen two-visit package
    has no protocol-compatible instance/change/cross-visit identity GT.
16. **What is the 3RScan asset state?** 27/44 selected visits are complete;
    final ranking remains blocked.
17. **Is Office held out?** Yes, with attempt count zero.
18. **What is the strongest paper claim?** Signed t1 visibility can safely gate
    rigid registered OVI history with zero free-space/background conflicts, but
    B4 geometric identity plus rigid recovery does not improve completeness on
    the frozen Apartment pair.
19. **What cannot be claimed?** No ReScene superiority, learned dense-recovery
    gain, nonrigid completion, Object/Change/identity score, Office
    generalization, or final 3RScan ranking is supported.
20. **What is the single highest-value next action?** Obtain and source-bind an
    official trained ReScene checkpoint; without it the learned identity
    hypothesis cannot be tested at all.

Evaluated result: commit `be02e08ec5517cb6c7126a042410da56a2258692`,
config SHA `09a62abf5d85bf9cc5f404e6127368aaaf7c39e40400f5a063cd20e4fed0fbec`,
run manifest SHA `7a731691f01b8c548ac144e57097ca4afa376c8c182f871cb3413fd370ff8c0a`.
