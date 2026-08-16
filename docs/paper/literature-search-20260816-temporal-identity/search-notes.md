# Search Notes

## Scope

- Target: AAAI-family AI/CV paper on causal online open-vocabulary current-state mapping.
- Search date: 2026-08-16.
- Sources: official proceedings, CVF, RSS, arXiv, and official project pages.
- Exclusion: MDPI sources and non-primary summaries.

## Query Families

- online dynamic metric-semantic mapping short-term long-term change;
- open-vocabulary object mapping dynamic scenes;
- object-level TSDF existence probability data association;
- cross-session object association partial views occlusion;
- persistent identity versus visibility dynamic scene understanding;
- active/dormant object re-identification current-state mapping.

## Screening Logic

Eighteen works were screened. Twelve were retained as direct baselines,
mechanism precedents, or current novelty threats. The main novelty threat is
OASIS-Map: it already uses dense semantic correspondences for multi-session
change and 3RScan moved-object association. CROVE must therefore emphasize
strictly causal online inference, current-map readout, visibility-backed
absence, and role-separated identity decisions rather than claim generic
cross-session object association.

## Evidence Gaps

- No result is inferred from a paper abstract.
- Comparative numbers are not transferred across protocols.
- The proposed A5 thresholds remain development settings until measured on
  Apartment and transferred once to Office.
