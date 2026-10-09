# Source evidence and limits of verification

Audit date: 2026-10-09. Repository base: `9188e16d4ee7d0460de43413dfe73c225a400f41`; the queried `research/ovimap-samv-local-probe-v1` ref resolved to this commit. Sources below were read through the GitHub connector. Blob values are **Git blob SHAs returned by GitHub**, not newly computed SHA256s of downloaded files. Runtime data/weight contents have not been inspected or executed by the author of this instruction package.

All newly fixed algorithm settings are hypotheses defined in this package. None of these sources establishes that the proposed query/update improves benchmark accuracy.

## Inspected repository evidence

Base URL for paths below:
`https://github.com/Orangekostar/oviovo/blob/9188e16d4ee7d0460de43413dfe73c225a400f41/`

| ID | Path / function | Observed fact | Decision supported |
|---|---|---|---|
| S01 | `src/static_ovmap/source_preserving_update/binding.py`, `bind`, `load_scene` | Old binder checks exact protocol, branch, 234/18 results and publication; reader reconstructs D2, reads full G1/D2 payloads and exposes source dictionaries | Separate new binder; use exact full parent, redirect memo writes |
| S02 | `src/static_ovmap/backbone_wave1/readouts.py`, `fuse_readout` | Calls actual `pool` / `simple_dependence` with source availability; records full probabilities | Reconstruct p0 and absence behavior rather than assuming equal weights everywhere |
| S03 | `src/static_ovmap/source_preserving_update/decisions.py`, `source_components` | Each source softmax precedes group averaging | Vocabulary diagnostic concerns a real operator; no assumed fixed pairwise scale |
| S04 | `src/static_ovmap/minimal_instance_repair/observations.py`, `pose_banks`, `SourceRowProjector`, `representative_rows` | .20m/15deg pose selection, full mesh first hit, .02m/.02 relative depth test, one representative row per pixel | Keep FULL raster identity; diagnose physical support separately |
| S05 | `src/static_ovmap/samv_local_probe/fc_worker.py`, `run` | Exact original FP32 FC operators; ordinary and empty-area fallback pooling; image-based dense cache sharing | Reuse visual operators, add own staged job interface and budgets |
| S06 | `src/static_ovmap/cvpr_compact/area_fallback.py`, `pool_region`, `region_vector` | Hard-mask normal path unchanged; area weighting only when hard support is empty | No invented high-resolution region model |
| S07 | `src/static_ovmap/cvpr_compact/region_worker.py`, `FCSession` | Loads physical FC model/text and accepts explicit cache behavior | Inherited model/session, explicit cache-cold protocol |
| S08 | `src/static_ovmap/source_preserving_update/outputs.py`, `relabel_g1` | Whole-instance class update; G1 recovered labels and owner partition kept | Fixed geometry/partition experiment, real output update |
| S09 | `src/static_ovmap/source_preserving_update/evaluation.py` | Actual mask registry, official current-class ranks, identity-qualified reuse, original ordered pooling; old method/count assumptions | New orchestration, same released metrics and actual output |
| S10 | `docs/paper/static_ovmap/SOURCE_UPDATE_HANDOFF.md` | Full parent path, retained scores/masks/vectors; read-only old results; actual CLI/publication conventions | Bind existing full data, avoid re-encoding N/Q/AnyUp |
| S11 | `docs/paper/static_ovmap/SOURCE_UPDATE_RESULTS.md` | All proposed SU variants failed joint upgrade; same-view coarse FC exceeded AnyUp mixing on Replica by 1.032pp APall | Freeze ordinary FC, test observation/aggregation rather than another upsampler |
| S12 | `configs/static_ovmap/source_preserving_update_v1.json` | Original ordered Replica8/CF18 lists and full-runtime parent conventions | Preserve full ordered cohorts and benchmark identity |

Git blob SHA registry:
```
S01 6c82e4edddebff5db9a3be71a83956f191b41965
S02 38ce43ee5780f1d65b89e193e1db8e07cd24ea7a
S03 6495946e78b20c2f955948d3360179303c9c7760
S04 9ff26300c523c8bbe5752077b4d71238d56726ac
S05 020cdbbf2890667cf57e3454d0218a10e67c32cf
S06 18afa2856f7508d1a5014d8a6dc6282cd892183e
S07 9114b1a83bc775b63281072b9fcb3e27280f799f
S08 a2b2cd35ef220cdf2f9bb6a5dad02782b678f7c5
S09 0bea3558925c9fd73ec11015718d16535cc95321
S10 6e8b0fbb64abce899d56e76a91338940557af63d
S11 9b4fd2118e2eca6cd7234587b74017c840994727
S12 58ec5b0694640da5eae3c7e540776df41bce8660
```

S02/S07 and some long files were read in the relevant function ranges rather than claimed as an exhaustive whole-repository audit. This package does not claim that every historical artifact has been independently verified.

## Primary literature / official documentation

### L01: OVI-MAP
Zilong Deng et al., *OVI-MAP: Open-Vocabulary Instance-Semantic Mapping*, CVPR 2026; arXiv v1 2026-03-27.
- `https://arxiv.org/html/2603.26541v1`
- Relevant section: 3.2, object-centric coverage representation and semantic aggregation.

The paper maintains a 180x240 object-centered spherical coverage map and measures newly occupied direction bins before extracting object semantics. Its original pipeline is incremental. Our COVERAGE control transfers the geometric criterion to a two-read archived-sequence setting; it is not an end-to-end paper reproduction. Claimed method differences should be narrower than 'new view selection'.

### L02: ActiveLang
Liyan Chen et al., *ActiveLang: Active Open-Vocabulary 3D Mapping with Semantic-Uncertainty-Guided Exploration*, arXiv:2610.09518, submitted 2026-10-07.
- `https://arxiv.org/abs/2610.09518`

The verified abstract describes uncertainty-guided autonomous exploration and online language-feature adaptation. It is a recent preprint in the verified source, not evidence of conference acceptance. It is a related-work boundary, not a model dependency or mandatory reproduction. The present task re-queries archived RGB-D and does not move a robot.

### L03: VLM calibration
Weijie Tu et al., *An Empirical Study Into What Matters for Calibrating Vision-Language Models*, ICML 2024, PMLR 235.
- `https://proceedings.mlr.press/v235/tu24a.html`

The paper studies VLM calibration across domains/label sets and temperature scaling. Our fixed-vocabulary expansion diagnostic is not a first claim about calibration; the current production temperatures remain frozen. Stable class-pair ordering and calibrated correctness are different properties.

## Math/engineering deductions, not external claims
1. With exactly two equally treated observations, pose-group equal averaging equals a direct mean for both possible partitions (one pair or two singletons). The redundant control is removed.
2. Source-wise softmax followed by mixing can change the relative ordering of two existing classes when other classes enter; fixed scaled-score averaging cannot change that pair's raw difference. The reference test supplies a concrete numerical counterexample; neither rule is thereby proved more accurate.
3. SUPPORT splits each observed site's area among observing views, so total weight equals covered area. No guarantee follows for correctness, arbitrary near-duplicate invariance, AP or cross-dataset generalization.
4. Existing full-map p0 is already aggregated; the new task cannot truthfully claim selective removal of historical observations without their actual retained records and weights. Only new records are withdrawable in this study.

## Remaining actual-execution prerequisites
The server must resolve the real parent files, models, active GPU, original metadata and scorer; build/test physical support projection and FC integration; and measure any new costs. Network access from the artifact-building container was unavailable, so source access was through connectors/web. No new repository clone, GPU experiment, remote write or benchmark run was performed when this package was prepared.
