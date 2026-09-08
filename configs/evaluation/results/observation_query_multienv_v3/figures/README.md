# Qualitative Fixed Views

## Visual contract

- Artifact: two fixed top-down confirmation plates.
- Core claim: inspect whether final instance ownership changes under the selected
  FULL checkpoint without hiding the failed environment.
- Reviewer question: does the aggregate gain correspond to visibly different
  instance partitions, and is failure still present?
- Evidence layer: qualitative success/failure boundary.
- Source data: saved final owner arrays from Frozen, selected BASE, and selected
  FULL runs on the two predeclared CONFIRM pairs.
- Panel map: camera-supported RGB, evaluator GT, Frozen, BASE, FULL.
- Projection: the same XY top-down max-Z raster for every panel in one row.
- Color: Okabe-Ito categorical colors; prediction colors are panel-local
  instance IDs. Light gray is background and mid gray marks GT voxels shared by
  multiple instances at the 5 cm evaluation resolution.
- Traceability: `qualitative_cases.json` freezes case order and run IDs;
  `manifest.json` records all summary, prediction, runtime, and output hashes.
- Claim boundary: GT is used only for its display panel and never modifies the
  saved prediction ownership arrays.

## Caption

Fixed-view instance partitions on the two predeclared confirmation pairs.
scene0009 is the positive case, while scene0449 preserves the zero-TP failure.
Colors identify instances within each panel and are not cross-method identity
correspondences. The figures visualize the saved final maps; quantitative
claims come from `confirmation_metrics.csv`.
