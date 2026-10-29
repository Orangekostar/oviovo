# Real semantic correction and harm cases

Visual contract: qualitative evidence for how the frozen semantic readout changes labels on identical native instance masks. The cases illustrate both correction and harm; they do not establish a benchmark gain or improved geometry.

- Source scope: the first scene in the frozen ScanNet validation order, seed 17, after its predictions and official scores are frozen.
- Comparison: multi-source U10 versus full-system U11. Also display the same object's U00 and U01 labels.
- Selection: use the existing class-independent strict-IoU > 0.5 diagnostic matches; select the smallest native owner ID among corrected instances and independently among harmed instances. An absent category remains explicitly absent.
- Panels: one correction and one harm, each with a 3D view of the same frozen owner mask, local geometric context and four actual labels. Object points are blue; contextual points are gray. Text explicitly identifies correct and incorrect labels.
- Geometry display: use the original evaluation XYZ in metres, a fixed 0.5 m context margin, equal axis scale, elevation 25 degrees and azimuth -60 degrees. Deterministic point subsampling affects display only; all numerical counts and IoU use the full frozen masks.
- Format: full-width supplementary figure, editable PDF text plus PNG preview; a JSON record and caption retain source identities, selection counts and point-display counts.
- Publication: export only the case record, receipt, caption and rendered figure. The source point cloud, raw mesh, RGB and complete GT remain in their original local dataset locations.
- Placement: accompany S7's corrections and harm diagnostics. All matched instances and the correction/harm counts are stated in the caption so the examples retain their scope.

Generation uses existing frozen artifacts. No model inference, image encoding, training or recipe selection is needed. The figure must be inspected after rendering; formal completion of the remaining benchmarks is a separate requirement.

The existing review phase generates and verifies the cases:

```bash
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py \
  --output-root "$OUT" --phase review --resume
```

Outputs are in `$OUT/cases/scannet200_val/scene0568_00/`: `data.json`, `receipt.json`, `caption.md`, `figure.pdf` and `figure.png`. The primary rendered-figure check is `$OUT/review/visual_cases.json`. Repeating review with unchanged sources verifies and reuses the case files without rewriting them. Changed sources or missing rendered files fail explicitly, preserving the existing case for inspection.
