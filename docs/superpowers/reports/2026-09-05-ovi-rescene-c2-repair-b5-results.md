# OVI-MAP x ReScene C2 Repair and Apartment B5 Results

Date: 2026-09-05

Evaluated code: `85248e62b4a0e780ed72ea339abb911806a564ef`

Final status: `C2_REPAIR_BLOCKED`.

The formal Apartment build used the frozen 0.05 m input-only calibration and
processed 854,429 native ReScene model rows. It recovered legal same-visit RGB
and normal support for 817,691 rows (0.9570028639009209), leaving 36,738 rows
unsupported. Of these, 6,357 had no valid-normal source candidate and 30,381
had candidates but no depth-consistent observation in their own visit.

The C2 contract requires complete RGB and normal coverage. Therefore no GPU
forward, B4 reconstruction, B5 projection, relation comparison, registration,
or Office run was executed. All corresponding measurements remain null. This
run establishes an input-coverage blocker only; it does not establish ReScene
identity quality or superiority over B4.

The external audited receipt is at
`/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_c2_repair_b5_v2/built-input/input_contract_v2.json`.
The compact evidence package is under
`configs/evaluation/results/ovi_rescene_c2_repair_b5_v2/`.
