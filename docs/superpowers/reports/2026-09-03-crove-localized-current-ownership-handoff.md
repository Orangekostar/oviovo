# CROVE P6 Localized Current Ownership Handoff

## Repository Identity

- Repository: `git@github.com:Orangekostar/oviovo.git`
- Base branch: `research/crove-dense-current-state-recovery`
- Base SHA: `dd3247f2d9e81e31a7508b8b692480dbb8855246`
- Working branch: `research/crove-localized-current-ownership`
- Frozen code/evidence SHA before this handoff: `bf64009`
- Final local/remote SHA: the commit containing this handoff, verified by
  comparing local `HEAD` with `refs/heads/research/crove-localized-current-ownership`
  after push; the exact value is reported in the upload verification output.
- `UPLOAD_STATUS`: `VERIFIED` only after that equality check succeeds.

The handoff cannot embed its own commit hash without changing that hash. The
frozen pre-handoff SHA above identifies all implementation, tests, experiment
reports, and decisions; the final branch ref additionally contains this file.

## Implemented Surface

- P6-A: strict diagnostic counterfactual analyzer, 24-variant plan/compose/
  collect CLI, source binding, mass checks, and non-promotion contract.
- P6-B/C: typed reversible 5 cm anchor ownership, causal visibility evidence,
  bit-packed masks, partial native OVI-MAP readout, export/bridge propagation,
  and the frozen L1 policy.
- P6-D: bidirectional geometry agreement, deterministic dense-moved gate,
  exact compact fallback, audit fields, and the frozen gate config.
- P6-E: strict official Dyn JSON/CSV provenance sidecar and checkpoint-limited
  runtime-attribution support.
- Tests cover validation, tampering, causality, reversibility, geometry gate
  boundaries, fallback exactness, export/bridge integrity, and T1 isolation.

The complete branch diff contains 37 files. No frozen T1 protected source is
modified.

## Experiment Decisions

| Phase | Apartment result | Decision |
| --- | --- | --- |
| P6-A | 24 variants; Object 0.348472-0.367952; Dyn invariant 0.069225; 0/24 all-gate | `NO_GO / DIAGNOSTIC_ONLY` |
| P6-C L1 | Obj 0.367307; Dyn 0.069225; Chg 0.087246; current 0.149449; Ghost 0.441689 | `NO_GO / REJECTED_RETAIN_A6` |
| P6-D hybrid | 0/104 dense accepts; Obj 0.348472; Dyn 0.069225; Chg 0.088458; current 0.149560; Ghost 0.443760 | `NO_GO / REJECTED_RETAIN_A6` |
| Combined | Both standalone candidates failed | `NOT_RUN_STANDALONE_GATE` |
| P6-E | exact 0%; ambiguous 100% of 42,175 official Dyn mass | `NO_GO / ATTRIBUTION_AMBIGUOUS` |
| ReScene | no exact evidence of identity-dominant failure | `NO_GO / NOT_NEEDED` |

Apartment retains A6 as the formal candidate. Its registered reference values
are Object F1 `0.372762`, Change F1 `0.060853`, current mIoU `0.142897`, and
Ghost `0.646883`; the P6 Dynamic floor is the measured P5 value
`0.06922505723328032`. No P6 candidate passes all five frozen gates. Office is
`NOT_RUN_HELD_OUT` and was not accessed.

## Primary Artifact Inventory

Every large checkpoint/PLY/sidecar member remains in result storage and is
transitively bound by the listed manifest. Reports contain the corresponding
measurements and decisions.

| Artifact path | SHA-256 | Bytes |
| --- | --- | ---: |
| `.../p6a/counterfactual_plan.json` | `3a43ab9fc8bb3aebd1c5b880f8df3baf4d08c84f86a1228f223a443ef200bb89` | 5,083 |
| `.../p6a/collection/counterfactual_manifest.json` | `41585bcfa8b3a48c8df0c88faf8307701b045c185dd658cae64b5c52d8a7f9f0` | 6,336 |
| `.../p6a/collection/counterfactual_metrics.csv` | `6053b069dc18e0a8589f81949fa3c2431298bc857cf5cc778c577c88032aa049` | 8,478 |
| `.../p6a/collection/counterfactual_anchor_features.csv` | `c157ef4dec7fbecf50ca5d9b76e28fd4e6293a36835e40b0165bc8c7a8702f67` | 1,519 |
| `configs/evaluation/crove_ovimap_localized_visibility_l1_v1.json` | `53f4d808bda65e12a88496e2d27c887b24d9b1c5b47699b300261f4f1a67ec3c` | 518 |
| `.../p6c/apartment_l1/composition/run_manifest.json` | `259f181e2e562cc6546e8ca08727049c48e71bde84f3bb05ba11bfa525e73bad` | 1,094,732 |
| `.../p6c/.../00001472-77804109999/current/snapshots/77804109999.000000_current.npz` | `1d4f7999be877f2b71503f4580f2ee8c339d5874a0e46a933428c4c521ac81af` | 12,031,464 |
| `.../p6c/.../00001472-77804109999/current/entities/77804109999.000000_current.jsonl` | `878fc1835281249079963565a7f30d25484c43b864d0215b488035495f00f207` | 51,857 |
| `.../p6c/apartment_l1/gate/common-repeat-1/summary.json` | `1c778edfc8fd039ca2dd0c34a76967d0f001f7e24ec4912459c2e36117bbfca5` | 53,661 |
| `.../p6c/apartment_l1/khronos/evaluation/official_metrics.json` | `eabb8f26ca478197acca4a89129a4089061cb4255778d09a15798e0c0b3d805c` | 1,147 |
| `.../p6c/apartment_l1/gate/gate_decision.json` | `05145bd2ccf9bc0d8a9e88142f48b78b729dcc34d2fd18aca70a1713ed53192b` | 5,073 |
| `configs/evaluation/crove_dense_moved_geometry_gate_v1.json` | `9cd0933b0418ebe2391830aaa468e5d2954e8fc168eb97355ba13077f67333c8` | 399 |
| `.../p6d/apartment_hybrid/composition/run_manifest.json` | `1a3098140282257f247f835307f6b3727a3055c67ee84a297d65899f2609a9c7` | 200,619 |
| `.../p6d/.../00001472-77804109999/current/snapshots/77804109999.000000_current.npz` | `a718a689d5569b2eb631880e35d87606511dcd2b0420352e96e6464028b5eace` | 13,146,647 |
| `.../p6d/.../00001472-77804109999/current/entities/77804109999.000000_current.jsonl` | `7f838d2ade25d5c6826cf98175a30fec4076de703eef451497bde0b13958f9a1` | 53,438 |
| `.../p6d/apartment_hybrid/gate/common-repeat-1/summary.json` | `e7561160f9f94ec1924f88cd49b2e9534dfdc454cf1b7977aaf7e15ef43a0891` | 54,000 |
| `.../p6d/apartment_hybrid/khronos/evaluation/official_metrics.json` | `ef16da07ccd5c2ff1ee8b9f5ebca95cc99ff036596b3ce21823c5969237f9b9c` | 1,146 |
| `.../p6d/apartment_hybrid/gate/gate_decision.json` | `0f02a566e95d904ce6381e196e36f4408340ca05b0e9ca3742071e7d77bd7fb6` | 5,092 |
| `.../p6e/p5_actual/dyn_provenance.json` | `78784d10bd7a247fa164fdb2aeb8ca3067196fa41b4ef6fc978ff3cb7517a8cd` | 1,381,648 |
| `.../p6e/p5_actual/dyn_provenance.csv` | `40dcbb5d8b5d1357b4915daf2acd86ed361f120e751897670df80538d75cee84` | 5,068 |

The `...` result prefix is
`/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership`.
The two common-v2 repetitions for each candidate are byte-identical.

## Reproduction Commands

Run from this repository worktree. These commands reproduce the new P6
composition/provenance stages; official Khronos commands and complete source
identities are preserved in each result's `run_status.json` and
`run_manifest.json`.

```bash
ROOT=/home/ww/oviovo_baseline_runs/20260903_crove_localized_current_ownership
SOURCE=/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/crove_a6_c069510_source/run_manifest.json
ANCHOR=/home/ww/oviovo_baseline_runs/20260901_crove_ovimap_anchor/anchor/apartment/anchor_manifest.json

python scripts/evaluation/run_crove_anchor_counterfactuals.py plan \
  --p5-runtime-diagnostics /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/apartment_candidate/runtime_diagnostics.json \
  --p2-attribution /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p2/authority_v3/attribution.json \
  --output "$ROOT/p6a/counterfactual_plan.json"

for VID in $(jq -r '.variants[].variant_id' "$ROOT/p6a/counterfactual_plan.json"); do
  python scripts/evaluation/run_crove_anchor_counterfactuals.py compose \
    --plan "$ROOT/p6a/counterfactual_plan.json" --variant-id "$VID" \
    --source-run-manifest "$SOURCE" --anchor-manifest "$ANCHOR" \
    --visibility-policy configs/evaluation/crove_ovimap_unbound_visibility_v1.json \
    --output "$ROOT/p6a/variants/$VID/composition"
done

python scripts/evaluation/run_crove_ovimap_static_anchor.py \
  --source-run-manifest "$SOURCE" --anchor-manifest "$ANCHOR" \
  --readout-role localized_visibility_candidate \
  --visibility-policy configs/evaluation/crove_ovimap_localized_visibility_l1_v1.json \
  --output-root "$ROOT/p6c/apartment_l1/composition"

python scripts/evaluation/run_crove_ovimap_static_anchor.py \
  --source-run-manifest "$SOURCE" --anchor-manifest "$ANCHOR" \
  --moved-geometry-mode geometry_gated_anchor_translation \
  --readout-role hybrid_dense_candidate \
  --dense-geometry-gate configs/evaluation/crove_dense_moved_geometry_gate_v1.json \
  --output-root "$ROOT/p6d/apartment_hybrid/composition"

python scripts/evaluation/build_crove_dyn_provenance.py \
  --composition-manifest /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/apartment_candidate/run_manifest.json \
  --bridge-manifest /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/bridge_apartment_candidate/bridge_manifest.json \
  --official-dynamic-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/results/dynamic_objects.csv \
  --visualization-objects-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/eval_visualization/objects.csv \
  --visualization-associations-csv /home/ww/oviovo_baseline_runs/20260902_crove_dense_recovery/p5/khronos_apartment_candidate/map/eval_visualization/associations.csv \
  --output-root "$ROOT/p6e/p5_actual"

python scripts/evaluation/evaluate_crove_ovimap_static_anchor.py \
  --composition-manifest "$ROOT/p6c/apartment_l1/composition/run_manifest.json" \
  --official-results-dir "$ROOT/p6c/apartment_l1/khronos/map/results" \
  --target-manifest /home/ww/oviovo_benchmark_assets/tesse_cd/derived/common_v2_targets/20260722-stage3-fb97-scope-a/manifest.json \
  --aliases configs/evaluation/semantic_aliases/tesse_cd_common_v2.yaml \
  --label-space /home/ww/oviovo_benchmark_assets/tesse_cd/derived/common_v2_label_spaces/tesse_cd_apartment_label_space.yaml \
  --output-root "$ROOT/p6c/apartment_l1/gate"

python scripts/evaluation/run_crove_anchor_counterfactuals.py collect \
  --plan "$ROOT/p6a/counterfactual_plan.json" \
  --variant-results-root "$ROOT/p6a/variants" \
  --anchor-manifest "$ANCHOR" --output "$ROOT/p6a/collection"
```

The recorded result roots are immutable and must be absent before replay. Use
an equivalent empty root for a new execution. P6-D uses the same evaluator
command with its own composition, Khronos results, and gate directories.

## Verification And Limitations

- Full repository: `5748 passed, 10 skipped, 5 warnings`.
- Focused T1/runtime gate: `60 passed`.
- P6-E provenance/failure/runtime attribution: `28 passed`.
- `python -m compileall -q src scripts tests`: PASS.
- Frozen T1 source-manifest verification: PASS.
- `git diff --check`: PASS.

Known limitation: the official Dyn export lacks per-error prediction identity,
so 100% of official mass remains ambiguous. An existing incomplete runtime
monitoring run was excluded because its formal outputs were not equivalent to
A6 (294/336 emitted files differed). This limitation is reported rather than
bridged with a heuristic. Large runtime artifacts are intentionally not added
to Git.
