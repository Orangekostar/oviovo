# CROVE Pipeline Figure Prompt

## Visual Contract

- Artifact: main-paper method overview.
- Target venue / format: AAAI-style two-column paper, full-width `figure*`.
- Core claim: CROVE routes each causal observation only to compatible temporal
  state components, while a separate frozen route preserves cumulative mapping.
- Reviewer question: what is technically different from an untyped dynamic-map
  update, and how does the restriction prevent cross-state contamination?
- Evidence layer: mechanism overview; no quantitative result claim.
- Source data: state-compatible routing design, router implementation, temporal
  runtime, and T1 non-interference tests.
- Statistics / uncertainty: not applicable; final T2 evidence is pending.
- Figure prototype: left-to-right causal pipeline with one central routing
  matrix, factorized state, and asymmetric dual outputs.
- Panel map: `(a)` evidence construction, `(b)` admissibility routing, `(c)`
  factorized update and dual readout, plus a narrow invariant strip.
- Caption role: define the mechanism and its safety boundary without claiming
  benchmark superiority.
- Manuscript placement: Method Overview, immediately after the state tuple and
  before transition equations.
- Output formats: editable SVG and PDF; 300 dpi PNG only for review preview.
- Traceability: label the router as `A(z_t, s)` and keep names aligned with the
  method notation below.

## Paper Pipeline

For identity `i` at frame `t`, the state is

```text
S_i^t = (I_i^t, E_i^t, P_i^t, D_i^t, G_i^t, O_i^t, R_i^t, M_t^cum)
```

where `I` is persistent identity/prototype, `E` is existence, `P` is current
pose, `D` is dynamic evidence, `G` is the current geometry epoch, `O` is
reversible object/background ownership, `R` is the current-state readout, and
`M^cum` is the frozen cumulative map.

```text
Strict-prefix RGB-D frame t + pose + frame-local predictions
                              |
                              v
              Causal observation/evidence construction
     +------------------------+---------------------------+
     |                        |                           |
     v                        v                           v
role-conditioned         visibility state           geometry/readout
association              present / visible absent   local geometry
active continuation      occluded / unknown         current center
dormant reactivation
delayed birth
     |                        |                           |
     +------------------------+---------------------------+
                              |
                              v
            Typed evidence z_t with provenance + support
                              |
                              v
               State-compatible router A(z_t, s)
                              |
       +----------+-----------+-----------+-------------+
       |          |           |           |             |
       v          v           v           v             v
   Identity I  Existence E  Pose/Dyn P,D Epoch G   Ownership O
       |          |           |           |             |
       +----------+-----------+-----------+-------------+
                              |
                              v
            Current readout R: present objects + background
                              |
                              v
                      Current snapshot t

Frozen static fusion ------------------------------------> M_t^cum
                                                           |
                                                           v
                                                Cumulative map/history
```

The diagram must make these prohibited mutations visually explicit:

1. occluded, out-of-view, or depth-unknown evidence updates no temporal state;
2. visible absence may update existence and ownership, but not identity or
   geometry;
3. unqualified appearance may not update the persistent prototype or validate
   motion;
4. the current observation center drives only the temporal readout and cannot
   mutate fused geometry or `M^cum`.

Qualified identity motion and local geometric motion retain their own
displacement/confidence pairs. They may jointly support `D`, but the figure
must not imply that confidence from one source validates displacement from the
other. The temporal branch consumes only frame `t` and prior state.

## Drawing Prompt

```text
Create a publication-quality full-width vector method figure for an AAAI-style
robotics and 3D vision paper. White background, flat scientific diagram, crisp
editable vector text, no gradients, no shadows, no decorative illustration,
no numerical results, no rankings, and no SOTA language.

Title inside the figure: "CROVE: State-Compatible Causal Evidence Routing".

Use a left-to-right composition with three labeled panels and a narrow safety
strip underneath. Make panel (b), the evidence router, the visual anchor and
slightly larger than the supporting panels.

Panel (a), "Causal Evidence at Frame t":
Show compact RGB and depth tiles, a camera pose, object masks, and dense
semantic probabilities entering a box labeled "Strict-prefix Frontend".
Split its output into three vertically aligned evidence families:

1. "Role-conditioned Identity" with three lanes labeled "active
continuation", "dormant reactivation", and "delayed birth". Draw appearance
and semantic tokens only through a small qualification gate.
2. "Visibility / Existence" with ray icons labeled "present",
"visible absent", and "occluded / unknown (neutral)".
3. "Geometry / Readout" with separate tokens labeled "local geometry" and
"current observation center".

Every token carries a small provenance tag "frame t". Do not show future
frames or ground truth entering the method.

Panel (b), "Admissibility Router A(z_t, s)":
Draw a compact evidence-to-state matrix rather than a generic neural-network
box. Columns are grouped and labeled:
"Identity I", "Existence E", "Pose P", "Dynamic D", "Epoch G",
"Ownership O", "Current Readout R", and "Cumulative M^cum".
Rows are the evidence families from panel (a). Use filled circular ports for
admissible routes and small gray cross marks for forbidden routes. Emphasize
four rules with short callouts:
"occlusion is neutral",
"visible absence -> E, O only",
"qualified identity -> I, D only",
"current center -> P, R only".

Inside the Dynamic D route, show two parallel mini-arrows:
"geometry (d_g, q_g)" and "identity (d_a, q_a)" feeding an OR gate. Keep each
distance-confidence pair visually bound; never draw crossed connections.

Panel (c), "Factorized State and Dual Readout":
Show a persistent identity token, lifecycle strip ACTIVE / UNCERTAIN / DORMANT,
a current-pose marker, a geometry-epoch stack, and a reversible ownership
ledger. Merge only admissible temporal state into a large output labeled
"Current Snapshot t" with two contents: "present objects at current centers"
and "recovered background".

Below it, draw a separate muted-gray lane from "Frozen Static Fusion" directly
to "Cumulative Map / History". Place a lock symbol at the lane boundary and a
label "no temporal mutation; exact T1 guard". The current-center arrow must not
touch this lane. Do not depict the cumulative and current outputs as the same
map or as two presentation views of one mutable state.

Safety strip underneath the three panels:
Use four compact icon-plus-text cells:
"strict-prefix causality",
"weak identity cannot contaminate prototype",
"rejected motion cannot contaminate geometry",
"current readout cannot modify cumulative map".
This strip describes invariants, not experimental results.

Use an accessible scientific palette with redundant shape/line encoding:
temporal/readout #0072B2,
visibility/lifecycle #E69F00,
ownership/background #009E73,
qualified identity #CC79A7,
cumulative/static #666666,
neutral fills #F2F2F2,
text #222222.
Use solid arrows for admitted routes, dashed gray lines ending in a cross for
forbidden routes, and distinct lane positions so the figure remains legible in
grayscale. Do not use red/green alone for any distinction.

Use square or lightly rounded module boxes with corner radius no more than 4
px, consistent arrowheads, no overlapping connectors, and enough whitespace
for 8.5-9 pt text at final two-column paper width. Keep all labels horizontal.
Export editable SVG and PDF with embedded fonts and a high-resolution PNG
preview.
```

## Manuscript Integration

Place the figure after the paragraph that introduces `S_i^t` and before the
component transition equations. Reference panel `(b)` when defining
`A(z_t,s)` and panel `(c)` when distinguishing current and cumulative outputs.

Suggested caption:

> **CROVE overview.** Given a strict-prefix RGB-D stream, role-conditioned
> association, visibility reasoning, and local geometry produce typed evidence
> with current-frame provenance. The admissibility router permits each evidence
> token to update only compatible components of persistent identity, existence,
> current pose and dynamics, geometry epoch, ownership, and temporal readout.
> The current snapshot uses current observation centers and recovered
> background, while an isolated frozen-fusion route preserves the cumulative
> map. Cross marks denote prohibited mutations; they are algorithmic invariants,
> not measured outcomes.

## Render QA Ledger

| Issue | Artifact | Page/section | Severity | Fix | Owner | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Matrix labels may become too dense | Main pipeline figure | Method Overview | Medium | Use grouped column headers and four short callouts; move the full matrix to Method text if needed | Figure author | Pending render |
| Current and cumulative lanes may appear mergeable | Panel (c) | Method Overview | High | Preserve vertical separation, lock boundary, and no shared outgoing arrow | Figure author | Pending render |
| Color-only route semantics | All panels | Method Overview | High | Retain solid/dashed/cross encoding and grayscale check | Figure author | Pending render |
| Unsupported performance implication | Caption and safety strip | Method Overview | High | Keep invariant wording and omit scores/SOTA until formal T2 is bound | Paper owner | Enforced in prompt |

## Evidence Boundary

This figure may state implemented transition rules and verified invariants. It
must not show unverified T2 values, claim superiority, or imply that the
admissibility contract has already outperformed SuperMap, Khronos, or other
baselines. Quantitative support belongs in the final source-bound result and
ablation tables.

No-fabrication status: PASS. The prompt contains no invented metric, baseline
result, or statistical claim.
