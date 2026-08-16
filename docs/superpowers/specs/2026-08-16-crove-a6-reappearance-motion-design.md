# CROVE A6 Reappearance-Motion Design

Date: 2026-08-16
Scope: TESSE-CD temporal branch only
Selection split: Apartment only; Office remains held out

## Problem

The recovered A4 Apartment run reaches object F1 0.401 but dynamic F1 only
0.0095. A code-path audit found that a bank-only dormant identity can be
re-identified at a new position and assigned a new geometry epoch, while the
same branch unconditionally records displacement and motion confidence as zero.
The exported trajectory therefore labels a verified cross-session move as
static.

## Hypothesis

A qualified dormant re-identification supplies two temporally separated
identity endpoints. For identity `i`, let

```text
d_i(t) = ||c_obs(t) - c_last(i)||_2
q_i(t) = cosine(f_obs(t), f_last(i)).
```

If the existing dormant re-ID gate accepts the pair, and

```text
d_i(t) >= delta_motion and q_i(t) >= tau_motion,
```

then the reappearance is a dynamic observation in the current export. The
within-session consecutive-frame requirement does not apply: dormant re-ID
already binds two identity-qualified temporal endpoints. A reappearance below
either threshold advances the ordinary static evidence path.

## Invariants

1. Only an already qualified dormant re-ID trigger can create this evidence.
2. Same-position reappearance remains static with zero motion confidence.
3. The existing geometry-epoch transition, lifecycle update, association, and
   cumulative T1 branch remain unchanged.
4. Motion confidence is the accepted appearance similarity only for a
   threshold-qualified reappearance; it is zero otherwise.
5. The update consumes only current and past state and introduces no future
   frame, annotation, or evaluator input.

## Predeclared Candidates

Both candidates use code after the role-separated active/dormant gate fix and
change no frontend input or T1 configuration.

| Candidate | Base config | Displacement | Identity confidence | Continuous motion frames |
| --- | --- | ---: | ---: | ---: |
| A6-R | A5-R | 0.10 m | 0.70 | 2 |
| A6-RM | A5-RM | 0.15 m | 0.80 | 3 |

The re-ID gate itself remains at appearance similarity 0.75 and maximum dormant
distance 3.0 m. Consequently A6-R favors recall, while A6-RM rejects accepted
re-IDs whose appearance evidence is below 0.80 before assigning a dynamic label.

## Evaluation And Decision

Run both candidates on the complete causal Apartment sequence. Compare against
A4, A5-R, and A5-RM using official Obj./Dyn./Chg. F1, common-v2 current mIoU,
ghost rate, background F@5cm, and identity diagnostics. Select a candidate only
if dynamic or change F1 improves without a material regression in object F1 or
current mIoU. Freeze the selected whole configuration before the one-shot Office
run.

The main failure risk is a false dormant re-ID between visually similar repeated
objects. Report re-ID trigger count, reappearance displacement distribution, and
per-event dynamic precision rather than attributing every dynamic gain to correct
identity recovery.
