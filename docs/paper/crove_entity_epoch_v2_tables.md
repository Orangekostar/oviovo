# CROVE Entity-Episode Dynamics V2 Paper Tables

## Entity-episode ablation

| Method | Relation source | Current mIoU $\uparrow$ | Ghost $\downarrow$ | BG F@5cm $\uparrow$ | Surface F@5cm $\uparrow$ | Correct recovery $\uparrow$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| T1 only | none | 0.1209 | **0.0000** | **0.4453** | 0.4201 | 0.0000 |
| **B3 (selected)** | none | **0.1359** | **0.0000** | 0.3664 | 0.4313 | 0.0000 |
| Entity-episode inherit | none | 0.1357 | **0.0000** | 0.3664 | **0.4357** | **0.0295** |
| Entity-episode + G1 | geometry | 0.1357 | **0.0000** | 0.3664 | **0.4357** | **0.0295** |
| ReScene ID-only | learned identity | **0.1359** | **0.0000** | 0.3664 | 0.4313 | 0.0000 |
| Entity-episode + ReScene | learned identity | 0.1357 | **0.0000** | 0.3664 | **0.4357** | **0.0295** |
| Entity-episode + memory-last | memory | 0.1357 | **0.0000** | 0.3664 | **0.4357** | **0.0295** |
| Entity-episode + memory-bank* | memory | 0.1357 | **0.0000** | 0.3664 | **0.4357** | **0.0295** |

Table note: results are deterministic visit-final metrics on one frozen TESSE-CD Apartment pair. Correct recovery is `1,207 / 40,960`; deletion and bad-recovery rates are zero for B3 and all entity-episode rows. `*`The real input has one final OVI observation, so the memory-bank row exercises the code path but is not multiview evidence. These values must not be inserted into official sequence Obj/Dyn/Chg columns.

## Evidence-control checks

| Check | Result |
| --- | --- |
| D4 current mask equals D1 | PASS |
| D4 per-point semantics equal D1 | PASS |
| D4 headline metrics equal D1 | PASS |
| D4/D5 relation source shared | 94 candidates, 10 accepted |
| Prediction consumes GT | no |
| Exact pair-bound learned forward | PASS, reused once |
| Office retuning | prohibited; none performed |
| Office confirmation | `RAW_MISSING` |

Recommended caption: **Entity-episode ablation on the frozen Apartment development pair.** Local positive correction recovers supported history and improves surface F@5cm from 0.4313 to 0.4357 without Ghost, but current mIoU decreases from 0.1359 to 0.1357. Relation backends do not add a scored map increment, so B3 remains selected.
