# Replica composition transfer

User request: run the strongest few completed methods on Replica and measure whether the metric changes transfer.

## Fixed design before Replica outcomes

- Start from ScanNet release `04e0287b0d05c946d7519f0b3100ca2e4716713f`; preserve its files/results.
- Run all eight existing scenes: room0/1/2 and office0/1/2/3/4. Use frames 0..1990, stride 10, 200 scheduled slots. These scenes have historical project exposure; this is frozen cross-dataset transfer, not a claim of untouched test data.
- Six methods: N0, Q_GAIN, S_SIGLIP2_AREA, CP_M2_EQUAL_RAW, CP_M2_EQUAL_CAL, CP_M4_GAIN_S2. M2_RAW won CAL and improved ScanNet confirmation; M2_CAL had strongest regression means; M4 tests the promising fixed-Q_GAIN SigLIP2 reread. Choose all now, not after Replica scores.
- Transfer the exact frozen ScanNet Q checkpoint AND scaler; transfer its final three temperatures unchanged for M2_CAL. M2_RAW keeps .07. No Replica training, fitting, selection or new thresholds.
- Use the original Replica loader and released Replica evaluator: prediction vocabulary REPLICA_51 IDs 1..51; instance AP uses the evaluator's original 48-class subset. Semantic metrics use the original full 51-class readout domain. Do not substitute the unrelated runtime-41 or ScanNet200 vocabulary. Compare within-Replica deltas, not raw absolute metrics across datasets.
- Existing old Replica captures lack current full causal membership snapshots. Capture eight native-v10 maps using the already bound extension, upstream patch, original RGB-D, CropFormer and FP32 native model. One shared source GPU-2 lock, 8 CPU threads, sequential visual jobs. No new data/model downloads.
- Freeze each N0 surface/owners/projection/serialized ranks for all six methods. Q_GAIN uses B200, native model, one frame barrier and final reconciliation. M4 rereads every paid Q_GAIN attempt with separate SigLIP2 state/model/cache, no native fallback. Static S2 uses original lineage top3 AREA and genuine availability.
- Reuse audited prediction/evidence/fusion/replay/evaluator primitives. Add a separate Replica runner/authorization layer; never fake ScanNet roles or mutate its frozen guards.
- Evaluate locked predictions with full-content persistent keys, preserve released trace parity and GT only on evaluation side. Report all 48 method-scene rows, scene-macro means, deltas versus N0/S2/Q_GAIN, worst scene, positive-scene counts and physical/logical costs. Missing rows remain missing, not zeros.

## Completion evidence

Eight complete capture receipts, 48 locked/evaluated records, correct Replica vocabulary and native export parity, unchanged transferred checkpoint/scaler/temperatures, exact Q_GAIN/M4 paid trajectory parity, no missing-row averages, compact labels and external hashes, and a reproducible report. No automatic deployment or additional method selection follows the outcomes.
