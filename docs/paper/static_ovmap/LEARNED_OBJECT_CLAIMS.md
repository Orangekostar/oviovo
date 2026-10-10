# Evidence boundaries

Strict gate: False; material gate: False.

Interpret the matched MA–FC8, VIEW–MA, SURFACE–VIEW and AUX–SURFACE comparisons together with H and the fixed second-seed pair. Passing the whole-map gate alone does not establish the novel mechanism. The complete contrast data are in Table2 and claims.json.

H novel-category support: SUFFICIENT, 22 original objects.

The40 adapter-heldout categories supply no positive classification targets in this readout training. Objects from these categories may still occur in image context or fixed corruptions; geometric membership supervision is separate. Their absence from foundation-model pretraining is not established; scene-level backbone pretraining coverage cannot generally be audited.

New TRAIN/DEV/H sensor RGB is registered to the depth grid using measured calibration. Regression retains the inherited parent images and camera convention; the two input pipelines are preserved separately when interpreting transfer from supervised proposals to predicted maps.

No online30FPS, cold end-to-end speed advantage, reconstructed-geometry improvement, untouched benchmark confirmation or global-SOTA claim. Cached-feature timings are conditional model-resident readout costs. Checkpoint DEV validation is included in training; selected-head DEV inference is separately measured. The original whole-map stage records a combined capture/readout time; its separate capture/readout clocks and capture VRAM were not recorded and remain null. Readout peaks cover the post-capture interval. Any separately labeled cache-only profile is conditional on existing features and excluded from scientific totals. Training wall time accumulates resumed invocations; memory peaks are from the final invocation and earlier resumed peaks were not aggregated. The full public predict invocation includes CPU payload construction and current-class rank rebuilding. It may reuse earlier capture/readout receipts; its invocation time and those earlier stage times do not form one cold end-to-end total.
