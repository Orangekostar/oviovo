# Paired evidence handoff

Branch: `research/ovimap-paired-evidence-v1`; base: `9c35333088551a9d4a67937936d4d9c347b2b207`.

Worktree: `/home/ww/crove/ovimap-paired-evidence`. Output: `/mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001`. Parent binding: `/mnt/shared/ww/ovimap-a7-evidence-upgrade-wave1/attempt_001/binding.json`.

Runtime: `/home/ww/miniconda3/envs/ovimap-map/bin/python`. Existing NumPy/SciPy and original released evaluator; no environment reinstall. Scene workers use spawn, 4 BLAS threads, default 3 concurrent CPU processes. GPUs remain unused because the protocol prohibits new neural inference.

Optional `--path-map mapping.json` accepts explicit old-root/new-root pairs. It creates recoverable directory symlinks only when the old root is absent and its parent exists, preserving embedded signed paths. It never replaces existing files/directories, and every consumed bound input still passes content-hash verification. No relocation was required for this measured run.

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_paired_evidence.py --phase all --resume --threads 4 --workers 3
```

Phases: bind → recover → cal → freeze → replica → diagnose → report → publish. Independent scenes parallelize; prediction locks precede evaluation/GT; the transfer lock precedes Replica. Parent files and source-temperature fits remain unchanged. Exact evaluator-byte relocation is recorded separately and identity reuse has explicit aliases.

The released pooler prints an outside-prediction-path notice when it consumes an unchanged parent mask manifest. The pinned evaluator still loads those masks; exact original-control scene and pooled metric parity was verified. These notices are retained in logs, not hidden by changing the evaluator.

New modules: binding/evidence replace old branch-specific binding and recover caches; residuals/dependence/pair_graph implement fixed operators; evaluation wraps original SceneEvaluator/pool; selection freezes CAL; workflow schedules scenes; diagnostics joins outcomes; reporting/publication build measured deliverables. Old guards remain intact.

Large prediction arrays, masks and released traces remain in shared storage. Compact sources, labels, probabilities, rows, pools, choices and diagnostics are published with a content manifest. Reproduction requires the bound original caches, not just GitHub. Missing external caches cannot be replaced with inference in this protocol.

Binding identity: `6e0751c377b470d1126e3cf241b19c53b12bce9fdb80a2541bef2c2d1bca19bd`. Transfer identity: `1b5c3dfca410a965249f77b2275a6e54517bbc9e5188e3f767180b2654456198`. Report input identity: `6bbd516e021f203e875195d60d1878a02aa7baea742659f1140afd8607051d88`.

External final publication receipt: `/mnt/shared/ww/ovimap-paired-evidence-v1/attempt_001/publication/final.json`. It records the containing commit SHA after a normal push and comparison; the receipt is not recursively committed.

Not executed: conditional information acquisition, E06, new geometry/model/training work. All scenes are exposed; independent confirmation remains future work, not a completed claim.
