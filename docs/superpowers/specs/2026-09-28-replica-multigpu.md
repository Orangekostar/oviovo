# Replica hardware scheduling extension

User explicitly authorized parallel scenes across GPUs on 2026-09-28.
The original transfer contract remains unchanged. Hardware-only worker contracts
clone it, changing CUDA device, device lock and shared-memory namespace only.

- GPU 0: office0, office2.
- GPU 1: office1, office3.
- GPU 2: finish room2, then office4.
- room0 and room1 retain their completed original results.

Each worker runs capture, preparation, Q_GAIN, M4, fusion and all six evaluations
in sequence. The original orchestrator waits for delegated scene receipts before
its final report. All five metrics (uAP, AP25, AP50, mIoU, mAcc) remain required.
Frames, budgets, temperatures, checkpoint, FP32 and evaluator are unchanged.

The pinned mapper uses global POSIX shared-memory names. A launcher sets only its
IPC module's prefix to an attempt-and-scene-specific value before executing the
unchanged upstream mapper. The perception worker receives the actual slot names.
The isolation test creates two tiny rings, verifies independent contents and
attachment by name, and cleans up only its own slots.

Runtime contracts and scene completion receipts are under the attempt's parallel/
directory. Original shared GPU lock paths prevent overlap on a single GPU.

Validation: python -m pytest tests/replica_transfer -q (13 tests passed).
