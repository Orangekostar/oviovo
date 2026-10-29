# node7 Replica progress V5

All eight existing node7 Replica scenes are reused. replica_3d is the loader directory name; its evaluation tuples are derived from published author GT, not original tuple bytes. No Replica dataset was downloaded again.

Native RAW: 32/64 standard cells, all eight scenes, four complete RAW pools. U01/U11 and Replica bootstrap remain incomplete because 14 training-role mappings remain unresolved.

Scored baselines: OpenMask3D 1/8, MaskClustering 5/8, OVI-MAP 8/8, Open3DIS 1/8.

Full Open3DIS office0: genuine ISBNet, Grounded SAM-HQ, SPP lifting, refinement and final classification; all four stages and CPU score exited zero, with all 480 author/common AP entries identical. The original CPU score succeeded; its proposed cache recovery was never launched.

Multinode: node3 GPUs 0/1 and node1 GPU0 execute office1/office2/office3; remaining scenes and CPU scores are queued. All seven scene inputs and portable layouts are ready. Original node7-only raw symlinks are preserved; new shared copies retain every original byte and calibration value. Failed CPU preflights and the controller metadata-visibility failure remain recorded; successful ISBNet is reused.

Full objective unfinished. Publication disabled; no authorization to publish partial results.

Progress review identity: 2753f9e67290e7741fe2ee23838fb101d3d55f3f394028b8ab046472ddad9794
