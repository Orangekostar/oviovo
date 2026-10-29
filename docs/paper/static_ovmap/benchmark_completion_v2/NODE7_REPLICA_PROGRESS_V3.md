# Node7 Replica progress, 2026-10-29

Existing node7 Replica RGBD, author point clouds and clean meshes are reused. No replacement Replica dataset was downloaded. Only the two missing author model checkpoints (ISBNet and GroundingDINO) were acquired; SAM-H and CLIP weights are reused.

The accepted office0 geometry contains all 589,517 original vertices. All 588,759 source quads were converted in original face order to 1,177,518 triangles using the locally frozen Habitat StanfordImporter convention. The unchanged author mesh segmentator generated 1,248 superpoints. Partial Open3D conversion v2 is explicitly rejected and preserved; only input v3 is accepted.

The dedicated Python 3.10 / Torch 1.12.1+cu113 / NumPy 1.23.5 environment, CUDA 11.3 compiler SDK and GCC 9 are isolated from active experiment environments. Unchanged PointNet2, ISBNet and GroundingDINO CUDA sources were compiled successfully. Added dependencies are pinned with wheel SHA256 hashes; no active numerical library was upgraded.

Real office0 ISBNet inference completed using all 582 checkpoint tensors, the unchanged author test dataset transformation and full original point order. It generated 64 actual proposals and a 589,517 x 32 dense feature tensor. The actual forward took 4.672315579955466 seconds and loaded no target GT tensors. This is a prerequisite of full Open3DIS, not a completed Open3DIS benchmark result. The initial CPU check failure and first GPU initialization/call failure remain recorded; the corrected second GPU attempt exited 0.

GroundingDINO CPU validation strictly loaded all 939 inference tensors, including all 200 embedded BERT tensors. The checkpoint contains 940 tensors: label_enc.weight (2001 x 256) is absent from every frozen inference Python source, and dn_number is zero. That single unconsumed tensor is explicitly audited and retained in the original checkpoint bytes. No arbitrary unexpected or missing inference tensor is accepted. Only the official small tokenizer/config files were acquired; no separate BERT model weights were downloaded.

Native Replica standard coverage is derived from completed score receipts and full original-point G3 comparisons, never from running processes or proposal counts. The current reviewed snapshot has at least 24 of the expected 64 cells (six scenes, U00/U10 with seeds 17/29). OVI-MAP has five scored scenes; the OpenMask3D and MaskClustering remaining queues finish all original inferences before scoring. The machine-readable review records the exact snapshot and child status.

Remaining work includes the full Grounded-SAM -> 3D lifting -> refinement -> final classification bridge, genuine final Open3DIS inference and G3 verification, remaining native/baseline scenes, 14 unresolved training roles for U01/U11, Replica bootstrap, pooled tables and complete publication provenance. Full Open3DIS environment/adapter and the original full objective are not complete. Publication remains disabled.
