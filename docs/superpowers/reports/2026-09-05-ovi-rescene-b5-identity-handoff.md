# OVI-MAP x Persist4D ReScene B5 Identity Handoff

Date: 2026-09-05
Branch: `research/ovi-rescene-persist4d-b5-identity`
Base SHA: `2136865993033e35c44dac12444363a4788e8452`
Result-package commit: `8e72f6df433941335f18f6bb447c9fcff36a3c13`
Verdict: `RESCENE_B5_BLOCKED`

## Required Answers

1. **Was the canonical Persist4D checkpoint found?** Yes.
2. **Exact path, size, and SHA?** `/home/ww/paper5/checkpoints/rescene4d_concerto_t2_repro.ckpt`; 754,917,862 bytes; `85ed1aba60320cd19798536b71b91dbc156b7ea60f838832bc0bbbdba131546e`.
3. **Regular non-symlink?** Yes; direct regular file, `symlink=false`.
4. **Official or local reproduction?** `SOURCE_BOUND_RESCENE_REPRODUCTION`; not an official author checkpoint and not paper parity.
5. **Which report and commit bind it?** Persist4D `artifacts/P2_G2_REPRODUCTION_REPORT.md`, 9,790 bytes, SHA `d891fb7fd53306d8ab65db81b9bb85f08664a9689de850ac7836143b238816bc`, at commit `1380c4b9f37bec7933126ccc9bd70067de166f6f`.
6. **Was exact Concerto initialization verified?** Yes: `/home/ww/.cache/persist4d/concerto/concerto_base.pth`, 433,987,358 bytes, SHA `845ec7dec97a5fabff8fadb5d9858ac6734347b612d1a4b574213419c139de07`, revision `c31f993a56129f2ba9c5d06a35957e3f05bff710`.
7. **Does it strict-load pristine upstream?** Yes at the model-topology boundary: all 796 model tensors are consumed with exact shapes by ReScene `fb2fe42eb8f1e926567c48eea9acb874e608ee10`.
8. **If not, what is the actual runtime source?** Not applicable. The selected inference source would be the pristine upstream commit above; C3 was not reached.
9. **Were keys dropped, renamed, or ignored?** No model keys; missing, unexpected, shape-mismatch, and ignored/dropped lists are empty.
10. **Actual input feature contract?** Nine channels: shared-centered XYZ (3), source camera RGB in `[0,1]` (3), unit source geometry normals (3).
11. **Does frozen OVI RGB satisfy it?** No. PLY RGB is the instance palette and snapshots do not retain per-point camera RGB.
12. **Are normals required?** Yes. PLYs contain normals, but the frozen OVI loader/snapshot does not preserve a source-bound point-normal alignment.
13. **How do raw coordinates enter the backbone?** Shared-centered floating XYZ enters both Pointcept coordinates and the first three model-feature channels; integer grid coordinates are not substituted for XYZ features.
14. **How is color normalized?** Actual model RGB is source camera RGB in `[0,1]`; dataset-normalized RGB is discarded by the Pointcept collator. No ImageNet/RIO or second normalization is legal.
15. **How are t, batch, and grid handled?** Shared pair center `[-9.390950173139572, -2.7622697500133873, -0.07036522589623928]`; sequence batch is 0; Pointcept visit batches are 0/1; temporal values are exactly 0/1; each visit uses its native 2 cm spatial grid.
16. **Adapter token count?** t0 552,612; t1 490,416; total 1,043,028.
17. **Would-be executor input token count?** t0 446,153; t1 408,276; total 854,429.
18. **Are merge/drop all zero?** No. Merge counts are 106,459 and 82,140; cross-entity merges are 2,973 and 2,069. Drop and duplication are zero.
19. **Does token conservation pass?** No: `BLOCKED_RESCENE_TOKEN_CONSERVATION`; the mapping is not a permutation.
20. **Did the one-pair backend pass?** Not run because C2 failed before GPU.
21. **t0/t1 token count?** Only preflight counts exist: adapter 552,612/490,416 and would-be model 446,153/408,276; successful backend counts are null.
22. **Query count?** null; no backend output exists.
23. **Runtime and peak GPU memory?** null/null for B5; GPU command count is zero.
24. **B4 relation counts?** Frozen total is 120. Per-state and 1:1 counts are null because old B4 did not serialize `PairRelation` and the conditional rebuild was skipped.
25. **B5 relation counts?** null; B5 was not run.
26. **Intersection, B4-only, B5-only?** null/null/null; comparison was not run.
27. **B5-only persistent_static?** null, not zero.
28. **B5-only persistent_moved?** null, not zero.
29. **How many B5-only 1:1 pass frozen registration?** null; no B5 relations exist.
30. **Can true ReScene identity accuracy be claimed higher?** No. C3 did not run and TESSE lacks legal source-bound cross-visit identity GT; relation difference would not itself establish correctness.
31. **Was any current map or B7 run?** No. Learned B7 is `NOT_RUN_BY_SCOPE`; no map metric was recomputed.
32. **Office attempt count?** 0; `HELD_OUT`.
33. **3RScan X/44?** 27/44 complete; 17 lack only `sequence.zip`; final ranking not run.
34. **Focused tests?** 55 passed, 0 failed in the canonical Python 3.13 environment. The unbound host `pytest` launcher first failed during collection because its Python 3.12 Torch installation was incomplete; no test body failed there.
35. **Full suite?** 6,009 passed, 10 skipped, 0 failed, 5 NumPy deprecation warnings in 815.93 seconds; one authorized run.
36. **Final branch/local/remote SHA?** The result-bearing local and remote SHA is `8e72f6df433941335f18f6bb447c9fcff36a3c13` on the branch above. The handoff-only closure commit necessarily advances the branch after this file is written; its exact local/remote equality is recorded in the external upload receipt and final delivery because a Git commit cannot embed its own SHA.
37. **Single highest-value next action?** Build a versioned, source-bound RGB-D-to-OVI-token camera-RGB and normal-alignment bridge plus an adapter-to-model reverse map that preserves every adapter token, then rerun C2.

## Artifact Ledger

| Artifact | Bytes | SHA-256 | Producer commit |
| --- | ---: | --- | --- |
| Checkpoint audit `/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/checkpoint-audit/checkpoint_audit_summary.json` | 2,961 | `9721e343415860d1d1aea67f62beee1f33faec3441b063b2f521ecd485449bea` | `ae992da04544d465470d7dd4e08f7c76adc8032a` |
| Native witness `/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/native-input-witness/native_t2_input_witness.json` | 2,505 | `60ae18c00a4b88eb8c67f3fc8afb44e773f5c1ebb373e127cf183a9b40dee448` | `7d8a6c4908d0d9e035a67d92aecfde56abfde6cd` |
| Input contract `/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/ovi-input-witness/input_contract_summary.json` | 5,578 | `ee9b64abc8173707603d0c081947a6cb09da9fa48cec15db83575cb94abe1e0c` | bound by pair receipt |
| Pair receipt `/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/apartment/pair/apartment_pair_receipt.json` | 7,249 | `2fc579636d02acfa1fa026e9bfd0fa6483770a14e13e19372046481b86a646d2` | `bebe0b268b8ba9b125f8198ac0f1f6ef5ee00c28` |

All producer commands, input hashes, and source bindings are recorded in the
machine-readable receipts. No checkpoint, pair arrays, raw sample, query masks,
or other large binary is committed to Git.

## Verification And Upload

Focused tests: `55 passed, 0 failed` using `/home/ww/miniconda3/bin/python`.

Full suite: `6009 passed, 10 skipped, 0 failed` using
`/home/ww/miniconda3/bin/python -m pytest -q`.

Result-package local SHA: `8e72f6df433941335f18f6bb447c9fcff36a3c13`.

Result-package remote SHA: `8e72f6df433941335f18f6bb447c9fcff36a3c13`.

`UPLOAD_STATUS=VERIFIED`

The exact post-handoff branch-head equality is recorded at
`/home/ww/oviovo_baseline_runs/20260905_ovi_rescene_b5_identity/upload_receipt.json`
after the final push.
