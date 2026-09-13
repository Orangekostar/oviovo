# R5 Room0 released dense branch

Status: all 172 selected frames and 497 fixed native B0 query slots completed;
real released GLA-CLIP, AnyUp, sparse refinement, storage audits and paired native
evaluation are finished. The branch underperforms native B0/S1a. **D1/D2/D3 remain
off**; lambda .1/.2/.3 all give identical final labels, so no winning lambda is claimed.
No distillation, codebook training, SPAR or second dense backbone was added.

## Sources and exact execution scope

GLA-CLIP commit 63d674626cb1ddc566e3177b9cb5b2dc7a212158; DINO commit
7c446df5b9f45747937fb0d72314eb9f7b66930a; AnyUp commit
351807a9c4287368732cc247f26c7c81c9139af4. Source and weight hashes in binding.json
are authoritative. OpenAI CLIP-B/16 checkpoint SHA matches the official filename;
DINO-B/8 and AnyUp paper checkpoints load strictly from real downloaded files.
The independent venv uses Torch 2.4.0/cu121; no existing native environment was changed.

The bridge executes the original GLA forward_feature body, before vocabulary
classification, with the released ProxyCLIP/DINO options. It resizes Replica RGB
to 336x592 on an 8-pixel grid, processes ten 224x224 stride-112 windows, and stitches
512-channel 42x74 features. This is a mapping adapter, not an exact MMSegmentation
benchmark reproduction (including its resize/logit-stitch conventions). The
temporary DINO QKV hook is removed after each call; all 172 receipts show zero
remaining hooks. Preliminary repeated single-frame features match exactly.

AnyUp uses the original paper checkpoint, non-NATTEN implementation, ImageNet RGB
normalization, output 680x1200 and q_chunk_size=4096. D1 uses bilinear interpolation;
D2 uses AnyUp. D3 uses sparse 4-neighbor edges only between positive identical native
owners, valid visible depth, and strict depth difference <.05 m. Unknown owner 0
does not propagate. Cosine affinity temperature is 1; lambda=.1/.2/.3. Sparse CSR
storage is O(edges); there is no dense million-by-million matrix. Linear pooling
allows one neighbor pass to provide the three exact mixing coefficients.

Production dense space:
`sha256:3780c96d3f71a23efe8b26ba7ad831a20aa506d0dd75ae2d42172c69fa6fd647`.
ROI control space:
`sha256:b3a4b81c546cead185693f86fc7faa95a2dce6931a6aa630f2c17c4dba4b38fc`.
Both differ from native SigLIP-1024. D0 uses original CLIP global encoding of an
unmasked native exclusive bbox, standard 224 center crop and the same CLIP text
ensemble. It is not the native SigLIP six-crop procedure or a frozen-encoder ablation.

## Query/support budget and representation

All branches attempt the same 497 selected observations from 71 eligible B0 objects,
K<=8, preserving source-query IDs and native area weights. Dense pooling uses final
native-owner pixels with measured depth consistency. 46 slots have no support,
leaving 451 valid dense slots and 62 eligible objects under the same minimum-two
rule. Lost eligibility: 14,27,42,49,77,91,92,97,127. Their geometry remains present
and unknown; no cross-space fallback or extra image query fills the missing slots.

Thus D0→D1 includes visibility/support effects as well as changing feature extraction.
An additional **D0_MATCHED_VISIBLE** control reuses ROI features on exactly the same
451 slots and eligibility as D1; actual slot-ID equality is checked. D1→D2→D3 have
identical support and budgets. The matched control changes vertex mIoU by only
.000192 and leaves semantic AP unchanged; missing support is not the main observed
ROI/dense quality difference here.

Per object, retain at most 8 feature observations and 4 diverse observed image-quadrant
prototypes. These are not canonical object parts, and prototype maxima do not enter
the reported classifier. D0 retains 272 valid prototypes; each dense condition retains
252. Including object IDs, each condition's arrays occupy 1,605,381 bytes across
71 object slots. Actual compressed files: D0 941,738 bytes; D1 1,343,024; D2 1,343,097;
D3 .1/.2/.3: 1,342,929 / 1,342,706 / 1,341,625. Whole six-condition file is
7,646,193 bytes; raw arrays 9,629,446 bytes; binding sidecar 101,655 bytes. The original
producer feature-payload byte counter excludes 568 bytes of object IDs; storage_audit.json
explicitly includes them. Frame/prototype candidate cache is 22,909,610 bytes, separate
from retained representation. Model weights and historical native cache are also separate.

## Measured results

All conditions use identical full native owners, including unknown geometry. B0
matches all 954,492 original projected semantic vertices, original instance masks,
class/score manifest and GT conversion exactly. Projection previously passed the
independent strict <5 cm 1NN audit. GT is read only by evaluation.

| Condition | Vertex mIoU | Vertex mAcc | Semantic instance AP | AP50 | AP25 |
|---|---:|---:|---:|---:|---:|
| B0 native | .332857 | .381587 | .154316 | .343233 | .368233 |
| S1a native reference | .362175 | .416451 | .186723 | .384900 | .409900 |
| D0 ROI | .185136 | .276953 | .109685 | .201864 | .251864 |
| D0 matched visible | .184944 | .276953 | .109685 | .201864 | .251864 |
| D1 dense + bilinear | .243299 | .330209 | .140832 | .207221 | .257221 |
| D2 dense + AnyUp | .244616 | .330210 | .144304 | .210694 | .260694 |
| D3 lambda .1 | .244616 | .330210 | .144304 | .210694 | .260694 |
| D3 lambda .2 | .244616 | .330210 | .144304 | .210694 | .260694 |
| D3 lambda .3 | .244616 | .330210 | .144304 | .210694 | .260694 |

Semantic-instance AP uses the original released area-normalized confidence and
48-class instance subset, distinct from the 51-class text/vertex vocabulary. All
released class-agnostic geometry diagnostics are unchanged: mIoU .5220, mP50 .6790,
mR50 .5978. These are not integrated AP. R4's B0 canonical diagnostic is unchanged
because the full-owner geometry is identical; it is not newly claimed paper-AP parity.

## Actual cost and decision

GPU 0, A40; 8 CPU threads. There are 497 original-CLIP ROI calls, 172 dense frames,
1,720 dense CLIP windows and 1,720 DINO windows. D1/D2/D3 share the dense forward;
do not count each branch as a separate full re-encoding or hide the window budget.

| Timed stage | Seconds |
|---|---:|
| ROI encoding | 15.117 |
| GLA dense forward | 14.209 |
| Bilinear upsample | .569 |
| Bilinear transfer + pooling | 239.456 |
| AnyUp upsample | 270.240 |
| AnyUp transfer + pooling | 233.145 |
| Sparse neighbors + three lambda pools | 1745.301 |
| Sum of measured stages | 2518.038 |

Model load/text encoding in the resumed invocation took 10.419 s separately. Peak
GPU allocation was 8,501,040,640 bytes, process peak RSS 9,925,677,056 bytes. Readout
alone took roughly .028–.044 s per branch. The two-frame prefix was completed and
resumed with exact binding/hash checks; prefix model startup is not included in
the resumed-load number. Host work overlapped R6 and later B1 on other GPUs, so
these are shared-host stage measurements, not controlled end-to-end FPS. Historical
front-end/mapping, serialization and evaluation are not included in the stage sum.

AnyUp yields a small within-branch AP increase (.003472), but the whole dense branch
remains below B0 and S1a. Sparse refinement incurs substantial CPU cost without
changing classifications; no scaling-up or distillation is justified by this result.
The current no-extra-3D-training main configuration remains S1a, with S2/G1/G2/D1/D2/D3
off. C1 and full-scene validation remain outstanding; no full-benchmark gain is claimed.

Artifacts: `artifacts/static_ovmap/room0_dense/`; exact commands, source identities,
172 frame receipts, readouts, metrics, storage audit and external payload hashes
are retained. Runtime directory:
`/home/ww/oviovo_baseline_runs/20260913_static_ovmap/room0/dense_run/`.
