# External Table3 Source Audit

Status: AUTHOR_REPORTED_NOT_REPRODUCED. External raw scorer/rank identities and
the author's ScanNet sequence membership are not independently verified.

The chosen source is the [CVPR2026 CVF final PDF, Table3, PDF page6/printed12611](https://openaccess.thecvf.com/content/CVPR2026/papers/Deng_OVI-MAP_Open-Vocabulary_Instance-Semantic_Mapping_CVPR_2026_paper.pdf).
All30 numeric fields for the three ordinary rows were also checked against the
[arXiv v1 PDF](https://arxiv.org/pdf/2603.26541v1). The two PDFs agree. The CVF table
page was rendered and visually inspected; extraction preserved the authored
mIoU/mAcc/AP25/AP50/APall column order and excluded30-fps variants.

The immutable supplied literature_reference.json/HTML transcription differs in
seven fields. Source selection is explicitly changed to the verified PDF
representation before the implementation freeze. The supplied package remains
byte-identical, and its original values/note are retained in the source receipt.
No values were averaged, columns swapped, or corrections inferred from expected
metric ordering.

| Author Method | Source Dataset | Metric | Supplied (%) | Both PDFs (%) |
| --- | --- | --- | --- | --- |
| Mask3D + OpenMask3D | Replica | AP25 | 16.8 | 6.8 |
| Mask3D + OpenMask3D | Replica | AP50 | 14.5 | 4.5 |
| Mask3D + OpenMask3D | ScanNet | mIoU | 18.6 | 8.6 |
| Mask3D + OpenMask3D | ScanNet | AP50 | 18.0 | 8.0 |
| Segment3D + OpenMask3D | ScanNet | mIoU | 14.5 | 4.5 |
| Segment3D + OpenMask3D | ScanNet | AP25 | 15.0 | 5.0 |
| Segment3D + OpenMask3D | ScanNet | AP50 | 13.9 | 3.9 |

The original ScanNet Mask3D AP25<AP50 caution is retained in the original source
document. The PDFs explicitly give AP25=10.4 and AP50=8.0; the discrepancy is
therefore documented from direct source evidence, not repaired by assumption.

Shared cache: `/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001/external/`.
Canonical source receipt: `reference.json`, identity
`7499a1092303d7e6eb59bf198283c84915d5358b87263e6c3ac0fbd089a578cb`.
CVF PDF SHA256: `52622c8aae5c864ed093f802af5cec936058c13f2910f7740015f08c6280bf58`.
arXiv v1 PDF SHA256: `abbb5e569218a0aecbe9155069f15a14e8044ef7d9474737b5a459dc7a8d3fc8`.

External precision remains one decimal place. These three rows remain a separate
author-reported context block with AUTHOR_PROTOCOL_AS_REPORTED; CF18 is only the
placement key for the internal table column, not a verified external cohort.
No paired effect, significance, global-best bolding or SOTA claim crosses this
provenance boundary. Mask3D/Segment3D provide offline mesh instance masks with
OpenMask3D RGB-D semantic labeling; ordinary OVO-SLAM is the online context row.
