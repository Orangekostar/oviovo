# Replica training-role extension: user-confirmed synonyms

On 2026-10-30 the user confirmed “都为同义” for the proposed semantic relations. This extension resolves the remaining binary base/novel roles used by the fixed CAP10, γ=0.025 intervention. It preserves the training vocabulary (200 classes), the trained base set (160 classes), the official Replica evaluation classes, point order, GT, models, and earlier resolved roles. It does not claim that the manual relations are author-original aliases.

| Replica name | Explicit global semantic IDs | Role | Evidence |
|---|---|---|---|
| camera | 1266 | novel | LabelMaker crosswalk |
| comforter | 54 | base | LabelMaker crosswalk |
| desk-organizer | 1183 (storage organizer) | base | User-confirmed synonym |
| indoor-plant | 40, 1189 | base | LabelMaker crosswalk; same role |
| panel | 69, 89, 154, 561, 1195 | base | Crosswalk plus user-confirmed synonym group |
| plant-stand | 484 | novel | LabelMaker crosswalk |
| pot | 226, 1234 | novel | LabelMaker crosswalk; same raw Replica ID |
| sculpture | 281 | novel | LabelMaker crosswalk |
| tablet | 1289 (iPad/tablet computer) | novel | User-confirmed synonym |
| tissue-paper | 79 (paper) | base | User-confirmed synonym |
| tv-screen | 33 | base | LabelMaker crosswalk |
| vase | 194, 1217 | novel | LabelMaker crosswalk; same role |
| wall-plug | 242 | base | LabelMaker crosswalk |
| rug | 53, 61 | novel | LabelMaker crosswalk; same role |

For `panel`, the original fine-grained source classes have different roles. Under the confirmed synonym equivalence, a group is exposed/base if any member belongs to the frozen trained base set. This is an explicit new semantic policy. No AP, target-label frequency, embedding similarity, or ID magnitude was used to choose a role. IDs outside the training axis are retained as sourced semantic evidence; they do not expand that axis. An unmatched name remains unresolved.

The external evidence is [LabelMaker's fixed crosswalk](https://github.com/cvg/LabelMaker/blob/44ff92d42ae9b0579f016bb7bc5cd4bd09a613a3/labelmaker/mappings/label_mapping.csv) (SHA-256 `d04f3d9fd7cd4c5a9bb92d836f51618b974e25063e1a3ecab51dcf6fe7d03e7e`) and the frozen official ScanNet global taxonomy (`68674782476d967c35e8d0776933b70e10d21f1ea8019eba20221d38275139c0`). Raw Replica taxonomy IDs in that crosswalk differ from the evaluator's reindexed 1–48 IDs; comma-separated names and IDs are paired in source order.

The previous partial release, C1 `dd82c2e0e795ab0becf2281431ff1ab8860870cd` and C2 `df81b1ba42ce508a19ce945b52a03264598b656d`, remains historical evidence. The new protocol must be assessed with its explicit semantic extension. Historical target-label access is already disclosed; this continuation does not retrospectively claim that target GT had never been accessed.
