# 校准与选模源码复核

2026-10-24：本次提前复核最终审核中的稳定校准组件。范围为冻结源码、原G1实测证据及保存输出的身份核验；本轮数值重跑、模型调用和训练均为0。完整benchmark、正式统计、最终发布及17章整体审核仍待完成。

| 合同要求 | 源码与既有实测证据 |
| --- | --- |
| CAP/MIX/NMIX及γ单位 | CAP复用原FP64角度/切向计算并恢复原向量dtype；NMIX内部FP64组合/范数后恢复原dtype，端点复制源分数。γ直接作用于余弦分数。 |
| 完整原件分母与unknown | 279原件、9个unsupported保留；unknown=0计入原分母，macro只含正样本类别，空组保持null，argmax平局沿原valid_ids次序。 |
| 同预算及物理计数 | 每方法/seed为165候选并集，各族55；共1331逻辑候选，精确复用旧MV220，实际CPU候选预测1111，基础算分100次。 |
| LOFO与两seed | 12物理家族；另11家族seed17选参，同一配方应用seed17/29，按原索引拼回OOF后计算指标。静态FC另有12折，总60折。 |
| 固定方法与强简单对照 | 主方法保持CAP10/γ0.025；CAP/MIX/NMIX各从已曝光CONF12 seed17的55格选择，迁移benchmark后不再选参。novel_label_free=false。 |
| 表格聚合 | T3从每seed独立指标平均，包括已独立计算的H；S5保留每seed RAW/LOFO。表注明确novel标签参与选参及条件识别与全GT AP的范围。 |

本轮逐段阅读`calibration.py`、`numeric_audit.py`、原`semantic_update_evidence_v1/calibration.py`与`metrics.py`，并核对`tables.py`的校准数据接入、T3/S5及每seed平均。两个继承模块与固定BASE Git blob逐字节一致；7份相关源码/合同/参考算子的SHA及mtime记入复核证明。校准summary引用的4份输出SHA均核对通过，1960份稳定文件及93份冻结文件SHA/mtime保持。

原G1实测范围明确：9组RAW与原记录一致；220个旧MV候选数组精确复用；24组真实向量NMIX核对及72组候选预测核对；60折对象顺序及共享配方核对；四个学习模型各一折独立重新选参；9组OOF指标在拼回原件后复算。未将四折独立选参扩大为全部60折独立重新搜索。原实测中反向CAP和零范数NMIX回退事件均为0；边界回归属于合成检查。

稳定证据：校准summary `73283e52268234d4503f5c349ee1634f37c529dc6d76e2ed16ee73209fb08f80`；原真实数值审计`6ed752da46f53834931983fa633b8adf2f611a9c368b49d486e274ac16c90a33`；RAW精确核对`120d9251ff20f18d622bd7a666476d38020058239ae94d8a7e7cff31822def79`；简单配方锁`8d779d17dccca8cbd52bfd1ea921099e522fadc74cf387dfd7570aed14b8a923`。

本次复核证明：`$OUT/review/G4_calibration_selection_source_primary_review.json`，`0ed4d8c4f6008c10e6e3121403b874d23096fea5acbb0eeea1236c3af1988969`。这是校准组件源码/既有证据审核记录；最终审核需再次核对来源未变，并结合完整场景、正式统计、外部依赖与实际发布结果判定七个完成标志。
