# 固定实验矩阵

所有条件在同一Replica-8＋CF18完整评价；新条件只改已有实例类别，不改owner分区与恢复对象类别。

| 配置 | 几何/分区 | 决策 | 意义 |
|---|---|---|---|
| SU00_D2 | 父D2 | 原生D2 | 52条继承基线的一半 |
| SU01_G1 | 父G1 v2 | 原D2＋G1恢复 | 统一性能保护参照 |
| SU02_HARD_MATCHED | G1固定 | 共同域内重放IR06 | 不是直接抄原IR06成绩 |
| SU03_STABLE_MATCHED | G1固定 | 共同域内重放IR07 | 不是直接抄原IR07成绩 |
| SU04_F_REPLACE | G1固定 | N/Q保留，F→A | 保留旧裁剪证据 |
| SU05_F_BLEND | G1固定 | F组内旧F与A等权 | 保留旧区域证据 |
| SU06_F_COARSE | G1固定 | F组内旧F与同视图C等权 | 同视图表示对照 |
| SU07_GLOBAL_BLEND | G1固定 | 整D2与A混合，A质量系数匹配SU05 | 普通集成对照 |
| SU08_PAIRED_DELTA | G1固定 | F加入同视图A−C残差，N/Q保留 | 配对表示变化对照 |

9×26=234条逻辑结果；9×2=18个池化。52条原基线可继承；182条匹配/新条件需要真实预测和评价，只有严格内容等价才复用计算。

## 预设配对

- `SU04_F_REPLACE − SU02_HARD_MATCHED`
- `SU05_F_BLEND − SU04_F_REPLACE`
- `SU05_F_BLEND − SU06_F_COARSE`
- `SU05_F_BLEND − SU07_GLOBAL_BLEND`
- `SU08_PAIRED_DELTA − SU05_F_BLEND`
- `SU03_STABLE_MATCHED − SU02_HARD_MATCHED`

每种方法同时报告相对SU01的全五项变化，以及CF18相对D2的变化。没有增益的方法也保留。原始IR06/IR07仅作为谱系/覆盖上下文，不额外计入九种主配置。
