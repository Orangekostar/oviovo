# OVMap 贡献归因实测结果

{"execution_status": "COMPLETE", "evidence_status": "ATTRIBUTION_COMPLETE_EXPOSED_ONLY", "publication_status": "NOT_ATTEMPTED_SNAPSHOT", "submission_evidence_complete": false, "missing": []}

本轮不新增训练、RGB编码、SAM提案或建图。所有回归场景已曝光；后续确认仅登记，未执行。

replica8: E_SOURCE APall +3.1148pp, mIoU +2.4509pp；R|E APall +0.6450pp, mIoU +0.5814pp；E_GROUP(D2−equal) APall +1.3346pp, mIoU -0.2893pp。
replica8: D2−锁定普通融合 E8 APall +0.4753pp, mIoU +0.6861pp；E8仅由四个独立融合DEV场景选择，未在REG上改权重。
replica8: selected FG2 seed17−FC8 整图 APall +0.0392pp, mIoU +0.3945pp；此行与同step1000识别对照分开。
replica8: selected FG2 seed29−FC8 整图 APall -0.6729pp, mIoU -0.4594pp；此行与同step1000识别对照分开。
replica8: 新恢复实例42，投影≥100点11；IoU50新增definite TP score entries 3、FP entries 3。覆盖、唯一几何匹配、官方score-entry分别统计；事件数不是互斥对象数。
scannet_cf18: E_SOURCE APall +1.1272pp, mIoU +2.6152pp；R|E APall -0.0770pp, mIoU +0.0247pp；E_GROUP(D2−equal) APall +0.5806pp, mIoU +0.4073pp。
scannet_cf18: D2−锁定普通融合 E8 APall +0.7545pp, mIoU +1.6413pp；E8仅由四个独立融合DEV场景选择，未在REG上改权重。
scannet_cf18: selected FG2 seed17−FC8 整图 APall +0.1929pp, mIoU +0.3919pp；此行与同step1000识别对照分开。
scannet_cf18: selected FG2 seed29−FC8 整图 APall +0.2686pp, mIoU +0.6638pp；此行与同step1000识别对照分开。
scannet_cf18: 新恢复实例132，投影≥100点63；IoU50新增definite TP score entries 0、FP entries 47。覆盖、唯一几何匹配、官方score-entry分别统计；事件数不是互斥对象数。
S seed17同1000step DEV A：FG1 38.8721% → FG2 46.0711%（+7.1991pp），初始化/抽样记录匹配，全200类竞争。
S seed29同1000step DEV A：FG1 37.6073% → FG2 46.1732%（+8.5659pp），初始化/抽样记录匹配，全200类竞争。
replica8 E6_E7 rank诊断：固定旧rank改类别 +1.2960pp；固定旧类别改rank +0.2022pp；交互 -0.1636pp。非对角仅诊断，不能加入正式主表或解释为训练因果份额。
scannet_cf18 E6_E7 rank诊断：固定旧rank改类别 +0.5273pp；固定旧类别改rank +0.1713pp；交互 -0.1180pp。非对角仅诊断，不能加入正式主表或解释为训练因果份额。
replica8 FG0_FC8_FG2_SURFACE rank诊断：固定旧rank改类别 -0.1343pp；固定旧类别改rank +0.0417pp；交互 +0.1318pp。非对角仅诊断，不能加入正式主表或解释为训练因果份额。
scannet_cf18 FG0_FC8_FG2_SURFACE rank诊断：固定旧rank改类别 +0.1125pp；固定旧类别改rank +0.0069pp；交互 +0.0734pp。非对角仅诊断，不能加入正式主表或解释为训练因果份额。
replica8 E6→E7：2000家族配对draw的ΔAPall 95%分位区间 [-0.2646pp, +3.5942pp]；defined=2000，undefined=0。已曝光探索性区间，非独立确认。
replica8 FG0_FC8→FG2_SURFACE：2000家族配对draw的ΔAPall 95%分位区间 [-0.1575pp, +0.3778pp]；defined=2000，undefined=0。已曝光探索性区间，非独立确认。
scannet_cf18 E6→E7：2000家族配对draw的ΔAPall 95%分位区间 [+0.3236pp, +0.8566pp]；defined=2000，undefined=0。已曝光探索性区间，非独立确认。
scannet_cf18 FG0_FC8→FG2_SURFACE：2000家族配对draw的ΔAPall 95%分位区间 [-0.0583pp, +0.4666pp]；defined=2000，undefined=0。已曝光探索性区间，非独立确认。

## 系统与论文原版参照

| cohort | method | apall | ap50 | ap25 | miou | macc | measurement_kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| replica8 | A0 | 8.63 | 21.48 | 34.59 | 27.24 | 32.67 | REUSED_EXACT |
| replica8 | A1 | 11.74 | 24.50 | 37.97 | 29.69 | 37.43 | REUSED_EXACT |
| replica8 | A2 | 9.44 | 23.69 | 36.93 | 27.96 | 33.65 | REUSED_EXACT |
| replica8 | A3 | 12.39 | 26.06 | 39.75 | 30.27 | 38.40 | REUSED_EXACT |
| replica8 | A4 | 11.74 | 24.50 | 38.15 | 29.82 | 37.86 | REUSED_EXACT |
| replica8 | A5 | 10.94 | 21.88 | 32.29 | 23.72 | 33.28 | REUSED_EXACT |
| replica8 | U2 | 12.39 | 26.06 | 39.84 | 30.17 | 38.15 | REUSED_EXACT |
| replica8 | G3 | 12.10 | 25.54 | 39.23 | 30.18 | 38.47 | REUSED_EXACT |
| replica8 | FG0_FC8 | 11.73 | 26.57 | 41.47 | 31.77 | 39.30 | REUSED_EXACT |
| replica8 | FG1_VIEW | 11.11 | 24.32 | 39.83 | 30.13 | 37.26 | REUSED_EXACT |
| replica8 | FG2_SURFACE | 11.77 | 26.16 | 42.65 | 32.17 | 39.97 | REUSED_EXACT |
| replica8 | S29_FG2_SURFACE | 11.06 | 25.27 | 41.25 | 31.31 | 38.72 | REUSED_EXACT |
| replica8 | FG3_AUX | 11.90 | 26.83 | 43.32 | 32.48 | 40.32 | REUSED_EXACT |
| replica8 | E0 | 8.63 | 21.48 | 34.59 | 27.24 | 32.67 | NEW_CACHED_ANALYSIS |
| replica8 | E1 | 7.16 | 16.91 | 28.10 | 23.46 | 28.89 | NEW_CACHED_ANALYSIS |
| replica8 | E2 | 10.27 | 20.25 | 30.19 | 23.37 | 32.30 | NEW_CACHED_ANALYSIS |
| replica8 | E3 | 8.52 | 19.41 | 31.67 | 25.35 | 30.99 | NEW_CACHED_ANALYSIS |
| replica8 | E4 | 11.42 | 24.80 | 38.60 | 30.56 | 38.19 | NEW_CACHED_ANALYSIS |
| replica8 | E5 | 11.27 | 23.60 | 36.52 | 29.00 | 36.44 | NEW_CACHED_ANALYSIS |
| replica8 | E6 | 10.41 | 24.32 | 37.73 | 29.98 | 36.62 | NEW_CACHED_ANALYSIS |
| replica8 | E7 | 11.74 | 24.50 | 37.97 | 29.69 | 37.43 | NEW_CACHED_ANALYSIS |
| replica8 | E8 | 11.27 | 23.60 | 36.52 | 29.00 | 36.44 | NEW_CACHED_ANALYSIS |
| replica8 | J0_FC8_FONLY | 10.61 | 24.89 | 38.07 | 28.33 | 36.97 | NEW_CACHED_ANALYSIS |
| replica8 | J1_FG2_S17_FONLY | 11.77 | 25.42 | 36.61 | 27.44 | 37.07 | NEW_CACHED_ANALYSIS |
| replica8 | J2_FG2_S29_FONLY | 11.42 | 24.96 | 36.79 | 27.47 | 37.06 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | A0 | 7.18 | 15.72 | 21.21 | 16.38 | 25.36 | REUSED_EXACT |
| scannet_cf18 | A1 | 8.31 | 17.96 | 24.98 | 19.00 | 28.38 | REUSED_EXACT |
| scannet_cf18 | A2 | 7.11 | 15.61 | 21.09 | 16.44 | 25.47 | REUSED_EXACT |
| scannet_cf18 | A3 | 8.23 | 17.83 | 24.76 | 19.02 | 28.49 | REUSED_EXACT |
| scannet_cf18 | A4 | 8.30 | 17.93 | 24.92 | 18.99 | 28.44 | REUSED_EXACT |
| scannet_cf18 | A5 | 8.32 | 17.39 | 23.76 | 17.94 | 24.02 | REUSED_EXACT |
| scannet_cf18 | U2 | — | — | — | — | — | MISSING |
| scannet_cf18 | G3 | — | — | — | — | — | MISSING |
| scannet_cf18 | FG0_FC8 | 7.99 | 17.58 | 26.02 | 20.10 | 29.25 | REUSED_EXACT |
| scannet_cf18 | FG1_VIEW | 8.16 | 18.09 | 26.00 | 20.51 | 29.71 | REUSED_EXACT |
| scannet_cf18 | FG2_SURFACE | 8.19 | 17.92 | 26.26 | 20.49 | 29.56 | REUSED_EXACT |
| scannet_cf18 | S29_FG2_SURFACE | 8.26 | 18.26 | 26.70 | 20.76 | 29.86 | REUSED_EXACT |
| scannet_cf18 | FG3_AUX | 8.22 | 17.80 | 26.16 | 20.37 | 29.53 | REUSED_EXACT |
| scannet_cf18 | E0 | 7.18 | 15.72 | 21.21 | 16.38 | 25.36 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E1 | 5.07 | 10.36 | 14.99 | 11.61 | 18.71 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E2 | 8.38 | 17.57 | 24.03 | 17.96 | 23.90 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E3 | 7.10 | 15.22 | 21.43 | 16.04 | 25.33 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E4 | 8.46 | 18.15 | 25.33 | 19.45 | 28.47 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E5 | 7.55 | 16.04 | 23.16 | 17.36 | 25.14 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E6 | 7.73 | 16.48 | 24.08 | 18.59 | 27.39 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E7 | 8.31 | 17.96 | 24.98 | 19.00 | 28.38 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | E8 | 7.55 | 16.04 | 23.16 | 17.36 | 25.14 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | J0_FC8_FONLY | 8.55 | 17.99 | 26.19 | 20.14 | 28.99 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | J1_FG2_S17_FONLY | 8.33 | 17.79 | 26.06 | 21.00 | 29.55 | NEW_CACHED_ANALYSIS |
| scannet_cf18 | J2_FG2_S29_FONLY | 8.45 | 17.77 | 25.89 | 20.91 | 29.21 | NEW_CACHED_ANALYSIS |
| PAPER_REPLICA | OVI_MAP_PAPER | 8.50 | 21.20 | 34.50 | 26.50 | 32.20 | PAPER_REPORTED_NOT_MATCHED |
| PAPER_SCANNET | OVI_MAP_PAPER | 7.20 | 15.70 | 23.40 | 17.50 | 27.60 | PAPER_REPORTED_NOT_MATCHED |

外部 OVI-MAP 数字来自论文 Table 3，标记 PAPER_REPORTED_NOT_MATCHED；不以这些差值宣称同协议胜出。本地 voxel 描述0.01m与论文0.1m的差异保留。U2/G3历史对照只有Replica记录，CF18相应辅助格为MISSING/null，不扩展原观测库或新增推理填充。12个正式新配置均完整覆盖26场景。

## 模块增量（显示为百分点）

| module | cohort | pair | apall | ap50 | ap25 | miou | macc | attribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E_SOURCE | replica8 | A0 -> A1 | 3.11 | 3.02 | 3.39 | 2.45 | 4.76 | PACKAGE_ONLY |
| E_SOURCE | scannet_cf18 | A0 -> A1 | 1.13 | 2.24 | 3.78 | 2.62 | 3.02 | PACKAGE_ONLY |
| E_GROUP | replica8 | E6 -> E7 | 1.33 | 0.17 | 0.25 | -0.29 | 0.80 | MATCHED_CONTRAST |
| E_GROUP | scannet_cf18 | E6 -> E7 | 0.58 | 1.48 | 0.90 | 0.41 | 0.99 | MATCHED_CONTRAST |
| R | replica8 | A1 -> A3 | 0.65 | 1.56 | 1.78 | 0.58 | 0.98 | CONDITIONALLY_MATCHED |
| R | scannet_cf18 | A1 -> A3 | -0.08 | -0.13 | -0.23 | 0.02 | 0.11 | CONDITIONALLY_MATCHED |
| S | replica8 | FG0_FC8 -> FG2_SURFACE | 0.04 | -0.42 | 1.18 | 0.39 | 0.67 | MATCHED_CONTRAST |
| S | replica8 | FG0_FC8 -> S29_FG2_SURFACE | -0.67 | -1.30 | -0.22 | -0.46 | -0.58 | MATCHED_CONTRAST |
| S | scannet_cf18 | FG0_FC8 -> FG2_SURFACE | 0.19 | 0.35 | 0.24 | 0.39 | 0.30 | MATCHED_CONTRAST |
| S | scannet_cf18 | FG0_FC8 -> S29_FG2_SURFACE | 0.27 | 0.68 | 0.67 | 0.66 | 0.61 | MATCHED_CONTRAST |
| E_GROUP_STRONG_CONTROL | replica8 | E8 -> E7 | 0.48 | 0.90 | 1.45 | 0.69 | 0.99 | MATCHED_CONTRAST |
| E_GROUP_STRONG_CONTROL | scannet_cf18 | E8 -> E7 | 0.75 | 1.93 | 1.83 | 1.64 | 3.23 | MATCHED_CONTRAST |
| ER | replica8 | E_add | 3.11 | 3.02 | 3.39 | 2.45 | 4.76 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | E_cond | 2.94 | 2.37 | 2.82 | 2.32 | 4.76 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | R_add | 0.82 | 2.21 | 2.35 | 0.72 | 0.98 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | R_cond | 0.65 | 1.56 | 1.78 | 0.58 | 0.98 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | interaction | -0.17 | -0.65 | -0.57 | -0.13 | 0.00 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | mean_marginal_E | 3.03 | 2.69 | 3.10 | 2.38 | 4.76 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | mean_marginal_R | 0.73 | 1.89 | 2.06 | 0.65 | 0.98 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | replica8 | total | 3.76 | 4.58 | 5.17 | 3.03 | 5.73 | DESCRIPTIVE_FACTORIAL_E_R |
| S_FUSION_S17 | replica8 | S_add_F_ONLY | 1.16 | 0.53 | -1.46 | -0.89 | 0.10 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | S_add_D2 | 0.04 | -0.42 | 1.18 | 0.39 | 0.67 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | FUSION_add_FC8 | 1.12 | 1.68 | 3.40 | 3.45 | 2.33 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | FUSION_add_FG2 | -0.00 | 0.73 | 6.04 | 4.73 | 2.90 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | S_FUSION_interaction | -1.12 | -0.95 | 2.64 | 1.28 | 0.57 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | mean_marginal_S | 0.60 | 0.06 | -0.14 | -0.25 | 0.38 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | mean_marginal_FUSION | 0.56 | 1.21 | 4.72 | 4.09 | 2.62 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | replica8 | total | 1.15 | 1.26 | 4.58 | 3.84 | 3.00 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | S_add_F_ONLY | 0.81 | 0.07 | -1.28 | -0.85 | 0.09 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | S_add_D2 | -0.67 | -1.30 | -0.22 | -0.46 | -0.58 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | FUSION_add_FC8 | 1.12 | 1.68 | 3.40 | 3.45 | 2.33 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | FUSION_add_FG2 | -0.37 | 0.32 | 4.46 | 3.84 | 1.66 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | S_FUSION_interaction | -1.48 | -1.36 | 1.06 | 0.39 | -0.67 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | mean_marginal_S | 0.07 | -0.62 | -0.75 | -0.66 | -0.25 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | mean_marginal_FUSION | 0.38 | 1.00 | 3.93 | 3.64 | 2.00 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | replica8 | total | 0.44 | 0.38 | 3.18 | 2.99 | 1.75 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| ER | scannet_cf18 | E_add | 1.13 | 2.24 | 3.78 | 2.62 | 3.02 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | E_cond | 1.12 | 2.23 | 3.67 | 2.59 | 3.02 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | R_add | -0.07 | -0.11 | -0.12 | 0.05 | 0.11 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | R_cond | -0.08 | -0.13 | -0.23 | 0.02 | 0.11 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | interaction | -0.01 | -0.01 | -0.11 | -0.03 | 0.00 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | mean_marginal_E | 1.12 | 2.24 | 3.72 | 2.60 | 3.02 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | mean_marginal_R | -0.07 | -0.12 | -0.17 | 0.04 | 0.11 | DESCRIPTIVE_FACTORIAL_E_R |
| ER | scannet_cf18 | total | 1.05 | 2.11 | 3.55 | 2.64 | 3.13 | DESCRIPTIVE_FACTORIAL_E_R |
| S_FUSION_S17 | scannet_cf18 | S_add_F_ONLY | -0.22 | -0.19 | -0.13 | 0.86 | 0.55 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | S_add_D2 | 0.19 | 0.35 | 0.24 | 0.39 | 0.30 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | FUSION_add_FC8 | -0.56 | -0.41 | -0.17 | -0.04 | 0.26 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | FUSION_add_FG2 | -0.15 | 0.13 | 0.20 | -0.52 | 0.01 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | S_FUSION_interaction | 0.41 | 0.54 | 0.37 | -0.47 | -0.25 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | mean_marginal_S | -0.01 | 0.08 | 0.05 | 0.63 | 0.43 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | mean_marginal_FUSION | -0.35 | -0.14 | 0.02 | -0.28 | 0.13 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S17 | scannet_cf18 | total | -0.36 | -0.06 | 0.07 | 0.35 | 0.56 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | S_add_F_ONLY | -0.10 | -0.21 | -0.30 | 0.77 | 0.22 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | S_add_D2 | 0.27 | 0.68 | 0.67 | 0.66 | 0.61 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | FUSION_add_FC8 | -0.56 | -0.41 | -0.17 | -0.04 | 0.26 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | FUSION_add_FG2 | -0.18 | 0.48 | 0.80 | -0.15 | 0.65 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | S_FUSION_interaction | 0.37 | 0.89 | 0.97 | -0.10 | 0.39 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | mean_marginal_S | 0.08 | 0.23 | 0.19 | 0.72 | 0.41 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | mean_marginal_FUSION | -0.37 | 0.04 | 0.32 | -0.09 | 0.45 | DESCRIPTIVE_FACTORIAL_S_FUSION |
| S_FUSION_S29 | scannet_cf18 | total | -0.29 | 0.27 | 0.50 | 0.62 | 0.87 | DESCRIPTIVE_FACTORIAL_S_FUSION |

E_SOURCE 是增加来源的证据包差值；E_GROUP 为融合规则差值。完整三源 D2 与普通(0.25,0.25,0.5)数学等价；缺源/全缺源另见八种availability表。A5是F＋R；E0纯N不默认等于A0原生标签。

## 预算与证据范围

| operation | actual | limit | scope | evidence_status |
| --- | --- | --- | --- | --- |
| head_calls | 3324 | 4000 | NEW_THIS_TASK | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| RGB_encodings | 0 | 0 | NEW_THIS_TASK | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| optimizer_updates | 0 | 0 | NEW_THIS_TASK | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| SAM_proposals | 0 | 0 | NEW_THIS_TASK | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| map_construction | 0 | 0 | NEW_THIS_TASK | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| main_scene_rows | 312 | 312 | LOGICAL_CONFIG_ROWS | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| main_pools | 24 | 24 | LOGICAL_ORDERED_POOLS | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| rank_off_diagonal_assigns | 100 | 104 | DIAGNOSTIC | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| bootstrap_draw_pairs | 8000 | 8000 | MATCHES_ONLY_EXPOSED_EXPLORATORY | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| confirmation_families_evaluated | 0 | — | NOT_RUN_BY_CONTRACT | NOT_RUN_BY_CONTRACT |
| phi_calls | 5265 | — | NEW_FROZEN_FEATURE_REPOOLING | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| ordinary_FC_region_readouts | 1941 | — | FROM_CACHED_GRIDS_NO_RGB_ENCODING | TECHNICAL_COST_OF_EXPOSED_ANALYSIS |
| confirmation_APall | — | — | NOT_RUN_BY_CONTRACT | NOT_RUN_BY_CONTRACT |

计时是实际缓存分析；各阶段存在嵌套/重叠，不能相加为冷启动端到端耗时或30FPS。本轮3324次冻结head包括3312次DEV干预和12次TRAIN工程检查；训练样例仅2个，不是重测TRAIN。

## 结构与对应

同step1000两seed的FG1/FG2记录、各自selected结果和5次shuffle均已保留。新原样组与继承embedding逐位一致：{'FG2_SURFACE': {'all_labels_equal': True, 'embedding_max_abs': 0.0, 'full200_order_equal': True, 'historical_receipt': 'cdab0d89d191c6ede818fb5efbc52690fe940786563cbc135479f5efa963001e'}, 'S29_FG2_SURFACE': {'all_labels_equal': True, 'embedding_max_abs': 0.0, 'full200_order_equal': True, 'historical_receipt': '2011d9573933d352e46aa8f536f3362146de480b9e8a3664820dd2d5c36ef5fa'}}。对应打乱的下降仅支持当前模型依赖，不能推断随机分组重新训练必然失败。

## 确认优先级

[
  {
    "lfo_positive_fraction": 0.71875,
    "mean_primary_delta_apall": -0.0004306322247506038,
    "missing": [],
    "module": "S",
    "primary": [
      {
        "cohort": "replica8",
        "delta": {
          "ap25": 0.011775133725517528,
          "ap50": -0.004152824593718907,
          "apall": 0.00039212360013982095,
          "macc": 0.006688108464337872,
          "miou": 0.003944950648836876
        },
        "pair": [
          "FG0_FC8",
          "FG2_SURFACE"
        ]
      },
      {
        "cohort": "replica8",
        "delta": {
          "ap25": -0.002215862283022396,
          "ap50": -0.012962899549627205,
          "apall": -0.006728993879773951,
          "macc": -0.0057945538093301385,
          "miou": -0.004593785712888376
        },
        "pair": [
          "FG0_FC8",
          "S29_FG2_SURFACE"
        ]
      },
      {
        "cohort": "scannet_cf18",
        "delta": {
          "ap25": 0.0023942820126028885,
          "ap50": 0.0034694153963100993,
          "apall": 0.0019285138093638327,
          "macc": 0.003044499397818534,
          "miou": 0.003919276699974411
        },
        "pair": [
          "FG0_FC8",
          "FG2_SURFACE"
        ]
      },
      {
        "cohort": "scannet_cf18",
        "delta": {
          "ap25": 0.006730386432017121,
          "ap50": 0.006778107896717456,
          "apall": 0.002685827571267882,
          "macc": 0.006099810412908857,
          "miou": 0.006638459186630474
        },
        "pair": [
          "FG0_FC8",
          "S29_FG2_SURFACE"
        ]
      }
    ],
    "primary_positive_cells": 3,
    "recognition_limited": false,
    "specific": [
      {
        "comparability": "MATCHED_STEP_INITIALIZATION_DRAW_SCHEDULE_FULL200",
        "delta": 0.07199074074074074,
        "metric": "DEV_A_NOT_MAP_AP",
        "seed": 17,
        "step": 1000
      },
      {
        "comparability": "MATCHED_STEP_INITIALIZATION_DRAW_SCHEDULE_FULL200",
        "delta": 0.08565867003367,
        "metric": "DEV_A_NOT_MAP_AP",
        "seed": 29,
        "step": 1000
      }
    ],
    "specific_positive_cells": 2,
    "status": "SHORTLIST_FOR_CONFIRMATION"
  },
  {
    "lfo_positive_fraction": 1.0,
    "mean_primary_delta_apall": 0.009575664188701316,
    "missing": [],
    "module": "E_GROUP",
    "primary": [
      {
        "cohort": "replica8",
        "delta": {
          "ap25": 0.002461233654124706,
          "ap50": 0.0017213562222243295,
          "apall": 0.013345520318658821,
          "macc": 0.008029042307467094,
          "miou": -0.0028931536041464367
        },
        "pair": [
          "E6",
          "E7"
        ]
      },
      {
        "cohort": "scannet_cf18",
        "delta": {
          "ap25": 0.00898407210451746,
          "ap50": 0.014812857068254581,
          "apall": 0.005805808058743811,
          "macc": 0.00992485411588917,
          "miou": 0.004073150778354018
        },
        "pair": [
          "E6",
          "E7"
        ]
      }
    ],
    "primary_positive_cells": 2,
    "recognition_limited": false,
    "specific": [
      {
        "cohort": "replica8",
        "comparability": "SAME_FIXED_SOURCE_FUSION",
        "control": "E8",
        "delta": 0.004752699782213696,
        "method": "E7",
        "metric": "apall"
      },
      {
        "cohort": "scannet_cf18",
        "comparability": "SAME_FIXED_SOURCE_FUSION",
        "control": "E8",
        "delta": 0.007545204631098049,
        "method": "E7",
        "metric": "apall"
      }
    ],
    "specific_positive_cells": 2,
    "status": "SHORTLIST_FOR_CONFIRMATION"
  }
]

该名单是 SHORTLIST_FOR_CONFIRMATION，不是已证明的CVPR核心贡献。仍需独立家族、真实提案、heldout40类与统一输入/硬件预算，以及同特征学习式融合对照。

完整四格、逐scene/family/class、错误路径、官方score-entry/多对一/唯一几何三种口径、显式rank交叉、LFO及8000draw-pairs详见相邻机器工件。

[工件索引](../../../../artifacts/static_ovmap/contribution_attribution_v1/artifact_index.json) · [贡献账本](../../../../artifacts/static_ovmap/contribution_attribution_v1/experiment_ledger.json)
