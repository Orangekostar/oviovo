# Object/semantic decoupling — Room0

科学状态：**COMPLETE_NO_NET_GAIN**。软件与 8 条件实验已完成；未建立跨场景有效性。

8 个新条件均为实际运行；历史对照标记为 REUSED_REFERENCE。数值为百分数，差值为百分点。

## A：实际性能

| Run | Method | OVI/SF | Active AP | Unique AP / AP50 / AP25 | Canonical AP75 | mIoU / mAcc | Unknown source / projected | Empty / small | Targets / views / charged new crops | Eval seconds |
|---|---|---:|---:|---|---:|---|---|---|---|---:|
| fp32_primary | OD_E1_SEMANTIC | 64/0 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.147 / 11.913 | 0/3 | 26/78/468 | 5.39 |
| fp32_primary | OD_E2_OBJECT | 64/1 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.938 / 42.361 | 0.143 / 11.909 | 0/4 | 0/0/0 | 5.92 |
| fp32_primary | OD_E3_OBJECT_SEMANTIC | 64/1 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.143 / 11.909 | 0/4 | 27/81/18 | 5.50 |
| fp32_primary | OD_E4_FINAL_RANK | 64/1 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.143 / 11.909 | 0/4 | 0/0/0 | 5.24 |
| fp32_repeat | OD_E1_SEMANTIC | 64/0 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.147 / 11.913 | 0/3 | 26/78/18 | 5.32 |
| fp32_repeat | OD_E2_OBJECT | 64/0 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.147 / 11.913 | 0/3 | 0/0/0 | 5.26 |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC | 64/0 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.147 / 11.913 | 0/3 | 26/78/0 | 5.37 |
| fp32_repeat | OD_E4_FINAL_RANK | 64/0 | 20.644 | 20.644 / 39.323 / 40.990 | 16.842 | 36.556 / 41.980 | 0.147 / 11.913 | 0/3 | 0/0/0 | 5.15 |

新裁剪数按同一目标首次使用的条件归属，E1/E3 共用输入不重复计费。原生 1024D SigLIP，E1/E3 使用同目标掩码六裁剪与 51 类文本；E2 不读取新语义。Canonical AP75 是独立类别无关诊断，非发布版 AP。

## B：受控效应

| Run | After − Before | Δ Unique AP | Δ AP50 | Δ mIoU |
|---|---|---:|---:|---:|
| fp32_primary | OD_E1_SEMANTIC − AT_O_AREA | +0.000 | +0.000 | +0.000 |
| fp32_primary | OD_E2_OBJECT − AT_O_AREA | +0.000 | +0.000 | +0.381 |
| fp32_primary | OD_E3_OBJECT_SEMANTIC − OD_E2_OBJECT | +0.000 | +0.000 | -0.381 |
| fp32_primary | OD_E4_FINAL_RANK − OD_E3_OBJECT_SEMANTIC | +0.000 | +0.000 | +0.000 |
| fp32_primary | OD_E1_SEMANTIC − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.036 |
| fp32_primary | OD_E1_SEMANTIC − LO_U00_LOCAL | +3.704 | +5.686 | -0.800 |
| fp32_primary | OD_E2_OBJECT − LO_U00_OVI_FILL | +0.000 | +0.000 | +0.345 |
| fp32_primary | OD_E2_OBJECT − LO_U00_LOCAL | +3.704 | +5.686 | -0.419 |
| fp32_primary | OD_E3_OBJECT_SEMANTIC − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.036 |
| fp32_primary | OD_E3_OBJECT_SEMANTIC − LO_U00_LOCAL | +3.704 | +5.686 | -0.800 |
| fp32_primary | OD_E4_FINAL_RANK − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.036 |
| fp32_primary | OD_E4_FINAL_RANK − LO_U00_LOCAL | +3.704 | +5.686 | -0.800 |
| fp32_repeat | OD_E1_SEMANTIC − AT_O_AREA | +0.000 | +0.000 | +0.000 |
| fp32_repeat | OD_E2_OBJECT − AT_O_AREA | +0.000 | +0.000 | +0.000 |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC − OD_E2_OBJECT | +0.000 | +0.000 | +0.000 |
| fp32_repeat | OD_E4_FINAL_RANK − OD_E3_OBJECT_SEMANTIC | +0.000 | +0.000 | +0.000 |
| fp32_repeat | OD_E1_SEMANTIC − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.029 |
| fp32_repeat | OD_E1_SEMANTIC − LO_U00_LOCAL | +3.164 | +3.255 | -0.647 |
| fp32_repeat | OD_E2_OBJECT − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.029 |
| fp32_repeat | OD_E2_OBJECT − LO_U00_LOCAL | +3.164 | +3.255 | -0.647 |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.029 |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC − LO_U00_LOCAL | +3.164 | +3.255 | -0.647 |
| fp32_repeat | OD_E4_FINAL_RANK − LO_U00_OVI_FILL | +0.000 | +0.000 | -0.029 |
| fp32_repeat | OD_E4_FINAL_RANK − LO_U00_LOCAL | +3.164 | +3.255 | -0.647 |

E1 仅改类；E2 改活动对象与所有权；E3 对 E2 仅改类；E4 对 E3 仅改源点面积排名。E1 包含重选视角影响。此为分阶段消融，不是完整析因设计，不报告语义与几何的纯交互量。

## C：对象与动作

| Run | Method | U00 added | Retained | Newly gained | Newly lost |
|---|---|---|---|---|---|
| fp32_primary | OD_E1_SEMANTIC | [5000, 5005, 10004, 15000, 17000, 24000, 33000] | [] | [] | [] |
| fp32_primary | OD_E2_OBJECT | [5000, 5005, 10004, 15000, 17000, 24000, 33000] | [] | [] | [] |
| fp32_primary | OD_E3_OBJECT_SEMANTIC | [5000, 5005, 10004, 15000, 17000, 24000, 33000] | [] | [] | [] |
| fp32_primary | OD_E4_FINAL_RANK | [5000, 5005, 10004, 15000, 17000, 24000, 33000] | [] | [] | [] |
| fp32_repeat | OD_E1_SEMANTIC | [5000, 5005, 10004, 15000, 17000, 33000] | [] | [] | [] |
| fp32_repeat | OD_E2_OBJECT | [5000, 5005, 10004, 15000, 17000, 33000] | [] | [] | [] |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC | [5000, 5005, 10004, 15000, 17000, 33000] | [] | [] | [] |
| fp32_repeat | OD_E4_FINAL_RANK | [5000, 5005, 10004, 15000, 17000, 33000] | [] | [] | [] |

对象表采用发布版 strict IoU > .5 的真实 first-match / duplicate-score-owner 事件。完整 .25/.5/.75 全对象记录、几何对应下改类对错、原始/最终 IoU 与范围见机器账本；不以最大 IoU 代替 TP。

## D：覆盖、失败与成本

| Run | Method | Accepted actions | Ignored events (all thresholds) | Invalid class | Run requests / batches / crops | Budget excluded |
|---|---|---:|---:|---:|---|---:|
| fp32_primary | OD_E1_SEMANTIC | 0 | 72 | 10 | 81/81/486 | 0 |
| fp32_primary | OD_E2_OBJECT | 1 | 72 | 10 | 81/81/486 | 0 |
| fp32_primary | OD_E3_OBJECT_SEMANTIC | 1 | 72 | 10 | 81/81/486 | 0 |
| fp32_primary | OD_E4_FINAL_RANK | 1 | 72 | 10 | 81/81/486 | 0 |
| fp32_repeat | OD_E1_SEMANTIC | 0 | 72 | 10 | 78/3/18 | 0 |
| fp32_repeat | OD_E2_OBJECT | 0 | 72 | 10 | 78/3/18 | 0 |
| fp32_repeat | OD_E3_OBJECT_SEMANTIC | 0 | 72 | 10 | 78/3/18 | 0 |
| fp32_repeat | OD_E4_FINAL_RANK | 0 | 72 | 10 | 78/3/18 | 0 |

成本列为该 run 在 E1/E3 联合去重后的实际成本，不能逐行相加。共 84 次前向、504 张裁剪；模型加载 1.65s，裁剪/预处理 6.21s，编码 14.31s；无新 2D/3D 分割推理。

主缓存仅接受一个 ADD_UNCOVERED_OBJECT，重复缓存没有接受动作。动作分数使用归档正实例像素；未知观测弃权，不代表完整可见像素。缺失覆盖以 owner/class0 保留坐标并计入 FN。区域整数混淆差逐项重建全局，包括 pred0。

全部原生 OVI 语义目标均未通过改类；新增 SF 目标的独立改类不代表修复原生错误。类别错误、缺少合格形状和匹配/排名问题分开记录；两份 Room0 缓存不是独立场景或盲验证。

确定性正/负例：唯一新增对象（主缓存 SF query20，canonical99）源点 91→78，评估点 40→33；E2 原类 wall-plug 的局部语义改善被 E3 改为 blanket 后抵消，实例因小于100点未进入 AP。E4 仅将该实例分数91改为78；没有99–101门槛穿越，几何未改善。该对象是全部接受改类集合，不是挑选的成功案例。

机器证据：[performance](../../../artifacts/static_ovmap/object_decoupling_v1/performance.json)、[effects](../../../artifacts/static_ovmap/object_decoupling_v1/controlled_effects.json)、[objects](../../../artifacts/static_ovmap/object_decoupling_v1/all_object_outcomes.json.gz)、[causes](../../../artifacts/static_ovmap/object_decoupling_v1/cause_ledger.json.gz)、[support/labels](../../../artifacts/static_ovmap/object_decoupling_v1/final_support_and_labels.json.gz)、[regions](../../../artifacts/static_ovmap/object_decoupling_v1/regional_deltas.json.gz)。
