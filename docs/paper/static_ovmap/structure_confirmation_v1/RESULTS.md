# FG2结构验证与确认

状态：COMPLETE。指标以百分比呈现，差值以百分点呈现；缺失结果为null。

本轮实验与工程交付完成，物理对应优势未获支持。固定1000步GT clean Top1的两seed均值为S_LL 38.71%、G_LL 39.43%、V_LL 37.99%、MV_ATTN 41.76%。S−G为−0.7168个百分点，95%家族区间[−1.7732,0.2137]；真实提案selected APall差+0.0781，区间[−0.0005,0.2974]也包含0。G的8V物理配对残留仅约2.2%–2.6%，保留几何metadata；结果不能归咎于未有效打乱，也不能据此断言所有几何信息无用。

存在类别保持的取舍，但不能归因于正确对应。GT fixed1000上S与G的novel Top1均为53.125%，受控MV为0.781%；MV的base Top1为53.95%，高于S的34.42%。S对MV的novel优势伴随base下降；G同样保持novel，因此不能将这种差异称为物理分组独有优势，亦不代表超过原版CAMFusion。

两级权重和投影顺序只有限定证据。CONF GT matched1000的inner/outer平均效应分别为+0.5376/+2.3297个百分点，interaction为+0.3584，LL−UU为+2.8674；这些是点效应，未额外搜索区间。S_PRE clean Top1为39.07%，高于S的38.71%。S相对普通表面均值的balanced Top1差+9.7120、区间[3.2617,13.8630]，但普通Top1与真实APall区间含0。不能声称两级都已证明必要或pool→phi是关键性能来源。

真实输入支持是主要外推限制：279个固定GT原件包含215base/64novel，9个无支持原件保留abstention；真实337提案中327有多视图支持，但IoU>.5仅16个唯一GT、8家族，其中novel仅3个唯一GT、3家族，为LIMITED_REAL_OR_NOVEL_SUPPORT。GT派生诊断不能替代真实支持，存在支持域差；新CONF是提供重建表面上的稀疏提案评价，旧26场景是已曝光完整地图回归。CONF只证明未参与当前开发，不证明预训练或全部历史独立性。

效率与部署结论保持克制：S小头216194参数（活跃199553），受控MV为3818240；8V驻留缓存总耗时S约69–70ms、MV约64–75ms，峰值allocated约1.52/1.53GiB，没有稳定系统级速度或显存优势证据，不能写30FPS。旧地图各cohort/AP与mIoU有取舍，不支持整体优于OVI-MAP；EXTERNAL_FULL_REPRODUCTION=NOT_RUN，部署维持N0，本轮完成不等于CVPR证据已齐。

## 核心对照

GT固定1000步，真实提案使用各方法预锁定DEV选择步；均为8视图上限。学习方法先分别评分两seed再平均，静态方法只测一次；不拼接seed，也不平均scene AP。

| 方法 | REAL选步 | GT Top1 (%) | GT balanced (%) | GT base (%) | GT novel (%) | REAL APall (%) | REAL AP50 (%) | REAL AP25 (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_LL | 1000 | 38.71 | 47.20 | 34.42 | 53.12 | 0.73 | 2.93 | 7.34 |
| V_LL | 500 | 37.99 | 47.04 | 33.95 | 51.56 | 0.74 | 2.95 | 7.01 |
| G_LL | 1500 | 39.43 | 47.63 | 35.35 | 53.12 | 0.65 | 2.79 | 7.32 |
| S_UL | 1500 | 37.99 | 45.71 | 34.19 | 50.78 | 0.67 | 2.86 | 8.54 |
| S_LU | 250 | 36.20 | 45.53 | 32.56 | 48.44 | 0.58 | 2.74 | 6.75 |
| S_UU | 1000 | 35.84 | 44.90 | 32.09 | 48.44 | 0.58 | 2.74 | 6.75 |
| S_PRE | 1500 | 39.07 | 47.61 | 34.42 | 54.69 | 0.71 | 2.91 | 7.13 |
| MV_ATTN | 2000 | 41.76 | 26.27 | 53.95 | 0.78 | 0.50 | 1.13 | 2.15 |
| B_FC | 静态 | 35.48 | 44.74 | 31.63 | 48.44 | 0.58 | 2.74 | 6.75 |
| B_SURFACE_MEAN | 静态 | 36.56 | 37.49 | 32.09 | 51.56 | 0.57 | 2.72 | 7.18 |
| B_PFE_ADAPTED | 静态 | 37.63 | 46.12 | 33.95 | 50.00 | 0.57 | 2.73 | 6.73 |

## DEV：匹配步、选择步与终点

| 方法 | seed | 口径 | step | A (%) | M (%) | C (%) |
| --- | --- | --- | --- | --- | --- | --- |
| G_LL | 17 | matched1000 | 1000 | 42.62 | 58.40 | 42.46 |
| G_LL | 17 | selected | 1500 | 44.39 | 58.83 | 45.23 |
| G_LL | 17 | terminal2000 | 2000 | 44.24 | 59.37 | 45.86 |
| G_LL | 29 | matched1000 | 1000 | 44.95 | 57.26 | 49.14 |
| G_LL | 29 | selected | 1500 | 43.86 | 57.53 | 46.36 |
| G_LL | 29 | terminal2000 | 2000 | 43.29 | 57.03 | 46.36 |
| MV_ATTN | 17 | matched1000 | 1000 | 63.56 | 54.29 | 63.46 |
| MV_ATTN | 17 | selected | 2000 | 65.26 | 53.39 | 66.86 |
| MV_ATTN | 17 | terminal2000 | 2000 | 65.26 | 53.39 | 66.86 |
| MV_ATTN | 29 | matched1000 | 1000 | 71.02 | 58.86 | 74.07 |
| MV_ATTN | 29 | selected | 2000 | 70.80 | 52.86 | 72.19 |
| MV_ATTN | 29 | terminal2000 | 2000 | 70.80 | 52.86 | 72.19 |
| S_LL | 17 | matched1000 | 1000 | 46.07 | 58.87 | 48.01 |
| S_LL | 17 | selected | 1000 | 46.07 | 58.87 | 48.01 |
| S_LL | 17 | terminal2000 | 2000 | 45.32 | 59.05 | 47.39 |
| S_LL | 29 | matched1000 | 1000 | 46.17 | 59.68 | 46.36 |
| S_LL | 29 | selected | 1000 | 46.17 | 59.68 | 46.36 |
| S_LL | 29 | terminal2000 | 2000 | 46.89 | 59.32 | 48.52 |
| S_LU | 17 | matched1000 | 1000 | 37.29 | 50.61 | 42.06 |
| S_LU | 17 | selected | 250 | 37.75 | 50.66 | 42.06 |
| S_LU | 17 | terminal2000 | 2000 | 37.29 | 50.61 | 42.06 |
| S_LU | 29 | matched1000 | 1000 | 37.29 | 50.61 | 42.06 |
| S_LU | 29 | selected | 250 | 37.75 | 50.66 | 42.06 |
| S_LU | 29 | terminal2000 | 2000 | 37.29 | 50.61 | 42.06 |
| S_PRE | 17 | matched1000 | 1000 | 43.63 | 56.39 | 45.23 |
| S_PRE | 17 | selected | 1500 | 46.14 | 57.68 | 50.79 |
| S_PRE | 17 | terminal2000 | 2000 | 44.85 | 56.98 | 48.01 |
| S_PRE | 29 | matched1000 | 1000 | 44.58 | 58.17 | 45.23 |
| S_PRE | 29 | selected | 1500 | 44.01 | 58.17 | 42.96 |
| S_PRE | 29 | terminal2000 | 2000 | 44.93 | 58.02 | 45.74 |
| S_UL | 17 | matched1000 | 1000 | 37.47 | 50.43 | 40.18 |
| S_UL | 17 | selected | 1500 | 40.45 | 53.33 | 45.74 |
| S_UL | 17 | terminal2000 | 2000 | 40.37 | 53.88 | 45.74 |
| S_UL | 29 | matched1000 | 1000 | 39.36 | 52.98 | 42.96 |
| S_UL | 29 | selected | 1500 | 39.93 | 53.78 | 42.96 |
| S_UL | 29 | terminal2000 | 2000 | 40.01 | 53.48 | 42.96 |
| S_UU | 17 | matched1000 | 1000 | 37.29 | 50.61 | 42.06 |
| S_UU | 17 | selected | 1000 | 37.29 | 50.61 | 42.06 |
| S_UU | 17 | terminal2000 | 2000 | 37.29 | 50.61 | 42.06 |
| S_UU | 29 | matched1000 | 1000 | 37.29 | 50.61 | 42.06 |
| S_UU | 29 | selected | 1000 | 37.29 | 50.61 | 42.06 |
| S_UU | 29 | terminal2000 | 2000 | 37.29 | 50.61 | 42.06 |
| V_LL | 17 | matched1000 | 1000 | 38.87 | 57.44 | 40.81 |
| V_LL | 17 | selected | 500 | 42.54 | 58.43 | 46.36 |
| V_LL | 17 | terminal2000 | 2000 | 41.94 | 57.51 | 46.36 |
| V_LL | 29 | matched1000 | 1000 | 37.61 | 55.18 | 37.53 |
| V_LL | 29 | selected | 500 | 39.06 | 54.38 | 40.69 |
| V_LL | 29 | terminal2000 | 2000 | 36.20 | 54.51 | 37.53 |

DEV包含69个base原件、4个家族、四条件，竞争词表为200类。仅seed17按A/M/C及最早步选择；seed29沿用同方法seed17步数。

开发期间在G首个DEV前修正了base过滤后的排列键索引。S/V的本任务既有权重与DEV分数未改，模型/优化器/RNG/采样内容等价证明及原始记录保留；G从安全断点重放，实际重放另计。详见review/dev_key_repair_contract.json与workload记录。

确认阶段修复了控制环境抢先加载CUDA库导致的SAM导入失败，以及地图payload的禁用元数据字段。两次SAM失败均在import torch、进入main前终止，新增编码为0；失败日志与身份保留。修复仅隔离SAM子进程链接库并移除payload中的GT_used声明，未改变SAM/FC数值、已锁模型或地图标签计算。训练续跑核验原16臂与80检查点后直接复用原汇总，不覆盖实际训练消耗。

真实DEV上S/G的2760组条件记录具有相同对象键、视图、group数和有效观测数量。G的实际同槽保留率如下，未强制排列必须改变每个token：

| seed | 有效token | 仍在原槽token | 同槽保留率 (%) |
| --- | --- | --- | --- |
| 17 | 7900644 | 293474 | 3.71 |
| 29 | 7891674 | 293629 | 3.72 |

## 训练与投影消耗

| 方法 | seed | updates | 名义head forward | phi forward | phi向量 | 完成臂秒(含DEV) | 参数 | 可训练参数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_LL | 17 | 2000 | 64000 | 64000 | 3172744 | 1089.74 | 216194 | 199553 |
| V_LL | 17 | 2000 | 64000 | 64000 | 284363 | 942.75 | 216194 | 199553 |
| G_LL | 17 | 2000 | 64000 | 64000 | 3172744 | 987.59 | 216194 | 199553 |
| S_UL | 17 | 2000 | 64000 | 64000 | 3172744 | 670.17 | 216194 | 182912 |
| S_LU | 17 | 2000 | 64000 | 64000 | 3172744 | 964.90 | 216194 | 199553 |
| S_UU | 17 | 2000 | 64000 | 64000 | 3172744 | 681.48 | 216194 | 182912 |
| S_PRE | 17 | 2000 | 64000 | 64000 | 7900644 | 873.12 | 216194 | 199553 |
| MV_ATTN | 17 | 2000 | 64000 | 0 | 0 | 1603.75 | 3818240 | 3818240 |
| S_LL | 29 | 2000 | 64000 | 64000 | 3174096 | 953.14 | 216194 | 199553 |
| V_LL | 29 | 2000 | 64000 | 64000 | 284017 | 945.20 | 216194 | 199553 |
| G_LL | 29 | 2000 | 64000 | 64000 | 3174096 | 920.72 | 216194 | 199553 |
| S_UL | 29 | 2000 | 64000 | 64000 | 3174096 | 672.54 | 216194 | 182912 |
| S_LU | 29 | 2000 | 64000 | 64000 | 3174096 | 968.43 | 216194 | 199553 |
| S_UU | 29 | 2000 | 64000 | 64000 | 3174096 | 679.30 | 216194 | 182912 |
| S_PRE | 29 | 2000 | 64000 | 64000 | 7891674 | 876.32 | 216194 | 199553 |
| MV_ATTN | 29 | 2000 | 64000 | 0 | 0 | 1610.82 | 3818240 | 3818240 |

phi次数由实际draw与已验缓存valid布局精确推导，单列为源码推导证据；未测得逐kernel训练耗时，填null。每臂用时含训练、DEV、缓存读取和断点写入；中断回退前的额外耗时未单独计时，不能把各臂用时之和当作全部尝试的总训练时间。物理head和重放以总计数器为准。

G分组中同一规范物理位置的跨视图配对残留，直接从已保存排列计算；不含模型seed/step重复，不新增head：

| 输入域 | 条件序号 | views上限 | 跨视图对 | 保留物理对 | 残留率 (%) | singleton组 |
| --- | --- | --- | --- | --- | --- | --- |
| REAL | 0 | 2 | 10244 | 223 | 2.18 | 3134 |
| REAL | 0 | 4 | 52968 | 1363 | 2.57 | 1103 |
| REAL | 0 | 8 | 183145 | 4756 | 2.60 | 798 |
| GT | 0 | 2 | 5149 | 80 | 1.55 | 6880 |
| GT | 0 | 4 | 24154 | 464 | 1.92 | 4817 |
| GT | 0 | 8 | 76726 | 1683 | 2.19 | 3285 |
| GT | 1 | 2 | 5477 | 105 | 1.92 | 6935 |
| GT | 1 | 4 | 25479 | 520 | 2.04 | 4695 |
| GT | 1 | 8 | 79955 | 1809 | 2.26 | 3162 |
| GT | 2 | 2 | 5129 | 78 | 1.52 | 6920 |
| GT | 2 | 4 | 24302 | 482 | 1.98 | 4839 |
| GT | 2 | 8 | 77651 | 1705 | 2.20 | 3256 |
| GT | 3 | 2 | 5284 | 102 | 1.93 | 6846 |
| GT | 3 | 4 | 24889 | 511 | 2.05 | 4756 |
| GT | 3 | 8 | 78752 | 1816 | 2.31 | 3190 |

singleton不进入配对分母；同槽token保留率与跨视图物理配对残留率是不同统计量。

## 新家族真实提案

| 方法 | seed | 口径 | step | APall (%) | AP50 (%) | AP25 (%) |
| --- | --- | --- | --- | --- | --- | --- |
| G_LL | 17 | selected | 1500 | 0.72 | 2.92 | 7.60 |
| G_LL | 29 | selected | 1500 | 0.58 | 2.66 | 7.03 |
| MV_ATTN | 17 | selected | 2000 | 0.20 | 0.49 | 1.91 |
| MV_ATTN | 29 | selected | 2000 | 0.80 | 1.76 | 2.38 |
| S_LL | 17 | selected | 1000 | 0.72 | 2.92 | 7.01 |
| S_LL | 29 | selected | 1000 | 0.73 | 2.94 | 7.66 |
| S_LU | 17 | selected | 250 | 0.58 | 2.74 | 6.75 |
| S_LU | 29 | selected | 250 | 0.58 | 2.74 | 6.75 |
| S_PRE | 17 | selected | 1500 | 0.72 | 2.91 | 6.96 |
| S_PRE | 29 | selected | 1500 | 0.71 | 2.90 | 7.30 |
| S_UL | 17 | selected | 1500 | 0.76 | 2.98 | 8.31 |
| S_UL | 29 | selected | 1500 | 0.58 | 2.75 | 8.77 |
| S_UU | 17 | selected | 1000 | 0.58 | 2.74 | 6.75 |
| S_UU | 29 | selected | 1000 | 0.58 | 2.74 | 6.75 |
| V_LL | 17 | selected | 500 | 0.73 | 2.93 | 7.00 |
| V_LL | 29 | selected | 500 | 0.75 | 2.98 | 7.02 |
| G_LL | 17 | matched1000 | 1000 | 0.72 | 2.91 | 7.59 |
| G_LL | 29 | matched1000 | 1000 | 0.58 | 2.66 | 7.07 |
| MV_ATTN | 17 | matched1000 | 1000 | 0.20 | 0.49 | 1.03 |
| MV_ATTN | 29 | matched1000 | 1000 | 0.87 | 1.88 | 2.53 |
| S_LL | 17 | matched1000 | 1000 | 0.72 | 2.92 | 7.01 |
| S_LL | 29 | matched1000 | 1000 | 0.73 | 2.94 | 7.66 |
| S_LU | 17 | matched1000 | 1000 | 0.58 | 2.74 | 6.75 |
| S_LU | 29 | matched1000 | 1000 | 0.58 | 2.74 | 6.75 |
| S_PRE | 17 | matched1000 | 1000 | 0.72 | 2.91 | 7.29 |
| S_PRE | 29 | matched1000 | 1000 | 0.71 | 2.90 | 7.30 |
| S_UL | 17 | matched1000 | 1000 | 0.76 | 2.98 | 7.07 |
| S_UL | 29 | matched1000 | 1000 | 0.58 | 2.75 | 7.51 |
| S_UU | 17 | matched1000 | 1000 | 0.58 | 2.74 | 6.75 |
| S_UU | 29 | matched1000 | 1000 | 0.58 | 2.74 | 6.75 |
| V_LL | 17 | matched1000 | 1000 | 0.71 | 2.86 | 6.93 |
| V_LL | 29 | matched1000 | 1000 | 0.71 | 2.86 | 6.89 |
| B_FC | None | selected | None | 0.58 | 2.74 | 6.75 |
| B_SURFACE_MEAN | None | selected | None | 0.57 | 2.72 | 7.18 |
| B_PFE_ADAPTED | None | selected | None | 0.57 | 2.73 | 6.73 |

固定重建表面上的稀疏SAM提案，完整GT分母、原官方matcher与池化；所有方法共享支持及constant1排序。该表不代表新场景在线建图。

共享几何支持：

| IoU > | 匹配提案 | 唯一GT | family | heldout唯一GT | heldout family |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 55 | 43 | 12 | 14 | 8 |
| 0.5 | 18 | 16 | 8 | 3 | 3 |

预登记覆盖参考：LIMITED_REAL_OR_NOVEL_SUPPORT。40个唯一GT/8家族、20个heldout唯一GT/4家族只是资源覆盖参考，不是显著性或成功门槛。

## 独立GT支持诊断

| 方法 | seed | 口径 | step | views | Top1 | balanced | base | novel | H | 原件 | abstention条件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G_LL | 17 | selected | 1500 | 2 | 38.35 | 48.41 | 34.88 | 50.00 | 41.10 | 279 | 36 |
| G_LL | 17 | selected | 1500 | 4 | 37.28 | 46.39 | 32.09 | 54.69 | 40.45 | 279 | 36 |
| G_LL | 17 | selected | 1500 | 8 | 39.07 | 47.67 | 34.88 | 53.12 | 42.11 | 279 | 36 |
| G_LL | 29 | selected | 1500 | 2 | 37.99 | 49.11 | 34.88 | 48.44 | 40.56 | 279 | 36 |
| G_LL | 29 | selected | 1500 | 4 | 39.43 | 47.03 | 34.88 | 54.69 | 42.60 | 279 | 36 |
| G_LL | 29 | selected | 1500 | 8 | 39.07 | 46.25 | 34.88 | 53.12 | 42.11 | 279 | 36 |
| MV_ATTN | 17 | selected | 2000 | 2 | 41.58 | 23.72 | 53.95 | 0.00 | 0.00 | 279 | 36 |
| MV_ATTN | 17 | selected | 2000 | 4 | 42.29 | 26.21 | 54.42 | 1.56 | 3.04 | 279 | 36 |
| MV_ATTN | 17 | selected | 2000 | 8 | 42.65 | 24.88 | 55.35 | 0.00 | 0.00 | 279 | 36 |
| MV_ATTN | 29 | selected | 2000 | 2 | 40.14 | 23.14 | 52.09 | 0.00 | 0.00 | 279 | 36 |
| MV_ATTN | 29 | selected | 2000 | 4 | 41.94 | 25.02 | 54.42 | 0.00 | 0.00 | 279 | 36 |
| MV_ATTN | 29 | selected | 2000 | 8 | 42.29 | 25.24 | 54.88 | 0.00 | 0.00 | 279 | 36 |
| S_LL | 17 | selected | 1000 | 2 | 38.35 | 48.86 | 34.88 | 50.00 | 41.10 | 279 | 36 |
| S_LL | 17 | selected | 1000 | 4 | 39.78 | 48.27 | 34.88 | 56.25 | 43.06 | 279 | 36 |
| S_LL | 17 | selected | 1000 | 8 | 39.07 | 47.22 | 34.88 | 53.12 | 42.11 | 279 | 36 |
| S_LL | 29 | selected | 1000 | 2 | 38.71 | 49.94 | 34.88 | 51.56 | 41.61 | 279 | 36 |
| S_LL | 29 | selected | 1000 | 4 | 37.99 | 47.98 | 33.02 | 54.69 | 41.18 | 279 | 36 |
| S_LL | 29 | selected | 1000 | 8 | 38.35 | 47.19 | 33.95 | 53.12 | 41.43 | 279 | 36 |
| S_LU | 17 | selected | 250 | 2 | 37.63 | 48.82 | 33.02 | 53.12 | 40.73 | 279 | 36 |
| S_LU | 17 | selected | 250 | 4 | 35.48 | 43.50 | 30.70 | 51.56 | 38.48 | 279 | 36 |
| S_LU | 17 | selected | 250 | 8 | 36.56 | 45.56 | 33.02 | 48.44 | 39.27 | 279 | 36 |
| S_LU | 29 | selected | 250 | 2 | 37.63 | 48.82 | 33.02 | 53.12 | 40.73 | 279 | 36 |
| S_LU | 29 | selected | 250 | 4 | 35.48 | 43.50 | 30.70 | 51.56 | 38.48 | 279 | 36 |
| S_LU | 29 | selected | 250 | 8 | 36.56 | 45.56 | 33.02 | 48.44 | 39.27 | 279 | 36 |
| S_PRE | 17 | selected | 1500 | 2 | 37.99 | 48.09 | 34.42 | 50.00 | 40.77 | 279 | 36 |
| S_PRE | 17 | selected | 1500 | 4 | 38.71 | 47.38 | 33.95 | 54.69 | 41.90 | 279 | 36 |
| S_PRE | 17 | selected | 1500 | 8 | 38.35 | 47.12 | 34.42 | 51.56 | 41.28 | 279 | 36 |
| S_PRE | 29 | selected | 1500 | 2 | 36.92 | 45.77 | 33.49 | 48.44 | 39.60 | 279 | 36 |
| S_PRE | 29 | selected | 1500 | 4 | 37.63 | 46.62 | 32.09 | 56.25 | 40.87 | 279 | 36 |
| S_PRE | 29 | selected | 1500 | 8 | 39.78 | 47.76 | 35.35 | 54.69 | 42.94 | 279 | 36 |
| S_UL | 17 | selected | 1500 | 2 | 38.71 | 49.31 | 34.88 | 51.56 | 41.61 | 279 | 36 |
| S_UL | 17 | selected | 1500 | 4 | 36.56 | 44.12 | 32.09 | 51.56 | 39.56 | 279 | 36 |
| S_UL | 17 | selected | 1500 | 8 | 37.99 | 45.65 | 34.42 | 50.00 | 40.77 | 279 | 36 |
| S_UL | 29 | selected | 1500 | 2 | 37.63 | 47.23 | 33.49 | 51.56 | 40.60 | 279 | 36 |
| S_UL | 29 | selected | 1500 | 4 | 37.99 | 43.89 | 33.49 | 53.12 | 41.08 | 279 | 36 |
| S_UL | 29 | selected | 1500 | 8 | 37.99 | 45.58 | 33.95 | 51.56 | 40.95 | 279 | 36 |
| S_UU | 17 | selected | 1000 | 2 | 37.63 | 48.82 | 33.02 | 53.12 | 40.73 | 279 | 36 |
| S_UU | 17 | selected | 1000 | 4 | 34.41 | 43.04 | 29.77 | 50.00 | 37.32 | 279 | 36 |
| S_UU | 17 | selected | 1000 | 8 | 35.84 | 44.90 | 32.09 | 48.44 | 38.61 | 279 | 36 |
| S_UU | 29 | selected | 1000 | 2 | 37.63 | 48.82 | 33.02 | 53.12 | 40.73 | 279 | 36 |
| S_UU | 29 | selected | 1000 | 4 | 34.41 | 43.04 | 29.77 | 50.00 | 37.32 | 279 | 36 |
| S_UU | 29 | selected | 1000 | 8 | 35.84 | 44.90 | 32.09 | 48.44 | 38.61 | 279 | 36 |
| V_LL | 17 | selected | 500 | 2 | 36.20 | 48.92 | 32.56 | 48.44 | 38.94 | 279 | 36 |
| V_LL | 17 | selected | 500 | 4 | 35.84 | 44.99 | 30.70 | 53.12 | 38.91 | 279 | 36 |
| V_LL | 17 | selected | 500 | 8 | 37.63 | 45.68 | 33.49 | 51.56 | 40.60 | 279 | 36 |
| V_LL | 29 | selected | 500 | 2 | 38.35 | 48.43 | 33.95 | 53.12 | 41.43 | 279 | 36 |
| V_LL | 29 | selected | 500 | 4 | 36.56 | 45.30 | 31.63 | 53.12 | 39.65 | 279 | 36 |
| V_LL | 29 | selected | 500 | 8 | 37.63 | 47.09 | 33.49 | 51.56 | 40.60 | 279 | 36 |
| G_LL | 17 | matched1000 | 1000 | 8 | 39.78 | 47.98 | 35.35 | 54.69 | 42.94 | 279 | 36 |
| G_LL | 29 | matched1000 | 1000 | 8 | 39.07 | 47.29 | 35.35 | 51.56 | 41.94 | 279 | 36 |
| MV_ATTN | 17 | matched1000 | 1000 | 8 | 41.58 | 26.66 | 53.49 | 1.56 | 3.04 | 279 | 36 |
| MV_ATTN | 29 | matched1000 | 1000 | 8 | 41.94 | 25.87 | 54.42 | 0.00 | 0.00 | 279 | 36 |
| S_LL | 17 | matched1000 | 1000 | 8 | 39.07 | 47.22 | 34.88 | 53.12 | 42.11 | 279 | 36 |
| S_LL | 29 | matched1000 | 1000 | 8 | 38.35 | 47.19 | 33.95 | 53.12 | 41.43 | 279 | 36 |
| S_LU | 17 | matched1000 | 1000 | 8 | 36.20 | 45.53 | 32.56 | 48.44 | 38.94 | 279 | 36 |
| S_LU | 29 | matched1000 | 1000 | 8 | 36.20 | 45.53 | 32.56 | 48.44 | 38.94 | 279 | 36 |
| S_PRE | 17 | matched1000 | 1000 | 8 | 39.07 | 47.63 | 34.42 | 54.69 | 42.25 | 279 | 36 |
| S_PRE | 29 | matched1000 | 1000 | 8 | 39.07 | 47.58 | 34.42 | 54.69 | 42.25 | 279 | 36 |
| S_UL | 17 | matched1000 | 1000 | 8 | 37.63 | 45.63 | 33.95 | 50.00 | 40.44 | 279 | 36 |
| S_UL | 29 | matched1000 | 1000 | 8 | 38.35 | 45.79 | 34.42 | 51.56 | 41.28 | 279 | 36 |
| S_UU | 17 | matched1000 | 1000 | 8 | 35.84 | 44.90 | 32.09 | 48.44 | 38.61 | 279 | 36 |
| S_UU | 29 | matched1000 | 1000 | 8 | 35.84 | 44.90 | 32.09 | 48.44 | 38.61 | 279 | 36 |
| V_LL | 17 | matched1000 | 1000 | 8 | 37.63 | 46.91 | 34.42 | 48.44 | 40.24 | 279 | 36 |
| V_LL | 29 | matched1000 | 1000 | 8 | 38.35 | 47.18 | 33.49 | 54.69 | 41.54 | 279 | 36 |
| B_FC | None | selected | None | 2 | 37.28 | 48.73 | 33.02 | 51.56 | 40.26 | 279 | 36 |
| B_FC | None | selected | None | 4 | 34.05 | 42.41 | 29.30 | 50.00 | 36.95 | 279 | 36 |
| B_FC | None | selected | None | 8 | 35.48 | 44.74 | 31.63 | 48.44 | 38.27 | 279 | 36 |
| B_SURFACE_MEAN | None | selected | None | 2 | 37.63 | 39.92 | 34.42 | 48.44 | 40.24 | 279 | 36 |
| B_SURFACE_MEAN | None | selected | None | 4 | 37.99 | 37.59 | 33.49 | 53.12 | 41.08 | 279 | 36 |
| B_SURFACE_MEAN | None | selected | None | 8 | 36.56 | 37.49 | 32.09 | 51.56 | 39.56 | 279 | 36 |
| B_PFE_ADAPTED | None | selected | None | 2 | 37.28 | 46.79 | 33.49 | 50.00 | 40.11 | 279 | 36 |
| B_PFE_ADAPTED | None | selected | None | 4 | 37.28 | 45.47 | 31.63 | 56.25 | 40.49 | 279 | 36 |
| B_PFE_ADAPTED | None | selected | None | 8 | 37.63 | 46.12 | 33.95 | 50.00 | 40.44 | 279 | 36 |

未观察原件保留主分母；heldout CE160为null。GT派生支持与真实SAM支持分别评价。

固定原件覆盖：

| 分组 | 原件 | family | 正样本类别 | 四条件记录 |
| --- | --- | --- | --- | --- |
| base | 215 | 12 | 59 | 860 |
| novel | 64 | 12 | 20 | 256 |
| all | 279 | 12 | 79 | 1116 |

全200竞争的损失及定义分母：

| 方法 | seed | 口径 | 分组 | NLL200 | defined_n | CE160 | defined_n | abstention条件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| G_LL | 17 | selected | base | 4.28 | 824 | 4.05 | 824 | 36 |
| G_LL | 17 | selected | novel | 4.21 | 256 | null | 0 | 0 |
| G_LL | 17 | selected | all | 4.26 | 1080 | 4.05 | 824 | 36 |
| G_LL | 29 | selected | base | 4.28 | 824 | 4.06 | 824 | 36 |
| G_LL | 29 | selected | novel | 4.21 | 256 | null | 0 | 0 |
| G_LL | 29 | selected | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| MV_ATTN | 17 | selected | base | 2.11 | 824 | 1.98 | 824 | 36 |
| MV_ATTN | 17 | selected | novel | 4.63 | 256 | null | 0 | 0 |
| MV_ATTN | 17 | selected | all | 2.71 | 1080 | 1.98 | 824 | 36 |
| MV_ATTN | 29 | selected | base | 2.18 | 824 | 2.05 | 824 | 36 |
| MV_ATTN | 29 | selected | novel | 4.70 | 256 | null | 0 | 0 |
| MV_ATTN | 29 | selected | all | 2.78 | 1080 | 2.05 | 824 | 36 |
| S_LL | 17 | selected | base | 4.29 | 824 | 4.06 | 824 | 36 |
| S_LL | 17 | selected | novel | 4.22 | 256 | null | 0 | 0 |
| S_LL | 17 | selected | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| S_LL | 29 | selected | base | 4.29 | 824 | 4.07 | 824 | 36 |
| S_LL | 29 | selected | novel | 4.21 | 256 | null | 0 | 0 |
| S_LL | 29 | selected | all | 4.27 | 1080 | 4.07 | 824 | 36 |
| S_LU | 17 | selected | base | 4.39 | 824 | 4.16 | 824 | 36 |
| S_LU | 17 | selected | novel | 4.33 | 256 | null | 0 | 0 |
| S_LU | 17 | selected | all | 4.37 | 1080 | 4.16 | 824 | 36 |
| S_LU | 29 | selected | base | 4.39 | 824 | 4.16 | 824 | 36 |
| S_LU | 29 | selected | novel | 4.33 | 256 | null | 0 | 0 |
| S_LU | 29 | selected | all | 4.37 | 1080 | 4.16 | 824 | 36 |
| S_PRE | 17 | selected | base | 4.29 | 824 | 4.06 | 824 | 36 |
| S_PRE | 17 | selected | novel | 4.22 | 256 | null | 0 | 0 |
| S_PRE | 17 | selected | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| S_PRE | 29 | selected | base | 4.29 | 824 | 4.07 | 824 | 36 |
| S_PRE | 29 | selected | novel | 4.22 | 256 | null | 0 | 0 |
| S_PRE | 29 | selected | all | 4.28 | 1080 | 4.07 | 824 | 36 |
| S_UL | 17 | selected | base | 4.31 | 824 | 4.09 | 824 | 36 |
| S_UL | 17 | selected | novel | 4.26 | 256 | null | 0 | 0 |
| S_UL | 17 | selected | all | 4.30 | 1080 | 4.09 | 824 | 36 |
| S_UL | 29 | selected | base | 4.31 | 824 | 4.09 | 824 | 36 |
| S_UL | 29 | selected | novel | 4.26 | 256 | null | 0 | 0 |
| S_UL | 29 | selected | all | 4.30 | 1080 | 4.09 | 824 | 36 |
| S_UU | 17 | selected | base | 4.39 | 824 | 4.17 | 824 | 36 |
| S_UU | 17 | selected | novel | 4.33 | 256 | null | 0 | 0 |
| S_UU | 17 | selected | all | 4.37 | 1080 | 4.17 | 824 | 36 |
| S_UU | 29 | selected | base | 4.39 | 824 | 4.17 | 824 | 36 |
| S_UU | 29 | selected | novel | 4.33 | 256 | null | 0 | 0 |
| S_UU | 29 | selected | all | 4.37 | 1080 | 4.17 | 824 | 36 |
| V_LL | 17 | selected | base | 4.26 | 824 | 4.04 | 824 | 36 |
| V_LL | 17 | selected | novel | 4.21 | 256 | null | 0 | 0 |
| V_LL | 17 | selected | all | 4.25 | 1080 | 4.04 | 824 | 36 |
| V_LL | 29 | selected | base | 4.27 | 824 | 4.04 | 824 | 36 |
| V_LL | 29 | selected | novel | 4.21 | 256 | null | 0 | 0 |
| V_LL | 29 | selected | all | 4.26 | 1080 | 4.04 | 824 | 36 |
| G_LL | 17 | matched1000 | base | 4.28 | 824 | 4.06 | 824 | 36 |
| G_LL | 17 | matched1000 | novel | 4.22 | 256 | null | 0 | 0 |
| G_LL | 17 | matched1000 | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| G_LL | 29 | matched1000 | base | 4.28 | 824 | 4.06 | 824 | 36 |
| G_LL | 29 | matched1000 | novel | 4.21 | 256 | null | 0 | 0 |
| G_LL | 29 | matched1000 | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| MV_ATTN | 17 | matched1000 | base | 2.14 | 824 | 1.99 | 824 | 36 |
| MV_ATTN | 17 | matched1000 | novel | 4.22 | 256 | null | 0 | 0 |
| MV_ATTN | 17 | matched1000 | all | 2.64 | 1080 | 1.99 | 824 | 36 |
| MV_ATTN | 29 | matched1000 | base | 2.12 | 824 | 1.98 | 824 | 36 |
| MV_ATTN | 29 | matched1000 | novel | 4.27 | 256 | null | 0 | 0 |
| MV_ATTN | 29 | matched1000 | all | 2.63 | 1080 | 1.98 | 824 | 36 |
| S_LL | 17 | matched1000 | base | 4.29 | 824 | 4.06 | 824 | 36 |
| S_LL | 17 | matched1000 | novel | 4.22 | 256 | null | 0 | 0 |
| S_LL | 17 | matched1000 | all | 4.27 | 1080 | 4.06 | 824 | 36 |
| S_LL | 29 | matched1000 | base | 4.29 | 824 | 4.07 | 824 | 36 |
| S_LL | 29 | matched1000 | novel | 4.21 | 256 | null | 0 | 0 |
| S_LL | 29 | matched1000 | all | 4.27 | 1080 | 4.07 | 824 | 36 |
| S_LU | 17 | matched1000 | base | 4.39 | 824 | 4.16 | 824 | 36 |
| S_LU | 17 | matched1000 | novel | 4.33 | 256 | null | 0 | 0 |
| S_LU | 17 | matched1000 | all | 4.37 | 1080 | 4.16 | 824 | 36 |
| S_LU | 29 | matched1000 | base | 4.39 | 824 | 4.16 | 824 | 36 |
| S_LU | 29 | matched1000 | novel | 4.33 | 256 | null | 0 | 0 |
| S_LU | 29 | matched1000 | all | 4.37 | 1080 | 4.16 | 824 | 36 |
| S_PRE | 17 | matched1000 | base | 4.29 | 824 | 4.07 | 824 | 36 |
| S_PRE | 17 | matched1000 | novel | 4.22 | 256 | null | 0 | 0 |
| S_PRE | 17 | matched1000 | all | 4.28 | 1080 | 4.07 | 824 | 36 |
| S_PRE | 29 | matched1000 | base | 4.30 | 824 | 4.07 | 824 | 36 |
| S_PRE | 29 | matched1000 | novel | 4.22 | 256 | null | 0 | 0 |
| S_PRE | 29 | matched1000 | all | 4.28 | 1080 | 4.07 | 824 | 36 |
| S_UL | 17 | matched1000 | base | 4.32 | 824 | 4.09 | 824 | 36 |
| S_UL | 17 | matched1000 | novel | 4.26 | 256 | null | 0 | 0 |
| S_UL | 17 | matched1000 | all | 4.30 | 1080 | 4.09 | 824 | 36 |
| S_UL | 29 | matched1000 | base | 4.32 | 824 | 4.09 | 824 | 36 |
| S_UL | 29 | matched1000 | novel | 4.26 | 256 | null | 0 | 0 |
| S_UL | 29 | matched1000 | all | 4.30 | 1080 | 4.09 | 824 | 36 |
| S_UU | 17 | matched1000 | base | 4.39 | 824 | 4.17 | 824 | 36 |
| S_UU | 17 | matched1000 | novel | 4.33 | 256 | null | 0 | 0 |
| S_UU | 17 | matched1000 | all | 4.37 | 1080 | 4.17 | 824 | 36 |
| S_UU | 29 | matched1000 | base | 4.39 | 824 | 4.17 | 824 | 36 |
| S_UU | 29 | matched1000 | novel | 4.33 | 256 | null | 0 | 0 |
| S_UU | 29 | matched1000 | all | 4.37 | 1080 | 4.17 | 824 | 36 |
| V_LL | 17 | matched1000 | base | 4.26 | 824 | 4.04 | 824 | 36 |
| V_LL | 17 | matched1000 | novel | 4.21 | 256 | null | 0 | 0 |
| V_LL | 17 | matched1000 | all | 4.25 | 1080 | 4.04 | 824 | 36 |
| V_LL | 29 | matched1000 | base | 4.26 | 824 | 4.03 | 824 | 36 |
| V_LL | 29 | matched1000 | novel | 4.20 | 256 | null | 0 | 0 |
| V_LL | 29 | matched1000 | all | 4.25 | 1080 | 4.03 | 824 | 36 |
| B_FC | None | selected | base | 4.39 | 824 | 4.16 | 824 | 36 |
| B_FC | None | selected | novel | 4.33 | 256 | null | 0 | 0 |
| B_FC | None | selected | all | 4.37 | 1080 | 4.16 | 824 | 36 |
| B_SURFACE_MEAN | None | selected | base | 4.51 | 824 | 4.29 | 824 | 36 |
| B_SURFACE_MEAN | None | selected | novel | 4.52 | 256 | null | 0 | 0 |
| B_SURFACE_MEAN | None | selected | all | 4.51 | 1080 | 4.29 | 824 | 36 |
| B_PFE_ADAPTED | None | selected | base | 4.33 | 824 | 4.11 | 824 | 36 |
| B_PFE_ADAPTED | None | selected | novel | 4.24 | 256 | null | 0 | 0 |
| B_PFE_ADAPTED | None | selected | all | 4.31 | 1080 | 4.11 | 824 | 36 |

每家族GT诊断clean Top1（matched1000、两seed点估计均值，原件主分母）及真实提案覆盖：

| family | GT原件 | novel原件 | S_LL (%) | G_LL (%) | V_LL (%) | MV_ATTN (%) | 真实IoU>.5唯一GT | 其中novel |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scene0410 | 22 | 9 | 45.45 | 45.45 | 47.73 | 38.64 | 2 | 1 |
| scene0447 | 26 | 8 | 30.77 | 32.69 | 34.62 | 36.54 | 2 | 1 |
| scene0456 | 22 | 5 | 36.36 | 34.09 | 36.36 | 34.09 | 0 | 0 |
| scene0460 | 20 | 4 | 35.00 | 32.50 | 30.00 | 45.00 | 0 | 0 |
| scene0199 | 16 | 9 | 50.00 | 50.00 | 43.75 | 31.25 | 0 | 0 |
| scene0109 | 19 | 4 | 42.11 | 47.37 | 44.74 | 47.37 | 3 | 0 |
| scene0108 | 34 | 3 | 42.65 | 44.12 | 35.29 | 50.00 | 2 | 0 |
| scene0158 | 33 | 5 | 30.30 | 30.30 | 31.82 | 34.85 | 3 | 0 |
| scene0182 | 16 | 2 | 40.62 | 43.75 | 43.75 | 62.50 | 0 | 0 |
| scene0501 | 38 | 11 | 36.84 | 36.84 | 35.53 | 27.63 | 1 | 0 |
| scene0444 | 11 | 2 | 18.18 | 18.18 | 18.18 | 59.09 | 2 | 1 |
| scene0628 | 22 | 2 | 54.55 | 56.82 | 54.55 | 56.82 | 1 | 0 |

每seed及各family的完整分类细项保留于[GT评分记录](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/confirmation/GT/score.json.gz)。家族数是统计聚类数，不将两seed作为双倍样本。

## 已曝光26场景完整地图回归

| cohort | 方法与读出 | APall | AP50 | AP25 | mIoU | mAcc |
| --- | --- | --- | --- | --- | --- | --- |
| replica8 | B_FC__F_ONLY_FIXED_ELIGIBLE | 10.61 | 24.89 | 38.07 | 28.33 | 36.97 |
| replica8 | B_FC__PARENT_D2_FIXED_ELIGIBLE | 11.73 | 26.57 | 41.47 | 31.77 | 39.30 |
| replica8 | B_PFE_ADAPTED__F_ONLY_FIXED_ELIGIBLE | 9.96 | 21.76 | 33.27 | 26.40 | 34.61 |
| replica8 | B_PFE_ADAPTED__PARENT_D2_FIXED_ELIGIBLE | 11.73 | 26.65 | 42.16 | 32.66 | 39.59 |
| replica8 | B_SURFACE_MEAN__F_ONLY_FIXED_ELIGIBLE | 10.44 | 21.87 | 31.46 | 23.18 | 32.41 |
| replica8 | B_SURFACE_MEAN__PARENT_D2_FIXED_ELIGIBLE | 11.08 | 25.17 | 39.20 | 30.54 | 37.28 |
| replica8 | G_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 11.77 | 25.27 | 37.10 | 27.76 | 37.69 |
| replica8 | G_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 10.86 | 24.47 | 40.60 | 31.08 | 38.46 |
| replica8 | G_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 11.42 | 24.74 | 36.63 | 27.29 | 37.37 |
| replica8 | G_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 11.79 | 26.13 | 41.22 | 31.54 | 39.07 |
| replica8 | MV_ATTN_seed17__F_ONLY_FIXED_ELIGIBLE | 3.15 | 7.39 | 9.99 | 8.21 | 14.82 |
| replica8 | MV_ATTN_seed17__PARENT_D2_FIXED_ELIGIBLE | 4.47 | 9.96 | 12.68 | 10.91 | 17.43 |
| replica8 | MV_ATTN_seed29__F_ONLY_FIXED_ELIGIBLE | 4.27 | 8.80 | 11.43 | 8.27 | 16.34 |
| replica8 | MV_ATTN_seed29__PARENT_D2_FIXED_ELIGIBLE | 4.67 | 10.32 | 13.40 | 11.07 | 17.79 |
| replica8 | S_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 11.77 | 25.42 | 36.61 | 27.44 | 37.07 |
| replica8 | S_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 11.77 | 26.16 | 42.65 | 32.17 | 39.97 |
| replica8 | S_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 11.29 | 24.54 | 36.24 | 26.93 | 36.68 |
| replica8 | S_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 10.99 | 25.14 | 41.11 | 31.29 | 38.71 |
| replica8 | V_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 11.03 | 23.38 | 34.22 | 25.34 | 34.46 |
| replica8 | V_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 11.11 | 24.32 | 39.83 | 30.13 | 37.26 |
| replica8 | V_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 10.88 | 22.72 | 33.54 | 25.39 | 34.73 |
| replica8 | V_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 10.99 | 23.71 | 39.28 | 30.00 | 37.28 |
| scannet_cf18 | B_FC__F_ONLY_FIXED_ELIGIBLE | 8.55 | 17.99 | 26.19 | 20.14 | 28.99 |
| scannet_cf18 | B_FC__PARENT_D2_FIXED_ELIGIBLE | 7.99 | 17.58 | 26.02 | 20.10 | 29.25 |
| scannet_cf18 | B_PFE_ADAPTED__F_ONLY_FIXED_ELIGIBLE | 8.57 | 18.27 | 27.68 | 20.97 | 29.87 |
| scannet_cf18 | B_PFE_ADAPTED__PARENT_D2_FIXED_ELIGIBLE | 8.60 | 18.45 | 26.72 | 20.73 | 30.10 |
| scannet_cf18 | B_SURFACE_MEAN__F_ONLY_FIXED_ELIGIBLE | 8.28 | 17.58 | 25.22 | 20.62 | 27.90 |
| scannet_cf18 | B_SURFACE_MEAN__PARENT_D2_FIXED_ELIGIBLE | 7.68 | 16.99 | 25.27 | 19.69 | 28.51 |
| scannet_cf18 | G_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 8.43 | 17.96 | 25.93 | 20.91 | 29.42 |
| scannet_cf18 | G_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 8.29 | 18.28 | 26.39 | 20.49 | 29.53 |
| scannet_cf18 | G_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 8.78 | 18.62 | 26.81 | 21.47 | 30.10 |
| scannet_cf18 | G_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 8.28 | 18.26 | 26.33 | 20.49 | 29.61 |
| scannet_cf18 | MV_ATTN_seed17__F_ONLY_FIXED_ELIGIBLE | 3.42 | 7.58 | 12.43 | 9.55 | 16.66 |
| scannet_cf18 | MV_ATTN_seed17__PARENT_D2_FIXED_ELIGIBLE | 4.46 | 9.82 | 15.58 | 12.07 | 19.50 |
| scannet_cf18 | MV_ATTN_seed29__F_ONLY_FIXED_ELIGIBLE | 3.64 | 7.87 | 12.56 | 9.93 | 17.59 |
| scannet_cf18 | MV_ATTN_seed29__PARENT_D2_FIXED_ELIGIBLE | 4.83 | 10.27 | 15.32 | 11.91 | 19.92 |
| scannet_cf18 | S_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 8.33 | 17.79 | 26.06 | 21.00 | 29.55 |
| scannet_cf18 | S_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 8.19 | 17.92 | 26.26 | 20.49 | 29.56 |
| scannet_cf18 | S_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 8.38 | 17.81 | 25.82 | 20.80 | 29.20 |
| scannet_cf18 | S_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 8.14 | 17.83 | 25.86 | 20.33 | 29.43 |
| scannet_cf18 | V_LL_seed17__F_ONLY_FIXED_ELIGIBLE | 8.41 | 17.85 | 25.75 | 20.92 | 29.68 |
| scannet_cf18 | V_LL_seed17__PARENT_D2_FIXED_ELIGIBLE | 8.16 | 18.09 | 26.00 | 20.51 | 29.71 |
| scannet_cf18 | V_LL_seed29__F_ONLY_FIXED_ELIGIBLE | 8.37 | 17.96 | 25.86 | 20.82 | 29.75 |
| scannet_cf18 | V_LL_seed29__PARENT_D2_FIXED_ELIGIBLE | 8.18 | 17.99 | 25.93 | 20.43 | 29.78 |

固定G1分区、旧eligible/NQ/温度及scene-specific词表。该回归不增加独立确认家族。

## 固定家族配对统计

| S减对照 | metric | 均值差(pp) | 2.5% | 97.5% | 区间含0 | defined | undefined |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S_LL/G_LL | GT_A | -0.4279 | -1.4417 | 0.5163 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_C | -0.7735 | -1.9485 | 0.3240 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_H_base_novel | -0.6709 | -1.7840 | 0.4098 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_M | -0.4682 | -1.3209 | 0.1414 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_all_balanced_top1 | -0.4277 | -1.4479 | -0.0378 | 否 | 1000 | 0 |
| S_LL/G_LL | GT_all_top1 | -0.7168 | -1.7732 | 0.2137 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_base_top1 | -0.9302 | -2.1320 | 0.2225 | 是 | 1000 | 0 |
| S_LL/G_LL | GT_novel_top1 | 0.0000 | -2.2059 | 2.6316 | 是 | 1000 | 0 |
| S_LL/G_LL | real_ap25 | 0.0181 | -0.2432 | 0.5213 | 是 | 1000 | 0 |
| S_LL/G_LL | real_ap50 | 0.1407 | -0.0009 | 0.5369 | 是 | 1000 | 0 |
| S_LL/G_LL | real_apall | 0.0781 | -0.0005 | 0.2974 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_A | 1.7967 | 0.4284 | 3.2475 | 否 | 1000 | 0 |
| S_LL/V_LL | GT_C | 0.5440 | -1.3839 | 2.6395 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_H_base_novel | 0.8805 | -1.3788 | 3.4431 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_M | 0.9290 | -0.3855 | 2.0681 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_all_balanced_top1 | 0.1622 | -2.2679 | 1.4640 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_all_top1 | 0.7168 | -1.4712 | 3.0364 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_base_top1 | 0.4651 | -2.0308 | 3.2088 | 是 | 1000 | 0 |
| S_LL/V_LL | GT_novel_top1 | 1.5625 | -3.5211 | 7.6092 | 是 | 1000 | 0 |
| S_LL/V_LL | real_ap25 | 0.3255 | 0.0028 | 0.5545 | 否 | 1000 | 0 |
| S_LL/V_LL | real_ap50 | -0.0257 | -0.0467 | 0.0873 | 是 | 1000 | 0 |
| S_LL/V_LL | real_apall | -0.0153 | -0.0276 | 0.0475 | 是 | 1000 | 0 |
| S_LL/MV_ATTN | GT_A | -6.2963 | -16.4499 | 2.7501 | 是 | 1000 | 0 |
| S_LL/MV_ATTN | GT_C | -5.0780 | -14.1442 | 2.9970 | 是 | 1000 | 0 |
| S_LL/MV_ATTN | GT_H_base_novel | 40.2532 | 34.1978 | 45.1607 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | GT_M | 20.7018 | 10.9503 | 23.9042 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | GT_all_balanced_top1 | 20.9381 | 11.2411 | 23.6628 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | GT_all_top1 | -3.0466 | -9.7582 | 2.5993 | 是 | 1000 | 0 |
| S_LL/MV_ATTN | GT_base_top1 | -19.5349 | -27.9374 | -12.5478 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | GT_novel_top1 | 52.3438 | 41.4445 | 63.8526 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | real_ap25 | 5.1899 | 2.1944 | 7.9174 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | real_ap50 | 1.8026 | 0.6113 | 3.5093 | 否 | 1000 | 0 |
| S_LL/MV_ATTN | real_apall | 0.2240 | -0.0229 | 1.0907 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_A | 2.7824 | -0.1853 | 5.5053 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_C | 1.7626 | -2.3805 | 6.0669 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_H_base_novel | 2.2092 | -2.0679 | 6.6184 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_M | 9.0713 | 4.2316 | 12.5640 | 否 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_all_balanced_top1 | 9.7120 | 3.2617 | 13.8630 | 否 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_all_top1 | 2.1505 | -1.8522 | 5.7095 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_base_top1 | 2.3256 | -2.7036 | 7.1429 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | GT_novel_top1 | 1.5625 | -7.3543 | 13.1591 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | real_ap25 | 0.1590 | -0.3467 | 0.6952 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | real_ap50 | 0.2042 | -0.0224 | 0.5120 | 是 | 1000 | 0 |
| S_LL/B_SURFACE_MEAN | real_apall | 0.1582 | -0.0066 | 0.3472 | 是 | 1000 | 0 |

四对共4000个预定抽样集合。先分别测两seed再平均差；GT主口径为1000步，真实AP主口径为DEV选择步。各指标报告未作多重比较调整的percentile家族bootstrap区间；边界保留4位小数，避免微小负值被显示为0.00。

## 确认集两级相对权重效应

| 任务 | seed | 口径 | metric | inner均值效应(pp) | outer均值效应(pp) | interaction(pp) | LL−UU(pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_all_top1 | 0.90 | 2.33 | 1.08 | 3.23 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_all_balanced_top1 | 1.11 | 1.21 | 0.96 | 2.32 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_base_top1 | 0.70 | 2.09 | 0.47 | 2.79 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_novel_top1 | 1.56 | 3.12 | 3.12 | 4.69 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_H_base_novel | 1.00 | 2.50 | 1.34 | 3.51 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_A | 1.57 | 2.92 | 2.53 | 4.49 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_M | 1.19 | 2.30 | 1.49 | 3.49 |
| CONF_GT_FULL_ORIGINALS | 17 | matched1000 | GT_C | 1.24 | 2.74 | 1.61 | 3.98 |
| CONF_REAL_SPARSE_AP | 17 | matched1000 | apall | -0.02 | 0.16 | -0.04 | 0.14 |
| CONF_REAL_SPARSE_AP | 17 | matched1000 | ap50 | -0.03 | 0.21 | -0.06 | 0.18 |
| CONF_REAL_SPARSE_AP | 17 | matched1000 | ap25 | -0.03 | 0.29 | -0.06 | 0.26 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_all_top1 | 0.90 | 2.33 | 0.36 | 3.23 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_all_balanced_top1 | 1.11 | 1.21 | 0.91 | 2.32 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_base_top1 | 0.70 | 2.09 | -0.47 | 2.79 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_novel_top1 | 1.56 | 3.12 | 3.12 | 4.69 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_H_base_novel | 1.00 | 2.50 | 0.68 | 3.51 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_A | 1.60 | 2.89 | 2.29 | 4.49 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_M | 0.88 | 2.61 | 1.72 | 3.49 |
| CONF_GT_FULL_ORIGINALS | 17 | selected | GT_C | 1.31 | 2.67 | 0.98 | 3.98 |
| CONF_REAL_SPARSE_AP | 17 | selected | apall | -0.02 | 0.16 | -0.04 | 0.14 |
| CONF_REAL_SPARSE_AP | 17 | selected | ap50 | -0.03 | 0.21 | -0.06 | 0.18 |
| CONF_REAL_SPARSE_AP | 17 | selected | ap25 | -0.65 | 0.91 | -1.30 | 0.26 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_all_top1 | 0.18 | 2.33 | -0.36 | 2.51 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_all_balanced_top1 | 1.02 | 1.27 | 0.77 | 2.29 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_base_top1 | 0.00 | 1.86 | -0.93 | 1.86 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_novel_top1 | 0.78 | 3.91 | 1.56 | 4.69 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_H_base_novel | 0.24 | 2.58 | -0.19 | 2.82 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_A | 0.90 | 2.78 | 1.19 | 3.67 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_M | 1.13 | 2.54 | 1.37 | 3.67 |
| CONF_GT_FULL_ORIGINALS | 29 | matched1000 | GT_C | 0.54 | 2.67 | 0.20 | 3.21 |
| CONF_REAL_SPARSE_AP | 29 | matched1000 | apall | 0.08 | 0.08 | 0.15 | 0.15 |
| CONF_REAL_SPARSE_AP | 29 | matched1000 | ap50 | 0.10 | 0.10 | 0.19 | 0.20 |
| CONF_REAL_SPARSE_AP | 29 | matched1000 | ap25 | 0.08 | 0.83 | 0.15 | 0.91 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_all_top1 | 0.54 | 1.97 | -0.36 | 2.51 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_all_balanced_top1 | 1.14 | 1.15 | 0.96 | 2.29 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_base_top1 | 0.47 | 1.40 | -0.93 | 1.86 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_novel_top1 | 0.78 | 3.91 | 1.56 | 4.69 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_H_base_novel | 0.57 | 2.25 | -0.18 | 2.82 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_A | 1.00 | 2.67 | 1.02 | 3.67 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_M | 0.81 | 2.86 | 1.53 | 3.67 |
| CONF_GT_FULL_ORIGINALS | 29 | selected | GT_C | 0.99 | 2.22 | 0.34 | 3.21 |
| CONF_REAL_SPARSE_AP | 29 | selected | apall | 0.08 | 0.08 | 0.15 | 0.15 |
| CONF_REAL_SPARSE_AP | 29 | selected | ap50 | 0.10 | 0.10 | 0.19 | 0.20 |
| CONF_REAL_SPARSE_AP | 29 | selected | ap25 | -0.56 | 1.47 | -1.11 | 0.91 |

UU仍学习自己的总路由质量；interaction单列，不与两个主效应重复相加。

## 实测成本

| 模型 | views | 含pool/phi/transfer median ms | P90 ms | head median ms | allocated P90 GiB | 参数 |
| --- | --- | --- | --- | --- | --- | --- |
| selected:G_LL:seed17 | 2 | 38.05 | 41.09 | 2.45 | 1.37 | 216194 |
| selected:G_LL:seed17 | 4 | 59.84 | 62.88 | 2.82 | 1.42 | 216194 |
| selected:G_LL:seed17 | 8 | 91.77 | 97.70 | 2.85 | 1.51 | 216194 |
| selected:G_LL:seed29 | 2 | 41.57 | 44.12 | 2.72 | 1.37 | 216194 |
| selected:G_LL:seed29 | 4 | 49.96 | 58.84 | 2.54 | 1.42 | 216194 |
| selected:G_LL:seed29 | 8 | 67.23 | 76.24 | 2.38 | 1.51 | 216194 |
| selected:MV_ATTN:seed17 | 2 | 35.53 | 37.62 | 2.56 | 1.38 | 3818240 |
| selected:MV_ATTN:seed17 | 4 | 49.11 | 53.21 | 2.63 | 1.43 | 3818240 |
| selected:MV_ATTN:seed17 | 8 | 75.11 | 84.12 | 2.70 | 1.53 | 3818240 |
| selected:MV_ATTN:seed29 | 2 | 34.67 | 36.36 | 2.54 | 1.38 | 3818240 |
| selected:MV_ATTN:seed29 | 4 | 44.76 | 47.10 | 2.55 | 1.43 | 3818240 |
| selected:MV_ATTN:seed29 | 8 | 63.80 | 73.86 | 2.58 | 1.53 | 3818240 |
| selected:S_LL:seed17 | 2 | 47.01 | 51.00 | 3.30 | 1.37 | 216194 |
| selected:S_LL:seed17 | 4 | 51.25 | 53.14 | 2.59 | 1.42 | 216194 |
| selected:S_LL:seed17 | 8 | 70.19 | 76.69 | 2.45 | 1.52 | 216194 |
| selected:S_LL:seed29 | 2 | 35.46 | 37.18 | 2.40 | 1.37 | 216194 |
| selected:S_LL:seed29 | 4 | 47.19 | 50.29 | 2.42 | 1.42 | 216194 |
| selected:S_LL:seed29 | 8 | 68.97 | 77.74 | 2.44 | 1.52 | 216194 |
| selected:S_LU:seed17 | 2 | 35.88 | 39.05 | 2.45 | 1.37 | 216194 |
| selected:S_LU:seed17 | 4 | 46.27 | 49.82 | 2.44 | 1.42 | 216194 |
| selected:S_LU:seed17 | 8 | 68.76 | 74.20 | 2.45 | 1.52 | 216194 |
| selected:S_LU:seed29 | 2 | 35.62 | 37.66 | 2.43 | 1.37 | 216194 |
| selected:S_LU:seed29 | 4 | 47.11 | 49.87 | 2.42 | 1.42 | 216194 |
| selected:S_LU:seed29 | 8 | 71.49 | 74.43 | 2.49 | 1.52 | 216194 |
| selected:S_PRE:seed17 | 2 | 36.58 | 38.21 | 2.57 | 1.37 | 216194 |
| selected:S_PRE:seed17 | 4 | 48.16 | 50.25 | 2.57 | 1.42 | 216194 |
| selected:S_PRE:seed17 | 8 | 71.34 | 75.56 | 2.60 | 1.52 | 216194 |
| selected:S_PRE:seed29 | 2 | 45.10 | 50.80 | 3.34 | 1.37 | 216194 |
| selected:S_PRE:seed29 | 4 | 49.59 | 63.53 | 2.76 | 1.42 | 216194 |
| selected:S_PRE:seed29 | 8 | 72.93 | 76.27 | 2.66 | 1.52 | 216194 |
| selected:S_UL:seed17 | 2 | 35.73 | 38.61 | 2.36 | 1.37 | 216194 |
| selected:S_UL:seed17 | 4 | 45.48 | 50.60 | 2.32 | 1.42 | 216194 |
| selected:S_UL:seed17 | 8 | 65.74 | 67.77 | 2.32 | 1.52 | 216194 |
| selected:S_UL:seed29 | 2 | 34.66 | 36.81 | 2.31 | 1.37 | 216194 |
| selected:S_UL:seed29 | 4 | 44.97 | 46.67 | 2.30 | 1.42 | 216194 |
| selected:S_UL:seed29 | 8 | 65.68 | 68.36 | 2.33 | 1.52 | 216194 |
| selected:S_UU:seed17 | 2 | 34.52 | 36.61 | 2.33 | 1.37 | 216194 |
| selected:S_UU:seed17 | 4 | 46.65 | 51.55 | 2.33 | 1.42 | 216194 |
| selected:S_UU:seed17 | 8 | 65.71 | 67.62 | 2.34 | 1.52 | 216194 |
| selected:S_UU:seed29 | 2 | 34.14 | 36.75 | 2.32 | 1.37 | 216194 |
| selected:S_UU:seed29 | 4 | 45.67 | 47.59 | 2.33 | 1.42 | 216194 |
| selected:S_UU:seed29 | 8 | 66.66 | 69.54 | 2.35 | 1.52 | 216194 |
| selected:V_LL:seed17 | 2 | 35.35 | 36.63 | 2.39 | 1.37 | 216194 |
| selected:V_LL:seed17 | 4 | 45.18 | 47.24 | 2.39 | 1.42 | 216194 |
| selected:V_LL:seed17 | 8 | 66.35 | 68.16 | 2.40 | 1.52 | 216194 |
| selected:V_LL:seed29 | 2 | 34.41 | 36.65 | 2.39 | 1.37 | 216194 |
| selected:V_LL:seed29 | 4 | 45.44 | 47.27 | 2.39 | 1.42 | 216194 |
| selected:V_LL:seed29 | 8 | 67.72 | 68.97 | 2.41 | 1.52 | 216194 |
| selected:B_FC | 2 | 32.71 | 34.64 | 0.42 | 1.37 | 0 |
| selected:B_FC | 4 | 44.72 | 48.67 | 0.42 | 1.42 | 0 |
| selected:B_FC | 8 | 63.94 | 67.93 | 0.42 | 1.52 | 0 |
| selected:B_SURFACE_MEAN | 2 | 32.95 | 35.08 | 1.21 | 1.37 | 0 |
| selected:B_SURFACE_MEAN | 4 | 43.50 | 44.90 | 1.21 | 1.42 | 0 |
| selected:B_SURFACE_MEAN | 8 | 65.55 | 69.56 | 1.23 | 1.52 | 0 |
| selected:B_PFE_ADAPTED | 2 | 33.96 | 36.39 | 1.10 | 1.37 | 0 |
| selected:B_PFE_ADAPTED | 4 | 43.85 | 45.48 | 1.10 | 1.42 | 0 |
| selected:B_PFE_ADAPTED | 8 | 64.03 | 70.80 | 1.11 | 1.52 | 0 |

模型已驻留、读取已缓存FC grid；head计时包含在总计时中。该表不是冷启动或系统30FPS证明。

新确认输入的共享计算只发生一次，不逐方法重复相加：

| 共享模块 | 图像编码 | decoder batch | 阶段秒 | allocated峰值 GiB | 边界 |
| --- | --- | --- | --- | --- | --- |
| SAM2 AMG | 48 | 192 | 48.08 | 3.06 | 含模型构造/加载及固定seed图提案，不含环境启动；全部方法共享一次 |
| FC全图捕获 | 635 | 不适用 | 309.14 | 1.80 | 完整固定bank捕获，含逐帧预处理、提取、投影与缓存写入 |

## 图与实际案例

| 图 | 文件 | 内容与口径 |
| --- | --- | --- |
| budget | [PNG](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/budget.png) · [PDF](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/budget.pdf) | 左：全部8方法的DEV A (%)；右：CONF固定GT原件的clean Top1 (%)、选择步与2/4/8视图前缀；实线seed17、虚线seed29。 |
| coverage | [PNG](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/coverage.png) · [PDF](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/coverage.pdf) | 左：真实AMG/提案/唯一GT的不同计数单位；右：固定base/novel GT原件的支持与abstention，不混用两个任务分母。 |
| factorial | [PNG](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/factorial.png) · [PDF](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/factorial.pdf) | 已曝光DEV A (%)；固定1000步，左右为seed17/29；两个面板共享颜色范围，格内为实测值。 |
| grouping | [PNG](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/grouping.png) · [PDF](../../../../artifacts/static_ovmap/structure_confirmation_v1/figures/grouping.pdf) | CONF12真实稀疏提案池化APall (%)；DEV选择步、8V；两seed分别显示，静态只计一次。 |

| 对照 | 分层 | 实际案例 | 缺少位置 |
| --- | --- | --- | --- |
| S_LL/G_LL | success | 0 | 2 |
| S_LL/G_LL | failure | 0 | 2 |
| S_LL/G_LL | unchanged | 2 | 0 |
| S_LL/MV_ATTN | success | 2 | 0 |
| S_LL/MV_ATTN | failure | 1 | 1 |
| S_LL/MV_ATTN | unchanged | 2 | 0 |

案例来源和确定性选取记录见[来源JSON](../../../../artifacts/static_ovmap/structure_confirmation_v1/records/figures/source_data.json.gz)。案例使用实测深度/mesh投影，三列共享相机、尺度和颜色；原始RGB不上传。

## 覆盖、消耗与缺项

| 阶段 | 状态 |
| --- | --- |
| preparation | COMPLETE |
| training | COMPLETE |
| selection | COMPLETE |
| predictors | COMPLETE |
| raw_confirmation | COMPLETE |
| confirmation_geometry | COMPLETE |
| confirmation_inputs | COMPLETE |
| confirmation_features | COMPLETE |
| real_predictions | COMPLETE |
| GT_inputs | COMPLETE |
| GT_predictions | COMPLETE |
| confirmation_score | COMPLETE |
| map_predictions | COMPLETE |
| map_score | COMPLETE |
| statistics | COMPLETE |
| cost | COMPLETE |
| qualitative | COMPLETE |

```json
{
  "workload": {
    "counts": {
      "SAM_images": 48,
      "confirmation_FC": 635,
      "confirmation_head": 97083,
      "cost_head": 3648,
      "engineering_head": 5,
      "main_head": 1050016,
      "map_head": 16236,
      "optimizer_updates": 32097,
      "replayed_updates": 98
    },
    "identity": "eb6e15f53f7ab148b1dc705076ed2adf7962ebdb6b66d947ccbe97a79532a124",
    "limits": {
      "SAM_images": 48,
      "confirmation_FC": 1152,
      "confirmation_head": 300000,
      "cost_head": 4000,
      "engineering_head": 200,
      "main_head": 1100000,
      "map_head": 131072,
      "optimizer_updates": 32400,
      "replayed_updates": 400
    },
    "status": "COMPLETE"
  },
  "missing": []
}
```

部署维持N0。所有控制、负结果和undefined单元均保留；本报告不据单项上涨声称整体优于OVI-MAP或达到CVPR录用标准。
