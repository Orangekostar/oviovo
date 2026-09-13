# Codex任务：OVI-MAP静态基准优化

你在 `Orangekostar/oviovo` 仓库执行开发。目标是用可归因、可复现的静态实验超越OVI-MAP，不继续扩展CROVE动态维护，不把评测变更当成方法收益。

本任务依据 `01_RESEARCH_REPORT.md` 和 `02_LITERATURE_MATRIX.md`。所有下列 `src/static_ovmap/*`、新脚本和配置均为**待创建的建议路径**，不是已经存在的仓库接口。`reference/static_core.py`只是通过8项小测试的独立参考核心，不是集成完成的mapper。

## A. 工作范围与版本

已读基底：`d5c0688bc662f8e65455cb9c62909de87c941b43`，远端分支 `research/crove-fine-current-map-v1`。先记录实际本地HEAD、远端SHA和工作树状态；保护现有未提交工作，不做reset、不覆盖用户分支。建议创建独立worktree/分支 `research/ovimap-static-benchmark-v1`。

外部OVI版本分开锁定：

```text
origin_impl 6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9
main        f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424
```

先使用已有身份一致的原生baseline/map/feature缓存。只有缺实际输入或配置不一致的单元需要重跑。不要为了建立本项目重建整个环境，不要把ROS/Python3.8和新VLM环境强行合并升级。

## B. 开工前必读与复用

只读与任务直接相关的文件，记录关键函数及实际调用者，不根据README的旧版本名称推断当前主链：

| 已存在且已核验的文件 | 本任务依赖 |
|---|---|
| `src/evaluation/baselines/ovimap_paper_audit.py` | Replica协议、manifest与真实指标产物 |
| `src/evaluation/baselines/ovimap_native.py` | 整数mask/raycast、inclusive bbox、颜色一致性 |
| `configs/evaluation/results/crove_fine_current_map_v1/table_values.json` | 现有Room0开发指标，不当作Replica8 |
| `src/oviv2/ovimap_static_anchor.py` | 确认它依赖时序状态，然后保持不改 |
| `src/modules/semantic_memory.py` | 旧身份提交逻辑；不直接复用其类别冻结策略 |
| `run_room0_v2_full_eval.py` | 确认它调用src.v2，不能当现成论文runner |
| OVI `scripts/utils/mesh_postprocess_utils.py` | 最后8次查询、少于2次跳过、5 cm投影、文本匹配 |
| OVI `scripts/vl_models.py` | 已有6个crop、各模型特征空间、bbox切片 |
| OVI `scripts/eval_inst_seg.py` | 实际输出mP/mR，不是自动等于积分AP |

本次不要求通读无关动态模块，也不要求跑全仓库数千项测试。最低必要验证见后文。

## C. 第一批任务：先交付一个真实S1结果，不先造大框架

### R0：协议与输入资产绑定

记录实际可用的原生mesh、实例ID、instance semantic字典、原始帧/mask/pose以及权重。每个文件说明是否有、本轮是否复用、版本来源。不要把历史文档中的绝对路径当作当前已经存在。

复用已有 `ovimap_paper_audit.py`，只补缺失的静态扩展：

- Replica8固定，200帧取0、10、…、1990。GT仅评测时打开。
- ScanNet按原始实现18场景固定；主分支注释缺 `scene0378_02`，不能漏掉。采样依真实序列/原始baseline配置显式导出，不能擅自套用Replica步长。
- 体素0.01 m；预测mesh到GT顶点1NN、距离严格小于0.05 m，未匹配点语义0。
- 语义词表Replica51/ScanNet200与valid_ids都要锁定；不能只改task字符串而保留错误ID映射。
- 原始 `mP@t/mR@t`、规范AP以及论文主表AP来源分开命名；缺失某个AP实现只阻止该列论文比较，其他有效实验继续。

输出 `artifacts/static_ovmap/protocol_manifest.json` 与 `baseline_asset_inventory.json`。路径是建议；符合仓库现有产物布局时优先沿用现有目录。

### R1：建立原始读出精确对照

在不改mesh和instance ID的条件下，用已锁定的特征缓存重现原始last8、vis_area加权和min_two_queries逻辑。

检查缓存每实例查询数分布；如果绝大多数对象少于8条，last8替换的可改善对象很少，先据此判断工作优先级。若历史缓存只保留最后8条，明确不能比较“全历史最佳8条”，先补观测记录，不伪造历史。

原始输出与适配后的baseline在相同格式下应一致。只需在相关小样本和可复用场景上验证，不额外做全仓回归。bbox一像素边界修正作为独立开关，不混入S1主对比。

### R2：实现观测bank和纯语义读出

建议新文件：

```text
src/static_ovmap/contracts.py
src/static_ovmap/cache_io.py
src/static_ovmap/observation_bank.py
src/static_ovmap/readout.py
scripts/evaluation/run_static_ovmap_readout.py
```

观测记录最低字段：

```text
scene_id, instance_id, frame_id, source_query_id
feature_space_id, feature_dim, feature
visible_area_px, crop_bbox_xyxy, depth_valid_ratio
mask_quality_proxy, sharpness_proxy
camera_direction, view_bin, geometry_support_frames
source_mask_id_or_path, source_config_hash
```

`feature_space_id`须绑定encoder权重、layer、预处理/crop策略。不同空间分开计算文本相似度，禁止混合平均；维度相同也不能豁免。

缺少原始mask时不要用GT补，缺少quality时不要偷偷填“完美质量”。可先跑仅用已有面积/方向字段的明确降级版本，记录字段缺失；后续通过原始输入补轻量质量计算并单列成本。

实现策略：

1. `last8`：原始顺序精确复现。
2. `random8`：固定seed，作为选择机制对照。
3. `quality8`：只看可观测质量。
4. `quality_coverage8`：固定K下质量加权视向覆盖，使用参考核心作为起点。
5. `all_views`：单列不同缓存/读出预算，不冒充K=8。

必须增加三个无需额外推理的拆分：S1a只换观测选择并保留原vis_area融合；S1b保持last8但换质量权重；S1c二者同时变化。

首轮禁止增加VLM查询，禁止替换编码器，禁止改变实例过滤。原native geometry和class-agnostic ID导出保持不变。任何几何或类别无关指标变化都要解释读出耦合，不能自动归因为语义更好。

离线候选池算法使用全部已发生的查询记录；作为 `STATIC_OFFLINE_READOUT`。实现在线版本时，必须保存基于当时信息得到的几何方向和质量量，并按前缀更新有界内存bank；仅在最终地图坐标上加cutoff不构成严格在线证明。离线与在线不混报。

**R0–R2完成后的首个交付**：至少一个真实开发场景、相同geometry/query/encoder的配对指标和观测选择诊断；同时保存原始与新语义输出。若无收益，说明哪些观测被替换、语义margin如何变化，而不是立即换大模型。

## D. 第二批任务：验证另外两种独立错误

### R3：单语义查询实例fallback

不删除几何层中的未知语义实例。将“语义查询只有一次”和“几何只观测一次”拆开；对具备独立多帧几何支持、一次高质量语义查询的对象尝试fallback。

新策略需单独开关，输出新加入实例数量、类别、几何支持帧数、预测置信度。阈值只能在开发条件选择；同时记录新增TP/FP，不能仅凭mIoU提升宣布成功。

### R4：多视角实例图

建议新文件：

```text
src/static_ovmap/instance_graph.py
src/static_ovmap/hierarchy.py
src/static_ovmap/native_export.py
```

从原生预测的superpoint/tracklet建节点，利用原始200帧的共视、深度一致重投影和whole-object掩码支持生成正负边。先实现merge，再实现有充分whole-object多视角证据的split；分别做消融。

所有cannot-link必须说明它为什么代表两个不同完整物体。不能将任意不同mask ID、不相交SAM碎片、不同暂时文本标签直接当物体互斥真值。传递性约束要覆盖整个合并分量，参考代码已有这部分。

层级选择采用固定的预测证据规则，不可用GT逐实例挑层。保存原始partition、变更映射和拒绝原因，以便出现负收益时直接恢复，不重跑前端。

原几何坐标不变；输出instance id可以变，但必须具备完整来源映射。内部使用足够宽的整数；输出到既有uint8/uint16格式前显式检查/重映射，不允许溢出。

先看类别无关AP50/高IoU指标和merge/split错误，再看语义AP。用GT做开发诊断是允许的，但GT只能进入评测/诊断程序，不进入生成边或选择实例的程序。

## E. 第三批任务：一个稠密分支，不堆三个

### R5：GLA-CLIP、VIP或SPAR择一

先确认真实发布checkpoint和当前源码接口。GLA-CLIP可从 `gla_clip_segmentor.py::GLA_CLIPSegmentation.forward_feature` 检查特征前向；完整segmentor读取词表，必须将分类前特征输出和查询分类分开。

VIP树中是 `dinosegmentor.py`、`eval_seg.py`，README示例可能仍写 `eval.py`；不能把不存在的命令写进最终操作手册。SPAR论文Legacy权重与更新预处理版分开记录。任一候选安装/资产明显阻塞时，记录原因并转另一个已发布候选，不消耗大量时间修三套环境。

建议新文件：

```text
src/static_ovmap/dense_features.py
src/static_ovmap/feature_refine.py
src/static_ovmap/query_scores.py
```

顺序消融：原始ROI编码 → 单帧稠密特征池化 → AnyUp替代插值 → 轻量几何净化。AnyUp使用正确RGB归一化，记录模型版本、output_size与chunk参数；NATTEN版本不能默认和原论文算子完全等价。

图净化仅在可信同owner且深度/可见性一致区域传播，unknown=0不传播。先T=1，lambda在0.1/0.2/0.3的有限开发选择中确定。参考 `refine_small_graph` 是<=4096节点诊断，生产使用稀疏矩阵/边或分块，不能对百万顶点开N×N。

每对象保存K条语义观测和少量局部原型，测实际存储。先证明质量收益，再考虑EmbodiedSplat式码本或SPAR蒸馏；未经收益验证不要启动大规模训练。

### R6：最高准确率的独立T3D轨道

SpaCeFormer只作为明确带额外三维预训练的轨道。先读真实 `space_former_seg.py` 和checkpoint接口，不将backbone误作完整实例推理。

输入相同RGB-D观测产生的原生点云，不使用GT mesh。确认坐标/颜色归一化、mask/score输出和NMS，并使用其SigLIP2文本空间匹配1152维clip特征。

先跑 `SpaCeFormer-only`，再比较 `OVI-only` 和 `SpaCeFormer+OVI proposals+ours readout`。额外训练数据、可能包含的ScanNet训练/测试场景与计算费用明确记录。无法核实预训练场景排除时不作“未见场景零样本泛化”主张。

### R7：难例复核可选，不默认装入主系统

WOW-Seg区域命名或SAM3视觉提示补mask只选择一个做有预算实验。需要固定触发比例、次数、文本到类别映射及同预算随机触发对照。输入整个评测类别列表生成mask必须标 `QUERY_CONDITIONED`；禁止与先建query-independent地图混报。

## F. 实验矩阵与报告

至少保持以下条件可独立选择，不要求在无收益时全部跑满：

| ID | 条件 | 单独回答的问题 |
|---|---|---|
| B0 | 原始OVI，原始readout | 同机同协议基线 |
| B1 | 重构OVI，原始readout | 版本漂移，不是方法消融 |
| S1 | B0+quality/coverage8 | 不增加查询是否改善语义 |
| S2 | B0+单语义查询fallback | 是有效召回还是多了FP |
| G1 | B0+merge | 碎片是否减少 |
| G2 | B0+merge+split | 粘连是否改善 |
| D1 | B0+一个稠密语言骨干 | 骨干替换效应 |
| D2 | D1+AnyUp | 上采样增益 |
| D3 | D2+几何净化 | 净化增益 |
| C1 | 冻结的有效组合 | 增益是否兼容，不按算术相加 |
| T0/T1 | SpaCeFormer独立/融合 | 额外三维训练轨道的潜力 |
| Q0/Q1 | 随机/不确定性复核 | 同预算选择机制是否有效 |

为每一结果记录scene、protocol、input_frame_ids、source/model/config身份、原始评测文件、模型训练条件、在线/离线类型、运行耗时和显存。原始缺失值记null和原因，不补零；同一配置完整列出所有主指标。

Replica Room0已经用于开发：主表仍报告8场景，额外7场景诊断不能冒称全新独立测试。开发调参不使用论文18场景的GT，训练/开发划分按物理场景而不是仅按scan id。

成本同时报告前端、VLM、融合、导出、整体总时长及冷/热缓存口径。结束计时前GPU同步并排空worker；不要把预缓存mask/特征的mapper吞吐写成端到端FPS。多个GPU按场景并行时记录CPU/RAM争用，最终性能对比用一致硬件条件。

## G. 验证与资源上限

保留与任务直接相关的最低验证：已有协议/导出回归、观测选择和不同语义空间测试、互斥传递性、小图不跨owner传播、一个真实场景smoke。边界bbox只需有针对性的已知裁剪测试。

禁止借此任务增加无关安全扫描、通用异常注入、全动态测试重跑或大规模随机压力测试。也不能因为“避免过度测试”而跳过真实数据试跑和评测输入一致性；这两项是研究正确性而非附加安全负担。

预测区间不是验收阈值，不允许为了追上预测值改评测或挑选不同模型。遇到负收益及时报告，并保留基线；缺数据/权重/评测列应局部阻塞，不无限追加审计框架。

## H. GitHub交接

每一实质阶段完成后提交代码、配置、真实小结果和交接Markdown到任务分支。不上传原始受限数据、未经授权模型文件、巨型mesh或私人凭证；大产物记录实际路径、大小、校验和与获取条件。仅对输入/结果清单作必要校验，不做全仓无差别重复哈希。

最终交接文档必须包含：

```text
任务ID和实际完成范围
本地/远端分支、最终commit、外部模型/数据版本
每项实验的状态：COMPLETE / NOT_RUN / BLOCKED / FAILED
实际执行命令（已经存在并验证，不是本文建议接口）
主表、消融表、成本表及逐场景文件
哪些提升来自模型、读取策略、几何修正或版本变化
负结果、未解决限制、后续最高优先级
代码/小结果/文档是否已push；远端SHA核验
```

不得仅写“全部完成”；只有 `git ls-remote` 等实际返回证明远端SHA匹配时才写PUSH_VERIFIED。不要为了把最终commit写进自身文件而反复制造新提交，最终SHA可在终端交接中报告。

## I. 本包参考代码运行

在本包的 `reference/` 目录运行：

```bash
python -m pytest -q test_static_core.py
```

只依赖NumPy和pytest。参考代码不能直接承诺替换任何现有类。迁入仓库后遵循实际项目接口与依赖版本，用以上实验验证方法收益。
