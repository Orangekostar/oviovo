# 从 CROVE 转向 OVI-MAP 静态基准的研究与实施方案

## 1. 决策结论

建议暂时冻结 CROVE 的动态维护，另开一条静态实验分支。最值得优先投入的不是重新堆一个“更大分割器＋更大VLM＋更大建图器”，而是：**保持 OVI-MAP 原生稠密几何，改进语义观测选择与读出；随后用共视约束修正实例粒度，最后加入一个几何约束的稠密语言特征分支。** 暂称这条工程路线为 `OVI-Refine`，该名称不代表已经发表或完成验证的方法。

并行保留一条更进取的路线：**SpaCeFormer 三维实例先验＋原生 RGB-D 重建几何＋上述语义读出**。这条路线更接近“优先追求最高准确率”，但属于使用额外三维预训练的方法，不能包装成与原 OVI-MAP 完全相同的训练条件。[P03]

截至本次核验，尚无实验能证明这套组合将“全面打过”OVI-MAP。下面的预测是工程规划区间，不是结果或统计置信区间。当前完成的是源码与文献审查、具体实验设计，以及独立参考核心；没有在用户服务器上运行新 benchmark，没有修改或推送用户工作树。

## 2. 证据范围与可复现版本

本次按**正式会议年份为2026**筛选21篇相关论文，部分首次预印本发表于2024/2025年。覆盖三维开放词汇实例、静态建图、二维稠密视觉语言特征、区域识别、特征上采样等互补方向。逐篇的论文、仓库、入口、公开程度和迁移限制见 `02_LITERATURE_MATRIX.md`。存在实现目录才计入源码公开清单；只有占位README的直接相关项目不凑数。

检查深度并不等同“完整复现21个系统”：论文方法与相关实验、官方仓库结构和复现入口已检查；对真正决定实施的 OVI-MAP 导出、特征编码、评测，以及用户仓库协议/适配、AnyUp、GLA-CLIP 和 SpaCeFormer 接口做了源代码级检查。CompetitorFormer 的正式收录/摘要和代码已核验，但终稿全文下载未完成；不使用其未核实数值做判断。部分仓库仍缺完整权重或发布步骤，已逐项列明。

| 对象 | 固定引用 | 用途 |
|---|---|---|
| 用户仓库 | `Orangekostar/oviovo@d5c0688bc662f8e65455cb9c62909de87c941b43` | 实施基底；分支 `research/crove-fine-current-map-v1` |
| OVI-MAP原始实现 | `OVI-MAP/OVI-MAP@6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9` | `origin_impl`，论文忠实性优先的起点 |
| OVI-MAP重构实现 | `OVI-MAP/OVI-MAP@f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424` | 当前多环境实现，作为独立版本记录，不混同原始实现 |
| 建议新分支 | `research/ovimap-static-benchmark-v1` | 仅建议，尚未创建或推送 |

源版本是2026-09-13核验的远端状态，不证明服务器工作树无未提交修改。[B02][U01]

## 3. 要打的究竟是哪张表

OVI-MAP 使用 RGB-D、位姿、200个采样帧，评估原生预测地图投影到GT顶点后的结果。Replica为8场景、51类；ScanNet为18场景、200类。官方README更正所有实例建图实验使用**0.01 m体素**，不能沿用论文中的体素尺寸笔误。[B01][B02]

| 数据集 | 类别无关实例：mIoU / AP50 / AP25 | 语义：mIoU / mAcc | 类别相关实例：AP25 / AP50 / APall |
|---|---|---|---|
| Replica | 36.3 / 50.8 / 76.7 | 26.5 / 32.2 | 34.5 / 21.2 / 8.5 |
| ScanNet | 41.2 / 24.0 / 37.4 | 17.5 / 27.6 | 23.4 / 15.7 / 7.2 |

这些是论文目标值，不是本次本地复现值。类别无关/相关AP不能混用，mIoU也必须写明是实例匹配IoU还是逐顶点语义IoU。[B01]

### 3.1 18场景名单已从原始实现找回

原始 `scripts/eval_inst_seg.py` 的ScanNet注释名单为18场景；重构主分支相应注释只有17场景，缺少 `scene0378_02`。应使用显式配置，不能照抄重构注释或误用标准ScanNet全验证集替代论文子集。[B05][B09]

```text
scene0011_00 scene0011_01
scene0050_00 scene0050_01 scene0050_02
scene0084_00 scene0084_01 scene0084_02
scene0168_00 scene0168_01 scene0168_02
scene0231_00 scene0231_01 scene0231_02
scene0378_00 scene0378_01 scene0378_02
scene0518_00
```

这闭合了论文“18场景”与原始代码名单的一致性；仍须从实际运行配置导出逐场景采样帧、数据版本和位姿/深度单位。不要假定ScanNet所有序列与Replica一样固定 `step=10`。

### 3.2 公开脚本中的Precision不是PR曲线AP

当前 `scripts/eval_inst_seg.py::assign_pred_inst_to_gt_inst` 对每个预测实例计算最佳GT IoU，再统计0.25/0.50/0.75阈值下TP/FP；最终输出 `mP`/`mR`，没有按置信度排序并积分PR曲线。它不能不加说明地称为标准AP。[B05]

处理方法不是擅自“修正”作者的评测来换取分数，而是保留三个明确命名的产物：原始发布脚本输出、论文AP对应实现/来源，以及统一标准AP诊断。用户仓库已有 `class_agnostic_ap_manifest` 约束，应优先复用并确认其真实产物，缺哪项只阻止那项论文比较，不阻止语义读出实验。[U03]

### 3.3 “全面超过”的验收定义

同一版本、场景、输入、类别集合和评测投影下，报告两数据集的类别无关实例指标、逐点语义指标、类别相关AP全部列；同一个选定配置不能按指标挑不同最优模型。准确率全面领先与准确率/速度/显存同时Pareto领先是两个目标，后者不能从更大模型直接推出。

另外，超过OVI-MAP不自动等于2026年全领域SOTA。新增三维模型、二维语言骨干、训练监督或离线全局处理，必须给出相应对照，而不是只有一条较弱基线。

## 4. 当前代码究竟支持什么

### 4.1 已有结果不能直接当作八场景胜利

`configs/evaluation/results/crove_fine_current_map_v1/table_values.json` 中，Room0开发条件的mIoU是0.4478887，AP50是0.4707106。协议名是 `crove_fine_current_map_v1_replica_room0_dev`，不是论文八场景协议。不能拿44.79直接对比论文26.5，或把47.07与不同定义的AP作大小判断。[U02]

同一提交中的2 cm结果是**对已有1 cm表面的体素采样**，不是2 cm建图实验；不能据此宣称重建成本下降或2 cm TSDF质量等价。[U01]

### 4.2 不应把“关闭动态开关”当成新静态主干

`src/oviv2/ovimap_static_anchor.py` 实现的是不可变OVI-MAP锚点与CROVE时序状态的组合，包含前缀身份绑定、移动几何和生命周期逻辑。它不是一个独立静态建图器。[U04]

`run_room0_v2_full_eval.py` 调用的是 **`src.v2.pipeline.SemanticMapV2Pipeline`**，而不是 `src.oviv2`；运行参数和采样也不是现成的论文八场景入口。README又描述更早的模块分层。三个年代的命名空间不能凭名字当成同一条当前主链。[U06][U07]

`src/modules/semantic_memory.py` 包含已提交身份标签和anchor门槛，适合历史身份维护，但新静态语义读出不应被这种提交策略锁死。将查询时语义打分与实例几何分离，避免一个早期错误类别阻止后续纠正。[U05]

### 4.3 应复用的资产

| 已检查的现有文件 | 已确认功能 | 本次绑定 |
|---|---|---|
| `src/evaluation/baselines/ovimap_paper_audit.py` | Replica-51八场景、200帧、模型与产物检查 | 复用协议与结果验证，不重写评测框架 |
| `src/evaluation/baselines/ovimap_native.py` | 无损mask、整数raycast ID、RGB一致性、inclusive bbox | 复用输入规范；新模块内部显式ID重映射 |
| `src/oviv2/ovimap_static_anchor.py` | 静态锚点＋时序组合 | 冻结；不把它当新的静态mapper |
| `src/modules/semantic_memory.py` | 对象身份语义提交与导出策略 | 仅参考数据结构；新语义bank独立实现 |
| `run_room0_v2_full_eval.py` | 旧V2单场景runner | 不冒充官方评测runner |
| 现有OVI-MAP原生结果/feature缓存 | 是否完整须服务器核验 | 优先作为不重建的S1输入，不盲目重跑全部场景 |

## 5. 官方实现给出的直接切入点

### 5.1 最后8次观测不等于最佳8次

原始与重构实现的 `mesh_postprocess_utils.py` 均执行：

```python
vis_scores = vis_scores[-8:]
feat_inst = feat_inst[-8:, :]
```

再按可见面积归一化融合。这截取的是该实例**最后8次语义查询记录**，不是最佳质量或最佳覆盖的8次。[B03][B10]

这给出了最便宜的因果实验：固定mesh、实例ID、原来发生过的查询和编码器，只改变保留哪些已缓存观测。若旧缓存只剩最后8条，则没有“免费恢复历史特征”这回事；需要从原始输入补提特征，并单独记录额外预算。

### 5.2 不足2次语义记录会被跳过

同一导出循环对 `len(frame_list) < 2` 执行 `continue`。[B03][B10] 这不能直接解释成“物体只被看见一次”，因为几何观测与语义查询次数不同。可用独立多帧几何支持，保留只有一次高质量语义查询的真实实例；应同时统计新增TP和新增FP，不能简单移除所有过滤。

### 5.3 六裁剪已经存在

`VLModel.encode_image_with_bbox` 已经使用3个扩张尺度，每个尺度同时编码原图与黑背景mask图，共6个crop；逐个归一化后等权平均。[B04] 因而“加多尺度crop”不是新的方法贡献。可以研究的是：哪些crop被背景污染、何时应看局部核心、何时保留上下文，以及怎样让一帧共享特征减少重复编码。

### 5.4 不是所有“校准”都能改变类别

源码canonical打分为 `min_k exp(s_c)/(exp(s_c)+exp(s_k))`。如果各候选类别共享同一canonical集合，这等于 `sigmoid(s_c-max_k s_k)`，对 `s_c` 单调；因此argmax类别不变。这是对源码公式的数学推导，不是实验结果。“打开canonical”不能被当作新的分类涨点模块；真正的AP排序或拒识标定要单独验证。[B03]

### 5.5 小修复与方法收益必须分开

用户适配器输出inclusive bbox，而官方编码器使用Python右开区间切片；存在右/下边界少一个像素及极小框退化的风险。先做几张确定性裁剪的对照，不应把尚未验证的边界问题说成主要性能瓶颈。[U08][B04]

## 6. 文献转化为五项实现，而不是21项拼装

21篇完整矩阵与源地址单列。优先阅读/实现联系最强的是：**SpaCeFormer、LEGO、GLA-CLIP、AnyUp、SPAR、GeoPurify和WOW-Seg**。其他论文帮助排除错误路线或提供备选，不意味着全部装入pipeline。[P01][P03][P08][P11][P13][P16][P20]

### M1：质量与覆盖联合的语义观测bank

**输入**：实例ID、原始帧ID、特征、feature-space身份、原始观测mask、像素面积、深度有效率、清晰度、几何共视/方向、查询记录来源。所有质量量必须来自预测与观测，不来自测试GT。

初始质量公式（待验证设计）：

`q_o = min(1, sqrt(area_o / area_ref)) × depth_valid_o × mask_purity_o × sharpness_o`。

`mask_purity`不是GT IoU，而是例如crop内属于该预测实例的比例、深度边界一致性或原始mask核心占比等可观测代理；具体定义需固定并做独立消融。面积上限避免大背景天然占优。

在相同K=8预算下，用质量加权的视向覆盖选择观测；再做同一语义空间内的归一化加权融合。比较 `last8`、`random8`、`quality8`、`quality_coverage8`、全部观测等设置。全部观测属于不同存储/读出预算，应单列。

选择与融合权重进一步拆成廉价缓存消融：S1a只换观测子集、仍用原vis_area权重；S1b保留last8、只换质量权重；S1c同时替换。这样不会把选择收益与加权收益混为一谈。

首轮不用新增网络。若只有`vis_area`但缺其他观测信息，先跑last8对比面积/方向版本；缺失字段明确记录，不能生成假的quality=1伪装为完整版本。后续新增轻量观测日志，再补全质量分解。

**目标**：先提高语义mIoU/AP，保留早期清晰视图，避免重复视图稀释。固定原始实例几何后，类别无关几何AP原则上不应变化；若变化，说明导出耦合或实例过滤变了，应单独列为M1b。

**接口建议**：新增 `src/static_ovmap/observation_bank.py`、`readout.py`；外部patch只读取`inst_sem_dict`并保持原有产物可追溯。参考核心已实现候选池选择，不等于已实现固定内存在线流式版本。

### M2：具备互斥约束的多视角实例粒度修正

**依据**：LEGO提供多粒度层级组织思路；CompetitorFormer处理查询冲突；OVI本身已有super-point合并，所以“再加一个合并步骤”并不足以形成区别。[P08][P07][B01]

节点选原生预测superpoint/tracklet，不从GT mesh提取。正边来自多次共视、深度一致的重投影重合、相邻表面和稳定whole-object观测；负边来自可信的“同一视图中确属不同完整物体”证据。**不能把任意两个不相交SAM片段设为cannot-link**，因为它们可能是同一柜子的两个部件。

聚合必须在整个连通分量上检查互斥约束，防止A不能与C合并却通过B间接合并。输出part/object层级及固定选择规则；保留原partition，导出whole-object层，而不是按GT逐实例挑最优层。用明确分离的多视角whole-object支持拆开粘连候选；只靠语义相似不合并两个椅子。

**新增建议**：`src/static_ovmap/instance_graph.py`、`hierarchy.py`。参考核心已实现传递性cannot-link聚合，但RGB-D边构建、层级选择和真实split尚未实现；这些是Codex的明确开发任务。

**目标**：减少碎片和粘连，主要提高类别无关AP50/AP75，再带动语义实例AP。过强合并会损害同类相邻物体，过強互斥会保留碎片；必须同时看两类错误。

### M3：单个稠密语言特征分支＋几何约束净化

**优先候选**：GLA-CLIP、VIP或SPAR中选一个；先不用三个同时集成。GLA-CLIP读取类别词表的完整segmentor要拆成分类前特征和query时文本打分。普通DINO仅做空间亲和不能自动代替文本对齐；VIP依赖dino.txt的条件要保留。[P11][P13][P15]

特征低分辨率时，比较双线性插值与AnyUp；AnyUp输入图像需ImageNet归一化，输出保持原特征空间，不能把不同backbone的特征混为一体。[P20]

对原生实例内部的少量局部特征原型，用几何、法向、可见性与owner共同构造稀疏亲和A，尝试一到两轮弱残差：

`F_new = normalize((1-lambda) F + lambda A F)`。

初始lambda可在开发集试0.1/0.2/0.3。这里只借鉴GeoPurify的几何净化思想，不等同复现其教师、18轮传播或学生训练；GeoPurify场景选择涉及语义统计，不能把整个方法说成不接触三维标注。[P01]

**硬约束**：不跨不同可靠实例传播，不让未知owner=0互相平滑，不向未观测空间补语义“真值”；用稀疏边或chunk，禁止N×N全mesh矩阵。

**目标**：小物体、边界与局部语义；收益必须在“同一几何、同一语言模型”的消融中显示，不能全部归因于更强骨干。

### M4：特征成本控制

语义分支有效后，再讨论SPAR蒸馏/单前向替代或EmbodiedSplat的稀疏系数思想，不能在主方法收益未知时先训练压缩器。[P13][P02]

存储采用每对象K个观测＋少量表面原型，不在每个百万级顶点保存高维语言向量。原型能否替代稠密特征须通过准确率—存储曲线验证。缓存key至少含scene/frame、encoder权重、layer、preprocessor和crop规则；缓存后的吞吐与包含特征计算的端到端吞吐分列。

### M5：有预算的难例语义/掩码复核

只对低margin、跨视角不一致或低召回风险对象调用WOW-Seg区域命名，或SAM3/X2SAM补充掩码。设每场景复核对象比例/次数预算；比较“同预算随机复核”，确认收益来自选择机制而非更多推理。[P16][P09][P18]

SAM3视觉提示与拿完整benchmark词表逐类生成mask不是相同设置。后一种可以研究，但需标为query-conditioned建图，不能与类别未知时建图混称。用户现有SAM3记录仅证明mock链路和环境情况，不能证明真实SAM3质量优劣。[U09]

## 7. 最高准确率备选：SpaCeFormer轨道

SpaCeFormer值得单独立强基线，而不是仅作为背景文献。已公开代码将backbone与实例分割模型分开；应检查 `space_former_seg.py` / `SpaCeFormerInstSeg`，不要只拿backbone输出当实例预测。其实例输出包含query logits、逐点mask及1152维SigLIP2语义特征，下游文本匹配和NMS仍要正确接入。[P03]

输入必须是相同RGB-D帧重建的原生点云/mesh，坐标单位和RGB归一化严格依实现；GT mesh仍只在评测投影阶段使用。先比较单独SpaCeFormer与单独OVI，再试互补proposal融合与M1/M3，避免一个复杂组合涨点后不知道是谁贡献。

OVI SigLIP-L语义向量与SpaCeFormer SigLIP2向量不可直接相加；即便两个向量维度碰巧相同，也必须确认encoder/layer/preprocessing一致。多模型集成采用各自text encoder得到的验证集校准分数，而不是跨空间平均。

这条路线使用额外三维学习与伪标注预训练。训练集与评测扫描可能重叠时须做scene级数据范围核验；不能把checkpoint名字中的“open vocabulary”理解为不存在训练数据交叠。若无法闭合预训练数据范围，保留为预训练模型辅助结果，不声称未见场景泛化。

## 8. 分阶段实施与决策

| 阶段 | 执行内容 | 产物/停止规则 |
|---|---|---|
| E0 | 查找已有native基线与manifest，固定origin/main版本、8/18场景和评测名称 | `protocol_manifest.json`、基线表；不重跑已经身份一致的结果 |
| E1 | 同几何、同查询、同编码器的last8对quality/coverage bank | 逐场景配对语义指标；完全无改善则先分析观测pool，不启动重训练 |
| E2 | 几何支持的单语义查询fallback；与E1分开 | 新增实例TP/FP、mask数量、语义增量；不能只看总mIoU |
| E3 | M2多视角边构建、互斥聚合和拆分 | merge/split错误与类别无关AP；效果为负则恢复原partition |
| E4 | 一个稠密语言分支，双线性vsAnyUp，净化开关 | 骨干/上采样/图净化三者分离消融；显存、额外query与时间完整记录 |
| E5 | SpaCeFormer独立轨道或难例复核 | 仅当主要瓶颈仍未解决时扩大模型；预训练和额外预算单列 |
| E6 | 冻结一个组合，跑官方全场景与完整列 | 统一主结果、消融、成本、失败例、交接与GitHub产物 |

不能把Room0继续调参后当未见场景。官方8场景均需报告，另外可以报告不含Room0的7场景诊断；它不能替换官方8场景主表。ScanNet开发条件应独立于论文18场景，并检查同一物理环境的不同scan/session不会跨开发/验证引入近重复。

首轮只需与新增接口对应的单元测试、一个真实小规模smoke和已有相关评测回归。没有理由为了M1重复跑数千个无关动态测试，更不需要做通用安全扫描或与任务无关的压力测试。一次协议检查、一次产物校验、足够的真实实验，比持续追加审计框架更有价值。

## 9. 预期效果：方向、范围与失败条件

以下数字是**在本地基线接近论文、模块在独立开发条件上呈正收益时的合理目标区间**。它们不是模型推导的置信区间，不是承诺；发生零收益或回退完全可能，各模块增益不能直接相加。

| 指标（%） | 论文锚点 | 组合方案规划目标 |
|---|---:|---:|
| Replica semantic mIoU | 26.5 | 30–34 |
| Replica semantic AP50 | 21.2 | 25–30 |
| Replica semantic APall | 8.5 | 10–13 |
| Replica class-agnostic AP50 | 50.8 | 55–61 |
| ScanNet semantic mIoU | 17.5 | 20–23 |
| ScanNet semantic AP50 | 15.7 | 18–22 |
| ScanNet semantic APall | 7.2 | 8.5–11 |
| ScanNet class-agnostic AP50 | 24.0 | 27–32 |

M1的预测可信度相对最高，因为可固定几何验证读出；但若多数实例不足8条缓存、早期特征本就差、质量代理不相关，则收益可能为零。M2对AP75的提升更不确定，强依赖原生mask粒度和几何误差。M3可能改善边界也可能把错误owner内语义一起平滑。T3D可能有更高上限，也可能因输入重建与训练分布偏差下降。

必须同时追踪mAcc/召回；高置信过滤很容易让AP或mIoU上升却损失罕见类别。因此“全面”由最终全部列决定，而不是由上面的目标表宣布。速度不做未经实测的FPS预测；以相同机器、相同冷/热缓存口径，包含前处理、VLM、worker排空和后处理的场景总耗时为准。

## 10. 本轮已经实现与尚未实现

`reference/static_core.py` 是独立原创参考代码，含可观测质量、固定K观测选择、同空间融合、传递性互斥聚合、小图owner/可见性门控净化。`test_static_core.py` 在本环境实际通过8项测试，日志随包提供。

未实现：真实数据缓存适配、完整在线有界内存bank、RGB-D图边生成、实例拆分层级、外部VLM加载、稀疏百万点图、原生mesh导出及官方benchmark集成。没有下载并运行21个模型，没有训练，没有论文新实测分数，也没有向GitHub推送本包。详细Codex开发安排见 `03_CODEX_IMPLEMENTATION_HANDOFF.md`。

## 资料与来源

论文[P01]–[P21]的题名、会议、原文、官方仓库与读到的实现入口，见同目录 `02_LITERATURE_MATRIX.md` 和 `literature_matrix.json`。

- [B01] OVI-MAP论文，主表及附录： https://arxiv.org/html/2603.26541v1
- [B02] OVI-MAP官方README，CVPR2026、版本重构及1 cm更正： https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/README.md
- [B03] 官方重构版语义聚合、投影与canonical打分： https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/utils/mesh_postprocess_utils.py
- [B04] 官方VLM六裁剪与模型映射： https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/vl_models.py
- [B05] 重构版实例Precision/Recall脚本与17场景注释： https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/eval_inst_seg.py
- [B06] 多环境perception worker： https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/perception_worker.py
- [B09] 原始版实例评测，含完整18场景注释： https://github.com/OVI-MAP/OVI-MAP/blob/6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9/scripts/eval_inst_seg.py
- [B10] 原始版同样使用最后8次观测： https://github.com/OVI-MAP/OVI-MAP/blob/6304b5e93c14e66f25a01a64d3f650e0f1ccb0b9/scripts/utils/mesh_postprocess_utils.py
- [U01] 用户静态精细地图证据提交： https://github.com/Orangekostar/oviovo/commit/d5c0688bc662f8e65455cb9c62909de87c941b43
- [U02] Room0开发结果： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/configs/evaluation/results/crove_fine_current_map_v1/table_values.json
- [U03] 已有论文协议检查： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/src/evaluation/baselines/ovimap_paper_audit.py
- [U04] 时序静态锚点： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/src/oviv2/ovimap_static_anchor.py
- [U05] 旧语义提交模块： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/src/modules/semantic_memory.py
- [U06] 旧V2单场景入口： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/run_room0_v2_full_eval.py
- [U07] 用户README和模块历史： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/README.md
- [U08] 原生mask/raycast/bbox约定： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/src/evaluation/baselines/ovimap_native.py
- [U09] SAM3已有真实/模拟运行状态： https://github.com/Orangekostar/oviovo/blob/d5c0688bc662f8e65455cb9c62909de87c941b43/docs/superpowers/reports/2026-06-04-sam3-room0-results.md
