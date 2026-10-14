# OVMap 已有有效模块贡献归因：Codex 最终执行指令 v1

**版本：2026-10-14。任务：完成证据分析工具开发、已有结果复核、有限融合选模、归因计算、真实可视化、候选贡献提名、交接及 GitHub 发布。**

> 执行本任务，不仅返回方案。本文取代旧的《00_投稿前贡献归因与证据准备方案.md》，不调用此前波次的 `all --resume`。用户已经授权向同仓库研究分支正常提交和推送。各条件分支已经定义，不再逐阶段请求确认。
>
> 本轮研究对象是**已有 E / R / S 模块及其已完成对照**，不是发明新算法。E 是 N/Q/F 读出包，D2 只是其中的组合规则；R 是固定表面上的既有恢复；S 是 FG2 表面读出。Wave2 控制器只作为已完成的辅助证据，不再训练、不重新挑其检查点。
>
> **新增训练=0；新增 RGB 特征编码=0；新增 SAM 提案=0；新增几何建图=0。** 缓存概率重组、冻结 head/phi 调用、官方评分、统计重采样和真实渲染均是实际新计算，必须计数。后续未调参场景只登记计划，不在本轮自动采集、编码、建图或评价。

## 0. 权威、阅读顺序与完成定义

阅读顺序：本文 → `01_源码证据与任务绑定.md` → `02_冻结执行配置.json` → `03_交付前审核报告.md`。`source_evidence_manifest.json` 给固定提交、函数、读取范围和已知 Git blob SHA。`04_合同自检.py` 只检查本包数学和任务定义，不能代替真实接入。

源码基点 `BASE=f4245ec0654577634fefb60f3344c6ca9c9335e8`；FG 科学父提交 `FG_BASE=bec4176d9a6784a89ea293a9803e5cf6fab178ec`。不能把历史不同波次的同名指标看成同一实验。本文中新路径与函数名是**必须实现的契约**，不是声称它们已存在。

新分支：`research/ovmap-contribution-attribution-v1`。
新模块：`src/static_ovmap/contribution_attribution_v1/`。
新入口：`scripts/evaluation/run_ovmap_contribution_attribution_v1.py`。
新配置：`configs/static_ovmap/contribution_attribution_v1.json`，复制本包 `02_冻结执行配置.json` 的原字节。
Git 文档／工件：`docs/paper/static_ovmap/contribution_attribution_v1/`、`artifacts/static_ovmap/contribution_attribution_v1/`。
部署保持 `N0_UNCHANGED`，不合并 main，不影响其他任务或进程。

任务完成与论文证据充分分开：全部已定义分析已执行／按规则标明不适用，且已正常发布，可为 `execution_status=COMPLETE`；这不代表 `submission_evidence_complete=true`。本轮不做新的独立确认，所以最终这个字段固定为 false，列出实际缺口，不预判录用。

## 1. 先纠正五个容易导致错误归因的事实

| 已核对依据 | 事实 | 必须采用的处理 |
|---|---|---|
| S01–S03 | A0/A1/A2/A3 已有 E×R 四格；A5 是 F-only＋R | 复用机器精度四格；不得把 A5 当成不含恢复的纯 F |
| S04–S07 | E 同时增加来源和改变融合；N/Q 共享 SigLIP | 不将 Native→D2 的全部差值归给融合权重或“去相关” |
| S10 | 融合开发场景不是 FG 的对象 DEV | 15 权重只在四个旧 BB 开发场景选；不能从只有 FC 的对象库伪造 N/Q |
| S25–S27 | 官方 rank 随预测类别变化；事件非一预测一事件 | 排名交叉使用独立诊断接口；官方匹配事件和唯一对象数分开 |
| S28 | 旧 conveyor 是各 owner 最大几何 IoU，可多对一 | 只能当描述性对应；另算 one-to-one 几何诊断；官方 TP 从原 scorer 取 |

FG2 是方法编号，不是两视图；FG4_ROUTE 不是四视图。预算写 FG2-2V/4V/8V。固定 TSDF 表面不等于固定 owner 分区；R 改变后者，S 的读出比较固定后者。F 的视图预算不等于全系统 RGB-D／N/Q 预算。

## 2. 工作区、依赖与数据角色

### 2.1 有限工作区准备

先读取适用于目标路径的真实 `AGENTS.md`。由当前工作目录解析仓库根和 origin，确认是 `Orangekostar/oviovo`；从 BASE 建独立 worktree。不 stash/reset/clean 其他工作区，不强推。BASE 缺失时只 fetch 对应提交一次。

分支／目录已存在且 task-id、spec 相同则恢复；与无关工作冲突时取最小未占用后缀 `-02`、`-03`，写入 task_paths。实际分支、目录一旦记录，resume 不再次选择。

默认 `TASK_HOME=<当前仓库父目录>/ovmap-contribution-attribution-v1`；worktree 为 `TASK_HOME/worktree`；运行输出 `OUT=TASK_HOME/attempt_001`。OUT 不得等于或位于任何父实验根内，也不放入 Git 工件目录。用户 CLI 明确路径优先。

CLI：`--spec --parent-root --output-root --wave2-root --area-root --compact-root --backbone-root --path-map --gpu --phase --resume`。
`--parent-root` **只表示 FG_ROOT**，不改写继承 binding 中原有 `parent_root` 的语义。
路径覆盖只改变物理路径，不改变数值协议／输入身份；先验证 JSON 原身份再在内存做 PathResolver 迁移。

### 2.2 依赖根的解析顺序

| 根 | 依据与解析方法 |
|---|---|
| FG_ROOT | CLI 优先；否则依次检查 `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`、`/mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001`，接受首个身份匹配者 |
| LR_ROOT | 从经验证的 FG_ROOT/source_binding.json 的 `lr_parent_root` 读取 |
| W2_ROOT | CLI；否则 S34 中 `/home/ww/crove/fg2-branch-utility-wave2-v1/attempt_001`；可用 Git 已发布 records/tables 替代纯分析所需的小记录，不能替代缺失的原模型输入 |
| AREA_ROOT | CLI；否则先沿现有 runtime/evidence-exploration 小 reference 中 `v2_results_root` 解析；最后检查 S01 指明 `/home/ww/ovimap-area-fallback-v2/attempt_001` |
| COMPACT_ROOT | CLI；否则 AREA_ROOT/experiment.json 的 `parent`，已知候选 `/mnt/shared/ww/ovimap-cvpr-compact-tables-v1/attempt_001` |
| BB_ROOT | CLI；否则 S09 的 `/mnt/shared/ww/ovimap-backbone-wave1-v1/attempt_001` |

只读取这些根及它们直接引用的本任务所需文件。不运行 `find /`，不扫整盘，不重验全部历史终态。父路径缺失则列明；允许用户环境的既有挂载恢复，但本任务不得 sudo 改挂载／全局代理／驱动，不以重新训练或重新编码“修复”缺缓存。

S35的本地配置写voxel_size_m=0.01，而原OVI-MAP论文Implementation Details文本写0.1m；这是需登记的协议描述差异，不自动认定哪一方真实运行有误。实际capture/config/上游命令是复现实验身份依据；不能以8.63接近论文8.50证明协议一致，也不因此在本轮重建地图。

机器记录实际可见性不由本指令保证。缺失依赖只阻塞对应阶段；报告、其他独立分析和发布仍执行。不得为了满足计数从相似实验取替代输入。

### 2.3 明确的角色分离

**融合开发 `FUSION_DEV`：**固定顺序为 `scene0056_00, scene0534_00, scene0445_00, scene0626_00`（S10）。使用各自 `BB00_NATIVE` 的同源 N/Q/F、原温度、原支持和 scorer。这四场景是历史开发数据，不是新确认集。

**完整回归 `REGRESSION`：**配置中的 Replica8（8 场景）和 CF18（18 captures／7 物理家族）。保持原顺序。只对已有输出做重组／评价，不重建地图。

**表面识别 `S_DEV`：**原 69 个 base 原件／4 家族，四条件、完整 200 类竞争；`S_TRAIN` 原 622 个 base 原件／24 家族用于既有监督说明和统计，不新增全量 TRAIN forward。其他旧 H/H2/Wave2 是已曝光辅助证据，不变成新的确认集。

**后续确认 `CONFIRM_NEXT`：**仅提名最多 12 个未参与本轮分析的 ScanNet 家族及资源计划，不读其 GT、不推理。无法证明全历史未用时，只称“未参与本轮开发”。

先写 `role_lock.json`，检查 FUSION_DEV 与 REGRESSION 无家族交叉。若真实父清单与固定规格不符，阻塞相应角色而非静默替换。

### 2.4 实际文件合同

AREA/COMPACT：AREA 的 `experiment.json`、`predictions/<scene>/receipt.json`、`evaluation/<scene>/receipt.json`、其 row/pool 路径；COMPACT 的 `resolved_inputs.json`、`contexts/<scene>.json`（不存在才按原规则取 binding.scenes）、`projected_views/<scene>/{manifest,registry,receipt}.json`及必要的 source descriptors。S03 中 `predictions` 是 method→manifest 路径字典，与 FG 的 `methods` 字典不同；新代码分别适配。

BB：`resolved_inputs.json`、`readouts/<scene>/BB00_NATIVE/receipt.json`、同目录 `sources/{N,Q,F}.json`、`config.json`、`evaluation_rows.json` 及其 scores/matches/trace。只用 BB00_NATIVE，不调用其他变体地图重建。

FG/LR：FG 的 `source_binding.json`、`export/weights_registry.json`、`evaluation/store.json`、`recognition/{dev,regression}`、`decisions/<scene>/{FG2_SURFACE,S29_FG2_SURFACE}.json`、`predictions/<scene>/receipt.json`；LR 的 `split_manifest.json`、`class_split.json`、`regression/<scene>/manifest.json`、`recognition/regression.json`。S 的新推理另需 `features/train-dev.json`、原 inputs.npz、raw/projected grids、FC/text/算子。

已有同一步记录：`FG_ROOT/training/seed{17,29}/{FG1_VIEW,FG2_SURFACE}/dev_1000.json` 及 arrays；原 arm receipt、draws/initial-model identity。只读已存记录；缺 step1000 不改用“表现更好”的另一步。selected 结果仍单列。

W2：优先 `conveyor.json`、`repeat.json`、`selection_lock.json` 等；Git fallback 在 `artifacts/static_ovmap/branch_utility_wave2_v1/records/` 下，gzip 同语义支持。复制小记录后在本任务 index 记录“从发布工件读取”，不写父 OUT。

scorer：每个已评分 receipt 的 context.evaluator/exporter/GT/projection 为权威；不能因 Git 中某个 eval_utils.py 看起来相同就换掉。相同 owner_id 只在同一 geometry/payload 域内可比较。

### 2.5 环境与最小FrozenFC上下文（不阻塞纯缓存分析）

`PY_CTRL`优先 `/home/ww/miniconda3/envs/ovimap-map/bin/python`；`PY_FC`读取父 `FC_python`（历史候选 `/home/ww/miniconda3/envs/oviovo-radseg/bin/python`）。GPU默认继承父binding的字符串，CLI覆盖需记录；FC worker设置 `CUDA_VISIBLE_DEVICES=GPU`，使用原 `execution_config(binding)['gpu_lock']`＋queued_gpu_lock，只绑定一张原A40。不循环nvidia-smi寻找别的卡，不升级CUDA/Torch；原FP32、禁TF32/AMP。

纯E/R/评分只用控制环境，不把FC权重当所有阶段前置。S推理前，深拷贝已验证FG binding，仅替换output_root/path_map/gpu，保留reference/assets/scenes/lr_parent_root及旧parent_root原语义，新增字段为task_id/fg_parent_root。复制原 `class_split.json` 到OUT（原字节），在 `OUT/external/upstream_sources` 建向LR同目录的只读用途符号链接；父 features/train-dev manifest原字节引用或复制，不改变其identity。FrozenFC从这些位置读取类别与原算子，缺失不能自动捕获新图像。

加载两份精确head时，迁移路径后的text_identity可能与父registry字典不同：先核对实际text文件bytes/SHA及类别顺序，再在OUT生成仅两head的runtime_registry，保留父registry身份，调用原load_exported。冻结骨干/文本/head全部eval且requires_grad=False。不得仅为绕过身份检查伪造text字段。

S使用普通父ObjectLoader或S29兼容层；不默认构造旧CachedObjectLoader，因为它要求旧格式工程parity工件。需要小LRU时计入总32GiB，而不是另加32GiB。核心历史parity用69个DEV原件的已保存labels/A/M/C；同路径短路严格相等，跨历史embedding沿rtol1e-5/atol1e-6。该基线与S干预原样组共用一次计算，不重复调用。

## 3. 必须实现的任务接口与阶段图

| 新文件 | 职责 | 可复用原接口 |
|---|---|---|
| inventory.py / binding.py | 角色、父件、同域账本、最小运行上下文 | PathResolver、verified、ConsumptionIndex；**不调旧 binder** |
| sources.py / selection.py | 8 固定读出、15 权重、锁定普通权重 | S04 pool/simple_dependence；不调旧 step>=250 选择器 |
| payloads.py / scoring.py | 固定 owner 输出、实际 scorer、动态池化 | PredictionPayload、native_ranks、S24/S25/S26 数值函数 |
| history.py / recovery.py | E×R 原精度归因与 R 匹配 | S02 支持约束、原 matches/trace |
| surface.py / interaction.py | matched-step、对应干预、S×融合 | FGHead、FrozenFC、已有 F 向量和新纯 F payload |
| attribute.py / rank.py / stability.py | 互斥错误路径、rank 四格、LFO/bootstrap | S25 manifest/scoring内核、S27 trace；不改原规则 |
| visualize.py | 真实案例和图表，固定样例规则 | 保存的 RGB/pose/预测/几何，非图像生成 |
| nomination.py / reporting.py | 证据分级、最多两个候选、后续确认计划、主表 | 只消费锁定结果，不重新调参 |
| runner.py / publication.py | 单 GPU 调度、恢复、真实退出和 C1/C2 | 原锁与标准 Git；不复制旧整阶段图 |

允许合并纯内部工具文件，公开阶段、记录与职责不变。

公开 phases：`preflight, inventory, history, fusion_dev_predict, fusion_dev_score, freeze, main_predict, main_score, surface, attribute, rank, stability, qualitative, nominate, report, publish, all`。

all 顺序如下：
1. preflight/inventory/history；历史已评分标签可复查，但不能用于选择权重。
2. fusion_dev_predict 锁定四开发场景的全部 15 权重输出 → fusion_dev_score → freeze 权重。
3. main_predict 锁定 26 场景所有已激活新配置 → main_score。不存在“涨点才准评分”的科学门槛，这是归因实验。
4. surface、attribute、rank、stability；依赖分支独立，不因权重开发缺失而跳过所有 E/R/S。
5. qualitative、nominate、report、publish。零增益、部分受阻时也执行后四者，真实标明缺项。

实现新入口并核对 `--help` 后，使用T0记录的真实路径运行：

```bash
"$PY_CTRL" -u "$REPO/scripts/evaluation/run_ovmap_contribution_attribution_v1.py" \
  --spec "$REPO/configs/static_ovmap/contribution_attribution_v1.json" \
  --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU" \
  --phase all --resume
```

有显式父根／path-map时追加对应已支持参数；`REPO`是新worktree绝对路径，不是owner/repo字符串。该命令是**开发后的验收入口**，不是声称当前仓库已有该脚本。

显式单 phase 缺前置就报缺失，不暗中训练、编码或调用旧 all。发布由外部调用者在科学控制进程真实退出后完成，详见第15节。

## 4. 贡献账本与历史四格：先复用，不重新跑已完成实验

### 4.1 每个结果单元必备字段

`experiment_ref, source_root, cohort, ordered_scenes, family_ids, geometry_identity, owner_partition_digest, semantic_digest, rank_digest, source_model_and_text_ids, source_availability, request_view_scope, supervision, temperatures, recipe, seed_step, scorer_context, pool_identity, exposure, measurement_kind`。

measurement_kind 仅取 `REUSED_EXACT`, `NEW_CACHED_ANALYSIS`, `PAPER_REPORTED_NOT_MATCHED`, `MISSING`。路径、标题、舍入分数相同不是 alias 证明；只有 payload/投影/真实类别表/rank/scorer/有序输入相同才复用评分。

### 4.2 原四格角色必须精确

从 AREA 真实 method 键读取：A0=`CT_A0_NATIVE`、A1=`CT_A1_E`、A2=`CT_A2_R`、A3=`CT_A3_ER`。这些精确键已由S35确认；同时核对A0的recipe确为 `existing=NATIVE,recovery=NONE`。实际父spec不同就报告协议不一致，不按相似名字替换。

A4=`CT_A4_NATIVE_RECOVERY`，A5=`CT_A5_FC_ONLY`，U2=`CT_H_U2`，G3=`CT_G3`（S35）；A5 包含与 A3 相同的 G1 恢复。逐级核对 A3→EV01_G1_V2→IR01_G1→FG REF_G1 的 owner/semantics/rank/geometry，失败则不同域分别报告，不强行串成收益瀑布。

只从机器精度 pool receipt 读取五指标 fraction；历史 Markdown 的两位小数不进入正式计算。缺池但该 cohort 的全部 scene matches/receipts 完整，可在本任务 OUT 用原 pool 重建；这是完整历史池重建，单列 RECONSTRUCTED_HISTORICAL_POOL，不训练/不重新出预测。

每个指标：
`E_add=P10-P00; E_cond=P11-P01; R_add=P01-P00; R_cond=P11-P10; I=P11-P10-P01+P00`。
可另报 `mean_marginal_E=(E_add+E_cond)/2`、`mean_marginal_R=(R_add+R_cond)/2`，两者和等于总增量。它们是所定义系统四格上的描述性边际，不是自然因果份额；不得再把 I 加到两个平均边际上。

## 5. 模块 E：固定来源消融和有限普通权重选模

### 5.1 固定 8 读出

在原 A0/D2 incumbent 分区，使用**历史同一份** N/Q/F scores、原每场景温度、完整 valid_ids。N 与 A0 的 native 导出标签可能因原 canonical/fallback 不同而不一致，必须检查；不能预先认定 E0=N 就精确等于 A0。

| ID | 算法 |
|---|---|
| E0 | N-only |
| E1 | Q-only |
| E2 | F-only（历史 F，不是新 FC8） |
| E3 | N/Q 可用来源等权 |
| E4 | N/F 可用来源等权 |
| E5 | Q/F 可用来源等权 |
| E6 | N/Q/F 可用来源等权 FC_EQ |
| E7 | 原 D2：原 pool＋simple_dependence('PE_D2_GROUPED')；全缺源沿用原生类别 |

E0–E6：先各路 `softmax(scores/T)`，仅在指定且 available 的来源中等权；指定来源全部不可用时 label=0（unknown），owner mask 仍保留。缺源不能补零 logits、不能偷偷调用 N 回退。数据声称 available 却非有限／长度不等于词表是数据错误，不把它当不可用。

E7 要复现真实父 D2，包括全缺源回退、原 dtype/温度与 argmax 顺序。一个对象中“source 分数可用”与“native label 存在”是不同事实。E6 全缺源时是未知，不保证与历史 fuse_readout('FC_EQ') 的 native fallback 相同；若因此不同就新增评分，不能误标 alias。

新 E payload 浅继承原几何、owner_ids、非incumbent背景；对 incumbent 区域写新类别（含0），用**原 native_ranks**生成当前类别 rank 后 lock。不要调用拒绝未知标签的旧业务 guard 来伪造正标签；可以新建极薄输出构造器，但数值 mask/排名规则不变。

### 5.2 普通权重对照 E8

权重固定为 `{(i/4,j/4,k/4): i,j,k为非负整数且i+j+k=4}`，共15组。各组只使用权重>0且available的来源，按可用权重和重归一化；有效权重和为0则 unknown。冻结原温度，不新增校准。

FUSION_DEV 只有四场景 **BB00_NATIVE** 的完整实际 N/Q/F、geometry、GT/scorer都可读，才执行15组开发池。先锁定全部四场景、全部权重预测，再读取标签评分；不按场景先挑赢家。

唯一选择顺序（fraction，容差1e-10）：`APall降序 → mIoU降序 → 非零来源数少 → 到(0.25,0.25,0.5)的L1距离小 → (wN,wQ,wF)字典序升序`。APall或mIoU未定义的候选不入选；全部未定义则无赢家，原因明确。

写 `weight_selection.json` 与不可变 `weight_lock.json` 后才进行新 REGRESSION 候选评分；未选中时freeze仍写status、null权重和原因，并将E8排除，不能因不存在赢家而缺整个锁文件。不能给Replica/CF18分别挑权重，也不能看S结果后换权重。历史回归已经曝光，如实声明这是新开发选择后的已知回归检验，不是首次未见测试。

缺任一开发场景或原同源上下文时：`E8=NOT_SELECTED_NO_FUSION_DEV`；不在CF18/Replica/69对象上改选，不跑15组全回归“固定曲线”再挑赢家。其余8固定读出照常运行。

完整三源时，D2与普通(0.25,0.25,0.5)**数学等价**；只有相同最终label/rank/scorer身份才复用物理评分。缺源时二者可不同，按8种availability pattern分别报告。不能宣称D2在数学等价子集优于它自身。

### 5.3 来源作用的定量描述

每源单独正确率只在相同的 `ALL_THREE_AVAILABLE` 对象集比较，再另给全输出未知／覆盖结果。对象标签用固定A0支持对GT的最大几何IoU>0.5定义描述性match；0.25<IoU<=0.5及不匹配者单列。不得把这批多对一匹配对象当官方TP。

预测分层固定为：ALL_AGREE；NQ_AGREE_F_DIFF；OTHER_DISAGREEMENT。类别和availability先于GT分层。每层给N/Q共同错误、F纠正比例、融合纠正/误伤、NLL、Brier、margin、原件/家族计数。NLL日志允许 `max(p_y,1e-12)`，记录该保护；不能回写预测概率。Brier为全词表概率对one-hot的平方和，不对类别数再归一化。

报告E7−A0是**证据包总增量**；E7−E6及E7−E8才接近融合规则增量。若只加F就解释了大部分收益，结论写异质证据增量，不把Q路径收益写成Q_GAIN优于随机策略（该问题本轮没有新的同预算查询对照）。

## 6. 模块 R：复用恢复四格与匹配轨迹

A0/A2、A1/A3 的表面mesh必须相同；恢复前incumbent的owner支持不得变化。新增域为 `(raw==candidate) & (A0.owner_ids==0)`，按原registry逐个核对。恢复的原始owner/行集合来自预测侧，不从GT补几何。

分开统计：源可用候选、target投影后>=100点可评价owner、unknown→known的目标点数、增加的实例数、官方definite TP/FP score-entry、新增unique GT被匹配数、歧义ties、旧TP保留/丢失。原trace重复事件不计为多个独立对象（S27）。无官方trace时只能报告其他可测项，不制造TP/FP数字。

**补充几何one-to-one诊断**：从原投影owner数组与原GT instance数组构造交集/并集；GT合法性、实例类范围和minimum=100按实际scorer上下文。预测包含所有投影>=100点的正owner，包括semantic unknown；预测0不是实例。GT类别不用于配对，仅筛合法GT集合。

对阈值0.25/0.5/0.75采用严格 `IoU>threshold`。在按owner/GT id排序后的二分图，优先最大匹配数，再最大IoU和；可用补dummy列的线性指派，合法边权为 `min(num_pred,num_gt)+1+IoU`，非法/dummy为0，求最大和后只保留合法边。tie保持固定库/输入顺序。报告Recall/precision/count，名称 `DIAGNOSTIC_ONE_TO_ONE_GEOMETRY`，绝不称official AP。

对E的所有固定owner方法，该几何诊断应相同；R可以改变。发现不一致先检查unknown过滤/投影/分母，而不是宣称语义改善几何。

原A4原生恢复、A5 F-only＋同恢复、U2旧恢复成本和IR02/IR03已有对照均按原条件列出。不同恢复候选/观测/操作库不是纯替代消融，明确comparability字段。无新恢复推理。

## 7. 模块 S：既有FG2作用和对应依赖

### 7.1 同一步与已选检查点分开

先读取 seed17/29 的 FG1_VIEW 与 FG2_SURFACE **step1000** 的原 DEV records/arrays，按 object_keys 和 class_id 配对，使用原 summarize 得 A/M/C与净修正。核对同seed初始模型与draw schedule、原数据、完整200类竞争。

只存在各自selected结果时，把selected表如实保留，matched-step为 `MISSING_STEP1000_RECEIPT`；不偷偷换步。dev_1000.json存在不代表step1000权重存在，所以不得拿selected.pt在地图上冒充统一步数实验。FG3作为已有强参照，FG4和Wave2只做历史上下文，不新增forward训练。

FC8→FG2是监督＋结构的包增量；FG1→FG2更接近分组设计差异。两者不可混称。

### 7.2 有限的对应干预（唯一授权新FG head推理）

只用S_DEV的69个base原件、四条件、prefix8、两个**原selected** FG2。权重：seed17键FG2_SURFACE/step1000，SHA `de7fc82edba5440d9c338f5253ecd3600a340271089600c9ff2d286e2711d963`；seed29键S29_FG2_SURFACE/step2000，SHA `501a6ad010fd2951c93a7638fce94c599b994c7fe9215adad82054919a84ea77`。

FC identity `6132b1b1da69a7a3bcd650586f0bc7e9aea2b64d8183a004b1a61c39892c8b44`；text SHA `8e3a532d4945493f7ecaab63168fab9232c8d9d0693c717893c7f9b73a88c4b4`。原文本类别顺序也须一致。路径迁移后先验text字节，再在OUT生成运行registry，不直接绕过字典身份检查。

复用 S29 的纯loader兼容层，或其父ObjectLoader；不启动 capture/capture_dataset。原四条件可能对应不同site，随机键必须含condition。固定5个shuffle seeds `[101,211,307,401,503]`。

对每个view，只在该view的valid slot索引内，使用同一个排列联合置换 `(local_raw,local_unit,metadata)`。valid布局、view_valid、fc_vectors、frame_ids、有效token多重集保持原样；每view的排列独立，破坏跨view对应。随机生成器用SHA256 `(task,key,condition,view_frame_id,shuffle_seed)` 派生，不用Python进程hash。无效padding不参与。

共同slot置换只在2个TRAIN工程对象做：**所有slot同步置换，包括local_valid与三项局部输入**，用同一映射保持物理组，只检验等变/输出近似恒等。它不是前述各view不同valid排列，不能混淆。浮点求和次序可不同，使用原历史容差rtol1e-5/atol1e-6；不要求这种重排序逐位相同。未改变输入的短路同路径仍要求逐位相同。

新调用预算：DEV至多 `69×4×2×(1+5)=3312` 次原head，工程样例另列，合计硬上限4000。不重跑全部TRAIN，不为打乱后不下降反复换随机种子。报告5次各值和均值，不选最好/最差。新phi和grid重池化分别计数。

推理破坏可能产生分布外输入；下降只说明当前模型依赖原对应，**不能证明**随机分组重新训练必然失败。5个shuffle种子不是5个模型或独立数据集。无真实grid/head依赖则S推理阻塞，历史S表和E/R仍执行。

### 7.3 S×融合的两组四格（只使用已有对象向量）

固定G1最终分区和LR regression同一个eligible owners列表。非eligible原件、恢复对象与背景全部取 **FG0_FC8的固定类别**，四格均相同；不把纯F干预域扩成全地图。

每seed的四格：`F=FC8/FG2(seed)` × `readout=F_ONLY/D2(N,Q,newF)`。D2三格（FC8、FG2种子17/29）已经有真实完整输出，应严格alias复用；本轮只新建三个F_ONLY配置：J0_FC8_FONLY（两seed共享）、J1_FG2_S17_FONLY、J2_FG2_S29_FONLY。

用LR `recognition/regression.json`中 `LR04_FC_8` 的scores和FG同文件的 `FG2_SURFACE/S29_FG2_SURFACE` scores；按owner、数组methods、valid_ids配对，不按偶然行号。Replica使用其原context文本顺序，不能套用TRAIN 200类；此阶段不需要重新编码或调用head。

F_ONLY仅在eligible域取F argmax；N/Q/F沿原pool及PE_D2_GROUPED；同一newF应来自同arrays。missing eligible数组时阻塞对应整格，不把该owner保持旧标签后假装完成。缺源规则与历史F_ONLY不是同一个任务域，J系列必须清楚命名。

计算每seed两个条件S增量及交互；只解释这个干预域。新控制器不进图，不因旧门槛未通过就补跑它。

## 8. 评分与计数合同：主表、诊断和开发搜索分开

### 8.1 最多12个新主评分配置

E0–E7八项＋可选E8一项＋J0/J1/J2三项＝最多12项。每项对完整26场景，最多312条逻辑新配置场景记录、24个新cohort池。即使与历史精确alias，这个逻辑格仍存在，但物理评分计数为0。

原A0–A5/U2/G3/FG1/FG2/FG3/REF表只引用，不挤进“新增算法”列表；历史缺池的matches-only重建另计。15权重只在4融合开发场景，最多60条开发逻辑行、15池。rank交叉最多两对×两个非对角×26场景＝104条附加诊断行，不能与主结果混用。主正式score核对每cohort首场景的原A1一次，共2次工程评分；不再另加多轮scorer pilot。

新主方法的全部可运行场景预测先锁后评分。某角色缺场景时，其他场景可以保留PARTIAL结果，但不得生成缩小cohort的“完整pool”；paired整图结论为UNASSESSED_PARTIAL_COHORT。不能因为某预测数值恰同原输出就跳过所需的输入身份核对。

### 8.2 官方评价适配

复用 S25 `SceneEvaluator`、S26 `released_pool_with_classes/fraction_metrics/equivalent_scoring_input` 和父兼容层。新建动态任务调度，不直接调用旧run/evaluate_scene硬编码286/11/22的程序。实际必要的兼容补丁以冻结父 `partition_evaluator` 已有实现为准，不修改原上游score规则。

`SceneEvaluator` 的mask registry需要枚举所有正owner，即使label=0；它不意味着每个owner有可用N证据。官方manifest排除unknown，但语义confusion保留预测unknown对有效GT的错误。对于小owner也保留完整语义输出，不能裁掉使mIoU提高。

原IoU向量从scorer载入并用父validate_overlaps核对；不按自记AP定义另实现。类缺失沿原null/nan语义，分数只保留fraction，显示列×100。全图语义confusion.sum必须等于有效GT target总数。

全池用原ordered pool，不平均scene AP。prediction身份与scorer的真实context/实际view一致才复用。必要的路径迁移先验证原bytes，再生成新索引，不保留被改写JSON旧identity。

## 9. 定量归因的三个标签系统必须分开

1. **官方TP/FP/FN与AP：**原 released matcher及trace，含duplicate/tie，不能把events条数等同于对象数。
2. **描述性对象对应：**每预测取最大几何IoU（S28），只用于统一对象上的类别正误与错误路径，可多对一。
3. **唯一几何匹配：**第6节的one-to-one诊断，类无关，不能代替官方实例AP。

### 9.1 从F到地图的互斥路径

对固定G1域比较 FG0_FC8→FG2(seed)，先给全部正owner记录是否eligible；not_eligible类别单独，不能删掉。对eligible owner按如下固定顺序分配一类：

A. `NO_VALID_GT_REFERENCE`：无合法GT可定义。
B. `GEOMETRY_BELOW_025`：最大IoU<=0.25；附F/融合是否同最大GT类别的辅助标签，不判作成功识别对象。
C. `GEOMETRY_025_TO_050`：0.25<IoU<=0.5；单列低几何支持。
D. 在IoU>0.5中，old fused正确而new fused错误 → `HARMED`。
E. old fused错误而new fused正确 → `CORRECTED`。
F. new F正确但new fused错误 → `F_CORRECT_FUSION_WRONG`。
G. new fused仍错误 → `FINAL_STILL_WRONG`。
H. 剩余 → `REMAINED_CORRECT`。

另保存 `F_wrong_to_right`、`F_right_to_wrong`、`final_label_changed`、official新增匹配事件等非互斥字段。HARM/CORRECTION优先，避免一个对象同时落多主类。上面“正确”指描述性GT类别，不是官方TP。

对E定义相同对象参考下的来源修正路径；对R用新增/原incumbent两张表。每表有所有owner、可评价owner、eligible、point覆盖四个分母；没有读取到完整域不能只发布成功子集。

### 9.2 模块归因输出

`gain_attribution`覆盖E×R五指标、S×融合每seed五指标。保存逐scene、逐family、完整pool和per-class。报告条件增量、交互、误伤和适用范围，不分配“百分之多少创新来自某模块”。

所有统计函数基于原记录或新锁定输出；不得从Markdown倒推小数填入正式表格。机械选出最高分只代表开发选择，不证明新颖性。

## 10. 标签／排名交叉：独立诊断入口，不滥用旧rank_mode

固定两对：①E6→E7（同A0分区）；②FG0_FC8→FG2_SURFACE seed17（同G1分区）。本轮不因某另一对更好看而更换。

四格 `D_ab=AP(labels from a,ranks from b)`。两个对角复用原official结果。两个非对角由新函数 `score_explicit_view(...)` 直接构造诊断manifest，再调用原 `assign_instances_for_scan/evaluate_matches/compute_averages`，或必要的原trace函数；输出 `rank_mode=DIAGNOSTIC_CROSSED_LABEL_RANK`。**旧SceneEvaluator.evaluate只允许两个预设mode，而且OFFICIAL模式会重新算rank，不能拿它评分后谎称保留旧rank。**

排名字典来自该pair两端完整payload.instance_ranks，按owner ID配对；非对角每一行的label决定是否纳入manifest，rank从另一端取，按原六位小数格式。原positive owner在另一标签中为unknown时，若完整payload已给0 rank可按其0值使用；若根本没有该owner的rank则该pair为UNSUPPORTED_MISSING_CROSS_RANK，不根据GT补rank、不换pair。owner集合或几何不同也不运行此pair。

非对角语义confusion只随label轴变化，必须与对应label端相同。对角parity先用新函数构造显式view，与旧官方view逐项相等后直接引用原receipt；已有matches的未观测evaluate_matches可作内存数值parity，不新增scene assign/mask评分。若view或AP不一致则停止rank阶段，其他主结果保留。

用第4节同样的平均边际式描述label/rank贡献，interaction单列。不同标签导致不同类别FP排序，不可把rank分量直接称作弊或无意义；只是区分评价路径。

R新增owner没有合法旧rank，**本轮不做R的交叉rank**；由新增FP/TP与rank插入轨迹解释，避免人为给新实例造旧confidence。

## 11. 统计稳定性：重用matches，不为重采样重跑模型

### 11.1 固定比较与LFO

比较：A1−A0、A3−A1、E7−E6、FG2(seed17/29)−FG0_FC8。五对在可用完整cohort上做leave-one-family-out，共每对8+7次减家族池化；所有原family结果保留。

CF18同一family的captures一并移除；Replica每scene为family。paired方法使用相同删留集合。报告完整Δ、每次LFO Δ、正值比例、极端变化，不能把某家族删掉后的最好分当主结果。

### 11.2 两个预先固定pair的bootstrap

只对E7−E6、FG2(seed17)−FG0_FC8运行各cohort 2000次family-level配对bootstrap，种子4314。每次有放回抽取与原family数相同的family，保留该family全部captures；方法两边完全同抽样。四组共8000次重采样对，不再按“哪一对显著”挑pair。seed29单独原值及LFO，不当第二套独立数据。

**不要调用拒绝重复scene的旧pool(root,...)去处理bootstrap，也不要平均scene AP。** 在内存读取原matches，按抽样复制scan域，给每份重复scan及递归出现的prediction filename独立namespace，保留score、count、intersection、GT id语义；调用原 `evaluate_matches` 和 `compute_averages`，加总对应confusion求mIoU。不在每draw创建真实mask文件，不每draw运行trace observer三遍。

实现前只做一个原始完整序列的数值parity，容差1e-12；以及一个小重复family检查文件名隔离。受控重采样不会产生新预测，只改变评估样本频数。不能用一个scene重复覆盖另一个scene的字典键。

输出paired Δ的2.5/97.5分位数、defined_draws、每draw有效GT类别数量范围。缺类draw沿原scorer的undefined/nanmean语义，不填0；报告这种区间的目标集合随抽样变化。整draw某指标无定义则该指标只统计defined并报告排除数量，不能反复重抽直到好看。

若matches不足：LFO可在完整scene清单上由既有manifest重算，bootstrap为 `BLOCKED_MISSING_MATCH_CACHE`；不改成scene-AP bootstrap冒充同指标。区间都是已曝光数据上的探索性不确定性，不能消除历史选模偏差，不用“显著”替代新颖性论证。

## 12. 定性结果：真实渲染与固定案例选择

qualitative阶段先调用第13节的纯shortlist函数并写入 `shortlist_prelock.json`（此时全部所需统计已完成），nominate阶段只复用该选择、不根据图像好坏更改。最多选两类模块，每类6个案例：2个成功、2个失败/误伤、2个无变化。无足够案例就少画并报告该类别不足，不调阈值补足数字。模块E的case pair固定E6/E7，S固定FG0_FC8/FG2种子17，R固定A1/A3；若选E_SOURCE则A0/A1。

分类来自锁定对象变化；每层以有效GT margin变化的绝对值距该层中位数最近排序，随后cohort固定顺序、family、scene、owner ID排序；R没有可比margin时用IoU距层中位数。优先不同family，直到该层目标2个；清晰度检查只允许缺图/遮挡到完全不可见等明确原因，记录每次跳过，不能因效果不够漂亮换例子。

每案例相同camera/尺度/颜色映射，显示真实对象支持、去模块/加模块输出、源分数或表面对应、GT仅作评价参考，附原class/newclass/IoU/rank/官方匹配状态及ID。无许可发布原RGB时省略该面板，使用允许的派生几何图；不上传原扫描或原始GT图。

优先保存的RGB/pose/预测快照或现有headless渲染环境。没有Open3D渲染能力时可用保存xyz和camera做真实点云投影的matplotlib图，注明POINT_PROJECTION；不安装大型GUI，不用图像生成模型生成/修补实验图。缺几何时只能输出真实数值证据卡，标 `VISUAL_BLOCKED_GEOMETRY`，不能假称有空间定性证据。

每case只保留一个组合PNG；另给四种统计图：E/R增量及交互、来源错误转移、S匹配步/对应干预、预算与实际成本。图表来源必须可由JSON重建；所有失败/unknown区域保留，注意力热图不是因果证明。

## 13. 选模、贡献分级与后续确认登记

### 13.1 本轮“选模”的确切含义

①按第5节选择一个普通融合权重；②FG权重/控制器不再训练或重选；③从E_SOURCE、E_GROUP、R、S的已有方法中选最多两个**后续确认优先候选**。这不是部署升级，也不是强行保证两个创新点。

`CLAIMS_LEDGER`对每项给三个独立字段：
- effectiveness：`POSITIVE_OBSERVED / MIXED / NO_POSITIVE_OBSERVED / UNASSESSED`；逐指标、逐cohort、逐seed保留符号，不要求所有格全涨。
- attribution：`PACKAGE_ONLY / MATCHED_CONTRAST / CONDITIONALLY_MATCHED / UNRESOLVED`；OOD打乱不能单独提升为matched结构对照。
- novelty：`KNOWN_EQUIVALENT / CONCRETE_DIFFERENCE_IDENTIFIED / UNRESOLVED`；附近邻原文位置、本项目新增操作及对照。数学等价不写首创；没有检索证据就UNRESOLVED。

表现与对照：E_SOURCE主对A0/A1、没有单独新颖性差值；E_GROUP主对E6/E7，普通替代E8优先（若已锁）否则E6；R主对A1/A3，A4与U2按原可比域辅助；S主对FG0_FC8/两seedFG2，结构对照固定step1000 FG1/FG2，对应干预仅补充。

### 13.2 可重复的优先排序

每候选只用可比且finite格，计算：specific_positive_cells（强替代/结构同条件差的正格数）、primary_positive_cells（主指标APall正格数）、对应主pair的LFO正值比例、主ΔAPall的cohort等权均值。数值比较tol1e-10，缺统计按−1排序而非补0。

E_GROUP/R的specific格限可比整图AP；S的specific格来自两seed匹配step1000的DEV A，注明与整图不是同指标，**只作排序优先级不混成性能分数**。S主ΔAP仍用两个seed×两cohort各格，cohort内先seed均值。

排除完全无正主格的项目；S若主AP无正但同budget结构A有正，也允许作为“识别限定候选”，不能叫map主贡献。排序依次以上四量降序、固定 E_GROUP→S→R→E_SOURCE 破平局，取最多2个，写 `shortlist.json`。缺必要对照时可以列备用，但status=EVIDENCE_INCOMPLETE。若都无支持，shortlist为空，完整分析仍结项。

命名 `SHORTLIST_FOR_CONFIRMATION` 而非 `PROVEN_CVPR_CONTRIBUTION`。KNOWN_EQUIVALENT的E_GROUP可以作为有效实现保留，但不能仅因排序靠前标CORE_CANDIDATE；最终核心说法必须同时写明具体区别和尚未确认范围。普通权重胜出时如实保留为强对照，不为保住D2重调温度。

### 13.3 外部近邻对照

阅读本包01中6个已核对的原始来源；输出E/R/S各至少一张“原文已有内容—本项目新增操作—数学等价/差异—所需对照—已完成结果—未解决问题”的表。

本轮不启动外部大模型复现，不进行无限文献搜索。若某具体新颖性仍需补充定位，只围绕该说法做最多4个额外原始来源检索。无法联网时使用已附来源与保守UNRESOLVED，不能引用二手AI总结当作原论文事实。

### 13.4 确认计划只登记不执行

需要一个具体 `CONFIRMATION_PLAN.md/json`：锁定候选配置/源权重/词表/骨干/对照、主指标、预先场景列表（可取得时）、真实提案与heldout类别评测、视图预算与全系统输入区别、资源需求和缺失项。

只读取现有官方ScanNet train队列（从LR既有upstream sources manifest定位）及有限暴露清单：FG/Q/控制器训练与选模roles、FUSION_DEV、REGRESSION、old_H、H2、已存在acquisition lock。将scene以`sceneXXXX`归family，排除所有已列家族；按SHA256(`OVMAP-CONFIRM-20261014|family`)排序取前12，每family取queue中capture号最小者。只锁名称，不打开其annotations，不检查预测结果再替换；少于12就照实列不足。

官方队列或暴露记录缺失则名单为null、附需要的精确输入；不能猜12个场景。只证明本轮未用时如实限定。`confirmation_execution=NOT_RUN_BY_CONTRACT`，无新增下载/编码/建图权限，不自动改成运行阶段。

## 14. 有限资源、恢复与四组测试

### 14.1 资源硬边界

单GPU沿父FC_python/原GPU lock；只S干预和必要真实接入使用GPU。CPU评分最多3进程，各BLAS线程1；scene assign/score、matches-only重算、pool调用分别计数，后两者不隐瞒但不冒称新增scene预测；主机缓存总上限32GiB。OUT启动空闲>=10GiB，5GiB留存底线，达到则停止新增大工件而不删除用户数据。

上限：12主配置／312主行／24池；15权重／60开发行／15池；104非对角rank行；2次额外scorer基线检查；4000次新冻结head；最多12定性案例；bootstrap固定8000 draw-pairs。

不得新增优化器、RGB特征、SAM、TSDF／实例生成、AnyUp、SAM-V或第三种门控。发现缺真实输入就受阻，不用随机输入填科研结果。已有大型score/trace缓存读取成本不称为0；不要把每个嵌套计时加总成端到端耗时。

### 14.2 恢复身份

每analysis unit的key包含：task/schema、该阶段numerical producer digest、spec参数、直接父inputs/SHA、role/method/seed/step/prefix、源available/词表/温度、owner/geometry/rank/scorer范围。报告文字修改不得导致重新跑神经网络；数值修复只失效受影响下游，保存旧单元和失败原因。

写临时文件后原子replace；相同身份且完整的输出直接复用。manifest包含绝对路径迁移时，原内容先验证，再生成本任务新描述。不同身份不能覆盖旧“COMPLETE”。一次有界恢复核对即可，不嵌套生成无限审计目录。

### 14.3 只做四组必要测试

T1：来源概率／15权重计数／完整源D2等价／缺源规则／全缺源unknown与D2 fallback不同；权重锁定与无开发集路径。
T2：E固定owner、R仅新增合法raw支持、J固定eligible域、unknown语义/词表、S联合token排列与有效数；真实head短路及两个工程对象的排列检查。
T3：四格与平均边际恒等、S×融合逐seed定义、官方pool原精度、一个rank对角parity、重复family命名空间、三种匹配口径分离。
T4：有限清单/alias/中断resume/预测先锁/选模不看回归/负结果仍发布/CLI及相对链接；实际发布用真实Git结果，不用模拟push冒充。

正常一次完整四组，修bug后仅重跑受影响组和一个最小实际回归。禁止全仓pytest、漏洞扫描、驱动升级、多GPU压力、整盘hash或无穷随机测试。代码/JSON的轻量校验不替代正式分析结果。

## 15. GitHub发布、状态与真实退出

必须上传：新源码、入口、四组测试、冻结spec、本包原指令/证据绑定/审阅说明、RESULTS/HANDOFF/REVIEW/CLAIMS_LEDGER/CONFIRMATION_PLAN、三张主表及归因细表、选择全15记录及锁、适用的FG同step表/5干预、rank/统计/成本、定性图、精简重算所需记录和artifact_index。

大mask/mesh/RGB/原始GT、基础模型/文本权重、环境、凭证、优化器都不上传。新模型权重不存在，不造checkpoint。小头仍可引用固定FG_BASE导出文件。Git每新blob<=40MiB，总新增任务工件<=150MiB；结果按scene gzip分块，必要时移除冗余中间张量，不能删正式方法/scene/负结果以压体积。

必须保存紧凑重算信息，不能只放私有OUT路径：scene/cohort/method/来源score及availability和有效ID映射；每class/IoU的y_true/y_score/hard_FN/has_GT/has_pred及confusion；规范化对象错误路径；原精度指标、selection、图表来源。重复完整matches超过预算时保留OUT并给bytes/SHA/再生成命令，Git中的PR充分统计需对原AP校验；重bootstrap仍以有完整matches的OUT为准，不能声称从不充分字段可重做全部几何匹配。

### 正常双提交

1. 完成所有能执行的科学阶段，记录每worker真实退出、负结果／未运行原因。外部调用者先观察科学all真实退出，写 `OUT/execution_observed.json`；负科学结果可退出0，工程/依赖未解决返回非零。
2. 生成报告、紧凑工件、一次成品审阅。将**数值源码**digest与报告源码digest分开。此时不预写将来的Git SHA。
3. 只stage本任务命名空间、入口、测试、config、docs/artifacts；`git diff --check`及一次新增文件大小/路径检查。创建 C1（代码＋结果）。
4. 正常 `git push -u origin HEAD:refs/heads/<actual_branch>`；`git ls-remote origin refs/heads/<actual_branch>`完整SHA必须等于C1。
5. 在Git内 `PUBLICATION_RECEIPT.json` 只证明已发生的C1发布，含C1/remote、数值源码、spec、artifact index及已观察的科学退出。更新HANDOFF中的C1永久链接，创建仅报告／收据变更的C2；正常push并核对remote=C2。
6. C2核对后在 **OUT外部**写 `publication/final.json`，记录C2、本地/远端一致与最终状态，不为将C2写进自己的commit而继续循环提交。

公开 `--phase all` 由协调父进程按上述方式调用科学子进程、观察退出、执行report/publish；它自己的最终退出由Codex/shell外部记录。不能让all尚在运行就伪造其退出码。显式publish仅消费完成工件，不启动科学worker。resume若C1/C2已存在且所有输入和远端一致，不新增worker、推理或提交。

正常网络错误最多一次重试；无权限／错误凭证／远端冲突保留本地结果，`PUBLICATION_BLOCKED`，不改全局身份代理，不force。远端有无关工作可按任务后缀另开分支，不覆盖。

### 状态三轴

- execution_status：`COMPLETE / PARTIAL_DEPENDENCY / FAILED_ENGINEERING`。指定可选E8因无融合开发集未选，是定义内分支；缺基础四格、正式固定E0–E7、必要S材料等必需输入，则对应阶段PARTIAL，不把全任务写100%完成。
- evidence_status：`ATTRIBUTION_COMPLETE_EXPOSED_ONLY / ATTRIBUTION_PARTIAL / NO_ADDITIONAL_SUPPORT`，每模块另列效果与新颖性状态。确认未运行始终明示。
- publication_status：`PUSH_VERIFIED / PUBLICATION_BLOCKED / NOT_ATTEMPTED_SNAPSHOT`。Git中C1快照不能冒称已证明C2；最终以C2内C1收据和OUT最终记录解析状态。

所有状态都附实际范围和数字；missing/null不变0。REPORT历史受阻状态与最终状态分栏，不残留互相矛盾的PASS/BLOCKED混合行。

## 16. 最终交付与一次审核

主报告回答：
1. E/R/S各自带来多少主指标增量，哪个仅条件有效？
2. E涨点来自来源增加还是具体组合？D2与普通权重哪些条件下等价？
3. R恢复了哪些支持/实例，TP/FP与覆盖怎样区分？
4. S在同预算结构比较中是否有优势，局部改善在哪条地图路径保留/消失？
5. AP提升中类别、排名及新增FP分别有什么证据？
6. 哪个最多两个模块值得下一步确认，已有说法应收窄在哪里？
7. 当前仍缺哪些独立场景/真实提案/novel/外部统一条件证据？

三张主表：系统与原版参照（外部论文行清楚隔离）；模块加/去/强替代与四格归因；预算/成本与泛化证据状态。补表保留全候选、逐家族、匹配口径和5次干预。不存在的确认指标空值＋NOT_RUN_BY_CONTRACT。

`HANDOFF.md`写实际根/环境/锁、直接输入链、源码/权重出处、真实CLI、表格重建命令、完成与受阻阶段、普通权重锁及候选、核心数值、发布收据边界。新登记的确认清单不写成已运行。

一次审核清单：原件/类/家庭分母；每类null；A5及N/A0区别；D2缺源；15开发与12主配置上限；S matched-step与selected分栏；counterfactual只作诊断；完整预测先锁；图表可重建；同源alias证明；CPU/GPU实际计数；四组测试；真实C1/C2；论文主张不超证据。

发现错误只修复受影响代码/文档，重新检查对应项目；不进入“再全审一次”的无限循环。最后回答必须有实际结论、主要数字、完成/受阻原因、训练/编码等新计数、分支及C1/C2完整SHA、RESULTS/HANDOFF永久链接、仍缺的投稿证据。

**现在按任务顺序完成开发、有限选模、证据归因和发布。不要把后续确认当成本轮已经补齐，也不要因为负结果或只剩一个简单有效模块而拒绝交付。**
