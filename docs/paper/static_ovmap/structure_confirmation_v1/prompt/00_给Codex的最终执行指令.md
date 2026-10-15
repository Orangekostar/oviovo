# FG2 结构验证与新家族确认：Codex 最终执行指令 v1

**日期：2026-10-15。执行任务，不仅返回设计。** 本文件与同包 JSON、方法定义和确认接口共同构成唯一合同；取代此前针对本轮的自然语言可选建议，不取代旧实验的历史事实。执行完开发、训练、选模、确认、回归、分析、交接和正常 GitHub 发布。负科学结果仍须完整交付；缺依赖如实标 PARTIAL，不制造成功空结果。

## 0. 本轮究竟验证什么

研究对象仅为既有 FG2：物理对应约束的组内观测聚合与组间对象读出。依次检验：

1. 同样的组数、有效观测和网络规模下，真实对应是否优于从头训练的非物理分组？
2. 组内观测权重、组间相对权重各有何作用？禁止将改变权重总质量混入“均匀分组”消融。
3. 是否优于固定输入上的普通表面平均、可见性区域特征聚合和受控学习式多视图融合？
4. 在预先登记的新家族、真实 SAM2 提案、heldout 正样本和固定地图应用中，收益是否保持？

不再研究 bias correction、不训练 Wave2 控制器、不调 D2 温度、不改几何关联或引入新大模型。E/R 已完成归因作为引用，不重跑其 8,000 次重采样。

**权限与预算发生了明确变化：**本轮最多 8 方法×2 seed×2,000＝32,000 次新优化器更新；训练与旧地图读取缓存，不新编码。确认阶段允许最多 1,152 次共享 FC 全图编码、48 张图的 SAM2 AMG 以及有授权的有限 ScanNet 原件获取。它不是上一轮“全程零训练／零编码”的延续。新 TSDF/在线建图和 N/Q 编码仍为零。

## 1. 权威与源码基点

- BASE=`2f0d8a4f7043557462f05875eca8b7f71eb66761`；FG_BASE=`bec4176d9a6784a89ea293a9803e5cf6fab178ec`。
- 仓库 `Orangekostar/oviovo`，新分支 `research/fg2-structure-confirmation-v1`。
- 新模块 `src/static_ovmap/structure_confirmation_v1/`。
- 入口 `scripts/evaluation/run_fg2_structure_confirmation_v1.py`。
- 配置 `configs/static_ovmap/structure_confirmation_v1.json` 复制 `02_冻结执行配置.json` 原字节。
- 文档 `docs/paper/static_ovmap/structure_confirmation_v1/`；工件 `artifacts/static_ovmap/structure_confirmation_v1/`。
- 阅读本包顺序：00 → 01 源码绑定 → 02 JSON → 03 方法精确定义 → 04 确认与评分接口 → 05 审核报告。06 参考程序是合成合同测试，不是实际实验实现。

先读目标目录适用 AGENTS.md。核实 origin 后由 BASE 建独立 worktree；不重置、清理或 stash 其他工作区。不改旧结果、旧部署或 main。名称被无关工作占用时最小未占用后缀 -02/-03，实际名称写 task_paths 后锁定；同 task/spec 已存在则恢复。BASE 本地没有时仅获取该 ref；不能以 default branch 的新代码替代。

## 2. 必须在作判断前读取的证据

01 中按任务绑定了训练、特征、样本、预测、评分和发布来源。重点：

- 原训练器强制四旧臂，并按旧门槛跳过 seed29，不适合调用其整体 train/all；只复用原算术、draw_schedule、optimizer_for、learning_rate、base_loss 及 checkpoint RNG 格式。[S02–S04]
- 父训练只保留 last/selected，不保证 step1000 权重仍存在。本轮从头训练默认开启，并保留每个评价步的真实 model state。[S02]
- 每个输入有 64 个物理位置上限，其中可能包括提案外上下文；不是纯部件监督。local_unit 已归一化，不能用它冒充未归一化的 phi(x)。[S05–S08]
- 旧 choose_views 是覆盖优先，不是时间顺序。2/4/8 是预算上限，FG2 是方法编号。[S06]
- 新确认旧计划已封存，但此前没有执行；新提案生成不能直接调用带旧 H2 gate、旧最大 8 图限制的整体 real_proposals.run。[S09–S12]
- 原新地图接口要求历史 parent_command/capture/completed schedule，不是任意新场景通用建图接口。本轮明确使用提供的无语义重建表面做新提案确认；旧 26 场景用于地图应用回归。[S13–S14]

禁止只因为目录存在就声称缓存/模型可用。每一判断记录“源码事实／设计选择／本机测量”之一。

## 3. 工作区、直接依赖与数据角色

默认 TASK_HOME 为当前仓库父目录下 `fg2-structure-confirmation-v1`，worktree=`TASK_HOME/worktree`，OUT=`TASK_HOME/attempt_001`；CLI 显式值优先。OUT 不能位于任一只读父根内。

直接根：

| 根 | 解析方式 |
|---|---|
| ATTR_ROOT | CLI；否则 `/home/ww/crove/ovmap-contribution-attribution-v1/attempt_001`；纯计划可由 BASE 发布的 gzip 工件补读 |
| FG_ROOT | CLI；否则依次已知 `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`、`/mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001`，只接受身份匹配者 |
| LR_ROOT、PR_ROOT | 从已验 FG binding 的 lr_parent_root、pr_parent_root 解析 |
| 原数据/下载权限 | 已验 binding 的 prior_data_root、data_roots 和其 acquisition_lock/plan，不猜共享盘 |
| FC、text、算子 | 原 export registry、assets、context；保持 FC model identity 与文本字节 |
| SAM2 | 已有 `model_assets.json` 路径见 S10、S11；原 pinned commit、config、checkpoint bytes |

只验证实际消费文件和其直接清单。路径迁移先验原内容身份，再在内存解析 path_map，不篡改父 JSON 或给修改后内容沿用旧 identity。

TRAIN：原全部对象清单顺序作为 draw 索引域（父清单包含非 base 原件）；采样仅 622 个 base 原件、24 家族。DEV：原 69 个 base 原件、4 家族、四条件、全 200 类竞争。不得过滤重排对象后直接复用原整数 draws。heldout40 正标签不参与训练或选模。

REG26：原 Replica8/CF18 固定顺序、7 个 CF 物理家族；是已曝光回归。CONF12：从 BASE 下 `artifacts/static_ovmap/contribution_attribution_v1/records/confirmation_plan.json.gz` 读取 `scene_nomination.selected`，验证 Git blob `41b2769612e89e6f9d4e67c26d29550d79d8c884`。从 ATTR_ROOT 读取时必须与该封存内容身份一致。**不得重新提名、更换缺失 capture 或依据标签补类别。**无需在本包猜写 12 个场景名。

先锁 `roles.json`：TRAIN/DEV/REG/CONF 的精确名称、家族、原清单和排除依据。新增发现与 TRAIN/DEV 同家族，阻塞确认并报告冲突，不静默换一家。只证明“未参与当前开发”，不声称从未预训练于这些类别或场景。

## 4. 环境与最小适配

控制环境继承 `/home/ww/miniconda3/envs/ovimap-map/bin/python`；FC 环境从父 FC_python 取值，历史为 `.../oviovo-radseg/bin/python`。一张父 A40、原 `.visual-gpu-<id>.lock`；串行 GPU 工作者；CPU scorer 最多3、BLAS1。不升级 torch/CUDA、不寻找空卡、不修改代理和全局 Git 身份。

构造新 binder，保留原 FC 所需 context/cohorts/scenes/assets/observation_root/lr_parent_root，新增 task_root/confirm_roots 等字段。不把 cohorts 全换 CONF12 导致 FrozenFC 无法读原 ScanNet200 文本上下文。复制原 class_split.json 字节；OUT/external/upstream_sources 只读链接到原算子快照。新的数学配置独立放 task_specification。

所有基础模型 eval、requires_grad=False；训练小头需要普通 tensor，不能用 inference_mode 生成长期训练输入。**不能把整个小头前向置于 no_grad：冻结 phi 仍须对组内权重的输入可微。**数据缓存可 no_grad，head/train 中保留 autograd。

训练原七键 `fc_vectors,view_valid,frame_ids,local_raw,local_unit,metadata,local_valid`。targets、class_id、family、condition 不进入 head。G 的确定性排列由外部数据适配器从 key/condition/frame 生成；排列是标签无关的输入变换，不是 head 获得 ID 特征。

复用现有 sparse cache 时核对模型/输入/条件/数值 producer；不能直接实例化要求旧 engineering/parity 文件的 CachedObjectLoader 并伪造该文件。实现薄 sparse 包装：每对象×条件只生产一次 full8 七张量，读出前切片；所有方法共享 CPU32GiB总预算（dense8+sparse24，不是两个32）。支持 raw grid 恢复不等于允许重编码旧训练图像。

## 5. 模型与消融合同

精确公式与边界见03；本节是方法清单，不能减少正式对照。

| ID | 作用 | 相对原 FG2 唯一主改动 |
|---|---|---|
| S_LL | 本方法 | 原表面分组、两级学习、pool→phi |
| V_LL | 视图结构对照 | 原 VIEW grouping |
| G_LL | 同组数非物理对照 | 每视图 valid-token 联合固定排列，训练/测试同规则 |
| S_UL | 组内消融 | 均匀组内权重，仍学习组间权重 |
| S_LU | 组间消融 | 组内学习；组间均匀，但保留本臂可微总路由质量 |
| S_UU | 两级相对权重均匀 | 仍学习自身总修正质量，不称 training-free |
| S_PRE | 投影顺序对照 | phi逐token→组内池化；无逐token额外L2 |
| MV_ATTN | 受控学习式近邻 | 256维、4层、8头、多视图 attention＋latent pooling；完整定义见03；不是 CAMFusion 原版 |

无训练对照：B_FC（原 fc region 平均）、B_SURFACE_MEAN（同局部 raw 的普通两级平均）、B_PFE_ADAPTED（以可见性将 region 向量组织到采样点后聚合）。后两者不得冠以原版 Open3DIS 名称。

主结构比较 S_LL/G_LL 同参数、同组数、同有效布局；S_LL/V_LL 是原结构包比较；S_LL/其他两级臂是操作必要性消融。MV_ATTN 参数量和实际算力独立报告，不声称与 FG2 同容量。除指定变化，不增 membership、consistency 新损失、自由训练 backbone、R 或 Wave2 gate。

## 6. 训练、检查点与梯度

默认8臂两seed全部从头训练一次；这是为得到相同训练条件和所有指定步的真实权重，不依赖旧 selected 的不同步数。只复用本任务已完成同 identity 臂；不从旧表现最佳权重 warm-start。任何纯计算等价的父臂复用提议必须在训练前证明所有步权重/optimizer/RNG/draw/数学相同，否则按默认训练，不能为了省预算只用历史摘要替代。

每seed从原顺序 objects 调原 draw_schedule(seed,updates=2000)；32,000原件抽样，prefix循环2/4/8、扰动循环1/2/3。原件一个 clean+corrupt；16micro累积／更新。保留原 base_loss：0.5 CEclean＋0.5 CEcorrupt＋0.1 cosine(corrupt,detach(clean))；CE仅160base、temperature0.07。不要载入会员标签训练G或均匀消融。

同家族 FG 变体复制相同初始 state；父 initialize(seed)；在构建所有额外对象后将优化随机流重置到 seed。MV 初始化 fork_rng(seed+701)，不改变公共 draws/RNG，末输出投影零初始化。G 数据排列不消耗公共 torch/numpy RNG。

AdamW分组、lr、warmup、cosine、eta坡度均复用原纯函数；grad_clip1。原第一步 lr=0、eta=0 是合法零更新效果，不把首步无非零梯度误判错误。step2以后做一次有效路径梯度核对；冻骨干无grad但小头应有定义梯度。

每个评价步250/500/1000/1500/2000永久保存 `model_stepXXXX.pt`，以及DEV records/full200 cosines；每100步保存 full resume（优化器、三类RNG和cuda、samplerindex、当前已完成验证）。每次更新完成后才推进 completed；中断在半个microbatch不能把未完成梯度或错误completed存成已完成步骤；恢复重做该步并计实际重放。保留 last 两代原子快照，最多400次额外重放更新，达到后停止PARTIAL，不默默无限重训。

所有输入缓存标签边界不变；变体实际参与梯度的参数分别记录。无梯度参数的 AdamW 行为沿父语义，不人工给disabled membership补零梯度。

## 7. DEV与锁定规则

主DEV使用prefix8，原 summarize 的 A/M/C，记录普通Top1/类别均衡Top1/净修正及四条件。所有臂同5检查点。

每方法只由 seed17 选择：A8降序→M8降序→C8降序→step早，tol1e-10，不设涨点才选的条件。seed29只能用该方法seed17锁定步；seed29全5步可保留作曲线，但不得换 selected。每方法即使比B_FC差也要报告，不删表。

另保留固定step1000全部8臂作为确认结构主对照，其中S/V/G是对应必要性的主比较；固定step2000所有臂作为训练终点DEV表。原历史selected仅旁列，不混入新公平表。

写全模型 `predictor_lock.json`：方法定义、权重sha、step、seed、class/text、训练来源、所有输入算子、静态对照、确认清单与主指标。锁定后任何数值改变生成新实验版本，不用确认GT重新挑step、阈值、视图、融合或场景。

**没有科学涨点进图门槛。**本轮是证据核验，所有完成臂进入预设确认；不因DEV负结果省略需要的反证。工程依赖问题只阻塞相应阶段，报告必执行。S/V/G/MV缺任一核心臂，确认可以留存已完成输出，但标PARTIAL_ARMS，不声称完整对照。

## 8. 真正执行新家族确认

本轮不是仅登记下一步。CONF12材料与权限可用就按04完成获取、共享输入、真实提案、推理和评价，无需再次询问。资源/权限缺失则记录精确文件/阶段，无权将其冒称确认完成。

顺序严格为：模型锁 → 原件有限获取 → label-free帧几何 → SAM2 masks与真实输入锁 → 真实全部预测锁 → 解析评测GT → 构造单独GT-support诊断并预测 → 两种支持分别评分。

新确认使用提供的原始未对齐重建mesh，非GT语义参与的支持提案。不称在线建图、不称仅4帧生成全场景；最多96帧只用于候选观测/几何可见性，最多8V用于每个对象的语义读出。GT-support数据明确为诊断，不替代真实提案。

共享原件不足不跳到下一场景；跨所有12family预测锁定后才读取真实提案的GT。每场景无人为“至少匹配足够才算成功”筛选，所有失败/未匹配/无语义支持保留。类别覆盖不足属于科学覆盖不足，不补挑有利场景。

## 9. 新确认的输出与主指标

选定模型＝8训练臂×2seed＋3静态＝19槽。全部按2/4/8前缀产生描述符（完全相同输入的别名只算一次）。额外同step1000全部8臂两seed在prefix8产生16槽；与selected相同的权重/输出严格复用。输出schema按方法/seed/step/prefix/对象键明确，不按数组偶然顺序拼接。

GT-support：clean普通Top1和类别均衡Top1为主要识别指标，四条件A/M/C为历史可比辅助；base/heldout/all都用全200竞争。匹配步主差、selected差和终点DEV不混表。未有有效视图的固定诊断原件报告abstention，主分母保留，common-supported子集另报。

真实提案：预测先锁。全部原始保留mask及来源列表作为分母；IoU>.25、>.5的匹配识别率/唯一GT均衡率另列。完整GT场景上用固定稀疏预测集合评分APall/AP50/AP25，prefix8；constant1 confidence、同一预测排序，不新加topk类别/语义confidence。重叠mask允许保留，不强行转exclusiveowner。原matcher会处理duplicate/FP/FN，按真实规则输出。最大35配置×12＝420逻辑scene行、35池；独立解释为**提供重建表面上的稀疏提案识别/分割评价**。

不报告该重叠稀疏提案集的“完整语义地图mIoU”；本轮没有定义无歧义的全场景仲裁和稠密输出。旧26scene仍提供完整地图mIoU。

heldout表只有真正heldout正样本才定义，不只是词表包含40类。类别无样本=null；H(base,novel)仅两者可定义时算。至少40个IoU>.5唯一GT、8family和20heldout原件/4family的数量仅是预先资源覆盖参考，不是显著性门槛；未达到照实为LIMITED_REAL_OR_NOVEL_SUPPORT，不补换数据。

## 10. 旧26scene地图应用回归

这个阶段验证“读出进入地图”，不是新确认家族的在线建图。只使用原LR/FG已保存的regression objects、raw grids、G1分区、N/Q、温度、投影与原evaluator。

固定11槽：S_LL/V_LL/G_LL/MV_ATTN的两seed，外加三静态；都用锁定selected、prefix8。每槽做两输出：
- F_ONLY_FIXED_ELIGIBLE：只在原eligible owner替换F argmax，其余与本轮B_FC地图参照一致。
- PARENT_D2_FIXED_ELIGIBLE：在同域替换F，调用原pool和PE_D2_GROUPED；不改N/Q和温度。

共最多22×26＝572逻辑行、44cohort池；同输入/label/rank/scorer完全相同可以复用评分，不从分数相似推断alias。需要预测的对象数从锁定原清单计算；不得为资源上限裁掉eligible对象，上限不足则该阶段PARTIAL_CAP。

新动态调度复用纯 build_payload / native_ranks / partition_evaluator / released ordered pool；不调用旧硬编码方法数及旧positive-signal gate的整体run。保留未知/背景/恢复类、原rank、原词表，Replica51与ScanNet200分开。所有scene预测锁后正式评分，不因某格差而换融合或不发布。

## 11. 推断性统计与机制图

只对CONF12四对 S-G、S-V、S-MV、S-B_SURFACE_MEAN，预先固定family配对bootstrap seed4314、各1000draw，共4000个抽样集合。每draw同family同时抽入两seed、两方法；每seed单算该方法指标，再求seed均值差，不能把两seed预测拼成双倍样本/双倍实例。GT识别与真实AP沿相同抽样集合，分别报告。区间的预定主口径为GT识别使用matched-step1000、真实AP使用selected；另一口径提供完整点估计，不重复搜索有利区间。

真实AP从原matches命名空间复制后调用原evaluate_matches；禁止平均scene AP。对象/四条件/视图非独立，原件与family是聚类单位。12family不保证功效；defined draw和缺类范围照实给出。主要表同时保留每seed、每family，不把CI当录用门槛。

组内/组间四格的效应与交互，从相同seed、相同数据和指标算：L/L-U/L等明确方向。由于U外层仍有可微gate，只解释相对表面分配，不写“完全没有学习”。G対比保留同组数/valid布局以及实际相同物理组残留比例；有重复feature或只有一个validtoken时排列可能无效，统计而非强造不等。

定性图最多12：在新确认主比较S/G、S/MV中各选成功2、失败2、无变化2；按层内margin变化距中位数、family/对象键确定。相同相机/尺度/颜色，真实投影或原RGB许可可发布范围，不用生成模型；GT只作评价。统计图包括分组对照、两级四格、预算曲线和真实支持/novel覆盖。各图来源JSON、失败和null保留。

## 12. 成本实验：不把小头参数少等同于整个系统快

训练计time/updates/forward/phi separately。推理成本选固定DEV hash前16原件，clean、2/4/8；所有锁定臂同对象，模型已加载，GPU同步，一次warmup+3次测量，报告median/P90峰值和有效view/group量。相同cachehit不能称为新编码；包含region pooling/phi/移入移出张量，另报仅head。

确认FC全图编码和SAM成本按共享内容真实计数，注明供所有方法共用；旧N/Q和场景构建不在本次冷启动计时，不虚构30FPS或减半编码。参数、head FLOPs估计如有需有工具和输入尺寸，不能用墙钟脚本总时长替代。

## 13. 资源、恢复和必要测试

8臂×2×2000更新＝32000，主训练clean/corrupt调用理论1,024,000；五DEV点理论22,080。不把大规模匹配训练称“过度安全测试”，但不得添加第9模型或继续训练至涨点。物理训练/验证head合计限1,100,000，重放更新≤400另列；确认head≤300,000；旧图head≤131,072；工程head≤200；成本测量head≤4000（独立记账，不挤入工程额度）。超过上限先停止新增计算并报告，不裁评价分母满足预算。

主机缓存总32GiB；OUT启动至少80GiB，数据计划下载+网格保留估算必须留20GiB。新运行空间≤200GiB、新下载≤120GiB。无法满足则对应阶段BLOCKED，保留已完成，不删用户文件。新FC确认≤1152、SAM图≤48、newmap/NQ/olddataencoding＝0。

每unit身份：task/spec、数值producer、directinput bytes、role/method/seed/step/prefix/permutation、模型/text、支持与scorer。报告或Markdown变化不应重启训练。数值修复只失效受影响输出，写旧身份与影响范围。GPUworker每臂/阶段完成写真实receipt；完整resume一次，有效锁存在就不新增worker/head/commit。

仅四组必要测试：
T1 输入/分组：七键、联合排列保持各view多重集、group sizes/valid布局、无标签前向、prefix隔离、原S/V FP32parity。
T2 数学/梯度：两级四格、own-mass均匀、phi可微输入但骨干无grad、投影顺序无隐含L2、no-local fallback、static/PFE/MV边界。
T3 训练/锁：相同初始化和draw；至少一个更新后的resume再一步一致；永久step文件和DEV结果身份；seed29不重选；确认锁和缺依赖分支。
T4 评价/发布：重叠proposal与全GT/unknown/重复实例、原pool数值一致、family namespace、有限资源计数、一次发布路径/大小检查及C1/C2真实回执。

常规一次四组；修复只重跑受影响组及最小真实回归。禁止全仓pytest、驱动升级、全盘hash、fuzz、漏洞扫描、重复全部历史终态核验、以合成数据填科研结果。

## 14. 文件任务分工与CLI

| 新文件 | 任务 | 直接复用 |
|---|---|---|
| binding.py / data.py | 原root、角色锁、精确七键稀疏输入 | 原FrozenFC与loader、PathResolver |
| models.py / grouping.py | 8训练臂、3静态、标签无关排列 | 原FGHead算术、unit/route/correct |
| training.py / checkpoints.py | 匹配优化、全部指定step、原子恢复 | 原draw/optimizer/lr/base_loss/RNG格式 |
| recognition.py / selection.py / export.py | 全候选、seed17锁、seed29固定复验 | 原summarize，不用旧choose门槛 |
| confirmation.py / sam_worker.py | 新12原件、无标签投影、AMG、真实支持 | SensorReader、SourceRowProjector、sam2_worker数值实现 |
| gt_diagnostic.py | 独立GTsupport输入与targets | instance_arrays、proposal_sites、site_observation |
| prediction.py / evaluation.py | sparseproposal全GT AP与旧26完整地图 | 原matcher/score/orderedpool与build_payload |
| statistics.py / costs.py / visualize.py | 对应必要性、factorial、family区间与真实案例 | 上轮统计命名空间纯工具，不调上轮全流程 |
| reporting.py / publication.py / runner.py | 证据三轴、全部交接、正常发布 | NUL Git路径协议、已有原子写工具 |

允许合并内部文件，公开阶段和产物职责不能省。CLI必须实现：`--spec --parent-root --attribution-root --output-root --path-map --gpu --phase --resume`，新增 `--scannet-root --sam2-root --sam2-python` 可选路径覆盖只改物理位置。`--parent-root`固定代表FG_ROOT。

阶段按02.phases顺序，`all`执行到底。确认缺资源时记录PARTIAL并继续可完成的旧图/报告/发布，不把异常吞成0科学成功。纯科学negative可exit0，未解决必需依赖非零；外部协调进程依然尽力生成完整交接。单phase缺前置直接报具体依赖，不暗中调用历史all。

## 15. 最终成品、主张与发布

必须交付：RESULTS.md、HANDOFF.md、REVIEW.md、CLAIMS_LEDGER.md、EXTERNAL_CONTROL_SCOPE.md、DATA_SCOPE.md、完整本包合同、冻结spec、源证据清单；训练曲线与所有DEV候选、modelselectionlock、同step与selected表、确认/回归/novel/覆盖/成本/interval/图片来源；可复核全预测cosines与评分PR充分统计、失败/null/alias、artifact_index。

新学得小头导出 model-only 的step1000、step2000、selected（重复只存一次），附loader和weight-sha；resume optimizer留OUT，不上传基础骨干、字体、原RGB/mesh/GT或凭证。每blob≤40MiB，总新任务≤256MiB。先算容量，分块压缩表与预测，不删正式负结果；仍超预算则记录发布PARTIAL并提供原件hash/命令，不偷偷违反上限。基础模型不复制到Git。

证据按效果/归因/外推分别判断：
- 真实对应优于同组数重训控制且确认保持，才支持物理对应优势；否则收窄为细粒度组织。
- L/L与某简化臂相同，不强说两层学习都必要；允许最后推荐简化版，但不在确认集重选方法来冒充预注册主方法。
- 比受控MV_ATTN好不等于超过原版CAMFusion；B_PFE_ADAPTED不代表完整Open3DIS；EXTERNAL_FULL_REPRODUCTION始终标NOT_RUN。本轮可以完成既定实验，但不能标所有CVPR证据已足。
- GT支持好而真实提案弱，报告DOMAIN_GAP，不写通用建图提升；旧26是回归，不是独立新场景建图；采样mesh不变就不宣称类无关几何更好。
- 新确认只支持当前有限家族与分类覆盖；不输出录用概率或用内部阈值替代论文价值。

普通发布事务：科学子进程真正退出→外部记录 observed exit→生成报告和一次成品审核→仅stage本任务代码/config/tests/docs/artifacts→`git diff --check`及NUL文件名/大小核对→C1提交并正常push，ls-remote完整SHA相等→Git内C1收据+HANDOFF永久链接→C2仅收据/文档，正常push并核验→OUT/publication/final.json保存C2及本地/远端状态。禁止force/main合并、无限自引用SHA提交；已有完成发布resume不再提交。网络最多一次重试，不改全局身份代理。

最终答复必须列：实际完成/受阻、核心方法与控制结果、base/novel/real支持量、训练/编码/head/评分消耗、匹配步与选择步、独立性范围、分支C1/C2完整SHA、RESULTS/HANDOFF永久链接。未看到的确认分数写null，不填预期值。

**现在按本合同完成实现与实验，不再停在一份下一步计划；但不得为保证正结论而改动任何锁定对象、预算或方法。**
