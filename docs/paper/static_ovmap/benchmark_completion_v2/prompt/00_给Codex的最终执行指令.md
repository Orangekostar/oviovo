# OVMap：Benchmark-first 补实验与发布执行合同

**任务标识：`ovmap-benchmark-completion-v2`。本文件替换旧“补实验执行包v1”，不是追加章节。**

目标是把版本一“多源语义融合＋锚定校准”补成可评价的完整方法：完成受控校准比较、实际地图联合消融、标准协议适配和代表性外部方法比较，再生成三张主文表和补充材料。不要继续训练物理分组、分支门控或凸包约束模型。结果可为负，缺数据必须明示，但不能把只整理了历史数字称为全部补实验完成。旧执行包的12GPU小时与2CPU小时上限是此前助手提出、未经本机测时支持的设计值，不作为用户既定配额继承；真实用户/调度器额度始终优先。

## 1. 唯一合同、文件与事实等级

同时读取本包 `01_源码依据与任务绑定.md`、`02_冻结实验配置.json`、`03_算子数据与评分合同.md`、`04_Benchmark与外部方法执行.md`。本文件规定工作流，02规定枚举及数值，03规定数值/数据接口，04规定外部完整流程。发现真实源码或数据与合同不符，保留输入、记为具体接口问题，仅阻塞依赖阶段；不得为了通过而修改方法、指标或数据范围。

来源：`Orangekostar/oviovo` 固定父提交 `9bbe040570ecbaec854e6d8aa94ba82a3a1afa2a`。01中的E01–E25为本次源码依据，X01–X06为官方方法说明，K为CCFA规范。新接口均标为**待开发**，不得汇报成仓库已有实现。

主文表固定且全包唯一：
- **T1：外部完整 baseline × ScanNet200 / Replica-48**，最终 Ours 一行；内部 A0/A1/G1/S/G/V 不占主体。
- **T2：同一完整系统的全对象域 MF×AC 四格**。
- **T3：同一 MV 的校准固定四格＋等预算简单替代**，列 Base/Novel/H。

所有历史来源、逐seed、GT诊断、局部eligible回归和工程状态放S1–S8或机器工件。主文不使用 Run17/Run29 两组列。正式论文表不放 commit、审核PASS、路径等工程字段。

## 2. 开始前的工作区绑定

1. 读取当前 `git status --short`、HEAD、origin；只检查将修改的任务路径。用户已有修改不动，不stash/reset、不force-push。
2. 本任务研究分支固定 `research/ovmap-benchmark-completion-v2`。优先复用这个分支的既有同任务worktree；否则从固定父提交新建隔离worktree。不要在main或旧已发布任务分支继续追加代码。
3. 新建模块 `src/static_ovmap/benchmark_completion_v2/`，入口 `scripts/evaluation/run_ovmap_benchmark_completion_v2.py`，配置放 `configs/static_ovmap/benchmark_completion_v2.json`。将本包保存到该任务docs/prompt目录，以供审阅。
4. CLI **不用语义多变的 `--parent-root`**。实现：
   ```bash
   PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py \
     --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" \
     --output-root "$OUT" --path-map "$PATH_MAP" \
     --asset-config "$ASSET_CONFIG" --phase all --resume
   ```
   以上是**本轮需实现的入口**，不是现在已有命令。`--semantic-root`可指旧OUT或本仓库公开A工件；`--structure-root`用于真实模型/旧地图依赖，可暂缺而不阻塞纯CPU校准。可选 `--gpu`、`--max-gpu-hours` 只用于用户/调度器实际约束。`07_运行资产清单示例.json`给出asset-config字段；null不表示资产存在，优先从已验证lineage解析，无需用户逐项重填已经能恢复的路径。
5. roots优先级：显式参数 > 该任务已锁定task_paths > 公开结构records/source_binding.json.gz中的有限lineage。只在这条依赖链解析路径并应用path_map；不扫描整个/home或全历史任务。旧训练/输入/结果目录只读。公开文档使用环境变量路径，不上传机器用户名/凭证。
6. 建立 `binding.json`：父提交、输入描述符、模型/文本身份、资产位置、旧LR/FG/ATTR继承链、GPU锁、解释器、官方数据路径。历史路径失效只允许明确映射或恢复同一资产，不可同名替换不同数据。

## 3. 必须开发的模块与调用关系

|待开发模块|直接复用的已核实接口|本轮需要补的逻辑|完成证据|
|---|---|---|---|
|`binding.py`|E01/E02/E07/E22|公开A恢复、路径映射、显式profile、父依赖解析；不运行父binder/all|binding与最小资产清单|
|`protocol.py`|官方split/valid class资源；E12训练manifest|场景/类别/训练与校准重叠、51→48协议隔离、目标词表映射|protocol_lock/overlap矩阵|
|`calibration.py`|E03–E06|新增NMIX，四方法同预算后处理；固定与搜索分开|逐原件预测、OOF、精确指标|
|`registry.py`|E07 `load_export`|只加载MV matched1000两份，读实际state核对；静态b只算一次|model_lock|
|`region_inputs.py`|E11/E13/E14/E25|MV仅区域特征快速路径，既有七字段接口等价；新scene输入制作|一次真实等价与帧/对象覆盖|
|`legacy.py`|E09/E10/E15|旧26场景局部替换新8配置，保留原非eligible标签；动态计数|208逻辑行/16池或明确缺项|
|`scene_bundle.py`|E18–E21几何和源构造原语|新标准scene、原OVI native分区、全对象NQF、不读取语义GT|全对象输入manifest|
|`standard_export.py`|官方mask格式及E16可借鉴逻辑|支持重叠mask、稳定对象ID、完整目标点域、常数confidence；不调用限制eligible的build_payload|U00–U11统一mask域|
|`external.py`|X01–X06官方入口|四种完整方法的独立环境/作者预测/格式适配，保留原模型管线|每方法job+实际结果或具体阻塞|
|`scoring.py`|官方实例evaluator；E09/E16池化逻辑|标准、旧局部回归分别评分；动态覆盖，匹配缓存复用|pooled AP和全分母|
|`statistics.py`|现有family namespace思路|限定两对比较、四个抽样集合，共用seed draw|真实区间与undefined数|
|`tables.py`|本包schema|从一个result_store生成MD/TeX/逐格来源|T1/T2/T3、S1–S8|
|`runner.py`/`publication.py`|E23/E24仅复用模式|独立DAG、断点原子写、失败隔离、实际子进程退出、C1/C2|执行收据与幂等恢复|

不得照搬旧任务的固定计数 `572/44`、`420/35` 或48张SAM限制。它们只解释旧协议。新阶段计数必须由已锁定方法×场景×seed集合推导。

## 4. P0：协议与输入清单先于评分

`bind`和`protocol`必须真实读取这些资源：
- 当前A `A/import.json`、`source_scores.npz`、`source_vectors.npz`、`candidates.json.gz`、`predictions.npz`。
- STRUCT `weights/manifest.json`、`records/model_selection_lock.json.gz`、`predictor_lock`、`class_split`、`split_manifest`、`source_binding`。优先公开版本，无须重跑训练。
- 有本地STRUCT时，其 map/readouts、map/prediction_lock、map/score、父LR regression/manifest和features/regression。
- 当前F骨干原模型/投影/预处理、N/Q骨干及Q_GAIN模型；每个学过标签的组件都纳入角色审查。
- 04规定的官方benchmark split、目标点序、有效实例类别与GT格式。

输出可机读的 `protocol_lock.json`：每个范围的scene列表、family列表、文本词表、评价合法类别、GT点序、min region、IoU阈值、标签映射、输入帧规则、训练/选模/校准标签角色、曝光性质。**文本有200类不代表实例评估恰好有200类**；以官方实例有效类清单为准。

角色审查规则：
- 分开记录“标签进入梯度训练”“用于超参/选模”“只见过指标”“未知”。含MV、Q_GAIN与任何训练式提案模块，不能只审MV。
- 官方val发生直接标签训练重叠的已训练组件，不可填“干净官方比较”的Ours单元格。可仍运行明确标为exposed的研究结果，但放独立附表。不得将剔除后的子集称全val。
- 当前CONF12全部已曝光且已有novel标签用于校准选择。新LOFO结果继续标探索性；不得反复分折就宣称新独立测试。
- `protocol`阶段按benchmark输出可运行/阻塞的独立scope记录；一个官方映射缺失不阻塞A缓存或另一benchmark。`method_lock`先锁共同模型，再锁各scope可用词表/映射，不能把一个scope失败提升为全局训练或校准失败。
- 本轮不重训以补救泄漏；若确需清洁权重，则交付 `requires_clean_training` 的明确输入/类别/命令依赖，继续其他实验，不执行旧B/C。
- Replica base mask必须在读取目标GT标签前由训练类别语义名单和官方映射确定。ID大小、测试频率、当前分数都不能决定它。

## 5. P1：校准与公平强简单对照，必须完成

### 5.1 直接恢复A，不重建缓存

调用E02的公开重建逻辑到新OUT，保留source_scores原FP32，source_vectors的`original_column_indices`映射回columns。只使用clean8V真实向量；公开包其他prefix/condition中的NaN不是模型失败，是未发布向量。禁止逆解embedding或用2/4V分数冒充2/4V向量。

从E01 `column_index`精确选择B_FC static selected和S/G/V/MV两seed matched1000。验证原件key/target/family/valid_ids一致，279原件9abstentions保留。源文字缓存不再L2归一化；embedding×text重构与原分数误差沿用1e-5检查。只核对消费资产，不全盘hash。

### 5.2 固定干预与搜索分开

T3固定面板只对MV matched1000计算 RAW、γ-only、CAP10-only、CAP10+γ0.025。前三个/组合若已存候选可精确提取；不要再跑旧110网格，也不要调用旧gate重新判断是否开B。

公平后处理面板：S/G/V/MV均给相同三族候选：MIX×11γ、CAP×11γ、NMIX×11γ，每族5个强度，共165/方法/seed。数值定义详03。B_FC只搜索11个γ，不复制静态结果当多个训练种子。总逻辑候选1331，实际复算/别名/reuse分别计数。

每方法每LOFO折只用其他11家族的seed17按 `H_macro → min(B,N) → B → 较少非恒等操作 → candidate_id` 选参数，容差1e-10；同参数应用到留出家族的两个seed。拼回279个OOF预测后重新算macro，不平均家族H。所有RAW/POST完整结果进S5；T3只展示同一个MV的算子对照。

针对三族各自55候选，另在**已曝光CONF12全体seed17**各选一个全局简单配方，写 `simple_recipe_lock.json`，用于之后的固定比较，不再看测试结果调参数。原最终CAP10/γ0.025仍作为事先决定的主方法配方，不因NMIX赢或输换成另一方案却仍叫同一方法。

这个阶段没有“必须涨点才运行地图”的门槛。无论正负都完成后续已具备数据的配对评价。

## 6. P2：最终读出锁定，不新增训练

通过 `load_export`加载 `MV_ATTN:seed17:matched1000`、`MV_ATTN:seed29:matched1000`，实际state必须 method=MV_ATTN、step=1000、seed匹配。不可使用 `load_registry(map_only=True)`，它会选selected（旧MV可能2000）。公开profile无法获取时该seed阶段明确BLOCKED，不能用其他步顶替。

锚点b来自同一输入的 `FrozenFCBase`；主方法CAP10、γ0.025、F8V、clean、原FC投影与预处理锁定。N/Q/F融合只用对齐类别分数，不在不同维度/模型向量间插值。

**区域-only等价输入优化（E25）：**MV只消费`fc_vectors/view_valid`形成base与attention，返回frame_ids/local_valid作元数据，不读取local_raw/local_unit/metadata数值。可以保留七key API，以形状 `(V,0,1536)/(V,0,768)/(V,0,8)/(V,0)` 的空局部张量标明“本方法未请求局部观测”，不是编造局部特征。用两个真实缓存对象比较完整输入与空局部输入的b/z，输出须在既定FP32数值公差内一致；通过后新场景只做实际区域FC向量，不生成不被MV使用的局部phi/Gram。若该优化不等价，退回原SparseInputs，不修改模型计算。

结果 `model_lock.json`记录两个真实权重、b算子、维度、预处理、词表、gamma作用时机、输入帧/对象支持、等价检查。b只计算一次；多种校准和四格复用同一份b/z。

## 7. P3：旧26场景的局部联合回归

先完成这条已存在数据链，不重建旧RGB、CropFormer、SAM或几何。使用STRUCT继承的LR `regression/<scene>/manifest.json`、`features/regression.json`与实际context。源text51/200维和原dtype不变。

在同一G1分区与原eligible集合上生成两seed×4配置：
- L00：eligible对象用MV原F；
- L10：eligible对象用N/Q/MV-F原融合；
- L01：eligible对象用校准MV-F；
- L11：eligible对象用N/Q/校准MV-F融合。

未选、恢复、背景对象按E10全部保持旧值。这是 **FIXED_ELIGIBLE_READOUT_INTERVENTION**，不是全局关闭多源。不得填入T2的GLOBAL四格；结果单列在S2/S7并标作用覆盖比例。

优先复用map/readouts中真正同step1000的原向量；其slot若selected2000，不能用其cosine当matched1000。缺同权重向量时只从现有区域缓存补两份MV前向；同一对象b和两seed各一次，不为4配置各跑模型。

沿E10 `build_payload`保存旧受限替换，沿E09 `partition_evaluator`和released池化重用数学；新wrapper按集合动态算8×26=208逻辑scene行、8×2=16池，不能调用旧572/44总入口。相同mask/labels/rank的物理评分可精确复用。

原26所有新增预测冻结后，才消费GT计算指标。保留每个scope的完整分母、unknown与未匹配点。模型/文本/输入匹配但结果不同应定位实际bug，不能为重现旧数字改rank或温度。

## 8. P4：标准benchmark全对象输入与完整四格

这一阶段**需要开发新适配器**，不是把旧cohort列表换成312/8。以04规定的原始OVI native分区作为新最终系统几何，不加入未稳定的G1恢复R。这是明确的新最终配置；旧A1/A3不是它的别名。

构造03规定的 `SceneBundle` 和不可变全对象mask域：
1. 生成/恢复原OVI原生native map与deferred capture；映射到评价点只用xyz几何，不读取semantic labels。
2. 在所有预测实例上，按原覆盖优先规则选择最多8个合格F观测。没有合格F观测的对象保留为unsupported，不按GT过滤。至少一个合格区域视图即按原MV有效输入处理，不沿用旧稀疏SAM的“少于2视图删除”规则；实际1–8视图数量完整记录。
3. 用真实capture/deferredmetadata构建N与Q_GAIN；E20/E21提供数值路径，新的scene manifest明确填全输入。Q的上限200attempts、实际消费数均记录；新场景候选耗尽可按原replay返回停止，不能伪造200成功或将Q=N。若原实现硬断言妨碍合法耗尽，只作有记录的边界适配，不改排序策略。
4. F原区域编码和b/z按相同内容缓存跨四格共享。新输入允许必要的新编码与提案处理，按实际content计数；不存在全程必须“0编码”的限制。
5. 冻结原件全部mask、所有source可用性和model_lock后计算U00/U10/U01/U11。每一格作用于**全部预测对象**，不能借E10旧eligible限制留下隐藏N/Q结果。

全局回退明确：MF关闭时只可用F，F缺失则unknown0；MF开启时用可用N/Q/F按原分组重归一，全部缺失则unknown0。没有有效F时AC不作用；不得把fallback N/Q输出当作F-only。合格F存在却MV输出非有限属于运行错误，不能用b悄悄替代。空输入正常unknown保留。

每场景、每模型仅算一次b/z/N/Q，各格只CPU变换及融合。最终U11直接同时供T1与T2使用。标准新输出使用每个原生预测mask的top1类别和固定confidence=1.0，借鉴E16的清楚协议；外部完整方法保留其原生confidence。这个标准导出策略与旧26的OFFICIAL_CURRENT_CLASS不同，分别报告，不能交叉拼分数。详见03/04。

标准例数以锁定的官方清单为准，预期ScanNet312、Replica8。若只有部分scene完成，任何pooled均标PARTIAL_SUBSET，T1全benchmark格保持缺失，不把遗漏scene当没有GT。不得因难例/低得分跳过场景。

## 9. P5：外部baseline，不只留四个名字

按04实现OVI-MAP、OpenMask3D、Open3DIS、MaskClustering四个独立adapter。优先下载/恢复作者已发布的完整预测在共同evaluator重评分，其次用作者预训练checkpoint跑完整pipeline。可复用相同内容输入，但不能给不同模型套同一视觉特征。

先固定外部repo提交、方法变体、模型、输入策略和所有路径，再执行。OVI使用作者原版 `origin_impl`；OpenMask3D用full-val脚本，非single-scene demo；Open3DIS不省refine后重新生成；MaskClustering保留CropFormer与CLIP-H/14。没有现成Replica支持时开发只改dataset IO/词表的适配并标port；不能声称作者发布过同协议Replica成绩。

当前无须同时训练/复现所有参考文献。OVIR-3D、Mosaic3D只作为经过核对的文献补充；未核实完整实例输出不进AP排名。主表baseline数由真实可比结果决定，不能挑容易击败的运行。

官方已有结果若split/labels/point域/evaluator一致可引用，标†在表注说明，缺compatible protocol才需新推理。用论文值不假装本地运行，亦不把demo结果换名为正式验证。

无收益不停止；只有真实许可/数据/权重/环境失败阻塞该方法。无法完成某baseline必须保留job recipe、完整错误与剩余步骤，不能一行“不可用”替代开发。

## 10. 资源计划：有限矩阵，不用虚构12小时保证

用户没有给本轮12h预算。撤销旧助手无实测依据的“全部补实验12GPU小时”限制；这不是无限实验授权，而是按02中的**有限方法×全部场景**完成原任务。不租云、不新增付费服务，不自动加训练。

- 同一用户既有GPU，最多一个GPU worker；CPU scorer最多3、BLAS1。
- GPU采用原资产配置/锁，不猜编号；当前GPU冲突时排队，不杀其他用户进程。
- 核心规模：CPU校准1331逻辑候选；旧26最多208scene格/16池；标准方法四格两seed最多2560scene格/16池；四外部baseline最多1280scene格，最终完整行复用四格末行，不额外跑。
- 每条新连接pipeline的第一个正式scene做测时兼工程检查，结果直接计入完整任务；不再额外拿toy反复试。
- 用实际首scene时间和剩余输入数量生成cost_estimate区间，记录硬件，不把官方3090的13.5h当本机实测。
- 如果存在用户/调度器真实额度，或显式 `--max-gpu-hours`，严格遵守并在checkpoint安全边界停止；保存剩余job计划为资源阻塞，不改变分辨率/视图/场景/训练以凑额度，不声称全部完成。
- 原始图像/网格按scene流式处理，缓存保留复现必需的区域向量、mask索引、b/z/分数；可删除**本任务创建且可重建**的大中间网格，不能删旧共享资产。磁盘初始及场景边界保留至少20GiB，估算最大下一scene临时需求并留两倍缓冲。
- 失败任务每同身份最多一次自动重试。若修复真实代码错误，新producer身份只重跑受影响分支，保留失败记录及实际消费；不全流程重来。

## 11. 选模、正式确认与标签使用

本轮的**主模型选择已明确**：matched1000两seed＋历史CAP10/γ0.025，不再选最漂亮的map step。新增只做后处理等预算选择，选择集仅限已曝光CONF12的声明角色，绝不从新的benchmark AP回调参数。

若目标是“最终novel类从未参与任何校准选择”，现有模型加旧CONF12配方不能自动满足。protocol_lock必须明确这一标签条件。新验证场景即使未曝光，也只能支撑锁定配方的场景迁移，不能改写其类别校准历史。

不自动另取12个场景当第三个benchmark；先完成标准312/8与上述真实输出。无干净标准权重或足够novel支持时，对应主张保持未支持，交接列出清洁训练/数据依赖。不要为了正结果启动已经终止的FREE/RADIUS/EVIDENCE。

## 12. 评分、统计和指标范围

所有指标内部用fraction，显示一次×100保留两位；每seed独立算后平均，静态一次。未知结果null，显示—；N/A仅用于不适用。没有正样本类别不捏造0准确率，组为空H=null；9无支持原件仍计错。

AP由全部锁定scene匹配记录调用该协议evaluator池化，不做scene AP平均。重复家族bootstrap需要给重复scene及prediction文件独立命名空间，防止dict覆盖。

新统计只两对 `U11−U10`、`U11−U01`，两benchmark各1000家族配对draw，共4000抽样集合。相同draw给两seed，每draw各seed先算AP再平均差值。两比较可复用底层匹配；无需给每个历史表重新跑8000次审计。稀有novel为空的draw记undefined，不反复抽到区间看起来显著。

Proposal recall使用所有有效GT完整分母；16/3只是已有匹配计数，不能用279诊断数作召回分母。只用几何阈值做条件识别是附加诊断，不能替代官方全分母AP。GT-only对象结果不叫真实提案结果。

## 13. 正式表格与交付数据

从唯一 `result_store.json` / `tables.json` 生成T1/T2/T3及S1–S8，不手工改TeX数字。每个非空单元格关联 `cell_provenance.jsonl`：table/row/column、raw value、显示单位、source commit/path/key、模型/预测/协议身份、聚合公式、数据角色。来源常量表不要复制多份。

T1：四外部baseline＋Ours；ScanNet200和Replica48各AP/AP50/AP25；同benchmark协议才能比较，额外监督/author prediction/port用必要表注。A0/A1/A3放S2历史面板。

T2：全局U00/U10/U01/U11；两benchmark至少AP/AP50。mIoU仅当同一个明确owner-to-semantic转换对全部四行都适用时在S2附报。旧L00–L11绝不填进这张表。

T3：同MV RAW、offset、CAP、CAP+offset固定四格；第二面板CAP/MIX/NMIX各同55候选的开发锁定结果。当前279探索和新benchmark条件识别分panel，caption写明选参标签角色，不自称独立。

S1协议与baseline完整身份；S2完整指标/历史/旧eligible；S3来源融合；S4结构负结果与两级消融；S5逐seed及RAW/POST/LOFO；S6提案覆盖；S7纠正误伤/融合/排名；S8实际frame/crop/encoding/head成本。方法/任务身份用自然名称，不将内部PASS/commit写进主表列。

论文缺格保持缺格。开发包保留缺失行；正式主表需要有最终Ours真实结果才可做性能结论，不以“模板已生成”声称投稿证据完成。

## 14. 四组必要检查，不做过度安全测试

**G1 数值与选择：**真实A结果精确重算原RAW/旧110候选身份，CAP边界/NMIX零向量回退、γ单位、H与LOFO标签隔离；统一一组，不给每个候选重复smoke。

**G2 模型与输入：**每seed一次真实state元数据；两个真实缓存对象检查区域-only等价；旧26首个Replica和首个CF18检查同b/源dtype/owner域。无需反复全盘/全部head初始化。

**G3 新数据/评分：**每新增evaluator/外部格式adapter用其第一个正式scene检查mask顶点序、类别ID、GT分母、constant/originalconfidence和官方metric；可重用正式结果，不缩模型版本冒充真实方法。全局四格检查 MFoff对象完全不读N/Q、mask域一致；实际读写traces用于判断，而非只搜索字符串。

**G4 汇总/排版/恢复：**跨表同prediction identity、perseed mean、null和alias、TeX编译及所有三张表看图检查；新runner在终态恢复时不创建worker、不导入GPU模型、不改科学产物或提交。外部格式不同的必要一次接入仍属G3，不为凑“4”忽略真实问题。

检查失败修真实原因，不改assert词串使报告变绿；合成测试与真实服务器结果分开写。

## 15. 完成状态与失败隔离

每阶段状态只用 DONE、BLOCKED_DEPENDENCY、BLOCKED_RESOURCE、FAILED_RUNTIME、NOT_APPLICABLE。DONE还需source/prediction/score coverage。阶段依赖失败只阻塞其后代，表格/报告/发布收集全部结果。

最终同时报告：`software_complete`、`calibration_complete`、`legacy_complete`、`standard_coverage_complete`、`external_comparisons_complete`、`publication_complete`、`claim_supported`。测完但没涨点是实验完成、主张可能不支持；缺数据不是机制负结果；发布成功不是标准benchmark完成。

不得让“有缺口的表已填好”覆盖 `standard_coverage_complete=false`。实际错误不能统一吞掉返回0；协调器保留子进程真实退出。负结果正常完成返回0；资源/依赖未完成可使用明确非零partial码并仍发布已有结果。

## 16. GitHub代码、结果、交接与恢复

新代码/配置/docs写入固定研究分支。大原始ScanNet/Replica图像、GT、用户凭证、外部版权模型不上传。已有公开小头用引用，不重复提交原512MiB工件。

提交真实新derived数据：b/z/valid_ids、每对象完整分数/预测/defined、GT-free mask索引或可复原支持描述、逐scene评价收据、官方match充分统计、model/protocol locks、job commands、训练/校准角色、tables、必要真实案例、资源消费。受数据许可限制不能公开的内容留完整本地路径与获取步骤，公开脱敏可复算统计。新增git科学导出最多512MiB、单blob40MiB是**发布策略**，超出的派生分片优先同仓release资产（已存在gh/token权限），否则明确本地保留，不删科学结果或谎称全部上传。

C1提交科学代码、配置、结果、表格、初版HANDOFF/REVIEW后正常push，读取远端ref核对，再随机选**一个新结果文件**读取确认不是只看命令成功。

C2只加入C1 publication receipt和交接永久链接，不能修改模型/数值。外部 `$OUT/publication/final.json`记录C2与实际发布后resume结果，避免第三次自引用提交。普通push失败最多两次，不force。

终态 `--resume`入口先检查本任务phase状态、producer与参数身份；未改变则不启动scorer/torch/cuda/训练/外部baseline，不重新生成时间戳造成dirty。若C1存在C2未完，只补发布；若科学任务被中断且有未完可运行job，继续该job；若任务已以PARTIAL_CLOSED关闭、缺资产job明确BLOCKED，未更改资产清单的终态resume仍只读取收据，不重新尝试所有失败依赖。新资产通过更新asset-config及显式refresh-assets解锁其后代；partial闭合不代表全部科研完成。

HANDOFF必须写清：跑了哪些范围/模型，哪些标准数据未完与原因；exact commands；解释器/输入/输出参数；所有表格字段；实际训练0及各编码/前向/CPU评分；剩余job清单；N0未变；能与不能写入摘要的主张。

## 17. 最终交付与审核结项

必须输出：代码与CLI、冻结配置、SOURCE_BINDING、protocol/model/simple locks、job_manifest、resource ledger、result_store、逐格来源、T1/T2/T3/S1–S8 MD+TeX及编译预览、RESULTS、HANDOFF、REVIEW、C1/C2收据、remaining_jobs。

给用户的最终答复先列实测结果与各范围覆盖，再给GitHub永久链接，不以测试条数替代科研成果。报告核查范围：运行/复算/引用/计划四者分开。旧物理对应主张未被本轮重新证明；D2普通权重、LOFO校准标签、完整benchmark边界如实保留。
