# FG2 下一轮：分支归因与分支效用准入实验

**版本：2026-10-14；状态：已审阅的实验设计，尚未执行真实新实验。**

本文件将下一轮限定为“保留原系统、一次定位、一个候选、强简单对照、条件性进图”。不是继续调整已经失败的 surface-bias 模块。将本文件与 `experiment_spec.json`、源码绑定文件一并交给 Codex；执行前由本地实际文件解析依赖，不把报告里的路径当成必然存在。本文设计的新文件、接口、预算、阈值均是拟定合同，不是仓库已具备的事实。

## 0. 决策摘要

下一轮名称：**分支效用准入（Branch-specific Evidence Admission，工作名）**。

先问：同一个视图对“整体对象向量”和“局部表面补充”的作用是否不同？若不同，是否必须把它在两个分支中同时保留/丢弃？

零训练诊断使用原 FG2 的可分解计算。只有 TRAIN 上存在预先规定的可恢复空间与分支偏好，才训练两个 1,345 参数的小控制器；FC、文本、FG2 全部冻结。控制器不生成新特征、不改变实例几何，只在原观测中决定 KEEP，或移除一个视图的全局贡献、局部贡献或两者。控制器所有输入均来自当前可用观测，不读取类别、场景、对象身份或 GT membership。

**资源约束：**新 RGB 编码=0、新 SAM 提案=0、新几何=0。前半段新训练=0；后半段条件成立时两配置×两种子×2,000=最多8,000次小控制器更新。这是本轮新预算，不沿用上轮“全程训练0”的描述。真实试验未运行前不报告任何增益。

## 1. 必须保留的事实与限制

1. 旧 M1 的 TRAIN/DEV 空间留出 R 为约−0.172/−0.133；M1/C2 锁定 λ=0，所有 pass_both=false。旧分支按负结果保留，不启动旧 M1 地图，不把本轮叫作它的补验。来源 S01/S02。
2. 原 FG2 的 DEV A：seed17 的2/4/8前缀为44.7222/38.7037/46.0711%；seed29为44.8843/39.7738/46.8908%。这是已曝光的开发现象，不是新独立确认。来源 S03/S04。
3. `choose_views` 首先选最大掩码，然后贪心补充表面覆盖；并非时间顺序。已有视图银行、提案和位置依赖原协议；不能据本轮宣称在线、因果时序或主动选视图降低编码成本。来源 S05。
4. 原 FG2 的 base 是 region FC 向量均值；local quality/key 在原单观测特征和metadata上计算。局部 group_vectors 与 alpha 不依赖 base，但 values=group_vectors-base 依赖。来源 S06。
5. 旧 DISAGREEMENT_QUERY 已比较 AREA/COVERAGE/VERIFY/DISAGREEMENT 和支持权重，未通过扩展条件。该旧结果不能自动否定所有新方案，但也不能把普通覆盖加权、分歧选择当成全新方法。来源 S07。
6. 阶段A没有证明“更多视图一定更差”“加法偏差普遍不存在”或“3D对应必然有效”。不硬要求新方法准确率逐前缀单调。

## 2. 工作区、数据和依赖

固定执行源码基点 `0f96c5e09c3ced9262105c1352cc428643f5676e`；固定 FG2 科学父基点 `bec4176d9a6784a89ea293a9803e5cf6fab178ec`。

拟建分支 `research/fg2-branch-utility-wave2-v1`；拟建模块 `src/static_ovmap/branch_utility_wave2_v1/`；入口 `scripts/evaluation/run_fg2_branch_utility_wave2_v1.py`。旧源码、缓存、结果和部署保持原样。若该名称属于无关工作，用最小未占用后缀并记录，不覆盖。

- FG_ROOT 从上一交接的真实 frozen-FC root 解析；优先候选 `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`，其次 `/mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001`。
- LR_ROOT 由经验证的父source_binding中的lr_parent_root解析。WB_ROOT是已完成surface-bias实验实际OUT，仅用于复用其输出。
- OUT 必须属于新任务，不放在任何父OUT或Git工件目录内部。迁移只用path_map，并先验证父JSON原身份再在内存解析。
- 控制环境优先 `/home/ww/miniconda3/envs/ovimap-map/bin/python`；真实FC worker使用父FC_python与原GPU lock。一张原A40，保持FP32、禁用TF32和AMP；不升级环境，不影响其他进程。
- TRAIN=622个base原件/24家族；DEV=69个base原件/4家族。保留四条件与原160/40划分。TRAIN统计、DEV选模、既有地图回归分开。
- FC identity、小头SHA、步骤、文本SHA见JSON及源码绑定。保留seed17 step1000与seed29 step2000，不在本轮重新挑FG2权重。
- 使用原每对象每条件的固定物理位置；不能根据子视图集合重新选site、mask或bbox。四条件的site索引不保证彼此同义，绝不跨条件拼接。

依赖只核验本轮实际消费文件；不重验整个历史工程。旧`preflight`可借鉴和调用纯依赖工具，但不要直接启动任何旧`all`、训练或发布入口。新binder保留旧字段语义，增加独立task_source_base，不将旧identity伪装成新任务identity。

## 3. 原 FG2 的精确分解与干预接口

设B(S)为视图集合S的原FC均值后归一化向量。局部前向返回U(S)=[u_s]、A(S)=[alpha_qs]。原Q=4、eta=0.5。

`Compose(B,L)=unit(B + eta * mean_q(sum_s alpha[q,s] * (u[s]-B)))`。

实际执行应复用父`correct(B,U-B,alpha,eta)`。无局部组时直接返回B，并定义m=0、g为768维零向量，以供特征计算。原同一集合对角项优先原head输出，Compose对角只用于校验。**禁止实现成 `unit(B_old + delta_new)`**，因为delta_new已经减去了另一base。

等价的分析式（只用于解释，不取代父FP32数值路径）：

`m=mean_q sum_s alpha[q,s]`，`g=mean_q sum_s alpha[q,s]*u[s]`，于是`z=unit((1-eta*m)*B+eta*g)`。

据此可以分别测量base方向、local加权方向和route总质量。注意route源码包含group_count分母，新增可见组可能改变质量；这只是待测机制，不提前认定是退化原因。

### 任意子集读取

先用父loader读取当前请求前缀（2/4/8），然后对七个张量的视图轴用同一索引集合切片；同时记录实际frame_ids。不得只改obj.views而漏改mask/uv对应关系。计算任何当前前缀特征前，先截去后续视图。

同一输入多次调用、原head与Compose对角应同数值路径一致；跨历史输出沿原embedding容差rtol1e-5/atol1e-6，并要求历史class/A/M/C一致。公式重排的float64合成相等不等于真实FP32逐位一致。

## 4. A0/A1：先定位前缀退化（零训练）

### A0：基线与输入身份

所有622+69原件×4条件×2种子，读取请求前缀2/4/8。FC_BASE每对象条件前缀只算一次，两个FG2种子共用。已有DEV FG2结果身份一致时复用；补算缺少的FC前缀，不重跑图像编码。

记录请求k和实际n；前缀别名（例如对象仅3视图，4与8一致）不得当成两次独立观察。单独报告实际至少4/8视图的配对子集，但主表保留全部原件。核验前缀nested、view_valid/padding、mask/feature变换一致。

### A1：2×2分支交叉

对2→4、4→8分别构建四格：

|格|base|local|
|---|---|---|
|00|旧前缀|旧前缀|
|10|新前缀|旧前缀|
|01|旧前缀|新前缀|
|11|新前缀|新前缀|

这些是受控读出，不是四个新模型；均只消费新前缀已提供的信息。用完整200类的GT margin作诊断，GT不得进入Compose。令h_ab为某个对象条件的margin，则：

`effect_base=.5*((h10-h00)+(h11-h01))`

`effect_local=.5*((h01-h00)+(h11-h10))`

`interaction=h11-h10-h01+h00`

前两项严格加和为h11-h00；interaction单独描述，不能再加一次。相同公式可用于线性汇总后的correctness，但主解释同时保留原始正确/错误转移。软件路径干预不称为世界中的因果证明。

输出全TRAIN/DEV、clean与三扰动分别、每家族、actual_views、group_count、mean(alpha.sum)、null、local g变化、base方向变化。原对象/家族是聚类单位，条件/视图不是独立样本。不按诊断挑最有利家族。

## 5. A2：构建“分支角色不同”的有限动作库（仍零训练）

当前S=前缀4或8；n=实际view_valid数。n<3时仅KEEP，保持原FG2。n≥3时，每个当前有效视图v构造：

- KEEP：Compose(B(S),L(S))。
- G(v)：Compose(B(S\v),L(S))；只取消该视图的整体region贡献，仍保留它对表面的观测。
- L(v)：Compose(B(S),L(S\v))；只取消局部表面贡献，保留整体region向量。
- B(v)：Compose(B(S\v),L(S\v))；两个分支一起取消。

最多3n+1个逻辑候选；每次最多移除一个视图。候选生成不看标签。L(S\v)在三类动作间共享，不能重算三遍；global-only无需新phi。同一subset在缓存中唯一。

### 效用监督与oracle边界

对TRAIN base对象定义 `U(a)=CE160(z_KEEP,y)-CE160(z_a,y)`，温度0.07。只在构建训练target或离线诊断时读取y；前向、动作生成和特征提取不读类别、GT membership、family或scene。

这是当前冻结FG2、给定集合、给定评价代理的条件效用；不是视图固有价值，也不是它在任意语言查询上的效用。

记录完整200类margin及correctness用于报告，但训练target只用base160 CE，heldout40不获得正标签监督。训练用CE160与测试完整200的差异如实说明。

GT最优动作只做TRAIN上的可恢复空间诊断，不是部署策略、不算真实方法、不作为AP上界。最优CE动作可能不等于最优Top1动作，分别记，禁止混用。

### 条件性训练的唯一触发规则

只用TRAIN、前缀8、原固定两FG2种子。按原件去重（四条件中有一个满足即可），每种子均满足：

1. 至少40原件、8家族有某动作U≥0.01。
2. 至少40原件、8家族有某动作U≤−0.01。
3. 至少20原件、6家族出现同一v的`max(U_G(v),U_L(v))>max(0,U_B(v))+0.01`。

正/负原件集合允许重叠，不把它们相加声称新样本数。这些阈值是资源决策，不是统计显著性或投稿门槛。DEV诊断不改变该规则。不满足时不偷偷降门槛；仍完成下节9个静态候选的DEV筛选及同配置复验，发布完整负/弱信号结果。

## 6. B：唯一学习候选——根据观测的分支效用选择一次编辑

### 6.1 控制器形式

每个非KEEP动作a提取24个标量，输入共享MLP `24→32→16→1`，两层GELU；共1,345参数，末层权重与bias初始化0。输出预测效用`U_hat(a)`，KEEP固定分数0。所有模型只改变选择，不改动候选z本身。

阈值tau通过DEV选定。如果`max_a U_hat(a)<=tau`，KEEP；否则采用最高动作。并列按KEEP、G、L、B、被编辑frame_id升序。不得利用GT、family、类别名或对象ID打破预测并列。

由于先构建动作embedding，再估计效用，推理需要至多n+1次局部head计算，而非只有一次MLP；新增phi、subset readout、动作向量数、搬运都计时。当前缓存证明不了编码节省或在线FPS。

### 6.2 16个共享标量（按此顺序）

1–3. G/L/B三个one-hot。
4. n/8。
5–7. 单视图归一化region向量与其他视图region向量的cosine均值、最小值、总体标准差。
8. 该单视图向量与`B(S\v)`的cosine。
9. 动作base与全量base的cosine。
10. 动作z与KEEP z的cosine。
11. `||z_action-z_KEEP||2`。
12. 全量local的`mean_q sum_s alpha`。
13. 动作local的同一质量。
14. 动作local g与全量g之差的L2范数（g定义见第3节）。
15–16. 被编辑视图有效位置的原quality_logits均值、总体标准差；无局部支持则0。

以上相似性用归一化副本计算，不改变父FC向量的原base算法。标准差均ddof=0。原有效向量为零/非有限是工程错误，不静默填造有效观测。

### 6.3 8个新增对应统计

仅用当前七张量中的valid、local_unit与metadata；不需要额外GT位置：

17. 视图v有效位置数/原slot数S（S<=64）。
18. 视图v有效位置中也被其他视图观察到的比例。
19. 仅v观察到的位置数/当前所有视图有效位置并集数。
20. v有效位置的occupancy均值（metadata第0项）。
21. v有效位置的clipped_depth_residual均值（metadata第1项）。
22–23. 对有其他视图支持的同一site，`1-cos(local_unit[v,s], unit(mean_{u!=v}local_unit[u,s]))`的均值和总体标准差；均值向量范数<=1e-12的site不参与，全部无定义时0。
24. 同一site的视向metadata[2:5]与其他视图视向的最大cosine，再对shared site求均值。

无局部有效支持时17–24置0；无shared site时18/22/23/24置0。17/18本身标明缺少观测，不能把缺观测当一致。metadata来自父生产者，不重采样、不把所有site都宣称为纯目标点。

### 6.4 两个同容量训练臂

- WITH_NEW_CORRESPONDENCE_STATS：完整24项。
- WITHOUT_NEW_CORRESPONDENCE_STATS：相同网络、初始化、目标、采样、步数，标准化后将最后8项置0。

后一臂是“不提供新增显式对应统计”，**不是完全不使用几何的系统**，因为它仍依赖原FG2和其quality。

归一化均值/标准差仅由每个冻结head种子的TRAIN动作行拟合，两个训练臂共用；标准差下限1e-6。DEV/地图不参与拟合，地图原样使用锁定normalizer。

### 6.5 训练合同

每臂2,000步，batch为16个原件-条件-前缀组；类均匀→原件均匀，prefix4/8等概率，condition概率0.5/1⁄6/1⁄6/1⁄6。按seed+3001一次生成draws，配对臂完全相同。样本没有合法非KEEP动作时该组损失为0、保留计数，不隐式重抽以改变分布。

每组对其非KEEP动作的SmoothL1(U_hat,U,beta=0.1)取均值，然后对16组等权平均。AdamW lr1e-3、weight_decay1e-4、betas0.9/0.999、eps1e-8、恒定LR，grad_clip1。初始化seed+2009；FP32，无额外分类/一致性/保持损失，无FG2梯度。训练只消费已缓存24维特征及utility，不重新调用FC/FG2。

两个seed、两个臂总上限8,000更新。每500/1000/1500/2000保存权重，每100保存恢复状态；完整标量曲线保留。seed29照相同规则训练2000步，但仅使用seed17选定step评价，不在seed29重挑checkpoint。

## 7. 强简单对照与选择，不让模型复杂度冒充创新

### 7.1 静态对照（无新训练）

base三种：原MEAN；每view原mask像素面积归一化权重的MASK_AREA；在当前归一化region向量中选与其他view平均cosine最大的COSINE_MEDOID（并列frame_id小者）。原始未归一化base向量按原region定义参与，最后按原unit处理。

local残差缩放gamma∈{0,0.5,1}；实际调用eta=0.5*gamma。原alpha/U不变，使用新base重算U-base。共9个候选，其中MEAN/gamma1精确是原FG2，MEAN/gamma0是当前前缀FC_BASE。缺必要mask元数据则明确阻塞该对照而不静默替代。

这是控制“简单改base或调残差就够不够”，不当成首次创新。

### 7.2 同一个完整控制器的耦合动作对照

COUPLED只允许KEEP/B(v)，使用WITH_STATS训练的同一网络、同一候选utility记录，但独立在seed17 DEV选择step/tau。其不能胜任G/L是有意限制的能力，不把参数量变化解释为原因。

最终四类：STATIC、COUPLED、NO_NEW_CORRESPONDENCE、WITH_NEW_CORRESPONDENCE。它们分别控制普通超参数、动作分支自由度、新增物理对应统计。没有训练触发时仅STATIC存在；不可伪造另三者的分数。

### 7.3 DEV选模合同

只使用69个DEV base原件、四条件、完整200类竞争；沿用原A/M/C定义。评价prefix4、8；prefix2策略恒KEEP，作为无编辑边界，禁止用4/8候选信息影响2。

每个可训练方法4个checkpoint×tau[0,0.01,0.03,0.1]=16个逻辑候选；加STATIC9个，总不超过57（别名不重算）。每类配置排序：A8降、M8降、C8降、A4降、平均编辑数少、checkpoint早、tau大，最后固定方法顺序；STATIC再按base MEAN/AREA/MEDOID、gamma1/.5/0破平局。平均编辑数定义为非KEEP动作比例；已证明不改变有效支持的no-op为0，不能根据GT或预测标签相同而改写计数。容差1e-10。

在读seed29的候选结果之前锁定每类step/tau或base/gamma，以及seed17最强对照C_STAR（在STATIC/COUPLED/NO_CORR三者最佳中按同规则）。seed29仅固定复验；不能更换C_STAR来抢救结果。

进图规则：相对同seed原FG2，A8≥+0.5pp；M8、C8、A4均不得下降超过0.5pp；四条件普通net_corrections8>0；同一配置两个seed均满足。此规则只决定本轮预算，不代表统计显著性。全部候选数值照实保留，不把未达标写成所有指标都恶化。

## 8. C：收益传播和条件性整图验证

### 8.1 既有FG2传播链诊断

先只用已有FG2与FC8地图readout/decisions/scorer记录，核对新F→F top1→N/Q/F最终类别→已匹配实例/指标的变化。模型和rank不改、不选新lambda，不为分析M1失败重新跑它的地图。数据访问范围是原26场景；缺缓存只记录该分析受阻，不妨碍TRAIN/DEV。

### 8.2 新候选的P3激活

WITH_STATS双种子达标：运行M及三个强对照（已存在者都保留，即使某对照未过进图门槛）。否则只允许已锁定C_STAR在自身双种子达标时以普通性能优化支线进图；不得让次强对照利用seed29结果取代C_STAR。全不达标expanded=[]，发布已完成结果。

最多4方法×2seed×26场景=208条逻辑新候选记录、16个cohort池。已知FG2/FC8/REF_G1/REF_D2原身份一致则引用；如果旧FG3结果用于强调优势，也只能身份核对后引用且标继承，不偷偷更换主基线。

### 8.3 地图协议

全部激活方法/seed/scene预测锁定后再评分。原eligible owners、owner分区、G1几何、恢复对象与背景保留。只替换F余弦，再原pool及PE_D2_GROUPED N/Q/F融合；原native_rank规则照旧，不增加语义confidence排序。Replica使用其context文本，ScanNet使用完整200；Replica不强套160/40。

原evaluator与ordered pooling数值函数复用，新写动态覆盖列表。不直接运行带旧固定行数的整套旧evaluation。首次正式评分中每cohort首场景额外核对一个原FG2 seed17基线，仅两次。

报告APall/AP50/AP25/mIoU/mAcc、四cell(delta两cohort×两seed)、base/heldout类别支持、方法编辑率/动作类型、完整原件分母。类别缺失保留原null语义；不平均场景AP替代pool。

地图不用于再选step/tau/gamma。新方法若只胜FG2但不胜简单对照，归类为性能改善或尚无独立机制优势。候选mask不变就不宣称类无关几何改善。

当前集均是已曝光开发/回归集，后续投稿还需预先登记的未调参场景及真实提案验证。此轮不临时换场景追好结果。

## 9. 计算与存储控制

前地图原head计算理论上限：`(622+69)*4*2*(3+4+8)=82,920`；设置85,000上限，工程调用单列。3为完整2/4/8三个state，4和8为对应前缀每次删一个的local state；不足视图、重复subset及已有精确state都复用。global分支交换和MLP训练不计入原head调用，但成本仍单列。

不要保存所有U[64,768]中间阵列跨全部对象；单个对象条件种子的各subset在内存处理后释放。训练只需24维features、CE utility与完整预测复查所需的小记录；动作z/完整cosines在OUT按块存储。Git保留每动作utility/Top1/目标margin、训练曲线、锁定候选的完整预测、normalizer/权重、表格；大动作embeddings留OUT且登记路径、大小、hash和再生成命令。不能只发布一个摘要数字。

主机缓存总上限32GiB；一个GPU；CPU scorer最多3。初始OUT空闲≥10GiB，保留5GiB；不足就停止新增数据，不删除用户文件。Git单新文件≤40MiB、新工件总预算150MiB，必要时gzip/分块，不能删正式候选行降低体积。

仅六个必要测试组：
1. 七键同索引子集、前缀与条件隔离、现有历史parity。
2. Compose对角、正确的cross-base减法、无局部组与KEEP恒等。
3. 动作库全/局部/两者编辑与subset去重、n<3不编辑。
4. target与forward隔离、24维特征定义、TRAIN normalizer、配对初始化/draws。
5. step/tau锁定、seed29不重选、数值别名与中断恢复。
6. 原N/Q/F及rank保留、动态覆盖、有限发布清单。

不跑全仓pytest、无限fuzz、全盘hash、驱动升级、更多GPU压力测试。一次成品审阅；修复后只复核受影响组。

## 10. 给Codex的模块安排与交付

|新文件/任务|职责|现有可复用接口|
|---|---|---|
|binding.py/subsets.py|新任务上下文、原head恢复、真实同索引子集|surface_bias...binding纯依赖工具；ObjectLoader；export.load_exported|
|decomposition.py|Compose、四格、效应分解|frozen_fc_routing.models.correct与FGHead输出|
|action_bank.py/features.py|3n动作与24维统计、标签独立target|原head；metadata；FC/text只给外部评分|
|utility_model.py/training.py|两个同容量MLP、单一效用回归、恢复|新小模型；不要调用旧整套trainer|
|selection.py|57候选、step/tau锁定、双seed扩展|preservation_routing.selection.summarize，不用旧step>=250 choose|
|prediction.py/evaluation.py|新F与旧NQ融合、208行动态上限|build_payload、simple_dependence、原scorer内核|
|reporting.py/publication.py|主表、交接与普通双提交|既有发布流程作为参考，不伪造旧task-id|

CLI约定（待实现，不声称现在可执行）：`--spec --parent-root(FG_ROOT) --previous-root(WB_ROOT) --output-root --gpu --path-map --phase --resume`。phase见JSON；all中依次执行，无需因已经定义的条件阶段再次询问。缺依赖明确BLOCKED，异常不吞；无论正/负结果都report/publish。新任务只实现本协议，不自动训练第三个模型或扩充数据。

交付主表六张：输入/基线；分支四格归因；训练效用与分支偏好覆盖；静态/学习选模与固定复验；进图主表或未激活原因；完整成本。辅助保留每家族/条件结果，三类典型转移（只global有害、只local有害、两者互补）按固定key规则取例，不按最好看的挑选。

Git文档：RESULTS.md、HANDOFF.md、CLAIMS.md、REVIEW.md、原提示词与锁定JSON。HANDOFF说明真实阶段、权重/normalizer/阈值、复现命令、标签边界、失败恢复点、仍缺独立确认。三轴分别为execution/science/publication；未激活不是漏做，新增训练若触发必须报实际数，不能继续填0。

发布按已授权的同仓库新branch正常push：C1代码/数据/报告→核对远端C1→C2记录C1收据与handoff→核对远端C2→OUT外部final.json。不得force push/合并main/无限提交自引用SHA。保留N0部署。当前设计交付并不意味着本对话已经运行或上传新实验。

## 11. 可支持的论文主张与停止边界

若M同时优于原FG2、STATIC、COUPLED和NO_CORR，并在固定地图得到稳定改善，则可以继续检验“物理表面对应辅助的分支特异观测准入”。这仍需外部近邻复核、真实提案、未调参家族和成本验证，不据此直接宣称首创/SOTA/可录用。

若只有STATIC提高，保留工程优化；若WITH与NO_CORR持平，不宣称新增对应有效；若WITH与COUPLED持平，不宣称分支解耦必要；若改进仅在4前缀而8不提高，不用它替代8视图主方法，可报告预算稳健性；若连有限动作库也没有可恢复空间，停止该微改动，不扩大为大网络。

先前的负结果限制旧加性校正，不证明本假设正确。本轮的价值要由动作干预、同容量对照、固定地图和泛化证据建立。
