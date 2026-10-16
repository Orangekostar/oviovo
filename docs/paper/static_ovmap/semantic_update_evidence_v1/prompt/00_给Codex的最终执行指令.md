# 语义更新约束：诊断、条件性匹配训练与新确认
## Codex 最终执行指令 v1

日期：2026-10-16。目标仓库：`Orangekostar/oviovo`。
**执行本合同，不是仅返回设计。**先完成 A 诊断，再按预先定义条件决定 B/C 是否激活；无收益也完成报告、交接和正常 GitHub 发布。未激活的阶段不要求制造结果。不能通过改阈值、删对照、换场景让主张成立。

权威顺序：本文件、02 配置、03 数学定义、04 数据与评分合同共同生效；01 提供依据，05 提供交付前审阅范围。若执行时发现真实矛盾，记录具体条款并按受影响阶段 PARTIAL，其他独立工作继续，不自行选有利解释。06 是 CPU 参考算子，不是已接入仓库的实现。

## 0. 本轮问题与禁止越界

已有事实：真实表面对齐 S 未在等规模 G 上确认独立优势；S/G 与自由 MV 读出的 base/novel 取舍明显。前者不再作为本轮主张。
本轮依次检验：
1. 原 MV 的 novel 失败主要是类别竞争偏置，还是 novel 内部判别下降？
2. 普通原模型混合、已知类偏置校准和角度限制能否解释 S/G 的优势？
3. 若仍有可利用差异，同网络、同证据、同监督下，“受证据限制的更新”是否超过只限制半径？
4. 若开发证据支持，锁定方法到新的家族与更密集的共享真实提案上确认。

不再训练 FG2 分组、Wave2 控制器、教师保持包，不调旧 D2、几何关联、文本或骨干；不把新实验命名为原版 CAMFusion、WiSE-FT、Mask-Adapter 复现。本轮没有保证新颖性或录用的含义。

## 1. 固定基点与工作区

BASE=`553758bda89031b05916ac81a2981b597dfb3bda`。
新分支：`research/ovmap-semantic-update-evidence-v1`。
模块：`src/static_ovmap/semantic_update_evidence_v1/`。
入口：`scripts/evaluation/run_semantic_update_evidence_v1.py`。
配置：`configs/static_ovmap/semantic_update_evidence_v1.json`，复制02原字节。
文档：`docs/paper/static_ovmap/semantic_update_evidence_v1/`。
工件：`artifacts/static_ovmap/semantic_update_evidence_v1/`。

先读适用 AGENTS.md，检查 origin 为用户仓库。由 BASE 创建独立 worktree，默认 TASK_HOME 为当前仓库父目录下 `ovmap-semantic-update-evidence-v1`，OUT 为 TASK_HOME/attempt_001。已有同 task/spec 工作则恢复；名称确被无关工作占用可用最小未占用 -02 后缀，在 task_paths 锁定。不覆盖、stash、reset 或清理用户其他工作区，不改 main/部署。

## 2. 必须先读的源代码

按01读取最新 `structure_confirmation_v1` 的 export、prediction_codec、prediction、gt_diagnostic、metrics、models、data、training、evaluation、acquisition、confirmation、binding、publication，以及上游 FrozenFC、base_loss、sam2_worker、原官方评分接口。只复用纯数值接口和明确数据格式，不能直接调用旧 bind/all：其 task、角色、12场景、35池、8臂等均有旧硬编码。

重要已核实事实：
- 公开 scores 保存无损 full200 cosines，但 `embeddings_omitted=True`；原 OUT 的预测 NPZ 则含 embeddings/defined。
- 旧 `partitions.H_base_novel` 是普通 Top1 的调和均值，不是类别均衡 H。
- 原 MV 的0.5残差系数没有约束自由输出向量的范数；不能把它等同于S/G的非负证据组合限制。
- 原训练是160类CE，且原对象列表含非base对象。新抽样不能在过滤重排后复用旧整数索引。
- 旧确认只有4个种子帧/家族、16mask/图；扩展新确认必须新建入口和任务worker，不能只修改JSON而继续调用旧16mask截断。
- 小头参数冻结不等于中间输入不需要梯度；本轮E层必须对共同网络输出q可微，禁止detach(q)或STE。

## 3. 依赖、权限与角色

`--parent-root`本轮唯一表示 STRUCT_ROOT，默认 `/home/ww/crove/fg2-structure-confirmation-v1/attempt_001`。这与上轮同名参数含义不同，必须在CLI help和交接写明。
FG_ROOT、LR_ROOT、PR_ROOT、ATTR_ROOT由STRUCT_ROOT的已验binding/原父清单解析。BASE下 `artifacts/static_ovmap/structure_confirmation_v1` 是公开分数/模型的只读后备，不是原NPZ路径。
先验证原字节身份，再在内存使用path_map；不得改父JSON再沿用旧identity。只检查实际消费清单和直接输入，不全盘扫描/哈希。

角色：旧TRAIN/DEV保持数据身份；旧CONF12明确为 `A_EXPLORE_EXPOSED`，可用于诊断与资源决策，不能称独立测试；REG26仍为已曝光回归。原heldout40为本项目既有训练未见类，但已被以前的CONF观察，不声称对整个研究过程完全未曝光。
若进入B，新增CORE152/PSEUDO8类别划分，正样本与CE分母都排除PSEUDO8；原class_split.json保留原字节，新角色另存core_class_split.json。
若进入C，另提名12个新物理家族；此处是新场景确认，不承诺基础预训练或全部历史的绝对独立性。

## 4. 环境和资源

一张继承GPU，原锁；CPU scorer≤3，BLAS1。控制/FC/SAM环境从父binding解析，不升级CUDA/torch，不改全局代理、Git身份，不搜索其他空卡。SAM沿最新隔离链接库的任务辅助方式，不向FC进程导入SAM。
总CPU张量缓存32GiB；A/B/REG不新编码旧RGB，必要时仅从合法原缓存恢复。
C允许最多1152次共享FC全图编码、144张SAM图；新TSDF/在线建图/NQ编码始终0。新下载≤120GiB、OUT≤220GiB、保留20GiB。已授权ScanNet获取按父锁执行；权限缺失仅阻塞获取，不虚构授权或空结果。

## 5. A：导入完整分数与向量

主诊断：旧CONF12 GT原件、clean、8V、matched1000；方法B_FC/S_LL/G_LL/V_LL/MV_ATTN，两个已有seed。应回读279原件、215原base/64原novel、9个无支持（这些数量是父数据检查，不得据此合成缺失记录）。静态B_FC只有selected静态列时，用该列作同prefix参照，不虚构matched1000列。selected、2/4V和四条件只作已有分数辅助，不能挑最有利子集触发训练。

优先读取原OUT `confirmation/GT/input_lock.json`、`prediction_lock.json`、逐对象NPZ。后备用公开 scores/manifest.json.gz + prediction_codec.decode；metadata中的columns从导出的GT预测锁解析。按key/method/seed/profile/step/prefix/condition连接，不能靠数组序号或混入REG的FP64分数。
保持原valid_ids顺序、文本字节、defined和unknown。NaN/undefined仍为abstention，不能填0后argmax。指标主分母包含原9个无支持，公共支持另报。

只做排名/混合/校准不加载大模型。角度分析优先读取原embedding；没有时才用已导出正确model-only状态和原稀疏输入补clean8V，最多2430理论head、硬上限2500。严禁从200个类别分数伪逆恢复768维embedding，严禁为补向量重编码图像。
公开权重有XOR-base gzip传输格式，必须用已验load_export/unpack_weights，不把.pt.gz当普通torch权重。

## 6. A：精确诊断与强简单对照

详见03。分别计算novel真实类别的完整词表排名、novel内部排名、最强竞争者是否base、相对FC的变化。novel-only只作诊断，不是可部署预测。

固定两族候选，不组成新的三重笛卡尔积：
- MIX_GAMMA：alpha∈{0,.25,.5,.75,1}；s=(1-alpha)s_FC+alpha*s_MV，然后减gamma×原base指示。
- CAP_GAMMA：将MV单位embedding相对b的角度上限设为{5,10,20,30,180}度，重算原文本分数，再减同gamma。
gamma固定11值见02，单位为原余弦分数，不是除以温度后的logit。
共110逻辑候选，严格相同预测可alias；不额外训练、不添加按对象oracle选择。

给每个S/G/V/MV绘制全体、base、novel的纠正-误伤统计、单位向量夹角及完整候选权衡曲线。对无定义向量保留缺项；不能把两个seed合并成双倍独立对象。

## 7. A：交叉拟合与进入B的唯一规则

旧CONF已经曝光。此处12次留一家族只是检查探索性结论对校准选择的敏感性，不能重新命名为独立确认。
每折只用其余11家族的seed17，从110候选中选H_macro最高者；相同候选应用于被留家族的两个seed。平局依次max min(B_macro,N_macro)、max B_macro、最少操作数、固定candidate_id词典序。H先在折内池化对象并按类平均，不取各家族H的简单平均。
将12折未参与该折选参的预测拼回固定279原件，重新计算每seed的B_macro/N_macro/H_macro。另报每家族H。S/G按原样，绝不为它们搜索另一个checkpoint。

B_GATE_OPEN当且仅当：S或G至少一者，对LOFO简单对照在两个seed上均有ΔH_macro≥0.01（容差1e-10），且两个seed上均至少8个可定义家族ΔH_macro>0；两者都满足时，按两seed的最小增量更大者、再固定S优先确定reference。这个reference只用于记录，不作为新模型的教师或warm-start。
不满足则 `COMPLETE_A_NO_RESIDUAL_SIGNAL`，B/C所有阶段 `NOT_ACTIVATED_BY_A`，仍生成结果、交接、发布。不能解释为已证明校准解决所有问题，只能说本轮未达到继续资源条件。缺scores/向量/类别导致不能计算，则PARTIAL_DEPENDENCY，不能伪称负科学结果。
不得换alpha/gamma范围或降低阈值救活B。本轮门槛是资源选择，不是显著性或CVPR标准。

## 8. B：开发未见类和共同输入

仅A开放后执行。候选PSEUDO类别是原base160内、TRAIN正原件≥4且DEV正原件≥1的类别；按SHA256(`SUEV1-PSEUDO-20261016|`+官方class_id字符串)、再数字ID排序取前8。仅消费旧TRAIN/DEV类别标签，不消费旧CONF或新确认类别标签选类。
不足8类、PSEUDO DEV不足2家族、CORE DEV不足8原件或8类，则B_DATA_LIMITED，报告并结束B；不换候选规则。
CORE=原base\PSEUDO（152）；FINAL_NOVEL=原heldout40。正样本采样与CE denominator都只有CORE。PSEUDO TRAIN原件完全不用于训练/蒸馏/一致性目标；可以保留在原index域但标记不可抽样。额外负词表也不偷偷包含PSEUDO。

保留原785对象列表及key顺序，用core标记建立新的draw_schedule；保存明确index→key、CORE正计数和抽样数组，不复用旧整数draws。各臂同seed抽样完全一致，2/4/8循环、clean+corrupt、16原件micro累积。
DEV仍保留69原base原件，分为CORE与PSEUDO进行全200竞争；40类在词表中，但其DEV正标签不参与选模。core_text另建，不覆盖FrozenFC原base_text/class_split。

## 9. B：三个同网络输出算子

严格实现03：FREE、RADIUS、EVIDENCE。三臂共享相同可学习网络结构、相同region与局部证据输入、参数量、初始state、损失和预算。只有输出算子不同。
局部字典来自当前输入的均匀有效raw表面池化→冻结phi→unit；不加载已受旧160类监督训练的S/G权重生成字典，也不将class/GT/membership/正确性作为输入。
共同网络是继承MV block的256维、4层、8头，对[基础向量token、region tokens、字典tokens]编码，3种可学习type embedding，1个latent pooling，末线性零初始化。三臂均读取同样的字典，不让E独占额外输入。
共同初始向量c=(1-.02)b+.02 mean(U)。另报告无训练B_C0=unit(c)与B_FC。这是避免纯b凸包顶点初始化的明确设计选择，不把它的变化归于E。
FREE输出unit(q)，RADIUS做半径0.5裁剪后unit，EVIDENCE做128步可微凸包近似投影，再同半径裁剪，最后unit。**它不是凸包与球交集的精确欧氏投影。**约束作用于归一化前向量；归一化后的单位向量不必在原欧氏凸包中。

## 10. B：数值接入与训练合同

06合成测试不替代真实FC接入。工程只用固定哈希前2个CORE TRAIN原件的clean、2/8V；每输入取q=c、q=c+.25d、q=c+.75d，其中d是按key种子生成的固定单位随机向量，不用标签选方向。
检查128步与256步输出L2≤.002、Frank-Wolfe gap≤1e-4，凸系数/半径/finite/非零梯度符合03。256步仅工程参考，不偷偷用于正式某一臂。
任一数值条件失败：保留诊断，B_NUMERICAL_LIMITED；不自动加迭代、换解算器、用straight-through或detach来继续训练。其他独立A报告照常完成。

三臂×2seed×2000更新=12000名义更新，默认全部从头；不得使用先前训练过PSEUDO类别的模型warm-start。最多200次额外重放更新单列，不为涨点延长训练。
复用原optimizer_for/learning_rate及base_loss算术，将显式152类core_text传入；loss=0.5CEclean+0.5CEcorrupt+0.1 cosine(corrupt,detach(clean))，温度.07。eta坡度与父协议相同。
FC及字典无grad；网络、q与有限步投影保留autograd。所有臂无AMP/TF32。原第一步lr/eta为0合法；第3更新检查有效输出梯度，不能仅以第1步判断失效。
每250/500/1000/1500/2000保存真实model-only及完整DEV分数；每100保存两代原子resume、优化器、RNG、sampler。中断半micro不保存成完成更新，重放计入实耗。
每臂同seed参数初值的digest一致；R/E没有额外学习参数。按实际参与参数报告，不把固定字典计入新模型权重。

## 11. B：选模、后处理和主指标

核心新指标：B_macro/PN_macro/H_macro；H_micro另列，不能调用旧partitions的H后改名字。class-macro仅对有正样本的类定义，未定义子集不能填0/1。未支持原件算错误，公共支持子集辅助报告。
每臂RAW checkpoint只按seed17 DEV H_macro降序→min(B_macro,PN_macro)降序→B_macro降序→最早step；tol1e-10。seed29沿用同step，不二次选。
每臂POST也获得同样5checkpoint×110后处理候选，参数只在seed17 DEV选择；gamma减CORE152，不能沿用旧160指示。POST的alpha、gamma或cap对所有对象/类别使用同一配置，禁止按GT切换。
RAW/POST检查点不同允许，但必须分表、各自保存；matched1000三臂另作结构对照，不拿selected冒充。
SIMPLE_STAR在FREE_POST、RADIUS_POST、B_FC、B_C0中按相同seed17 DEV准则选择，seed29沿用同来源与参数。E不参与选SIMPLE_STAR。

锁predictor_lock：raw/post/matched1000全部模型sha、类别角色、字典/算子/文本、后处理、SIMPLE_STAR、场景计划及资源。锁后不使用旧CONF/REG/newCONF结果改任何选择。
旧CONF若评价新模型，只能在锁后作探索补表，不参与本轮门槛或主确认；默认不额外评价，避免浪费。

## 12. B：旧地图回归和是否进入新确认

所有完成B臂无论是否有收益，都按04执行原26场景固定地图回归。RAW6槽+POST6槽+B_FC+B_C0=14逻辑槽；每槽F-only及旧D2两输出，最多728行/56池。新字典同当前对象输入，原N/Q、温度、owner支持、未知和原rank不变；新gamma只作用于F分支CORE类别，不修改N/Q。
这用于分析接入后的变化，不是选择模型或决定是否给某个控制组算分。源不足则独立标REG_PARTIAL，不伪称无需回归。

C_GATE_OPEN必须同时满足两个seed：
- E_RAW相对RADIUS_RAW的DEV H_macro至少+0.005；
- E_POST相对已锁SIMPLE_STAR的DEV H_macro至少+0.01；
- 上述每个比较中B_macro和PN_macro各自下降不超过0.01；
- E_POST不是alpha0/纯BFC别名；E_POST所用checkpoint的原始输出在至少10%可用DEV原件上与未约束q的单位向量L2>1e-5；
- 数值无未解决错误，RAW+POST模型/资源完整。
REG、原FINAL_NOVEL40和新家族标签不参与门槛。没过则COMPLETE_B_NO_ADVANTAGE，不再获取新数据；发布完整比较，不认为“没有证据”等同数学等价。

## 13. C：新家族提名和获取

只在C开放后使用既有官方ScanNet v2 train名单与有授权下载器。排除在有限项目清单中已出现的家族：旧TRAIN/DEV/H/H2、REG、旧CONF12、ATTR计划/已执行名单、既有acquisition selected、父binding lineage明确名称。仅逐层读取这些直接索引，不全盘grep所有文件；不能证明全历史时按“排除已知使用、本轮未开发”表述。
从剩余家族按SHA256(`SUEV1-CONF-20261016|`+family)及family排序取12，每家族最小数值capture。先锁名字，再查本地/HEAD/下载；缺文件不换下一个family/capture，不根据GT覆盖重新选。
存planned名单、排除及证据、官方名单sha、raw file路径。获取失败只影响该阶段，保留旧B成果；12缺任一则新确认PARTIAL，不发布看似完整的12家族均值。

## 14. C：共享真实提案与GT诊断

详见04。原无语义重建mesh和SensorReader用于几何投影；每家族最多96候选帧，12个AMG seed索引floor(j*(n-1)/11)，去重。SAM原设置及checkpoint不变，只将保留上限明确从16改32；最多144图/4608保留提案。
确切相同source_rows支持可按哈希去重并保留所有来源；不做新的跨视图合并/GT union，不按模型语义分数筛选。lift失败、无多视图支持全部留在覆盖分母；预测unknown，不删除后报准确率。
全12场景真实输入锁+全部真实预测锁后才解析GT。GT诊断每家族合法原件按固定匿名hash最多64，先选后检查可见性，abstention不替换，最多768原件。GT支持诊断不是无标签真实提案。
CORE/PSEUDO/FINAL_NOVEL三个集合分开；最终主泛化比较是CORE152与原FINAL_NOVEL40，PSEUDO8作为开发未见类别单列，全部预测始终200类竞争。不将PSEUDO加到最终novel里制造样本数。

## 15. C：评分、统计、成本

REAL：14个selected RAW/POST/static槽、8V，最多168场景行/14官方池；固定constant1 confidence，原matcher处理重叠、duplicate/FP/FN，完整GT分母，不报告稀疏提案的全图mIoU。
GT：同14槽2/4/8V，额外matched1000六槽仅8V，四条件。matched1000与selected/post分表。主GT指标clean H_macro_CORE_FINAL、两项类均衡准确率；H_micro/普通Top1/A/M/C为辅助，所有原件分母保留。
在source-input只读锁定后计算两个seed各自指标再平均，不把seed预测拼起来。真实proposal匹配Top1仅作条件诊断；正式AP所有提案/全GT。
04规定四对每1000个family配对抽样集合，共4000；同集合用于GT与真实AP，原matcher函数按独立namespace重算pooled AP，不平均scene AP。明确多指标探索性区间，未作多重比较调整；bootstrap不构成必然有效性证明。

真实支持参考仍为IoU>.5唯一GT40/8家族、FINAL_NOVEL唯一GT20/4家族，仅是覆盖参考，不是统计成功门槛。未达到标LIMITED_REAL_OR_NOVEL_SUPPORT，不扩大提案预算或换家族救分。
成本固定DEV hash前8个可用原件clean8V，RAW三臂及BFC/BC0，一次热身+3测量；含共同字典/Gram、网络、有限投影、FP64求解与搬运（若有）及总读出，另报只网络。不得把离线Gram预计算或projection排除后宣称更快。
定性最多8例：E_POST相对SIMPLE_STAR成功/误伤各2，E_RAW相对RADIUS_RAW成功/误伤各2；按margin变化距层内中位数排序选择，family/key打破并列，缺层报告，不挪成功填失败。真实投影/许可范围RGB，不用生成图。

## 16. 开发任务与必须交付的接口

| 新文件 | 必须实现 | 可复用依据 |
|---|---|---|
| binding.py / readers.py | STRUCT/FG/LR索引、公开score codec、OUT向量、原角色 | 01 S02–S07、S13 |
| diagnostics.py / calibration.py | 排名分解、110候选、LOFO、角度、A门槛 | 03；原metrics仅作历史核对 |
| class_roles.py / evidence.py | 8类开发未见划分、共同字典/Gram/中心 | 03；原data/observations |
| models.py / constraints.py | 同网络三臂、有限FISTA、radius、梯度与可行性 | 03/06；原MV blocks |
| training.py / selection.py | CORE sampler、明确CE152、checkpoint、RAW/POST锁 | 原纯draw/optimizer/loss/checkpoint |
| regression.py | 14槽两融合、原eligible支持与scorer | 原fixed-G1数值内核 |
| confirmation.py / sam_worker.py | 新名单、12seed×32cap、所有输入/预测锁 | 原Sensor/AMG/投影纯函数 |
| metrics.py / statistics.py | CORE/PN/N40、Hmacro、AP、familybootstrap | 原evaluator/namespace_matches |
| reporting.py / publication.py / runner.py | 三种终态、交接、两提交与无工作恢复 | 原发布事务，不复用其SCOPES硬编码 |

允许合并内部文件，不得省去公开阶段和产物。CLI至少：`--spec --parent-root --output-root --phase --resume --path-map --gpu`；可选 `--scannet-root --sam2-root --sam2-python`仅改物理路径。全流程必须区分未激活、依赖受阻、数值受阻、完成负结果。新文件均为本轮拟开发，不暗示已有仓库实现。

## 17. 四组必要测试与恢复

T1 数据：score无损解码、columns/key对齐、NaN/abstention、macro/micro、CORE排除和原索引域。
T2 数学：共同初始化、非零梯度、simplex/半径可行性、有限迭代数值误差、角度反向边界；不把06当真实接入证据。
T3 训练：同初值/抽样/损失、一完整更新后的保存恢复再一步一致、80式固定路径不得硬编码、本轮30指定checkpoint文件、seed29锁定、缺依赖门槛。
T4 评价/发布：完整预测分母、原scorer/namespace、资源和alias、有限文件检查、C1/C2事务与终态0工作恢复。

仅一次全4组，修复只重跑受影响组和最小回归。禁全仓pytest、fuzz、驱动/库升级、全盘扫描、反复历史哈希。合成结果不填科研表。
所有资源分counter，A≤2500head，B训练/DEV≤420000（含重放），工程≤200，REG≤24000，C≤220000，成本≤512；训练重放≤200更新。字典phi/Gram/投影迭代单计，不能混成零计算。
报告文字修改不影响数值identity；改数学必须新identity并明确失效下游。所有数值producer、input、class roles、model、text、profile、prefix、solversteps绑定。依赖恢复只检查恢复对象，不能重跑完成的父all。

## 18. 报告、审核与GitHub发布

必须交付 RESULTS.md、HANDOFF.md、REVIEW.md、CLAIMS_LEDGER.md、DATA_SCOPE.md、EXTERNAL_CONTROL_SCOPE.md、完整本合同/配置、模型结构与loader、全候选锁、A原件错误路径/LOFO、B每臂曲线与梯度/投影审计、C/REG全表、成本/interval/案例源、资源与恢复记录。
若A闭合：B/C表写NOT_ACTIVATED，明确0新训练/编码，不伪造模型。若B闭合：交付已训练权重和回归，C不获取原件。若依赖受阻：PARTIAL表保留，发布不被负结果阻止。
只上传本任务代码、文档、配置、model-only模型、派生cosines和紧凑评分统计。每blob≤40MiB，任务总≤512MiB；optimizer/raw图/mesh/GT/基础权重/凭证不入Git。无需复制父整套230MiB成果；引用固定提交和必要身份。新模型可复用原无损传输，但必须带独立architecture/loader。

审核不是只写PASS：逐条引用实际文件/字段；随机小量（非每文件）核对表到机器记录，检查所有训练候选存在、原件分母、数据角色和数值边界，逐张看最终最多8案例。没有看到的运行不能声称独立复现。新颖性只列已查近邻和具体差异，不给保证或录用概率。

科学子进程真实退出→外部记录exit→报告/一次审核→仅stage新SCOPES→diff --check和NUL路径/大小→C1普通提交push并核对完整remoteSHA→Git内C1收据和交接永久链接→C2仅文档收据push核对→OUT/publication/final.json记C2。
禁止force、merge main、反复自引用SHA。网络最多一次重试，不改全局身份代理。出版失败独立于科学结论；已完成发布的resume只验当前任务源和直接已消费身份，不启动worker/head/新commit。最终零工作数字必须来自真实恢复运行，不在发布前预写成已验证。

## 19. 最终答复必须说什么

明确运行到A/B/C哪一段、为何激活/未激活；先报主比较与强简单对照，再报辅助正数；区分Hmacro/Hmicro、matched/selected、旧160/新152+8+40、GT/REAL/REG、新提案密度与旧协议。
给出真实更新/重放、head/phi/projection、编码/AMG、逻辑评分/物理重算、原件/类别/家族及失败分母；分支C1/C2完整SHA、永久RESULTS/HANDOFF链接。
**本轮目标是判断是否存在超出强简单方法的独立技术价值，不是必须产生正结果；完成执行不等于投稿证据或新颖性已获保证。**
