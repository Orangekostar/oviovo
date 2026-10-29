# 缺缓存的处理与本次修复

2026-10-23：完整分类分数及已审核池化表覆盖50/312场景（400矩阵），补充表1041行、32页，见 [SNAPSHOT50.md](SNAPSHOT50.md)。本地公开格式包保留47场景及45场景表格快照，见 [PUBLIC_PREVIEW47.md](PUBLIC_PREVIEW47.md)。Replica仍0/8。GPU0/GPU2双卡队列继续运行，原控制器T状态为有意接管；当前资源配置见 [MULTI_GPU.md](MULTI_GPU.md)。下文较早覆盖数字及单卡表述属于历史时点；科学源码、模型和协议继续冻结。

此前第44场景审核时，逐场景自有/OVI核验覆盖 **44/312场景、352/2496个自有配置格**；完整分类分数41场景、池化表及补充预览40场景、公开索引37场景。scene0169_01四个原作业均实际退出0，分割/建图/F/NQ墙钟分别279.14/1724.70/133.11/272.07秒。保留73个owner，F可用64个；MF关闭/开启时unknown分别9/4，两种子一致。完整GT为32实例、228074点，200次查询及完整帧屏障通过。OVI核对67个作者分类对象，CPU读出/评分88.98秒（非端到端）。三项自动元数据子调用均实际退出0；主代理核对实际日志、共享GT链及1410份保护文件不变，见 `review/G4_scene44_continuation_review.json`（`8b1f78a58c68d47606eb2da926e9107aa0d53370a9ff23a491773c3660136adc`）。原队列已继续第45场景scene0655_00。45场景完整分数与汇总入口仍未执行，原生产源码保持冻结。

此前第43场景scene0169_00的完整审核保留在 `review/G4_scene43_continuation_review.json`（`e7a6de66a074c1101c8741f5c311a95c9a1c48aa2898bb97e1d96d7c9aaac293`）。第44场景追加主审曾把 `unknown_owners` 计数字段误当列表而退出1；修正后检查退出0，失败记录单独保留于 `review/G4_scene44_continuation_inspection_attempt1.json`，不属于科学作业失败。

第43场景缺少独立的历史容量预检查收据，诊断保留于 `review/G4_live_controller_capacity_record_gap.json`。后续容量监测现已接上：独立脚本通过19项隔离检查及真实只读预检查后，按第44–312场景有限范围启动，每5秒沿用现有估算公式检查输出盘。监督进程1238934/start52877230，监测进程1238936/start52877236；启动后12次采样均充足，当前可用5466.38GiB、要求81.73GiB。主代理核对1410份保护文件不变，原控制器及其他队列保持原PID/start、未重启或收到信号。见 `review/G4_capacity_guard_launch_primary_review.json`（`0bed44626d1de1d79d125e942a192282ed3ce556b849e783a85b6e8569a64ef8`）。空间不足时只暂停核实身份的控制器，当前子作业可继续完成；不自动恢复。历史缺项没有回填，当前采样不证明历史边界已检查，整段监测的最终退出尚未发生，完整资源审核仍为部分。

缺少特征缓存时，复用现有预训练模型，从原始 RGB-D、位姿和实例建图输出补算。缓存身份一致的结果继续复用；不重新训练，不改变冻结实验配置。正式标注、评测点顺序、类别角色证据和指定 checkpoint 是独立输入，不能通过重算缓存补出；缺项只阻塞对应任务，其余场景继续执行。

后续元数据审计已接续到第42–311场景的有限CPU入口；15项隔离检查及最终来源真实预检查通过，1410份保护文件保持SHA/修改时间不变。监督进程和审计子进程已实际存活；首个真实native审计子调用已退出0，核对scene0565_00的200份输出、6334906字节，现等待该场景自有评分。它只顺序复用三份既有审计脚本，不启动科学生产作业或改变覆盖数；第312场景仍由原后处理负责。见 [AUDIT_CONTINUATION.md](AUDIT_CONTINUATION.md)。

此前第42场景逐场景自有/OVI核验覆盖 **42/312 场景、336/2496 个自有配置格**；本地完整分类分数仍为41场景/328份矩阵，池化表及补充预览40场景，公开索引37场景。scene0565_00分割、建图、F、N/Q作业均实际退出0，耗时分别298.40、1175.66、158.90、431.50秒。118个原owner保留，F可用106个；MF关闭时12个unknown、开启时3个unknown，两种子一致；完整GT为34个实例、167558点，200次查询及完整帧屏障核验通过。OVI核对130个作者分类对象，CPU读出/评分67.98秒（非端到端）。自动审计的native/own/OVI三子进程均实际退出0；主代理检查其退出记录、日志身份及共享GT链，1410份保护文件保持SHA/修改时间不变。见 `review/G4_scene42_continuation_review.json`（`a1164b3e6aeccddda6a84ee52579d7a50b141ce8729f64cf05938aa5fd9dc5f5`）。原控制器、后处理、CPU对照及审计接续均实际存活，继续第43场景scene0169_00。

此前第41场景时，逐场景核验和本地完整分类分数均覆盖 **41/312 个 ScanNet 场景、328/2496 个配置格**，OVI共同协议对照覆盖同样41场景；汇总表及补充预览仍对应已审核40场景，公开格式预览仍为37场景。scene0580_01四个原科学作业均实际退出0，分割、建图、F、N/Q耗时分别278.46、1004.67、149.50、377.14秒。保留114个原owner，F可用102个；MF关闭时12个unknown、开启时4个unknown，两种子一致；完整GT为21个实例、165009点。原生200份输出共3439745字节。OVI核对112个作者分类对象，CPU读出/评分84.29秒（非端到端），新增GPU编码及训练为0。见 `review/G4_scene41_continuation_review.json`（`99fb741195834c6f4552c5c897b6e57fb62059008129f0db63c93f1b574ff892`）。93份冻结源码/测试/元数据保持SHA和修改时间不变，原三个控制/评分进程均实际存活，继续第42场景 scene0565_00。

第41场景完整分数新增8份矩阵、912条对象/方法记录及182400个类别分数，全部200类独立复算在1e-12容差内一致，标签/置信度/导出标记精确一致；64条新无支持记录保留undefined。当前总计328份矩阵、23496条记录、1872条undefined。1034份科学输入/历史证据、640份旧矩阵/manifest的SHA和修改时间不变，重复准备复用全部657份当前工件。实际子进程与工具session81700均退出0，未运行模型、读取GT/原始几何、重算AP或导出。见 `review/G4_full_scores_scene41_primary_review.json`（`a080c4b5b38ff1615206907acca156fb9c23dd3ce6abb19fb5334739a65fd501`）。40场景旧摘要按身份归档；本地分数进展不提升40场景表格、37场景公开索引或完整任务状态。

40场景汇总追加核验已完成：4898个数值单元、155个来源文本及57个缺失格通过核对，主代理已查看更新的T3预览，T1/T2沿用字节完全一致的已看图版本。S1–S8补充表实际编译为852行、28页，全部PDF显示行与冻结表格一致；主代理查看八张首页、S6续页及S7全部20页联系图和末页原尺寸。1384份保护文件及93份冻结源码/测试/元数据的SHA和修改时间不变。见 `review/G4_40scene_snapshot_primary_review.json`（`f4666fe2422e594c010de69489df3d5728215278c65f1239544afb779861bc7e`）及 `review/G4_supplement40_preview_primary_review.json`（`35dfa28e222790da5fb7af694acb55e0b8e4e167bc66d4c3d9e1402242214707`）。S6第41行只表示该场景原生proposal已存在，三项GT相关格仍为空，不计入40场景正式评分。公开索引仍为37场景，未执行本轮导出或C1/C2；完整312/8、正式统计及最终合同审核未完成。

此前逐场景核验覆盖 **40/312 个 ScanNet 场景、320/2496 个配置格**，OVI-MAP protocol port 同样覆盖40场景；CPU对照队列已完成第38–40场景，等待第41场景 scene0580_01。最新 scene0580_00 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时275.05、954.16、136.85、350.22秒。112个原owner全部保留，F可用96个；MF关闭时16个unknown、开启时3个unknown，完整GT分母为20个实例、158047点。八组方法/种子共享proposal，MF-off的N/Q访问为0，200次查询及全部帧屏障核验通过。123份检查文件SHA及修改时间保持不变，见 `review/G4_scene40_primary_review.json`。该时点完整分类分数和11张汇总表均覆盖40场景；本地公开格式预览仍对应37场景快照。

此前完整分类分数补齐至40场景、320组配置、22584条对象/方法记录，其中1808条保留undefined。第38–40场景新增24份矩阵、1384条记录、276800个类别分数，全部200类独立复算在1e-12容差内一致；标签、置信度及导出标记精确一致，100条新增无支持记录保留。1007份科学输入/历史证据、592份旧分数工件的SHA及修改时间不变；重复准备复用全部641份当前工件。真实子进程与工具session83027均实际退出0，未运行模型、读取GT或原始几何，也未重算AP或重建公开导出。37场景旧摘要按身份归档，主代理证据为 `review/G4_full_scores_scenes38_to40_primary_review.json`（`7e717ee224d94f9ac434426b4d3495fb87b5e341ad4272b5264824f88dedd0de`）。当前表格已刷新并核查至40场景，公开预览仍是37场景，不能把本地40场景分数及表格交付说成40场景公开导出已完成。

第40场景200份分割输出共3250982字节，逐文件size/SHA、前端与捕获帧序核对通过，296份检查文件保持不变，见 `review/G4_scene40_native_map_actual.json`。OVI对照核对108个作者分类对象，CPU读出及评分81.53秒，新增GPU编码与训练为0；102份文件身份核验通过，见 `review/G4_OVI_scene40_actual.json`、`review/G4_OVI_scene40_primary_review.json`。作者分类对象数不等于自有方法112个owner。主代理追加核对93份冻结源码/测试/元数据不变，见 `review/G4_scene40_continuation_review.json`（`72dd77fb30692c1b879b9b624d35f67a363b190c6459da96d2dadcd8a9b95608`）。这些元数据审核不独立重算AP、重新解码预测数组或读取GT标签值；原队列继续后续场景，完整任务未完成。

此前第39场景 scene0701_02 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时272.29、930.55、83.86、178.37秒。32个原owner全部保留，F可用29个；MF关闭时3个unknown、开启时1个unknown，完整GT分母为5个实例、110945点。八组方法/种子共享同一proposal身份，MF-off的N/Q访问为0；200次查询及全部帧屏障核对通过。主代理检查123份文件SHA及修改时间不变，见 `review/G4_scene39_primary_review.json`。当时池化表格、完整分类分数与本地公开预览为37场景快照，没有将39个逐场景收据说成39场景汇总已完成。

第39场景200份分割输出共2195878字节，296份检查文件保持不变，见 `review/G4_scene39_native_map_actual.json`。OVI对照核对29个作者分类对象，CPU读出及评分59.71秒，新增GPU编码及训练为0；102份文件身份核查通过，见 `review/G4_OVI_scene39_actual.json`、`review/G4_OVI_scene39_primary_review.json`。作者分类对象计数不等于自有方法完整owner计数。主代理追加核对93份冻结源码/测试/元数据不变，见 `review/G4_scene39_continuation_review.json`；这些元数据审核不独立重算AP、重新解码预测数组或读取GT标签值。

此前第38场景 scene0701_01 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时271.56、1000.41、87.22、171.33秒。保留29个原owner，F可用27个；MF关闭时2个unknown、开启时0个unknown，完整GT为4个实例、77206点。八组方法/种子共享proposal，200次查询及帧屏障核对通过，123份文件SHA及修改时间不变，见 `review/G4_scene38_primary_review.json`。

第38场景200份分割输出共1905035字节，296份检查文件保持不变，见 `review/G4_scene38_native_map_actual.json`。OVI对照核对29个作者分类对象，CPU读出及评分57.77秒，新增GPU编码及训练为0；102份文件身份核查通过，见 `review/G4_OVI_scene38_actual.json`、`review/G4_OVI_scene38_primary_review.json`。主代理追加核对93份冻结源码/测试/元数据不变，见 `review/G4_scene38_continuation_review.json`。一次任务内元数据核验因主代理错误并发启动依赖步骤、尚缺原生核验收据而退出1；原失败保留在 `review/G4_scene38_metadata_audit_attempt1.json`，随后按顺序执行退出0。该失败没有启动科学评分或改动生产模块。

此前第37场景 scene0701_00 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时272.36、991.72、100.60、182.48秒。保留30个原owner，F全部可用；全部方法unknown为0，完整GT分母为4个实例、73550点。八组方法/种子共享proposal，200次查询及全部帧屏障核对通过。122份不同路径文件SHA及修改时间不变，见 `review/G4_scene37_primary_review.json`。

第37场景200份分割输出共1816119字节，逐文件size/SHA及前端、捕获帧序核对通过，296份文件保持不变，见 `review/G4_scene37_native_map_actual.json`。OVI对照使用32个已核对作者分类函数的对象，保留完整GT及相同点评测域；CPU读出及评分耗时48.57秒，新增GPU编码及训练为0。身份链和引用数组SHA通过，102份文件保持不变，见 `review/G4_OVI_scene37_actual.json`、`review/G4_OVI_scene37_primary_review.json`。该耗时不是端到端耗时；元数据核查没有独立重算AP、解码预测数组或读取GT标签值。

此前第36场景 scene0583_02 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时275.35、1088.79、114.04、263.37秒。保留50个原owner，F可用46个；MF关闭时4个unknown、开启时0个unknown，完整GT为18个实例、89810点。200份分割输出共3341673字节；OVI对照核对62个作者分类对象，CPU读出及评分48.05秒。逐场景证据分别为 `review/G4_scene36_primary_review.json`、`review/G4_scene36_native_map_actual.json`、`review/G4_OVI_scene36_actual.json` 和 `review/G4_OVI_scene36_primary_review.json`。

此前第35场景 scene0583_01 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时276.42、1110.16、120.67、256.98秒。59个原owner全部保留，F可用46个；MF关闭时13个unknown、开启时2个unknown，完整GT分母为19个实例、98239点。八组方法/种子共享同一proposal身份，MF-off的N/Q访问为0；200次查询及全部帧屏障核对通过。主代理检查123份文件SHA及修改时间不变，见 `review/G4_scene35_primary_review.json`。

第35场景200份分割输出共3750222字节，逐文件size/SHA及前端、捕获帧序核对通过，296份文件保持不变，见 `review/G4_scene35_native_map_actual.json`。OVI对照使用59个已核对作者分类函数的对象，保留完整GT及相同点评测域；CPU读出及评分耗时47.22秒，新增GPU编码及训练为0。身份链和引用数组SHA通过，102份文件保持不变，见 `review/G4_OVI_scene35_actual.json`、`review/G4_OVI_scene35_primary_review.json`。该耗时不是端到端耗时；元数据核查没有独立重算AP、解码预测数组或读取GT标签值。

此前第34场景 scene0583_00 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时278.05、1179.48、123.81、270.65秒。69个原owner全部保留，F可用63个；MF关闭时6个unknown、开启时3个unknown，完整GT分母为23个实例、146782点。八组方法/种子共享同一proposal身份，MF-off的N/Q访问为0；200次查询及全部帧屏障核对通过。主代理检查123份文件SHA及修改时间不变，见 `review/G4_scene34_primary_review.json`。

第34场景200份分割输出共3885352字节，逐文件size/SHA、前端输入输出帧序和捕获scheduled/completed/actual帧序全部一致，296份文件保持不变，见 `review/G4_scene34_native_map_actual.json`。OVI对照使用68个已核对作者分类函数的对象，保留相同完整GT和点评测域；CPU读出及评分耗时61.71秒，新增GPU编码及训练为0。预测、评分、GT、matches及引用数组/原生来源的身份链通过，102份文件保持不变，见 `review/G4_OVI_scene34_actual.json`、`review/G4_OVI_scene34_primary_review.json`。该耗时不是端到端耗时。这些核验均未独立重算AP、重新解码mask或读取GT标签值；全部60个生产模块及23个原测试模块保持不变。

本轮新增三个任务内只读核查入口 `bootstrap/verify_native_scene_metadata.py`、`verify_completed_scene_metadata.py`、`verify_completed_ovi_metadata.py`，均以冻结协议的场景序号定位输入，第34–40场景最终执行均退出0。它们只生成自己的审核收据，不启动模型或评分器；原收据存在时拒绝改写。完整分类分数现对应41场景，11张表现对应40场景；本地公开预览仍对应下述已验证37场景快照。

此前第33场景 scene0025_02 的分割、建图、F、N/Q 四个作业均实际退出0，分别耗时305.31、1266.16、148.74、420.48秒；200帧顺序和200次查询预算核验通过。全部124个原owner保留，F可用106个；MF关闭时18个unknown、开启时11个unknown，完整GT为35个实例、177900点评测域。39份已检查文件SHA及修改时间不变，见 `review/G4_scene33_primary_review.json`。分割200份输出共5360887字节，逐文件size/SHA及输入输出帧序核对通过，捕获帧序也与冻结200帧计划完全一致；209份已检查文件保持不变，见 `review/G4_scene33_native_map_actual.json`。该场景检查覆盖元数据、trace、引用数组SHA及匹配身份，不重复解码mask或独立重算AP。该收据创建时，原控制器进入第34场景 scene0583_00。

此前第33场景OVI对照使用120个已与作者分类函数核对的对象，保留完整35个GT实例和177900点评测域；CPU原生读出与评分耗时79.57秒，新增GPU编码及训练为0。主代理核对冻结预测、评分、GT、匹配、脚本及引用数组SHA，全部60个生产模块和23个测试模块的SHA及修改时间不变。证据为 `review/G4_OVI_scene33_actual.json` 和 `review/G4_OVI_scene33_primary_review.json`。这不是端到端建图耗时。该收据对应有限CPU入口完成本批首场景时的状态。

原第28–32场景CPU入口在第31场景遗漏runner已有的 `score_compat.install()`，触发作者无GT分支的辅助 `ar_current` 未初始化错误。原失败进度和日志保持原样。有限恢复入口 `bootstrap/score_ovi_scenes31_to32_recovery.py` 补装既有运行时适配，复用第31场景冻结预测，并已完成第31、32场景；恢复进度为DONE，进程已结束。后台启动没有捕获整个进程的最终退出码，完成依据是逐场景成功收据及终态进度。第31场景16.61秒仅为恢复评分耗时，首次失败的精确耗时未保存，不能作为完整效率。

恢复前的真实诊断复现了原错误；适配仅补辅助NaN赋值，完整AP AST与作者源码保持不变。把有GT类别临时放在原函数遍历首位、再恢复类别轴得到的1980个AP数组位置与适配结果完全一致，95份已检查文件SHA及修改时间不变，见 `review/G4_OVI_scene31_AR_recovery_diagnostic.json`。第一次诊断错误地禁止作者模块正常导入PyTorch，该附加检查自身失败；改为检查CUDA未初始化后通过，原失败记录保留。主代理恢复审核见 `review/G4_OVI_scene31_recovery_primary_review.json`。

此前完整分类分数补齐至37场景、296组配置、21200条对象/方法记录，其中1708条无支持记录明确保留。第33–37场景新增40份矩阵、2656条记录、531200个类别分数，全部200类独立复算在1e-12容差内一致；原标签、置信度和导出标记精确一致，228条新增无支持记录保留undefined。818份科学输入/旧证据、512份旧分数产物SHA及修改时间不变，重复生成复用全部593份当前产物。实际执行退出0，见 `review/G4_full_scores_scenes33_to37_actual.json`；这一步新增模型推理、训练、GT或原始几何读取均为0。

11张表已按37场景刷新，核查4549个数值格、155个来源文本和54个缺失格。当前子集池化AP为U00 7.35、U10 9.53、U01 12.37、U11 12.91、OVI-MAP protocol port 9.55（%；U00–U11为两个种子的池化结果均值），不代表完整标准集。评分、统计、baseline、表格和review阶段实际退出码为20/20/20/0/0；更新视觉检查后再次review实际退出0。285份受保护元数据、源码、测试和旧公开预览输入SHA及修改时间在刷新审核时不变。主代理已查看新的T3预览，T1/T2的TeX、PDF和PNG与此前看图版本字节完全一致；完整范围主表缺失格保留。证据为 `review/G4_37scene_snapshot_primary_review.json`、`review/G4_actual_aggregation.json`、`review/visual_QA.json`。60个生产模块和23个测试模块未变，沿用此前实际通过的97项CPU测试，不重复运行未改变的测试。

本地公开格式预览也已扩展并验证至37场景：8950个文件、8912个不同分片、338819233字节，满足512 MiB总量及40 MiB单分片限制。直接从分片还原新增40份矩阵、2656条记录，类别轴及冻结标签/置信度/导出标记一致；原256份标准及176份Legacy矩阵的1714个已核查payload保持原样，复用原数值证据。本次共检查1875个来源payload和1873个不同分片，没有重新读取本地科学输入。导出DONE收据记录1077份文件SHA及修改时间保持；导出进程最终退出码未在恢复后的工具会话中观察，不另推断为0。分片验证和主代理身份链核查实际退出0，主代理另核对284份受保护文件不变。证据为 `bootstrap/37scene_preview_export.json`、`review/G4_37scene_scores_export.json` 和 `review/G4_37scene_public_preview_primary_review.json`。此前32场景索引按身份归档，详见 [CLASSIFICATION_SCORES.md](CLASSIFICATION_SCORES.md)。尚未执行C1/C2。

第33–37场景的有限CPU对照入口 `bootstrap/score_ovi_scenes33_to37.py` 已完成全部五场景，进度为DONE；后台进程整体最终退出码未观察，完成依据为五份实际成功收据及终态进度。入口先安装同一AR兼容处理，并要求此前恢复批次DONE；不得重复启动。入口审查、启动记录分别为 `review/G4_OVI_scenes33_to37_entry_review.json`、`bootstrap/ovi_scenes33_to37_launcher.json`。完整分数、刷新、导出及分片核验均已按前述证据实际完成。此前只有32场景时的真实前置拒绝检查仍保留为历史准备证据：`review/G4_full_scores_scenes33_to37_preflight.json`、`review/G4_37scene_delivery_preflight.json`；不将这些预检本身算作数值完成。

该批任务目录内的有限串联入口 `bootstrap/run_37scene_delivery_chain.py` 复用既有原进程/五场景完成门槛。8项隔离真实子进程检查先失败后全部通过，另一次真实等待前检退出0且没有启动串联子进程；这不是新增生产回归或G3。主代理完整入口审查及93份受保护文件SHA/修改时间核对通过，见 `review/G4_37scene_chain_entry_primary_review.json`。入口仅启动一次，PID924113/start51278115、tool session89994；五个串联步骤均实际退出0，整体实际退出0，进度为DONE/READY_FOR_PRIMARY_VISUAL_REVIEW。它完成三个元数据核验、完整分数复算及刷新后停止，没有执行导出或发布；主代理随后已完成看图及公开预览核验，证据见上文。启动收据和终态进度分别为 `bootstrap/37scene_chain_launcher.json`、`bootstrap/37scene_delivery_chain/37scene_chain_progress.json`。不得重复启动。原GPU控制器及后续全范围CPU队列继续运行。

2026-10-23 01:57 UTC，后续有限CPU入口 `bootstrap/score_ovi_scenes38_to312.py` 已实际启动，PID805492/start50709689；前批五份有序完成证据、评分及完整GT身份现已通过，已安装既有AR兼容处理。后续批次已完成第38–40场景，当前状态为WAITING_FOR_ORIGINAL_SCORE，等待第41场景scene0580_01。它绑定前批入口的原PID/start和源码SHA。冻结队列包含第38–312场景共275项；最多直接评分前274项，最后一项始终交给原后处理程序，避免控制器退出瞬间的竞争。前批失败、缺失或意外退出不会触发替代启动。

15项独立临时目录编排检查实际退出0，包括前批等待、完成、失败、进程消失、记录损坏及最后场景归属；这不是新增科学G3或生产回归结果。主代理完成完整差异及AST审核，原逐场景数值与验证代码一致；真实前批仍在等待时，预检未调用install或score，93份受保护文件SHA及修改时间不变，其中包含全部60个生产模块和23个测试模块。证据为 `review/G4_OVI_scenes38_to312_handover_tests.json`、`review/G4_OVI_scenes38_to312_entry_review.json` 和 `bootstrap/ovi_scenes38_to312_launcher.json`。新入口只启动一次，不代表后续275场景已经完成；后台整体最终退出码当前尚未产生。该次启动核验时，原GPU控制器和前批CPU入口保持运行，第34场景进入mesh生成；随后完成的第34场景及当前第35场景状态见上文。

完整312场景、正式Replica48、外部完整基线、全量配对统计、最终提示词逐项审核和C1/C2发布仍未完成。以下保留历史诊断与外部资产查询，数字均对应其记录时的范围；已核验逐场景完成数见上文；[RESULTS.md](RESULTS.md) 与表格对应已验证的37场景汇总快照。

本轮按原合同第17章补齐此前缺失的S1–S8编译预览：8表、794行、26页，S6为2页、S7为18页。单独渲染入口保留原表注、表头、面板和每行TeX原字节；16次TeX编译及24次PDF检查/渲染命令全部实际退出0。主代理已查看8张首图、S6续页、覆盖S7全部18页的3张联系图及其末页，无可见裁切或重叠。受保护的21份表格来源和93份既有源码/测试/元数据不变；受控重复CLI执行退出0、未启动子进程，81份文件SHA及修改时间不变。这些新预览位于本地tables目录，尚未进入当前37场景公开索引；最终tables后须重跑渲染入口，再进行review/export。命令及证据见 [SUPPLEMENT_PREVIEWS.md](SUPPLEMENT_PREVIEWS.md)。

后续衔接检查复现了首版渲染器把文件时间/审核收据更新误判为缓存变化的问题；仅修复独立渲染脚本的复用逻辑。真实缓存临时副本上的3项检查先2项通过、修复后全部通过，仍拒绝改动的表值和损坏的PDF，未启动子进程。按新脚本身份实际重建退出0，26页PNG与此前看图版本逐字节一致，794行PDF显示值再次独立核对一致；原21份表格/审核来源及93份源码/测试/元数据不变。新的实际CLI重复执行退出0并复用缓存，原失败与旧源码均保留。只读导出候选检查选中33份预览/摘要文件、原始共2200888字节；没有执行导出或改动当前公开索引。主代理证据为 `review/G4_supplement37_reuse_fix_primary_review.json`；这些布局检查不计入97项生产测试或科学G3。

发布阶段另发现环境没有 `gh` 或既有GitHub API凭据，原远端内容核验因此无法读取公开仓库。已在 `bootstrap/public_remote_reader_candidate/` 准备精确补丁：无gh时通过固定提交的官方公开Contents接口读取，继续核对字节SHA；原gh路径及失败处理保留。新增5项传输检查与原4项发布测试合并实际9项通过，另一次真实父提交读取取得3628字节，SHA与本地完全一致。主代理审核见 `review/G4_public_remote_reader_candidate_review.json`，合并测试证据为 `bootstrap/public_remote_reader_candidate/deployable_green.json`；这不是新结果发布或C1/C2证明。补丁尚未应用，全部60个生产模块和23个原测试模块保持不变；待原科学队列及后处理结束后，在最终源码/提示词审核及C1之前接入。无认证的release溢出分片仍按既有规则保留在本地，不能宣称已上传。

进一步在60个真实模块的独立副本中核对补丁边界，实际退出0：55处静态调用及4处动态调用的8种闭合方法展开，共63个数值producer身份全部保持不变；全局终态身份按预期改变。原生命周期代码正确拒绝旧身份的合成终态，重新闭合的合成目录恢复返回预期partial码20且文件SHA/修改时间不变，没有导入科学库。276份实际受保护文件保持不变，实际任务未创建terminal，也没有接入补丁或发布。证据为 `review/G4_publication_candidate_producer_boundary.json`；这是独立副本诊断，不是实际发布后恢复证明。

随后核对发现，发布路径列表未收集新增的补充表渲染脚本。已保留上述旧候选，另在 `bootstrap/publication_delivery_candidate_v2/` 准备包含公开远端读取修复及单条渲染脚本路径的合并补丁。临时Git仓库/本地bare远端的实际发布测试先复现1失败、9通过；补齐路径后10项全部通过，核对渲染脚本进入C1、无关脚本排除、C2仅含收据和交接、恢复不新增提交。精确补丁在临时源码副本实际应用后与候选逐字节一致，100份实际受保护文件保持SHA/修改时间；全部60个生产模块和23个原测试模块仍未修改。

新合并候选的60模块副本核查实际退出0，63组数值身份仍不变；303份实际文件保持不变，全局producer按预期改变，旧合成终态被拒绝、重新闭合后只读恢复返回20。主代理审核为 `review/G4_publication_delivery_candidate_v2_primary_review.json`（`02f86bab19d530310cba7f5c5e2be32eb53830aa31ce0823356379075abff121`），新边界证据为 `review/G4_publication_delivery_candidate_v2_producer_boundary.json`。后续应使用这个合并补丁，不先应用旧补丁再重复应用它；仍须等原科学队列及后处理结束、重新做最终源码/提示词审核后接入。10项是候选发布检查，不是新增科学G3或原97项生产回归；没有实际C1/C2、远端新结果发布或真实终态恢复。

2026-10-23 另核对MaskClustering和Entity两个官方GitHub releases入口，均返回有效空列表，没有新增权重或Replica GT/点序。主代理核对两份响应共4字节的size/SHA及请求字段，93份既有受保护文件保持不变，见 `assets/external_asset_discovery/official_release_assets_20261023/primary_review.json`。这仅覆盖两个公开发布列表，不代表其他来源不存在；未更改资产绑定。

另检查官方CropFormer README指向的公开演示站元数据入口：直接访问Hugging Face API网络不可达；镜像同路径返回3931字节通用HTML首页，未取得固定提交、文件清单或应用源码。镜像首次按JSON解析失败，原响应已保留，并从保留字节补记错误收据，没有重发请求；当次精确请求耗时未留存，明确记为未知。主代理核对4份响应/收据文件的SHA与字节数及原README链接，见 `assets/external_asset_discovery/cropformer_linked_demo_20261022/primary_review.json`。该次未获得模型、未判断替代权重兼容性、未更改实验绑定；这些有限访问失败不能证明其他来源不存在。

本次另外核对3个公开模型/数据集元数据检索响应：EntitySeg结果仍包含受限的作者仓库，CropFormer和Mask3D模型查询未返回记录。没有取得新的原始checkpoint或Replica实例GT，也没有改变资产绑定；这些有限查询不能证明其他来源不存在。响应字节、字段与SHA经主代理复核，见 `assets/external_asset_discovery/public_model_search_20261022/primary_review.json`。现有可补算场景继续运行，缺源文件的任务保留具体依赖。

随后检查公开Replica索引返回的首100条记录，并对8个相关仓库固定提交、检查完整文件清单；主代理复核16份元数据或说明文件的字节及SHA。Ethondong候选的room_0.npz实际下载成功、匹配公开LFS哈希，但仅有origin、voxel_size、tsdf三项数组，不能提供实例标注；只读取数组名称、形状和dtype，没有读取GT标签值。其余已检查说明分别涉及原始Replica归档、RGB-D子集、2D训练数据或四场景裁剪；未下载的大型归档仍明确记为内容未核实。没有候选被认定为本任务的正式48类实例GT/点域，也未更新资产绑定。完整范围和逐候选判定见 `assets/external_asset_discovery/replica_public_metadata_20261022/primary_review.json`，不据这些有限查询宣称不存在其他来源。

随后通过 HTTP 范围读取两个固定提交 ZIP 的完整目录：KevinConnorLee 的压缩包32035项、GS3LAM 的压缩包64051项，手工结构解析与 Python ZIP 目录读取逐项一致。前者目录包含 RGB-D、轨迹及8份网格，后者另外含逐帧二维语义图。仅解码两份 room0 网格的文件头，均没有实例或语义标签字段；GS3LAM 的 room0_pc_unseen.npy 仅解码头部，类型为 float64、形状为(1146,3)，没有读取数组值。另一个公开 room0 PLY 的头部与 GS3LAM 对应头部相同。9份响应的范围、字节数和SHA核对通过，实际检查退出0，见 `assets/external_asset_discovery/replica_public_metadata_20261022/archive_headers/primary_review.json`。没有下载完整ZIP，因此未验证全归档LFS哈希，也未检查其余场景的网格内容；这些候选仍不能证明本任务所需正式48类实例GT及评测点顺序，资产绑定保持不变。

另一候选镜像 lumpin/replica 的固定提交 `e347ffb99c81d908fcbf0be00cf4f16b9e3c40f9` 列出17个 Replica v1.0 分卷，总33859047808字节。已取得完整文件清单和首卷前1 MiB；主代理核对响应SHA、206范围及最初三个TAR头，只解码1536字节归档头，首个数据文件是 `apartment_0/textures/27-color-ptex.hdr`，没有解码文件内容。[Replica官方说明](https://github.com/facebookresearch/Replica-Dataset#replica-dataset-1)列出原始发布的实例语义网格及对应名称映射，因此继续核验这一候选。原下载现已完成17/17卷、33859047808字节，实际退出0；逐卷下载校验及随后独立size/LFS SHA复核均通过，最大请求次数为2，没有残留未完成分卷。进度与逐卷收据位于 `assets/external_asset_discovery/replica_public_metadata_20261022/lumpin_archive_probe/full_archive/`，元数据审核为其上级 `primary_review.json`。CPU流式归档检查实际退出0，耗时约292.95秒，共576个目录项。8个目标场景均有 `habitat/mesh_semantic.ply` 和 `info_semantic.json` 目录项；8份PLY头均含面级 `uint16 object_id`。主代理核对目录与文件头SHA、源文件状态及实验锁未变，证据为同级 `archive_catalog/primary_review.json`。检查没有解析GT标签值或映射JSON内容。正式48类GT及评测点序兼容性仍未建立，资产绑定保持原样。作者Open3DIS issue5确认其预处理会采样原始点集，但具体采样索引与转换脚本尚未取得；OpenMask3D issue5说明其处理后GT通过邮件分享，当前未联系作者。4份公开作者回复/索引响应已保存并校验于 `replica_preprocess_source_queries_20261022/`，不据此猜测点顺序或生成替代正式GT。下载脚本重入会重置尝试计数的问题，已在独立新入口 `full_archive/download_full_archive_resume.py` 修复：保留旧收据、消耗已开始但中断的尝试、每卷累计最多两次请求，并保留完整文件复用的原历史。三项隔离CPU回归在旧版全部失败、新版全部通过；新入口面对真实存活的原下载进程实际退出20，未创建第二个下载。证据见同目录 `retry_repair_review.json`。原下载源码保持原样，原进程现已正常完成全部分卷；尚未执行真实归档断点恢复，以后若需恢复应使用新入口，不能重跑会清空历史的旧脚本。

新增[提示词阶段审核](REQUIREMENT_AUDIT.md)，对应原17章的39个核验项，明确保留完整科研、正式统计、外部真实接入及C1/C2未完成。逐项核对当前19场景152组冻结manifest、GT冻结引用、评分身份及9816条对象/方法来源记录；共享mask身份、MF-off零N/Q、无源unknown和完整GT元数据一致。scene0414_00的U11两seed为精确预测别名，因此是152组逻辑结果、151份实际评分输入。初版审核误要求逻辑manifest与复用评分manifest相同，已按原评分器的明确alias_of及相同prediction_identity修正审核逻辑，未改生产评分；原失败和修正依据保留。251份已检查元数据的SHA和修改时间均未改变。此检查不重读全部mask数组或GT标签，不替代首场景实际数组/官方入口及此前公开数组复算证据；机器清单见 `review/G4_requirement_audit_partial.json`，没有生成最终审核或终态关闭记录。

此前18场景本地发布预览为235485946字节，144份标准矩阵和176份旧实验矩阵均已从公开格式分片恢复，与同一导出中的原预测逐对象或逐正owner顶点核对；记录见 `review/G4_legacy_full_scores_export.json`。表格刷新后的索引再次逐项核对1266个已验证输入及其分片字节一致，没有重复读取原始数据。旧实验实际核查保留6438份科学输入的SHA和修改时间，353份新增分数产物重复导出不改写；实现位于 `legacy_score_artifacts.py`，通过 `publication_export.py` 接入，新增3项确定性测试。此前5项案例产物的恢复核查见 `review/G4_real_cases_export.json`。这是历史本地预览；完整实验和正式统计仍在等待剩余场景及缺失输入。下文保留各阶段的历史诊断与恢复记录，当前覆盖以本节和RESULTS为准。

资源检查另修复了 `runtime.py` 的失败记账：外部GPU子进程非零退出，或退出0但缺必需输出时，失败收据现在保留原有 `gpu_worker_launches` 字段，使实际墙钟耗时进入累计与显式配额检查。2个新增回归在旧代码均失败，修复后完整97项测试通过；测试只运行隔离的CPU子进程并模拟GPU任务标记，没有真实GPU锁、模型或训练。实际现有76份作业收据均成功、无重试；加2份早期worker记录，新旧统计同为14568.03188873519秒已完成GPU worker墙钟上界，85份科学记录SHA/修改时间不变。不能把这个含CPU与加载开销的上界当作纯GPU计算时间。实际核对与主代理审核分别见 `review/G4_failed_GPU_cost_actual.json` 和 `review/G4_failed_GPU_cost_primary_review.json`；16个在运行数值产出模块、任务身份与重试规则保持不变，原控制器未重启。本地发布预览仍为前述18场景快照，最终C1将重新导出最新检查记录。

新增作者资源候选为 OpenYOLO3D 的 Replica 数据包。已固定作者提交 `24f344d001d28d1cdd4087d3769c0d53c34d64cd`；原 raw 文件访问失败后，通过 GitHub Contents API 恢复3份说明/下载脚本，并逐份核对 Git blob SHA。作者说明包含逐场景彩色 PLY 和 ground_truth 文本，但尚未取得数据字节，不能认定其类别 ID、点顺序与本任务一致。数据包的 Google Drive 两个公开入口均网络不可达；MaskClustering 所需576d0b权重的公开镜像文件入口返回403，仍未取得权重。访问与源码证据分别保存于 `assets/external_asset_discovery/OpenYOLO3D_author_bundle/candidate_access.json`、`source_transport_recovery.json` 和 `assets/external_asset_discovery/author_alternate_public_endpoints.json`。这些记录只证明来源与实际访问结果，不表示数据或权重已补齐；当前不继续重复请求同一失败下载。

Replica 的8个原始 RGB-D 场景均已核查并整理为作者 reader 可读取的目录：每场景2000帧，原始相机1200×680、深度除数6553.5、全部原始 c2w 位姿，保留作者 step10 的200帧计划。这只是输入准备，不能记为正式评测完成。

Replica 还缺两类证据：正式48类实例标注及对应原始评测点顺序；14个类别与训练语义空间的可靠对应关系。下载到的 OpenScene 三元组只有坐标、颜色、语义，没有实例标签，不能充当正式实例 GT。作者数据获取失败已保留记录；这些缺项只阻塞相关 Replica 结果，其他任务继续。

对14个未定类别再次逐项核对已绑定的官方ScanNet TSV，发现4个直接同名的原始类别条目：camera/1266、pot/1234、vase/194、rug/61；这4个ID均不在冻结的200类文本轴中。pot条目的WordNet key实际为 `sleeping_bag.n.01`，与其类别文字不一致，需单独核实。这些是候选类别元数据，尚不足以确认全体14类的base映射；没有自动把轴外类别归为novel，也没有采用NYU粗分组或GT频率。核查只读类别元数据，TSV、协议、模型和绑定的SHA及修改时间均不变，记录见 `assets/external_asset_discovery/replica_role_metadata_review_20261022.json`。正式类别角色和活跃协议尚未更新。

第四个 ScanNet 场景的评分另遇作者脚本错误：首个类别没有 GT 时，辅助 `ar_current` 未赋值。新加的 `score_compat.py` 仅在运行时补该辅助变量，完整 AP AST 和作者源码字节保持不变。修复前后，已有3场景24组真实 AP 数组完全一致；第四场景的8组冻结预测已通过 CPU 评分恢复，没有新增 GPU 推理。

`score_recovery.py` 的单 CPU 监控会处理仍在运行的旧队列中同类已记录错误，保留原始失败日志并限制重试次数。新 CLI 调用直接安装该兼容适配。相关10项兼容、I/O 与 G4 测试以及4项发布测试通过；正式实验和最终审核尚未完成。

代码位于本任务 namespace 的 `score_compat.py`、`score_recovery.py`、`replica_layout.py`、`mask_clustering_io.py`、`baseline_io.py`；运行及交接方式见 [HANDOFF.md](HANDOFF.md)。

MaskClustering 现已接通 `mask_clustering_full.py` 的五阶段原方法调度，以及 `mask_clustering_score.py` 的预测冻结、完整 GT 转换、共同评分、作者逐类别×IoU AP 核对和8场景池化。聚类参数、step10、三尺度 CLIP-H14 特征、mask 重叠和原始置信度保持作者定义；评分不依赖 Ours 的 bundle 或训练类别校准。合成连接测试不进入正式池化。

实际 `baselines` 预检已为8场景保存 `external/MaskClustering/pipeline/replica48/*/availability.json`，没有启动 MaskClustering GPU 作业。已补齐 CLIP-H14 权重，3944692325字节，SHA256与作者公布值 `9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4` 一致。独立环境已完成下述 CPU 依赖与入口核查；最新预检仍缺作者 Mask2Former576d0b 权重、正式 Replica 实例 GT/评测点域。因此真实 Replica G3 和正式结果尚未完成。作者报告值逐数值标注†，只有完整本地池化可填对应正式格。

外部方法单独执行磁盘检查，不因 Ours 缓存已完成而跳过。room0 的实际元数据估计要求60.430 GiB空闲；此前检查有5604.999 GiB，记录于 `review/G4_OpenMask_capacity.json`，没有启动模型。ScanNet 当前已完成11场景88组评分，第12场景 scene0414_00 正在执行；完整范围与正式对比仍未完成。

后续按可恢复的输入继续执行：ScanNet 按冻结的312场景名单复用本地原始文件，并通过已有授权逐场景获取缺失的原始文件，再用冻结权重补齐建图、图像特征与读出缓存；断点续跑时复用身份一致的产物。312是评测名单规模，不表示全部场景的原始RGB-D文件都已在本地。现有单 GPU 队列保持运行，避免重复启动。Replica 的正式实例 GT/评测点顺序、14个类别的训练空间对应证据，以及外部基线的独立环境和原始权重分别列为依赖；只有缺少这些输入的任务等待，不能用语义三元组、猜测标签或论文数值代替本地正式结果。

OpenMask3D 的 `openmask_io.py` 已通过作者原始相机、深度尺度、完整位姿、NumPy float16 mask 文件、重叠 mask、零特征、原始恒定置信度和彩色 PLY 检查。控制环境缺少 imageio，仅通过任务目录中的包及 dist-info 链接复用现有 imageio2.37.3，测试进程单独设置 PYTHONPATH，没有修改已有环境。

`openmask_full.py` 现已接通原始提案、SAM/CLIP 图像特征、原始文本特征和原始语义查询四阶段，保留120个提案、DBSCAN0.95和原始多视图设置。`openmask_score.py` 要求所有真实阶段作业成功后才能冻结预测；与 MaskClustering 共享的 `replica_baseline_score.py` 保留完整 GT、精确点顺序、逐类别×IoU 作者 AP 核对和官方池化。首场景返回 DONE 但未通过真实 G3 时，也会停止后续场景；合成连接测试不能放行或进入正式池化。

最新56项 CPU 回归测试通过，记录为 `review/G4_cpu_validation.json`，身份 `d17195c144022745ff9a7c254f33f92c472245855aa4c142de572e5286aece9b`。119个 OpenMask3D 作者代码/配置文件与固定 Git 源码一致；16个正在运行的数值产出模块字节保持原样。真实预检为8场景保存 `external/OpenMask3D/pipeline/replica48/*/availability.json`；已补齐 SAM ViT-H 和 CLIP-L14@336 权重，下载记录见 `assets/external_asset_discovery/OpenMask3D_public_models.json`，其中 CLIP 的 SHA256 与作者公布值一致。OpenMask3D 独立环境已建立，基础依赖检查通过，原始 CUDA 算子编译尚未全部完成，不能仅凭 Python 文件存在认定完整环境可用。仍缺任意场景 Mask3D checkpoint 和彩色评测 PLY/实例 GT。本次外部方法 GPU 作业为0，真实 Replica G3 和完整正式对比仍未完成。主表编译和检查记录见 `review/visual*.json`，当前覆盖见 [RESULTS.md](RESULTS.md)。

MaskClustering 独立环境的离线克隆因 conda 包缓存不完整失败，已保留日志；在线克隆随后实际退出0。在任务环境中补齐 OpenCLIP2.20.0、PyTorch3D0.7.3 等缺失依赖，保留 PyTorch2.0.0/CUDA11.8、torchvision0.15.0、NumPy1.26.4。CPU 上真实 `ball_query` 返回预期邻居与距离，原始图聚类/特征入口及生产使用的精确 Replica reader AST 导入通过，`pip check` 退出0。环境冻结清单和检查记录汇入 `review/G4_MaskClustering_environment.json`；这些检查不证明 GPU 完整推理或正式 G3 已完成。

真实入口检查发现项目 `src/utils` 的普通 Python 包遮蔽作者同名 namespace 目录。`mask_clustering_full.py` 现仅在作者调用上下文内绑定作者 `utils`，退出时恢复宿主全部同名模块；作者文件没有修改。两种宿主导入状态的回归先失败后通过，完整52项任务测试通过。补齐权重和软件环境只能恢复推理依赖，不能替代 Replica 缺失的正式实例标注与评测点顺序。

OpenMask3D 使用任务目录中的新环境，保留作者 Torch1.12.1/cu113、torchvision0.13.1 及其他显式版本；官方匹配 torch-scatter 二进制包和固定版本的编译源码均已下载。旧 PyYAML5.4.1 的隔离构建实际报错 `cython_sources`，已按 PyYAML 官方 issue736 对构建子进程约束 `Cython<3`，保留原版本的安装实际退出0。随后12项依赖导入、真实 CPU scatter_mean、原始特征入口定义和基础 pip check 均通过。Python3.10.20 是本任务的兼容性选择，作者推荐3.8.5。原始 MinkowskiEngine0.5.4 的 Sequence 导入错误已实测重现；`openmask_full.py` 仅在作者调用上下文中提供相同的标准库 ABC 别名，退出时恢复宿主，作者源码保持原样。异常退出和已有别名两种回归通过，全部54项任务测试通过。部分环境证据和完整冻结清单在 `review/G4_OpenMask_environment.json`，明确保留正式 G3 为 false。

后续原始 CUDA 构建先暴露 cuRAND/cuSOLVER 头文件缺失，已从 NVIDIA CUDA11.6.2 官方清单下载匹配组件并核对官方 SHA256，补充文件仅位于任务目录。原始 CUDA 编译单元的头文件预检实际退出0，再以 MAX_JOBS=1、SM8.6 和空可见 GPU 编译；CUDA11.6 编译器与 Torch11.3 运行时的次版本差异单独记录。完整算子、原始 proposal 入口和 GPU 推理仍须实测。过程和失败日志保留于 `bootstrap/env_creation/OpenMask3D_*`。

进一步实际导入发现，项目的普通 `datasets` 包遮蔽作者缺少 `__init__.py` 的同名目录，使原始 proposal 入口无法导入 `datasets.scannet200.scannet200_splits`。`openmask_full.py` 现仅在作者调用上下文中绑定固定源码目录，并在正常/异常退出时恢复宿主全部同名模块。两个回归先失败后通过；实际 Python3.10 环境也成功导入原始模块并验证恢复。完整56项 CPU 测试和逐文件核查记录于 `review/G4_OpenMask_namespace_primary_review.json`，不改变作者文件、算法或当前数值产出模块。

完整 MinkowskiEngine 构建在第20/21个文件暴露 Torch1.12 的头文件顺序问题：`CUDAUtils.h` 使用了尚未完整定义的 `at::Tensor`。最小编译样例实测原顺序失败，使用编译参数 `-include ATen/core/Tensor.h` 后退出0；原始作者源码保持不变。19个已编译对象及失败日志已归档，再按新编译参数构建，证据见 `bootstrap/env_creation/OpenMask3D_Tensor_header_preflight.json`。该预检只证明头文件修复，完整算子和原始入口仍须等实际检查结果。

后续完整构建已成功编译原失败的 `spmm.cu`，但第21个 Python 绑定文件出现 nvcc/pybind11 枚举模板推导错误。独立四行绑定样例在 C++14、C++17 及无 Tensor 预包含时均重现该错误，说明新错误不能通过前述头文件修复消除。任务目录中的 GCC9.3 工具链实际安装成功，但同样未修复该错误；独立下载的官方 pybind11 2.9.2 头文件也未修复问题，因此这两项均未用于新的完整构建。失败日志和收据全部保留，未启动完整基线推理。

现已找到作者在官方 [issue414](https://github.com/NVIDIA/MinkowskiEngine/issues/414#issuecomment-952498318) 给出的同一错误修复：用显式 `__eq__` lambda 调用原 C++ 等号。独立样例在原工具链下实际编译、加载成功；相等、不等及异类型 `NotImplemented` 行为通过检查，记录为 `bootstrap/env_creation/OpenMask3D_author_eq_compatibility.json`。任务构建副本的243个固定源码文件中仅 `pybind/extern.hpp` 发生变化，并保留 `py::is_operator()` 以维持原异类型比较语义；全部数值源码与原 Git blob 一致。原始 checkout 和20个成功编译对象继续保留，固定 OpenMask3D 模型源码未改。完整依赖正在该独立副本中重新构建，真实算子/原提案入口检查将在实际成功安装后执行；不能把此最小样例当作正式 G3。

10场景子集的共同协议池化 AP 为：U00 10.02、U10 11.44、U01 12.94、U11 13.23、OVI-MAP protocol port 11.09（%）。这些仅是10/312的部分结果，不能作为完整标准集结论。已更新11张表，查看三张当前主表预览，并核查1444个数值单元、155个文本来源和54个缺失格；记录见 `review/visual_QA.json` 和 `review/G4_actual_aggregation.json`。

另核查作者在 Open3DIS issue5 中提供的 Replica 原始数据和预处理脚本链接，实际访问返回401，记录于 `assets/external_asset_discovery/Replica_author_preprocess_source_access.json`。取得该脚本及原始数据后仍须核对正式评测点顺序和完整实例标注，不能凭“采样一半”描述自行生成替代 GT。

对同一作者预处理包公开链接补做一次普通GET检查，设置4096字节读取上限和15秒超时，实际仍返回401。未获得归档或转换脚本；三份科学资产锁的SHA及修改时间不变。该证据补足此前仅有HEAD检查的范围，见 `assets/external_asset_discovery/Replica_author_preprocess_GET_probe.json`；不再重复请求同一入口。

最新恢复：MinkowskiEngine、pointnet2、Detectron2 的完整编译与安装均实际退出0。首次实际导入发现 `libcusparse.so.11` 不在任务环境的查找路径；只在任务环境中绑定现有只读 CUDA11.6 的 cuSPARSE11.7.2.124 和 cudart11.6.55，保留原失败收据。实际稀疏量化、手算卷积结果2/6、等号/哈希/异类型行为、pointnet2 原始绑定、CPU ROIAlign2.5 和原始 proposal/Hydra 入口均通过。`pip check` 实际仍退出1，且仅有作者指定 Hydra/OmegaConf 与 Detectron2 元数据的两项已知冲突；严格检查只接受这两项及两条精确的旧依赖弃用警告，10项检查拒绝额外依赖、未知消息、重复或缺失冲突。没有升级版本或修改作者模型。实际结果见 `bootstrap/env_creation/OpenMask3D_native_runtime_recovery.json`，主代理核查见 `review/G4_OpenMask_native_primary_review.json`。这证明任务环境的原生 CPU 算子和原始入口可用，完整 GPU 推理、正式 Replica G3 仍等待原始 checkpoint、彩色评测 PLY 和实例 GT。

第11场景的捕获清单与冻结200帧计划完全一致，真实 F/NQ 特征、冻结预测和8组评分均已完成。最新11/312子集的共同协议池化 AP 为 U00 9.17、U10 10.52、U01 12.56、U11 12.78、OVI-MAP protocol port 10.86（%）；仅适用于该子集。11张表和生成文档已更新，三张新主表预览均由主代理实际查看；1559个数值单元、155个来源文本和54个缺失格核查通过，见 `review/G4_actual_aggregation.json`。标准全量、Replica、最终审核和发布仍未完成。

后续 GPU 依赖检查已在继承的物理 GPU0/A40 锁内完成：MinkowskiEngine 稀疏卷积输出2/6，pointnet2 的最远点采样、球查询和特征聚合，以及 Detectron2 ROIAlign2.5 均符合手算值。单个诊断子进程实际墙钟13.934748秒，已通过统一 GPU 作业收据计入资源核算。检查没有打开 checkpoint、场景或 GT，没有预训练模型前向、图像编码或训练更新；这是合成算子检查，不能作为完整基线推理、正式 G3、模型峰值显存或处理 FPS。证据见 `bootstrap/env_creation/OpenMask3D_native_GPU_operators.json` 与 `review/G4_OpenMask_GPU_operator_primary_review.json`。

此前发现并隔离复现一项软件缺口（现已按下文修复）：更新 `datasets.scannet200_val.label_mapping_path` 时，资产刷新未安排协议和 Legacy 后代，显式协议阶段仍直接复用旧锁。隔离复现未更改正式 binding、protocol 或正在运行的数值产出代码；见 `review/G4_asset_refresh_semantic_lock_gap.json`。新映射证据需要在显式、安全的刷新边界重新判定受影响类别，同时保留未受影响场景及历史预测/评分的完整身份链；在该修复实测通过之前，不能承诺提供新映射文件便会自动解锁14项 Replica 角色。软件完成和任务完成标志继续为 false。


语义资产刷新现已补齐受影响的协议与 Legacy 调度，并新增 `semantic_refresh.py`：使用原有 canonical/WordNet 规则重新判定角色；原始类别→ID 的 GT 转换、训练词表和已解决角色必须保持一致。旧来源通过显式 path_map 移位时仍须匹配原 SHA；首次补入原先缺失的 TSV 只能发生在尚无 ScanNet GT 收据时。旧协议及受影响的预测/评分收据按 SHA 归档，未受影响的预测和评分继续复用。刷新中断可用相同资产清单恢复。共同模型和报告不会因为原先缺失的可选 alias 而整体阻塞。

独立测试验证：在已有四格 raw 预测上，只用缓存 F/N/Q 和原共享算子补四格校准导出；raw 文件逐字节及修改时间不变，MF-off 的 N/Q 访问为0，重叠 mask 和缺 F owner 保留。独立 GT 内容不变，只更新其预测冻结引用；OVI 原匹配和指标复用。Legacy 只打开已扩展 Replica 场景的评分入口，未受影响 CF18 指标原样保留。两类中断恢复均通过。完整73项 CPU 回归通过，详见 `review/G4_cpu_validation.json`；这些连接用例不进入正式实验。

真实 CLI 也已在独立输出目录执行显式 refresh，并复用 scene0414_00 的8份实际冻结预测。16个实际 manifest/数组文件的 SHA 和修改时间均不变；旧 Replica 的323个对象、646份双种子读出缓存逐一匹配模型身份和数组 SHA。没有启动新模型、读取 GT 标签或修改正式实验锁。证据见 `review/G4_semantic_refresh_actual_cache_73.json` 与 `review/G4_semantic_refresh_primary_review.json`。当前14个 Replica 角色依然缺少正式对应证据；该接口支持有来源的 TSV canonical/WordNet 扩展，不能凭此猜出缺失角色，也不是通用手填 role-manifest 接口。

正式队列已完成12/312场景、96/2496个配置格；12场景子集的共同池化 AP 为 U00 9.38、U10 10.79、U01 12.76、U11 12.99、OVI-MAP protocol port 9.94（%）。11张表已按这些实际结果刷新，完整标准集主表仍留空。正式 Replica 实例 GT/彩色评测点域及部分作者 checkpoint 仍缺失，完整科研、最终审核、C1/C2与终态恢复没有完成。

新增资产应在当前控制器结束后显式刷新；现有长队列继续运行，不在途中替换其协议：

```bash
"$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py \
  --output-root "$OUT" --asset-config "$ASSETS" --refresh-assets --phase all --resume
```

MaskClustering 的实际前端入口与配置解析已退出0：复用已绑定的 CropFormer 环境，确认原始 MaskFormer/HorNet、200个查询、0.5置信度阈值，未构建模型或打开 checkpoint。物理 GPU0/A40 的继承锁内，作者原始邻居查询返回 `[0,1,-1,…]`，原始深度有效性、图边与节点合并也符合手算值；实际诊断子进程墙钟14.964982秒已由统一作业收据核算。这只是合成算子检查，不是完整模型推理、正式 G3、模型显存或 FPS。证据见 `bootstrap/env_creation/MaskClustering_frontend_CPU.json`、`MaskClustering_native_GPU.json` 及 `review/G4_MaskClustering_native_primary_review.json`。

新核查的作者官方 Google Drive 模型目录也实际返回网络不可达（15秒超时），记录于 `assets/external_asset_discovery/MaskClustering_author_model_drive_access.json`。所需576d0b权重仍未下载；不能用当前建图的03823a权重替代。正式 Replica 实例 GT/评测点域和14项语义角色证据仍分别缺失。现有 ScanNet 队列继续从原始数据补算缓存，scene0575_00 已完成建图并进入 F 特征生成；这些依赖不会让整个任务停止。

统计阶段另补齐长任务中断恢复：`statistics.py` 现在只在一次共享 family 抽样的3方法×2种子全部完成后，原子保存6个池化 AP；恢复时复用完整抽样，未完成的一次从头计算。检查点绑定协议、实际 match、算子和随机计划身份，输入改变会在新评分前拒绝复用。原随机种子6173、1000次抽样、重复场景命名空间、逐种子差值后取均值、undefined 不重抽的规则不变。检查点位于 `statistics/checkpoints/<benchmark>/`；输入锁的 DONE 只表示检查点身份锁定，不能当作统计完成。

三个新增中断/变更回归先失败后通过，完整76项 CPU 测试通过。隔离目录还复用了2个真实场景的官方 match：原实现计算3次共享抽样18次池化评分；新实现中断于第3次抽样的第2个评分之后，恢复复用前2次完整抽样，仅重算未完成的一次。所有已比较的 AP 与场景命名空间完全一致，旧检查点字节和修改时间不变。实际证据为 `review/G4_statistics_resume_actual_matches.json`；它是恢复诊断，未写入正式统计或结果表。正式全量数据仍未齐，实际 `statistics` CLI 继续返回依赖未完成，没有用2个场景代替312场景的正式统计。

最新正式部分覆盖为13/312个 ScanNet 场景、104/2496个配置格；共同协议子集池化 AP 为 U00 9.03、U10 10.40、U01 12.51、U11 12.79、OVI-MAP protocol port 9.81（%）。11张表已从同一结果库刷新，主代理已查看三张新主表预览，完整范围主表缺失格继续保留。当前原队列运行第14个场景 scene0575_01；Replica 仍为0/8，完整统计、最终审核与发布尚未完成。

另修复导出清单遗漏：原清单未包含实际 `readout_cache` 和融合温度，不能只凭原导出包复算预测。`publication_export.py` 现包含2872份真实 b/z 缓存及其收据、已恢复的 A 校准源数组；新增 `readout_replay.py` 仅快照冻结预测所用的逐场景温度、原始文本和 N/Q 分数，核对父 context 文件 SHA 和分数记录身份，不复制整个 context 或原始 GT/RGB/权重。缺失或改变的输入会在替换有效导出索引前报错；相同输入重复导出不改写快照。

4项新增回归先失败后通过，完整80项 CPU 回归通过且无跳过。真实导出预览包含6898个文件、6885个不同分片，含索引共183528504字节（约175 MiB），满足单分片40 MiB、总量512 MiB约束。从这些分片恢复后，104组标准预测的6192个 owner 标签、26个 Legacy 场景176组预测的10196个 eligible owner 标签全部与原冻结结果一致；Legacy 未干预点的预测也保持一致。2872份读出缓存和校准数组通过 SHA 校验。实际复算进程退出0，证据为 `review/G4_readout_replay_actual.json`，身份 `7242422648771d81f47914101ea210496b9c83cd367d628d5a4fc167d44cbd47`。没有运行模型、读取标准评测 GT 或改变实验配置；这是本地导出与复算验证，尚未提交或上传 C1/C2。

随后原队列完成 scene0575_01，当前正式部分覆盖为14/312、112/2496配置格，正在执行第15个场景 scene0575_02。14场景子集池化 AP 为 U00 9.00、U10 10.36、U01 12.55、U11 12.84、OVI-MAP protocol port 9.50（%）。11张表已刷新；三张主表预览已实际查看，1904个数值单元、155个来源文本和54个缺失格核查通过。上述导出复算验证对应更新前13个标准场景的快照；本地最新科学结果和最终发布导出仍须分别按其身份核对。

本次全范围评分刷新还定位并修复了调度状态缺口：尚未生成预测收据的306个场景原被记为 `FAILED_RUNTIME`，实际已完成的14个场景评分全部为 DONE。`runner.py` 的 score-only 入口现先以 `lstat` 检查预期预测收据，只有该收据不存在才返回 `BLOCKED_DEPENDENCY` 并给出补预测命令；损坏 JSON、断链和已有收据引用的文件意外缺失仍为运行失败。原诊断保留于 `review/G4_score_dependency_status_gap.json`，实际修复证据为 `review/G4_score_dispatch_actual.json`，主代理审核为 `review/G4_score_dispatch_primary_review.json`。

6项依赖/错误隔离用例中，旧实现复现3项失败、3项错误保留检查通过；修复后全部通过，完整86项 CPU 测试通过且无跳过。真实全范围评分 CLI 返回部分完成码20，阶段状态为 `BLOCKED_DEPENDENCY`，14个已完成场景、112个配置格原样复用，306个未预测场景正确等待。700个预测、GT、评分、match、锁及历史错误文件的 SHA 和修改时间全部不变；306份历史 traceback 保留且未被改写。没有新模型作业或新场景 GT 评分。仅改动调度入口，16个数值产出模块及原18个测试文件保持不变；冻结配置和协议未改。当前仍在执行第15个 ScanNet 场景；完整标准集、Replica、外部比较、正式统计和 C1/C2 尚未完成。

随后 scene0575_02 的8组正式评分完成，评分收据身份为 `7c79b61ea983ff42bcd5fed49402b9edde526e6d9995d72f6bcdbafc6105ca10`。15场景子集的共同协议池化 AP 为 U00 8.91、U10 10.26、U01 12.51、U11 12.81、OVI-MAP protocol port 9.41（%）。11张表已从当前结果库刷新；三张主表预览已实际查看，2019个数值单元、155个来源文本和54个缺失格核查通过。结果库身份为 `482747f12725117a25643b07b53615b5e83267907080d87cc78de4eb8575446c`，核查证据为 `review/G4_actual_aggregation.json`。本次是新增场景数据汇总，生产代码及测试未变；完整范围主表仍留空，未将部分结果作为全量结论。原队列继续执行 scene0426_00。
