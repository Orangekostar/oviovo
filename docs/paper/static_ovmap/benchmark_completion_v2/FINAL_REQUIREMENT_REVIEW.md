# 当前完整合同审核（2026-10-29，科研未结项）

Node7 的原 Replica RGB-D、轨迹和 mesh 已复用；8 场景逐点评测域与公开作者真值已连接并通过实际 G3。四个外部方法均已完成 Replica 全 8 场景。原生标准实验完成 2528/2560 项，其中 ScanNet 2496/2496、Replica 32/64。剩余 65 个逻辑任务依赖有来源的训练类别映射，详见 remaining_jobs.json；此结论不代表完整科研或发布完成。

本表保留原00第1–17章、03的A–G、04第1–6节的39项审核范围。继承的源码及数值审核沿用其真实历史收据；本次重新核对 Node7 接入、完成池、当前表格、状态报告及源文件变化。V6 核对100份执行文件及合并后104份不同源文件；原98份来源中92份未变，6份改动为4份元数据生产模块和2份既有测试。最后一处为测试模块缓存隔离修复，生产算法未变。完整31文件测试范围207项及17项子检查通过，无跳过项。另新增1份只读导入模块，原绑定、协议、模型、预测和AP数值保持不变。本次报告修复的19项检查和10项子检查通过，不与历史检查相加冒称独立检查总数。

当前 result_store：`4e7fea288940a759a1c8ac031e81604cfa6b3fdbf9a3bd0568ded36bf693b63c`；tables：`d24eabd9ba073b866fc0d5b9a08af0a8b875d637234b1695d447bcf94788570d`；源状态：`a225c28284ffe14d817aa61559a3d24e1a77fa106e5593bd6ec4f1ccc8e5e3b8`。

| 项 / 00合同章 | 当前判定 | 实现、实测与适用范围 |
|---|---|---|
| R01 / 1–2 | DONE | 固定BASE/研究分支、25父Git blob、原配置/提示词；SOURCE_BINDING与原hash清单。C1限本任务。 |
| R02 / 2–3 | DONE（实现） | binding/common/runner/lifecycle；显式roots/path_map、资产刷新和stdlib终态入口。 |
| R03 / 3 | 部分（实现已验证） | 当前104份不同源文件约束；完整软件测试通过；Replica真实RAW及四外部方法G3已完成。完整终态、发布及恢复验收仍未完成，software_complete保守false。 |
| R04 / 4 | DONE（评测协议/点域） | 固定312/8、文本200/实例198、48/旧51分离。8场景全点序、颜色、公开作者编码GT和派生四字段tuple已验证；不称派生文件为作者原始tuple字节。 |
| R05 / 4,11 | DONE（角色披露） | MV/Q_GAIN/CropFormer/FC/native组件的梯度/选择/指标/未知角色；不保证基础模型预训练去污染。 |
| R06 / 4,8 | BLOCKED_DEPENDENCY | 正式协议仍有14项base角色未定；3个有出处候选未正式应用，其余11项缺唯一有效对应；不依据GT、ID大小、词表缺席或分数猜测。 |
| R07 / 5 | DONE | G1：279原件、9unsupported、9组RAW精确一致、220旧MV候选复用。 |
| R08 / 5,14 | DONE | 1331逻辑候选、1111实际CPU候选；真实NMIX24向量/72预测核验；退化边界另列合成检查。 |
| R09 / 5,11 | DONE | 60折原序/共享配方、9组OOF指标；4学习模型各1折独立重选，不扩称60折全部独立重选。 |
| R10 / 5–6,11 | DONE | simple_recipe_lock；主方法CAP10/gamma .025，不按benchmark AP改选。 |
| R11 / 6,14 | DONE | 两seed真实matched1000 state；2真对象×2seed的b/z误差0、top1一致、原FC dtype。 |
| R12 / 7,14 | DONE（可执行范围） | Legacy同分区/eligible，非eligible保留；同b/原dtype/实际冻结rank及176完整分数manifest。 |
| R13 / 7 | BLOCKED_DEPENDENCY | 176/208行、12/16池；Replica AC缺32行/4池。 |
| R14 / 8 | DONE（原生输入） | ScanNet312与Replica8原生输入已完成；finite-pose≤200、F≤8，真实capture/deferred/NQ；零F/单视图owner保留。 |
| R15 / 8,14 | DONE（已执行范围） | ScanNet全mask四格、Replica RAW两格；MFoff trace零NQ、constant1/unknown。Replica U01/U11未补写，仍依赖角色映射。 |
| R16 / 8,12 | DONE（已执行范围） | ScanNet及Replica预测冻结→独立完整GT→评分身份链；保留全部原点序与分母，不按support筛GT。 |
| R17 / 8,13 | 部分 | ScanNet312/312、2496/2496；Replica8/8、32/64。T1/T2同U11；Replica RAW完整两seed池已进入表格，缺U01/U11保留空值。 |
| R18 / 9 | DONE（身份/缺项） | 四作者pin、完整方法/输入/监督/命令recipe；本地完整池优先，作者引用标†。 |
| R19 / 9,14 | DONE | OVI-MAP首场景作者格式及ScanNet312、Replica8本地完整官方matches池；注明共同协议port与constant1读出。 |
| R20 / 9 | DONE（当前证据） | OpenMask3D ScanNet兼容作者15.4/19.9/23.1标†；Replica完整任意场景原方法8/8本地推理、G3与全场景池完成，未继承旧13.1。 |
| R21 / 9,14 | DONE（当前证据） | Open3DIS ScanNet312完整作者ISBNet-GSAM-refined预测已本地评分；Replica8完整原方法本地推理、严格权重加载、G3、官方全场景池及只读导入已完成。 |
| R22 / 9 | DONE（当前证据） | MaskClustering ScanNet作者semantic列12.0/23.3/30.1标†；Replica完整CropFormer/聚类/CLIP-H14原方法8/8、实际G3和全场景池完成。 |
| R23 / 9,14 | DONE（Replica四方法） | OVI-MAP、OpenMask3D、Open3DIS、MaskClustering均8/8；全部采用完整有序场景的官方matches汇总，未平均场景AP；每方法逐场景作者AP480项误差0。 |
| R24 / 10 | DONE（有限执行） | 原有限矩阵、训练0、无付费服务；用户后续多卡/节点授权覆盖初始单卡；独立私有目录/显式等价导入。 |
| R25 / 10 | 部分（历史缺口保留） | 容量守卫及充足空间；第43场景历史容量收据缺口不回填；未测整流程峰值/在线吞吐。 |
| R26 / 11 | DONE（边界披露） | CONF12 novel曝光、LOFO探索、普通D2、N0未变；不重训清洁模型。 |
| R27 / 12 | DONE（已执行范围） | fractions、每seed独立再均值、null/静态/alias、官方matches池化及全分母。 |
| R28 / 12 | 部分 | ScanNet142家族、两比较×1000draw、0undefined，重复scene/文件namespace及两seed AP差均值保留；Replica2000draw仍缺U11/U01完整输入。 |
| R29 / 12–13 | 部分（几何已完整） | ScanNet312完整proposal recall与条件识别；Replica8完整几何诊断332有效GT/152匹配，条件CAP行仍待角色映射；既有五节点导入证据保留。 |
| R30 / 13 | DONE（当前真实/缺格表） | 唯一store、11表及逐格来源；36275数值格、155文本格、39null；T1/T2同U11结果键。 |
| R31 / 14 | DONE（G1范围） | G1_real_numeric_audit/G1_old_RAW_exact；真实复算与参考合成检查分开。 |
| R32 / 14 | DONE（G2范围） | model_lock和MODEL_INPUT_SOURCE_REVIEW；不重复已通过模型forward。 |
| R33 / 14 | DONE（已执行范围G3） | ScanNet与Replica RAW、四个Replica外部方法均有真实格式/评测/作者AP核验；辅助AR只补NaN分支，既有AP精确不变。未执行的Replica校准变体不计为验证完成。 |
| R34 / 14 | G4表格DONE；终态恢复待实测 | 三主表当前字节已看图；八补表6033行/163页逐行内容、字体及边距核验；5张全页contacts、8个首页和S6/S7末页共10张整页样本已检。未检完整论文整版。 |
| R35 / 15–16 | 待实际发布 | 当前0 runnable、65缺项显式保留；129项旧队列的64项完成均连接真实评分/完整池证据。C1/C2、远端新增数值文件回读和外部final尚待实际执行。 |
| R36 / 16 | 部分（来源已留存） | 原4543文件与Node7历史片段按实际存储路径去重并验证，最终合并清单含6875文件、123960769字节。真实成功/失败和关闭的来源汇入过程均保留；不将文件数解释为独立科学实验数。 |
| R37 / 15–16 | 待发布后实测 | stdlib终态源码/既有隔离检查；本次须验证实际exit20、零worker/科学导入/写入、SHA/mtime与HEAD不变。 |
| R38 / 16–17 | DONE（当前交接） | RESULTS/HANDOFF/REVIEW/CLAIMS与remaining_jobs已按当前320场景和四外部方法更新；删除固定GPU0限制，成本、配方、缺项及N0边界保留。 |
| R39 / 17 | 完整目标未完成 | 七维独立；partial发布不缩小原科研目标，不把完成审核当完成全部科研。 |

## 03 / 04逐节对应

| 合同 | 当前依据 / 边界 |
|---|---|
| 03.A | R07：原key/column/valid_ids、clean8V axis，不逆解其他prefix向量。 |
| 03.B1–B6 | R08–R10：原cosine gamma、FP64内部/原dtype恢复、defined/null、相同预算/选择顺序。 |
| 03.C | R11：原7-key region接口、空局部维、真实state及b/z等价。 |
| 03.D | R14/R16：真实SceneBundle/geometry/frames/capture，GT score-only。 |
| 03.E | R15/R17：全owner U四格、实际可用源、缺F unknown，AC不改mask。 |
| 03.F | R04/R06/R27–R29：官方类别/点域/投影、家族namespace；Replica点域及几何已完成，校准与配对统计仍阻塞。 |
| 03.G | R30/R36：result→prediction→input/model/protocol，显式转换，不改旧签名JSON。 |
| 04.1 | R04/R17：完整官方312/8与旧CF18/Replica51分开。 |
| 04.2 | R18–R23：四固定完整变体，作者预测/引用/本地运行分开。 |
| 04.3 | R20–R23：dataset IO port不减模型/省refine/用demo替代。 |
| 04.4 | R14–R17：原OVI native提案全mask四格，不拼Open3DIS提案。 |
| 04.5 | R24–R25：真实有限执行、多卡/节点授权、失败/成本/容量；历史缺口不回填。 |
| 04.6 | R05/R17–R23/R39：同协议/监督/完整范围；缺数据和负结果分开。 |

replica_3d是预处理目录名。当前已从本地Replica及公开作者编码GT生成并核验同点序的四字段tuple，明确标为派生文件。剩余base语义角色需要独立、有来源的对应关系。

关键实际收据：

| 内容 | identity |
|---|---|
| v3实际源码转换 | bc8c75bcebe720ef55a569b1ca162a88fec4f473e7a0ff859d71977bc2b7c5a7 |
| 显式资产刷新 | fe278c0e09ab32b97e3f6174a904cc6d390b6299ed9a6bc5eb99e9ee4d579e90 |
| 输入/预测/评分/统计刷新关闭 | b8772a1cf542fda8b245c2fd96b82eadefa25277deabefb63e551dc8e380284b |
| ScanNet完整statistics | 7132c270455b2ff1f0d7932ac034b0fb85bf55443000ee367f0c5791bf2553f2 |
| ScanNet完整diagnostics | 5339c06454babbfdf063cbe8ba5dabe40483d35bbbdf21e6a1440223a523b4fe |
| 五节点diagnostics关闭 | fbf9c3b5d18dd9daee854b5f25d086ca7bb09b2e3c7619697ae0547a80c1014c |
| G4数值汇总 | 861db7006ba18c599078dc5be3944b09ed17395e53214b995e30920e4c51ad55 |
| 三主表看图 | ec0d34abdb4962d1568edcb1767a2d54e78cf54f6488d081796e3c7955d0cfc6 |
| 补表编译 | 3b2990be3f206b70d6f2999fc5d6dd7fcea21865ef6f6d742443219388e3c991 |
| 补表机器内容核验 | d001559ef3919d8be0e0312e07d32b1feb490167af746b581281b1f25b951502 |
| 补表主代理看图 | 1bc9690a09d6fdfa2aaf3a0cd76d2dfd1a634552cd7d84f1c2955e57f16b5bab |
| 4543文件有限执行来源 | 9efbb8a88f88c72026a7d7618c61de0036ecb11d1c0862c9bb457a1ab86b2063 |

组件源码审核见CALIBRATION/MODEL_INPUT/GEOMETRY_SCORE/NATIVE_QUERY/LEGACY/STATISTICS/EXTERNAL_DISPATCH/RUNNER_LIFECYCLE_SOURCE_REVIEW.md。历史99文件门与v3应用、后续V4资源路径修复、V5报告修复和V6测试隔离修复形成显式来源链。当前执行文件100份、合并不同源文件104份；review/primary_final.json仍保留完整验收未完成状态。

本次原始review入口重跑收据：`4c4feb6857cc36e371b872d206f80c73825393895c476181a177b5856091eda5`；V5/V6状态和历史快照均保留于node7_replica_connection_20261029。旧报告未被当作当前缺项依据。

本次39项合同审核已完成，审核完成不代表39项科研要求全部DONE。发布前完整测试收据：`74895316b2c977622c7a6f5d9fd65df4f303057d4227ddbd84dfdca0a834489a`；实际退出0：`94b81a067054a7111f7eaae49786b6757a0b101a4b7f52f6c90a05afe2b68ad7`。
标准完整类别分数2528/2560、旧协议176/208，与冻结预测标签逐项一致；新增32份Replica RAW分数manifest，未运行新模型或AP。收据：`63a6acc269b59c231999b67a86fc110ca2ccc7a820169353108565cfc44ee64d`。
最终执行来源清单：`e90b6e7e14c72c09b1cc5509a924f0faebcce3871c929f859acbc1470cdc55eb`。C1/C2、远端内容回读和实际终态结果以publication/final.json为准，当前不预先声称发布或科研完成。
