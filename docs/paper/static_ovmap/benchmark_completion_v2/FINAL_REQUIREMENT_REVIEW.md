# 当前完整合同审核（2026-10-30）

标准矩阵2560/2560、旧协议208/208和16/16池、四个Replica外部方法各8/8、两数据集配对统计及320场景诊断均完成。全部结果使用既有模型与输入；本次角色扩展后的计算新增模型调用、训练和下载均为0。227项及17项子检查在当前规范工作区通过，无跳过。

用户确认的语义扩展见replica_role_extension_v1.md。该扩展保留原训练轴和已解决角色；panel为包含已见类的同义组并按any-base处理。此策略不称为作者原始唯一映射，也不依据target AP选择。

下表覆盖原00第1–17章、03.A–G、04.1–6的39项合同。C1准备时发布与终态为READY；最终真实C1/C2、远端内容回读和resume结果在$OUT/publication/final.json，避免自引用第三次提交。原两次部分结果提交完整保留。

| 项 / 00合同章 | 当前判定 | 实现、实测与适用范围 |
|---|---|---|
| R01 / 1–2 | DONE | 固定BASE/研究分支、25父Git blob、原配置/提示词；SOURCE_BINDING与原hash清单。C1限本任务。 |
| R02 / 2–3 | DONE（实现） | binding/common/runner/lifecycle；显式roots/path_map、资产刷新和stdlib终态入口。 |
| R03 / 3 | DONE（软件检查） | 当前V8规范工作区227项及17项子检查通过，无跳过。四项元数据兼容修改已独立测试；实际发布与终态验收另以外部final收据验证。 |
| R04 / 4 | DONE（评测协议/点域） | 固定312/8、文本200/实例198、48/旧51分离。8场景全点序、颜色、公开作者编码GT和派生四字段tuple已验证；不称派生文件为作者原始tuple字节。 |
| R05 / 4,11 | DONE（角色披露） | MV/Q_GAIN/CropFormer/FC/native组件的梯度/选择/指标/未知角色；不保证基础模型预训练去污染。 |
| R06 / 4,8 | DONE（用户确认的角色扩展） | 14项未定角色已按固定公开crosswalk和用户“都为同义”确认解决。panel采用any-base曝光；manual关系明确披露，词表200/base160、旧34项角色不变；未知类别不自动归novel。 |
| R07 / 5 | DONE | G1：279原件、9unsupported、9组RAW精确一致、220旧MV候选复用。 |
| R08 / 5,14 | DONE | 1331逻辑候选、1111实际CPU候选；真实NMIX24向量/72预测核验；退化边界另列合成检查。 |
| R09 / 5,11 | DONE | 60折原序/共享配方、9组OOF指标；4学习模型各1折独立重选，不扩称60折全部独立重选。 |
| R10 / 5–6,11 | DONE | simple_recipe_lock；主方法CAP10/gamma .025，不按benchmark AP改选。 |
| R11 / 6,14 | DONE | 两seed真实matched1000 state；2真对象×2seed的b/z误差0、top1一致、原FC dtype。 |
| R12 / 7,14 | DONE | 旧协议同分区/eligible、非eligible保留、同b/原dtype/冻结rank；208份完整类别分数与实际eligible预测精确一致。 |
| R13 / 7 | DONE | 208/208逻辑行、16/16完整池；新增Replica32个AC单元，原RAW预测及指标保留。 |
| R14 / 8 | DONE（原生输入） | ScanNet312与Replica8原生输入已完成；finite-pose≤200、F≤8，真实capture/deferred/NQ；零F/单视图owner保留。 |
| R15 / 8,14 | DONE | ScanNet312与Replica8全owner四格×两seed；同mask，MFoff trace零NQ，constant1、零F unknown均保留。 |
| R16 / 8,12 | DONE | 所有新AC预测先冻结，再连接完整GT评分；历史RAW及原GT字节不变，历史已访问target GT不被追溯称为从未访问。 |
| R17 / 8,13 | DONE | 标准2560/2560：ScanNet2496、Replica64；320场景、每scope8个完整全场景池，T1/T2同U11两seed。 |
| R18 / 9 | DONE（身份/缺项） | 四作者pin、完整方法/输入/监督/命令recipe；本地完整池优先，作者引用标†。 |
| R19 / 9,14 | DONE | OVI-MAP首场景作者格式及ScanNet312、Replica8本地完整官方matches池；注明共同协议port与constant1读出。 |
| R20 / 9 | DONE（当前证据） | OpenMask3D ScanNet兼容作者15.4/19.9/23.1标†；Replica完整任意场景原方法8/8本地推理、G3与全场景池完成，未继承旧13.1。 |
| R21 / 9,14 | DONE（当前证据） | Open3DIS ScanNet312完整作者ISBNet-GSAM-refined预测已本地评分；Replica8完整原方法本地推理、严格权重加载、G3、官方全场景池及只读导入已完成。 |
| R22 / 9 | DONE（当前证据） | MaskClustering ScanNet作者semantic列12.0/23.3/30.1标†；Replica完整CropFormer/聚类/CLIP-H14原方法8/8、实际G3和全场景池完成。 |
| R23 / 9,14 | DONE（Replica四方法） | OVI-MAP、OpenMask3D、Open3DIS、MaskClustering均8/8；全部采用完整有序场景的官方matches汇总，未平均场景AP；每方法逐场景作者AP480项误差0。 |
| R24 / 10 | DONE（有限执行） | 原有限矩阵、训练0、无付费服务；用户后续多卡/节点授权覆盖初始单卡；独立私有目录/显式等价导入。 |
| R25 / 10 | 历史限制保留 | 容量守卫及充足空间；第43场景历史容量收据缺口不回填；未测整流程峰值/在线吞吐。 |
| R26 / 11 | DONE（边界披露） | CONF12 novel曝光、LOFO探索、普通D2、N0未变；不重训清洁模型。 |
| R27 / 12 | DONE | 每seed官方matches全有序场景池化再取seed均值；fraction单位、静态/alias/null和完整GT分母保持。 |
| R28 / 12 | DONE | ScanNet142家族、Replica8场景家族；各1000次共享配对抽样计划、两比较各1000有效draw、0undefined。保留重复namespace，两seed AP差后均值；不称两个比较为独立抽样计划。 |
| R29 / 12–13 | DONE | ScanNet312原诊断数值未变并显式重新绑定；Replica8场景14条件行/场景，332有效GT、152几何匹配，全几何分母与缺F保留；固定配方不按benchmark选择。 |
| R30 / 13 | DONE | 唯一结果store生成11表及逐格来源；完整T1/T2、Replica条件识别及配对统计均已接入，缺失/不适用指标保留null。 |
| R31 / 14 | DONE（G1范围） | G1_real_numeric_audit/G1_old_RAW_exact；真实复算与参考合成检查分开。 |
| R32 / 14 | DONE（G2范围） | model_lock和MODEL_INPUT_SOURCE_REVIEW；不重复已通过模型forward。 |
| R33 / 14 | DONE（当前G3） | Replica64单元common/raw-author缓存matches上30720项AP误差0；另有独立原GT raw/contiguous-author逐点评测核验。ScanNet及四外部方法原G3保留；两次附加核验脚本失败和修正后真实成功均留存。 |
| R34 / 14 | DONE（表格与案例QA） | 三主表当前PDF/PNG/TeX已主代理看图；八补表全部行/数值、嵌入字体/边距/页数机器核验，全部页面contacts及整页样本看图；ScanNet案例图字节未变。未声称检查完整论文版面。 |
| R35 / 15–16 | READY；实际发布另验 | 65项旧缺项均连接真实完成收据；当前remaining_jobs为空。历史C1/C2保留，本次新C1/C2、远端新数值分片回读须由publication/final.json证明。 |
| R36 / 16 | DONE（有限来源） | 继承6875份历史执行胶囊，新增本次角色确认、协议刷新、成功/失败实际退出、源文件、检查和图表证据；发布清单逐文件SHA校验，文件数不等于独立科学实验数。 |
| R37 / 15–16 | READY；实际终态另验 | 227项软件检查含完整负结果exit0终态；本次实际发布后须原CLI resume退出0，零worker/科学导入/写入且科学文件SHA/mtime与HEAD不变；实际结果以外部final为准。 |
| R38 / 16–17 | DONE | RESULTS/HANDOFF/REVIEW/CLAIMS按320场景完整矩阵更新；同义策略单独披露，旧版/标准/作者引用区分，N0与成本/历史边界保留。 |
| R39 / 17 | 科研矩阵完成；发布验收外置 | 有限科学计算均已执行，全部完成维度独立报告；claim_supported仍false，不将覆盖完成、正向消融或发布视为物理对应/无泄漏泛化已获证明。 |

03.A–C对应R07–R11，03.D–E对应R14–R17，03.F对应R04/R06/R27–R29，03.G对应R30/R36。04.1对应R04/R17，04.2–3对应R18–R23，04.4对应R14–R17，04.5对应R24–R25，04.6对应R05/R17–R23/R39。

CONF12 novel标签参与选择，LOFO为探索证据；未重训清洁模型。Replica四字段tuple来自公开作者GT和原场景同点序转换，不称为作者原始tuple字节。第43场景历史容量收据缺口不回填；未测整流程峰值或在线吞吐。作者引用值保持†，完整方法变体/监督来源见原recipe。组件源码审核沿用已验证历史记录，并由V7角色扩展和V8元数据兼容边界、当前完整检查补充。
