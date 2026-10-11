# Frozen FC routing：提示词逐项验收

主代理最终验收：**所有已激活的科学阶段、产物和提示词要求核对通过，科学终态为 `COMPLETE_NO_TARGET_GAIN`，实际 `all --resume` 退出码为 0**。本文件记录提交前验收；正常推送及完整本地/远端 SHA 核对由提交后外部 `attempt_001/publication/final.json` 证明，不在提交内预写自身 SHA。

本次执行依据 2026-10-11 提示词包，而非前一轮 PR 的训练或选模门槛。17 个原件及配置逐字节相同；spec SHA-256 为 `e56a00ad94cc4422aea41d0e672dabc1dc61afaefb29760e858fc4ddef22631e`。

| 提示词要求 | 实现与验收证据 | 当前结论 |
|---|---|---|
| 指定基点、独立 worktree、新分支，保护父任务 | 基点 `76db115aa7df652a0b2dc4575df52e6a74a63549`；分支 `research/ovimap-frozen-fc-routing-v1`；新增目录，不修改父模块 | 已核对 |
| 原包和 JSON 配置完整保存 | `engineering/prompt_integrity.json`；17/17 文件与 ZIP 相同，主代理另行核对字节和哈希 | 已核对 |
| PR→LR→SU 真实依赖，不要求合格 R，不伪造旧任务身份 | 自有 `binding.py`；核对两代 publication、真实退出及结果身份；直接复用已验证底层数值函数 | 已核对 |
| 固定 24/4/4 数据角色、160/40 类、原观测和扰动 | `prepare.json`、`split_manifest.json`、`class_split.json`、`dependency_manifest.json`；TRAIN 785/622、DEV 87/69、old-H 101 | 已核对 |
| H2 训练前只检查名字与可用性，保留固定真实提案计划 | 原 PR `H2_plan.json` 原样复制；有限谱系曝光清单，不宣称全局未曝光 | 已核对 |
| 单 A40、环境隔离、排队 GPU 锁 | 实际 CUDA 工程记录；既有 FC/controller/SAM2 环境；不以 NVML 警告判定 CUDA 不可用 | 已核对 |
| 普通冻结 FC 为基底，四个相同形状的 216,194 参数头 | `models.py`；真实 FC8 一致性、phi 输入梯度和层参数检查；无 MA/R/可学习值映射 | 已核对 |
| 七个 JSON 白名单张量，无标签/家族/实例身份进入 forward | `data.py` 与 `models.py`；监督通过独立 target sidecar；实际工程及聚合测试 | 已核对 |
| 严格 pool-then-phi、VIEW/SURFACE、独立 quality/membership、绝对路由 | 生产数值实现及测试；eta=0、无支持、rho=0 的精确 FC 身份 | 已核对 |
| 原采样、原优化器和调度，四臂同初始化和 draws | 固定 seed+1009、批 16/微批 1、FP32、无 TF32、原 LR/eta 调度；`engineering/training_coverage.json` | 八臂实际初始化/采样及冻结数值源码一致性核对通过 |
| 四臂 seed17 各 2,000 步；合格后全部 seed29 复验 | 真实 `updates.jsonl`、draws、五次 DEV 检查点及每臂 receipt | 八臂各 2,000 步，共 16,000 步；曲线完整 |
| DEV 固定门槛，step0 不入选，seed29 不更换 seed17 提名架构 | `selection.py`、两份 nomination、`predictor_lock.json` | FG2_SURFACE 固定提名，seed17/29 均通过，完整路径已触发 |
| D0 使用既有 PR 分数，不新训练或按错误列表筛选 | `prior_audit/receipt.json`；GT margin 和 top1-top2 gap 分开保存 | 已执行 |
| D1 完整 622+69 base 对象、四条件，GT 只作诊断 | `diagnostic_D1/receipt.json`；ALL/TARGET_ONLY、fallback 和共同有效支持记录 | 已执行 |
| D2/D3 对所有训练后的选定头执行，FG4 无论是否合格都干预 | `mechanisms.py`、`transfer.py`；质量系数和路由系数共同统计；norm undefined 双侧排除 | TRAIN/DEV/old-H/H2 完成；真实提案双种子 FG4 干预完成，site purity 明确未计算 |
| 真正同架构复验才触发整图；11 方法、26 场景、286 行/22 池 | 自有预测及评估循环；保持 G1 几何/非适用对象/恢复对象，只替换 F 后原 D2 分组 | 286 行/22 池完成，所有预测先于评分锁定 |
| 原 scorer、当前语义类别 official ranks、同一 payload 的实例与语义指标 | `engineering/two_scene.json`；两场景实际输出/正 owner registry/rank/confusion 一致性；`evaluation/store.json` | 工程核验和完整科学整图评分完成 |
| H2 固定四家族、真实提案固定八帧，锁定输出后使用评价标签 | 自有 `holdout2.py`、`real_proposals.py`；支持不足与依赖失败分别处理 | H2 115 对象，其中 20 novel 对象跨 4 家族；274 次 FC 编码；真实提案 67 个共同支持，7 个 GT 匹配 |
| 所有训练臂导出真实小型权重、完整标量曲线、严格加载 | 自有 `export.py`；相对权重路径、FC/text 身份、两 DEV 对象实际 roundtrip | 八组权重严格重载，输出逐位一致，误差为 0 |
| 三类表格四种格式、四份报告，量纲与可支持的结论明确 | `reporting.py`；分开 DEV 提名、曝光整图结果、复验和 GT 诊断 | 16 张表、64 个格式文件完整且哈希相符；四份报告与本验收报告已核对 |
| 有限检查、真实过程退出、保留失败和重放成本 | 12 个测试组；20 个丢弃工程更新；两真实 scorer pilot；独立父进程 observer | 12/12 通过，最终实际退出 0；三个 all 尝试的完整 stdout 及失败历史均归档 |
| 正常提交/推送、完整本地远端 SHA 相等后写外部 publication | 最终过程证明归档 `terminal/`；外部 `publication/final.json` 不进入自身提交 | 发布方式及内容验收通过；推送结果以提交后外部 receipt 为准 |

数值解释遵循正式 JSON：forward 白名单实际列出七项；基础损失保留规定的 0.1 student consistency，不再叠加其他一致性、保持或对应损失。这两点在交接中明确记录。

发生过一次技术重启：初次训练完成 3 个未检查点更新后，修复“恢复时恰在 DEV 边界”和“恢复时已经到 2,000 步”的记账问题。原源码、三行标量曲线、失败过程观察和重放记录全部保留在 `history/resume_boundary_fix/`、`observations/` 与 `training/restarts.json`。重启使用同 seed、相同原始 draws，前 3 行标量记录逐项相同。重放更新与科学更新、20 个工程更新分开报告。

稀疏缓存只保存原 loader 的同一张量；两真实 TRAIN 对象、两条件、2/4/8 前缀的 12 项输入/输出/head 梯度/真实 phi 输入梯度均逐位相同，未新增优化步数。所有四臂使用相同缓存协议。

确认阶段还修复了两项实际接口问题：H2 下载器收到字符串而非 `Path`；SAM2 子进程继承了另一 CUDA 版本的动态库路径。原错误、源码、生成任务和进程退出已保存于 `history/confirmation_interface_fix/`。修复仅涉及路径类型和子进程库隔离，未升级环境或修改科学设置；独立导入对照复现了库路径污染导致的失败，隔离后通过。恢复过程中复用全部已完成训练与整图结果，新增科学训练步数为 0。

固定提名 FG2_SURFACE 的两种子 DEV A 为 46.0711% / 46.8908%，FC8 为 37.2885%。全部主轮与复验方法均未满足既定整图升级门槛，不能据 DEV 提升宣称实例图全面提升。研究基线保留 REF_G1，部署为 N0_UNCHANGED。路由干预 Top-1 使用相关条件记录的普通均值，与家族加权 A 分开呈现。

H2 中固定提名的普通 ALL Top-1 为 31.3043% / 30.6522%，FC8 为 26.0870%；HELDOUT 为 40.00% / 41.25%，FC8 为 31.25%。这支持本次有限对象上的识别改善，不构成独立整图 AP 或显著性证明。真实提案的七个匹配对象中，所有九种方法均答对五个；其中 IoU > .5 只有两个对象。该诊断未提供方法优劣证据，不能把 67 个几何支持对象当成 67 个有标签的评价样本。

FG4 在两种子选定 DEV A 上均未超过 FG2。DEV 普通 Top-1 相对 matched-mass/matched-norm 控制约高 1.087 个百分点，但 old-H 与真实提案证据不足以建立稳定的路由优势。D0/D1 属于原因探索和 GT 干预诊断，未被用于改动训练或筛选模型。

生产源码的默认 Git whitespace 检查通过；原包 CSV 的 CRLF 字节按要求保留，整体检查显式启用 `cr-at-eol`。

对发布产物的机械核验与主代理复核均通过：2,233 个主索引文件逐一匹配字节数和 SHA-256，八份权重各约 0.87 MB，八条正式训练曲线均为 2,000 行，历史三行重放曲线单列。没有超过 100 MiB 的单文件；不包含原始扫描、GT 图、基础模型权重、稠密缓存或优化器 last.pt。六份训练数值源码仍与科学运行绑定值一致，父模块未修改。

最终科学结果身份：`f1675869670130b358fd07f27fbaff59c3f2d3a47dbd98790f58d7746a6f7fdb`。外部父进程实际退出观察身份：`8818de9daabae52a4b41dd6bd552e491ea51894e0290d4309fbc18edd567a91c`。主产物索引在报告阶段生成；随后生成的真实终态证明单独由 [terminal/index.json](../../../artifacts/static_ovmap/frozen_fc_routing_v1/terminal/index.json) 索引，避免把上一轮失败退出冒充当前终态。

机械文件核验与主代理产物核对记录见 [release_review](../../../artifacts/static_ovmap/frozen_fc_routing_v1/release_review)。没有尚缺的已激活科学阶段，也没有因负结果追加种子、轮数、场景或配置。
