# Evidence Exploration Implementation Plan

**Goal:** 完整执行固定 9 方法 × 26 场景研究；验证 CF18 改善与 Replica 五项无回退目标，正常发布真实正负结果。

**Architecture:** 在 `static_ovmap/evidence_exploration` 新增独立适配器。只读父实验身份、几何、G1 视图、FC 特征与官方评测，预测器不接触 GT；同一结果库生成选择、分析和三张表。

**Tech Stack:** Python、原 FC PyTorch 2.4.0 环境、固定 AnyUp 351807a、Open3D 4 线程、原 released scorer。

**Spec:** `docs/paper/static_ovmap/evidence_exploration_v1/spec/CODEX_FINAL_EXECUTION_EN.md` 与逐字复制的 `configs/static_ovmap/evidence_exploration_v1.json`。

## 固定约束

9 个 ID、26 个场景、234 行/18 全池化；128 候选/场景、2 控制/目标、1 原 G1 视图。触发 h<4 或 purity<0.5；lambda=.25、margin=.01；输出 padded H/W 的 1/4；AnyUp FP32、256→128→64 仅真 OOM 降块。模型、文本、D2、排名、几何不改；不调参、不训练、不新增建图。单 GPU 2，三 CPU 评测进程、每进程 BLAS≤4，父 FC interpreter=`oviovo-radseg`，控制器=`ovimap-map`。部署 `N0_UNCHANGED`。

## 执行任务与验收

- [ ] T1 `binding.py`：读取父 reference/172 行14池结果库和实际 v2 预测/区域；记录 52 行/4池基线、来源与重定位、不可变算子身份。新 binder 不调用父 freeze，不要求未消费的历史 worktree。`--phase bind --resume` 完成后重用必须检查 identity。
- [ ] T2 `analysis.py` 缓存诊断先行：读取锁定输出、released matches/trace，记录恢复候选几何/类别/忽略/重复/TP/FP/并列、原 rank 变化；每 cohort 最多一个固定旧 rank 诊断池。GT 仅用于预测后的诊断。
- [ ] T3 `evidence.py`：重用原 G1 请求/掩码，验证重新 build_registry 身份；从真实 cached dense shape 计算原 signed support 与 area purity。选中帧完整 owner raster/原 depth，验证目标掩码一致；半开 bbox 扩张、面积/owner 排序取最多2合法控制。
- [ ] T4 `anyup_adapter.py`：固定 checkpoint严格载入、不改上游；独立 ImageNet guidance 对齐 FC。保留原 Q/K/conv/窗口/平均头，用块注意力流式积累普通/owner+depth/owner-only 目标与控制 pooled raw V；bilinear 使用同输出网格和 pool/head。office1 与 scene0011_00 真接口试测，至少一真实帧 full-frame ordinary/neutral 一致到 1e-5 且类别一致。
- [ ] T5 `verifier.py` 与 `outputs.py`：固定触发、竞争 cosine 中心化、显式 feature_available/proposed_class/accepted；触发外逐字返回 B1。原 `construct_output` + 冻结 D2 类别；defer 不追加 owner，官方 current-class rank 重算。
- [ ] T6 `workflow.py`、控制脚本：实现 bind/diagnose/assets/prepare/encode/predict/evaluate/select/time/tables/publish/all 与 gpu/path-map/parent-reference/resume。任务内容身份决定缓存；原始失败/实际计数保存，真失败叶最多2重试。每帧 FC≤1、AnyUp Q/K≤1；3变体共享 attention。不把程序缺陷标为算法无效/模型资源阻塞。
- [ ] T7 `evaluation.py`：在预测锁定后使用 SceneEvaluator、expanded_native_registry、fraction_metrics、trace_class_metrics、released_pool_with_classes。182 新 outcome，完全相同输入直接 alias；完整 18 ordered pools、所有5指标/类/场景，包含合法零恢复。
- [ ] T8 `selection.py`：用未舍入 fraction与1e-10容差核对10项无回退、CF APall>D2且AP50≥D2；.001 fraction另标实质目标。固定 lexicographic/simplicity 顺序，失败保留 B1；按固定 comparator map冻结≤4计时方法。
- [ ] T9 `timing.py`：Replica8 ×2逆序、≤64冷调用。同 GPU/CPU、模型和公共输入驻留；R2 search、额外几何、FC、AnyUp、输出均在计时内，不消费旧恢复 caches。记录 exclusive host stages、非可加 GPU events、峰值 allocated/reserved、加载成本、全部重复及计时后正确性校验。
- [ ] T10 `publication.py`：单结果库→3表 MD/CSV/JSON、4指定报告、cell provenance、成本/诊断/失败。Git≤50MiB，大模型/几何/特征留共享存储。逐项 requirement review；实现/资源/覆盖/科学目标/计时/选择/发布状态分开。提交与正常 push，full HEAD==ls-remote，收据存提交外。

## 验证

先对实际 production adapters 写 focused properties，再实现；测试覆盖空控制、同类邻居风险、未知深度中性、窗口零保持、trigger-off恒等、availability/defer独立、固定incumbent、新增registry、同网格与通道、alias和单位。真实 pilots只查正确性，不选参。完成后的 review 对主提示词 0–12 节及每个交付物逐项登记实物证据，不以测试替代234行/18池/计时/推送。

原文允许 PDF 可选；优先完整数值/源码/真实实验交付。主代理掌握所有科学/架构/集成/review；只有已定义接口的机械叶可使用 mechanical_worker。
