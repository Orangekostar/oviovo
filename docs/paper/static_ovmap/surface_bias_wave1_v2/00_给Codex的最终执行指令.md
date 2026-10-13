# FG2 表面对应视角偏移校正：Codex 最终执行指令 v2

**版本：2026-10-13 / 审定版 v2。任务性质：实现、真实缓存实验、校正强度选模、条件性整图复验、结果交接与 GitHub 发布。**

> 本文是给执行代理 Codex 的命令，不是论文结果。执行本任务，不要只回答方案。所有相关条件阶段已获授权，不再逐阶段请求确认。任何结果均需完成交接和正常 Git 发布；依赖或权限缺失时发布已完成代码及明确的受阻报告，不得假称完成真实实验。
>
> **本轮选模是选择校正强度，不是训练新权重。新增优化器更新、新 RGB 编码、新 SAM 提案均为 0。** 不自动启动下一轮小头适配。用户要求避免的是重复安全审计、无关测试和浪费，不是省略数值正确性、对照或发表所需的结果核对。

## 0. 权威、边界与阅读顺序

先读本文，再读 `01_源码证据与任务绑定.md` 和 `02_发布前审核报告.md`。机器配置 `surface_bias_wave1_v2.json` 与本文一一对应。`reference_bias_solver.py` 只是已做合成自测的 NumPy 参考，不是完整仓库实现，也未在真实 FC 上验证。不得执行旧包中的 `auto_push=false`、旧训练阶段、旧的 286 行固定计数或旧选模函数。

本文取代先前 `FG2_下一步执行提示词.md`。实现前检查数值约定是否与 JSON 一致；如附件损坏造成冲突，先恢复本包原件，不能看过 DEV/地图结果后再选择更有利的解释。仓库事实以本节固定提交的源码/工件为准；数学模型、门槛和新路径属于本次设计，不声称已经存在或已验证有效。

- 仓库：`Orangekostar/oviovo`。
- 源码基点：`bec4176d9a6784a89ea293a9803e5cf6fab178ec`，以下简称 `BASE`。
- 首选分支：`research/fg2-surface-bias-wave1-v2`。
- 新模块：`src/static_ovmap/surface_bias_wave1_v2/`。
- 新入口：`scripts/evaluation/run_fg2_surface_bias_wave1_v2.py`。
- 新配置：`configs/static_ovmap/surface_bias_wave1_v2.json`。
- Git 中的报告和提示词：`docs/paper/static_ovmap/surface_bias_wave1_v2/`。
- Git 中的紧凑工件：`artifacts/static_ovmap/surface_bias_wave1_v2/`。
- 既有部署保持 `N0_UNCHANGED`；不合并主分支，不修改 CROVE/Persist4D，不把实验候选替换生产配置。

上述新路径是待创建的工作约定。先遵循本地实际存在且适用于目标路径的 `AGENTS.md`；不要声称已读取不存在的文件。

## 1. 已核实的事实及本轮决策

| 源码/工件依据 | 已核实事实 | 执行决策 |
|---|---|---|
| `frozen_fc_routing/models.py::FGHead.forward` | `local_unit+metadata` 生成 quality/key；`local_raw` 经组内池化后进入 `phi` | 只替换 `local_raw`，原样调用原 `FGHead.forward`，不重写注意力 |
| `frozen_fc_routing/common.py::FORWARD_KEYS` | 严格七键输入 | 不向 forward 加第八个键、类别、family 或 membership target |
| `learned_object_readout/data.py::prepare_family` | `site_xyz[condition,site,3]`，每条件可选不同位置 | 偏移和空间划分均按对象×条件×前缀计算，禁止跨条件缓存复用 |
| `learned_object_readout/features.py::ObjectLoader.load` | 缓存中 raw 为未归一化特征；local_unit 是原 raw 的归一化 `phi` | 本轮有意采用“原始 key、校正 value”，不得重算 local_unit |
| `frozen_fc_routing/export.py` 与权重注册表 | 精确键为 `FG2_SURFACE`、`S29_FG2_SURFACE` | 复用原 seed17 step1000、seed29 step2000 权重，不重新选 checkpoint |
| `preservation_routing/selection.py` | A/M/C 可复用；`choose/qualifies` 限制 `step>=250` | 仅复用 `summarize`，新写 lambda 选择，禁止伪造 step 绕过门槛 |
| `frozen_fc_routing/runner.py, binding.py, common.py` | 旧任务强绑定分支/协议，包含训练；旧 source binding 检查 PR/LR 原终态 | 不运行旧 `all --resume` 或旧 binder，构建新的最小执行上下文 |
| `frozen_fc_routing/prediction.py` | 只替换历史 F，再执行 N/Q/F 的 `PE_D2_GROUPED` 融合 | 图上评估必须保留这条后处理链，不能直接用新 F 的 argmax |
| `disagreement_query/outputs.py::build_payload` | 明确保持 G1 owner_ids，按新类别调用原 rank 规则 | 固定最终实例支持；不能新做几何重分组或自行替换 confidence |
| `frozen_fc_routing/evaluation.py` | 旧调度硬编码 11 方法、286 行、22 池 | 新写动态覆盖调度，复用原 scorer 数值内核，不伪造旧行数 |

证据定位与文件 Git blob SHA 见附表。路径存在、源码逻辑已核对，不代表用户服务器上的所有原始缓存、GPU、FC 权重和 Git 写权限都可用；P0 负责有限检查这些运行条件。

## 2. 工作区与依赖解析：一次做对，不遍历整台服务器

### 2.1 工作区

从 Codex 当前工作目录执行 `git rev-parse --show-toplevel`；若不在仓库，只在用户给定工作区和已知交接中的 worktree 候选内检查，不运行 `find /`。核对 `origin` 确实为本仓库；接受 SSH/HTTPS 等同一 owner/repo 的标准形式，不改全局 Git 配置。

确保 `BASE` 对象存在，缺失时只执行一次对应 fetch。新建独立 worktree 和上述分支。旧 worktree 有未提交内容也不清理、不 stash、不 reset。若本任务同名分支及 worktree 已有本版 task-id/spec 身份，恢复该任务；若名字被无关工作占用，用最小可用后缀 `-02`、`-03`，记录实际分支，不覆盖旧分支。

下面是待解析变量，不要把未核实路径硬编码为“已存在”：

- `REPO`：本任务新 worktree。
- `FG_ROOT`：父 frozen-FC 实验的实际 `attempt_001`。
- `LR_ROOT`：从父 `source_binding.json['lr_parent_root']` 得到的实际目录。
- `OUT`：新实验输出目录，必须不等于、不位于任一父实验目录内部，也不在 Git 工件目录内。
- `PY_CTRL`：优先使用已知控制环境 `/home/ww/miniconda3/envs/ovimap-map/bin/python`。
- `PY_FC`：以父 binding 的 `FC_python` 为准，交接候选为 `/home/ww/miniconda3/envs/oviovo-radseg/bin/python`。
- `GPU`：默认继承父 binding 的字符串 gpu；仅一个设备，不通过反复扫描挑空闲卡。使用原 `execution_config(binding)['gpu_lock']` 和 `queued_gpu_lock` 排队。

已知 `FG_ROOT` 候选按顺序为：
1. `/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`
2. `/mnt/shared/ww/ovimap-frozen-fc-routing-v1/attempt_001`

仅接受包含所需父 manifest、身份匹配的候选。CLI 显式路径覆盖优先于候选，路径迁移通过 `--path-map` 记录。若目录不存在或必要文件缺失，写 `BLOCKED_DEPENDENCY`；不能重新训练/编码来补缺口，不能从其他实验选择相似替代物。

若用户没有给新输出位置，定义 `TASK_HOME=FG_ROOT.parent.parent / "fg2-surface-bias-wave1-v2"`，新worktree为 `TASK_HOME/worktree`，OUT为 `TASK_HOME/attempt_001`。FG_ROOT缺失或该位置不可写时，只为开发/受阻报告选当前仓库父目录下同名TASK_HOME；不得据此替换真实科研依赖。现有目录属于无关任务时与分支一起采用最小可用后缀。记录最终实际路径后，resume不再重新选择位置。PY_CTRL不可用时，系统Python仅可用于代码整理、JSON报告与发布；真实FC阶段仍须原PY_FC环境，不能拿系统环境冒充原数值环境。

### 2.2 父文件的只读依赖合同

**P0–P2 必需：**

- `FG_ROOT/source_binding.json`、`FG_ROOT/recognition/dev/{FC8,FG2_SURFACE,S29_FG2_SURFACE}.json` 及其指向的紧凑输出数组。
- 注册表：优先父 `FG_ROOT/export/weights_registry.json`，缺失时用固定提交中的 `artifacts/static_ovmap/frozen_fc_routing_v1/export/weights_registry.json`；按注册表相对路径加载两份导出权重。禁止依赖尚不存在的新权重。
- `LR_ROOT/split_manifest.json`、`class_split.json`、`features/train-dev.json`、`data/generated/<scene>/manifest.json`、对象 `*.inputs.npz` 及该 manifest 引用的 raw/projected 缓存。
- 父 binding 的 FC 模型/算子/text/context 文件，以及 `LR_ROOT/external/upstream_sources/MaskAdapter/fcclip/fcclip.py`。`FrozenFC.__init__` 实際从 `OUT/external/upstream_sources/...` 读取算子，因此在 OUT 下创建指向父目录的只读用途符号链接。
- 复制 `class_split.json` 到 OUT，因为 `FrozenFC` 从 OUT 读取它；其它需要兼容的 manifest 同样复制原始字节，不修改其 identity。

**仅 P3 必需：**

- `LR_ROOT/regression/manifest.json`、`regression/<scene>/manifest.json`、`features/regression.json` 及相应输入/特征数组。
- `FG_ROOT/recognition/regression.json`、父各场景 `predictions/<scene>/receipt.json`、原输出数组、scorer receipts。
- 父 binding 中每场景 context、capture manifest、surface 数组、D2/G1 payload、N/Q/F scores、projection 及原 evaluator 所需文件。

P0 可只核实 P3 顶层清单存在性；到进入 P3 时才读取其大型数据。P3 缺失不得阻止已经可完成的 P1/P2，也不能反复扫描父实验的所有历史审计。

### 2.3 新 binding 的构造规则

先用原 `verified` 验证父 JSON，再在内存做路径解析；不得改写文件后保留旧 identity。深拷贝已验证的父 binding，保留 `reference/assets/FC_python/scenes/cohorts/baseline_rows/lr_parent_root` 及 load_scene 所需已有字段，只替换本任务 `output_root/path_map/gpu` 等运行项；新增字段用 `task_id/fg_parent_root/new_spec` 等明确名字。不要把继承字段 `parent_root` 任意改成 FG_ROOT，因为旧数值读取链仍可能使用其原语义。

自己生成 `OUT/binding.json`，绑定父 JSON identity、实际解析路径、两份小头、FC/text、场景及源码。不要调用父 `bind/fixed_spec/validate_lineage/prepare/train/diagnose`。

已核实权重：

| 注册表键 | seed | step | 权重文件 SHA-256 |
|---|---:|---:|---|
| `FG2_SURFACE` | 17 | 1000 | `de7fc82edba5440d9c338f5253ecd3600a340271089600c9ff2d286e2711d963` |
| `S29_FG2_SURFACE` | 29 | 2000 | `501a6ad010fd2951c93a7638fce94c599b994c7fe9215adad82054919a84ea77` |

FC identity 必须为 `6132b1b1da69a7a3bcd650586f0bc7e9aea2b64d8183a004b1a61c39892c8b44`。对应文本文件 SHA-256 为 `8e3a532d4945493f7ecaab63168fab9232c8d9d0693c717893c7f9b73a88c4b4`，注册表描述 669136 字节；实际类别 ID 顺序也要核对。

`load_exported` 会比较整个 text_identity 字典，迁移后的路径可能不同。处理方式：先证明实际 text 文件的 bytes/SHA 与父描述相同，再在 OUT 生成仅含两头的运行注册表，以实际 FC/text 描述、已核实的小头相对路径重新计算 identity，同时记录父注册表 identity。之后按真实接口调用：

```python
head = load_exported(runtime_registry, exact_key, fc.device,
                     FC_identity=fc.model_key,
                     text_identity=fc.session.text_identity)
```

不能仅为通过字典相等检查而把不同文本文件伪装成父文件。小头只有约 0.87 MB，运行注册表可复制这两份权重；Git 发布时允许引用固定 BASE 的原权重，不重复上传八份无关权重。

### 2.4 缓存与环境

复用 `frozen_fc_routing.data.ObjectLoader`，不用其 `CachedObjectLoader` 默认构造器，因为后者要求本任务尚未生成的旧格式 sparse-parity 文件。需要加速时，新写轻量张量 LRU，key 包含父输入 SHA、FC/feature 身份、condition、requested_prefix；不复制所有大缓存。单进程主机缓存总量不超过 32 GiB，其中新七张量 LRU 至多 2 GiB；不运行新的 `capture/capture_dataset/FCSession` 图像提取方法。

父 `ObjectLoader.load` 从已有网格重新池化和调用 phi，不是新的 RGB 编码；这部分时间照实记录。缺少原始网格只能报告受阻，不准“顺手重算”。

沿用原 FP32、冻结模型 eval、禁用 TF32，无 AMP；偏移求解 CPU float64 后转回原 dtype。CPU评分并发最多3，GPU worker启动前设置OMP/MKL/OPENBLAS线程数各为1。CUDA/NVML 警告本身不构成阻塞，实际 import、一次张量运算和真实 head 前向才是依据。禁止升级驱动、Torch、全局环境，禁止为了本任务跑安全扫描、全仓测试或训练压力测试。

## 3. 必须实现的接口与文件分工

| 新文件 | 必须实现的职责 | 复用位置 |
|---|---|---|
| `binding.py` | 解析上述依赖，写本任务 binding，生成运行注册表；惰性处理 P3 依赖 | 父 PathResolver/verified、权重注册表 |
| `solver.py` | Huber 位置估计、共享支持图、KKT 相对偏移、C1/C2、明确回退 | 随包参考求解器；本指令第4节 |
| `adapter.py` | 冻结父 head 包装器，仅改 raw，返回预测及独立审计信息 | 原 FGHead，不复制其网络 |
| `diagnostics.py` | P1 空间交叉拟合、供体清除、零/中心化/独立支持对照、分层汇总 | site_xyz/uv 与 sample_raw 变换 |
| `selection.py` | 原 A/M/C 汇总；lambda 选择、种子29固定复验、P3 激活表 | 只复用 preservation_routing.summarize |
| `prediction.py` | P2 对象预测、前缀诊断、P3 F 替换与真实 payload | 父 prediction.build_predictions 的数值流程 |
| `evaluation.py` | 动态方法列表，固定 mask 的原 scorer/ordered pool | 原 partition_evaluator、released_pool_with_classes |
| `reporting.py` | 主表、作用量、阶段状态、交接、紧凑归档 | 原工件 schema 可参考，不复制旧终态 |
| `publication.py` | 有限 staging、正常双提交发布、验证远端、失败保留 | Git 正常命令；第9节 |
| `runner.py` | CLI/恢复/条件阶段/子进程退出及 finally 报告发布 | 原 GPU lock 与环境分工 |
| 新入口脚本 | 设置根和 src import 路径，调用新 main | 原入口脚本形状 |
| `tests/test_surface_bias_wave1_v2.py` | 第8节有限测试集 | 不触发全仓测试 |

允许为避免重复拆分而合并 `reporting/publication` 的内部工具；公开 phase、工件与职责不变。新文件不得假冒已存在的接口；`--help`、实际调用记录和测试必须与最终代码一致。

公开 CLI 必须支持 `--spec --parent-root --output-root --gpu --path-map --phase --resume`；`--parent-root` 在新入口中明确指 FG_ROOT。phase 为 `all,preflight,parity,diagnose,select,repeat,prefix,predict,evaluate,report,publish`。运行过程无交互提示；显式单阶段若前置缺失，说明缺失而非偷偷跑其它科学阶段。

入口实现并通过 `--help` 后，使用T0记录的真实变量执行（有path-map时另加对应参数）：

```bash
"$PY_CTRL" -u "$REPO/scripts/evaluation/run_fg2_surface_bias_wave1_v2.py" \
  --spec "$REPO/configs/static_ovmap/surface_bias_wave1_v2.json" \
  --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU" \
  --phase all --resume
```

该命令是新入口的交付契约，不声称目前仓库已存在它。运行日志、最终返回码和实际解析参数由新runner及其调用方保存。

## 4. 算法合同：唯一的第一轮数值定义

### 4.1 输入与包装器

输入仍为七键：`fc_vectors, view_valid, frame_ids, local_raw, local_unit, metadata, local_valid`。有效原始特征形状 `[V,S,1536]`，`local_unit` 为 `[V,S,768]`；实际 V≤8、S≤64。有效性 `valid = local_valid & view_valid[:,None]`；active views 为至少一个 valid 位置的视角。有效输入非有限是数据/工程错误，停止受影响阶段，不伪装成“不适用”。无效 padding 不参与计算，可原样保留。

包装器内部只浅拷贝七键字典，并替换 `local_raw` 为新张量，再调用父 head。其余六项逐元素保持原样，原缓存不被修改；不重算 local_unit。`lambda=0`、方法估计不支持时直接返回 `parent_head(original_inputs,phi)`，不是 FC8。输出与审计分开保存，不能通过新增输入键改变父 forward 契约。

原参数 `eta=0.5`、原 query/quality/membership/null、原冻结 phi 和 FC8 分支一律不变。key/quality/alpha/null 在同一输入不同 lambda 下应逐位一致。此处有意让 local_unit 对应未校正 raw，属于 value-only 干预，不是假称两者仍数学同源。

### 4.2 Huber 原语

对非空有限矩阵 `P[N,C]` 转 float64：

1. `m = median(P,axis=0)`；`eps = 1e-12 * max(1, ||m||2)`。
2. `scale = max(median(||P_i-m||2), eps)`；只在初始化算一次，不随 IRLS 更新。
3. `cutoff = 1.345 * scale`。
4. 最多10次：`r_i=||P_i-m||2`，`w_i=min(1,cutoff/max(r_i,eps))`，`m_new=sum(w_i P_i)/sum(w_i)`。
5. 更新 m；若变化范数≤`1e-8*max(1,||旧m||2)`，结束。无额外归一化、裁剪、温度或语义过滤。

这些是本轮冻结的设计常数，不是声称已证明最优的统计参数。

### 4.3 M1_CORRESPONDENCE_GRAPH

每对象、每条件、每前缀单独估计。对 active views 的每对 `(v,u)`，只用同一条件内同位置的有效观测；P1 还要求位置属于当前 fit_sites。共同拟合位置少于8，不形成边。位置差 `x[v,s]-x[u,s]` 的 Huber 中心为 `d_vu`，边权严格等于共同位置数，不额外使用 GT/类别/文本或特征norm权重。

active graph 不连通、active少于2或数值求解失败：整个对象回退原 FG2；不要对各分量分别任意定 gauge 后拼接。

构造图 Laplacian L 和 RHS H：边 `(v,u,w,d)` 为 L 的 v/u 对角加 w，非对角减 w；H[v]+=w*d，H[u]-=w*d。解：

```text
[L  1] [b ] = [H]
[1ᵀ 0] [nu]   [0]
```

active bias 等权均值为0；仅最后消除float64舍入残差，不换参考视角。校正有效位置：`x_tilde[v,s]=x[v,s]-lambda*b[v]`，再转回父 raw dtype。未使用/无效位置不参与估计，不人为补观测。

### 4.4 两个强简单对照

- `C1_FULL_VIEW_CENTER`：每active视角对其全部valid位置取 raw 均值 `m_v`，令 `b_v=m_v-mean_active(m)`，按相同lambda减去。少于2active回退；不要求M1图连通。不能削弱它的支持范围。
- `C2_SITE_HUBER_CONSENSUS`：逐表面在valid视角集合上计算上述Huber中心 `mu_s`；`x_tilde=(1-lambda)*x+lambda*mu_s`。该表面少于2个视角则保留原值；不借其它表面共享偏移。其key/attention仍原样。

M1、C1、C2 全部仅作用value且使用同一lambda网格 `[0,0.125,0.25,0.5]`。lambda=0共用原FG2基线记录，不重复宣称是多个新方法。

### 4.5 数值失败与可证伪边界

有效原输入错误不能被掩盖。只有新校正造成有效输出非有限、父 phi/unit 出现非有限或范数≤父定义 `1e-12` 时，记录 `NUMERIC_CORRECTION_FALLBACK` 并用原 FG2；不得宽泛吞掉所有异常。实现中先区分原输入、原head成功与新校正失败；真正代码bug应修复并失效受影响的派生结果。

零均值相对bias无法识别所有视角共同的绝对偏移。完整可见且等权池化时，校正可以严格抵消；原 pooled raw 不变，phi不能凭空带来变化。差分均值对同一支持集合的配对顺序重排不敏感，所以配对重排可退化，不能要求其一定降分。上述边界必须在报告中保留，不能为造效果改 gauge/attention。

## 5. P0：有限准备与真实接入核对

### 5.1 只做一次的准备

记录实际分支/BASE/环境/GPU/锁/父依赖。仅校验消费到的 manifest、两头、text和必要算子身份，使用现有 memo 避免每次重复hash巨型文件。不要把父2233个工件、全数据盘、其它环境加入验收依赖。一次磁盘检查：OUT可用空间至少10 GiB才开始；达到5 GiB保留空间时停止新增大工件，报告资源阻塞，不删除用户文件。

### 5.2 真实 parity

从 TRAIN base 对象按 `key` 排序取前2个作为工程样本，执行两个种子、4条件、前缀2/4/8。它们仅用于实现核对，不选科学参数。

同进程同输入，lambda=0/回退与父head要求 `rtol=0,atol=0`；value-only下quality/alpha/null也要求逐位相同。历史导出与新环境对比另列：沿用已存在预测路径的 `rtol=1e-5,atol=1e-6` 检查历史embedding，而不是把它与短路同路径的严格恒等混淆。69个DEV base原件上，历史FC8与两FG2的class predictions及A/M/C必须一致；若历史近tie导致标签不同，先定位环境/算子差异，不能借放宽容差选模。

DEV基线全覆盖与P2原FG2计算共用一次结果，不重复跑。历史路径为 `FG_ROOT/recognition/dev/<key>.json`，数组由该JSON的arrays描述解析，按`object_keys`配对，禁止按偶然行号对应。它可能含87个对象，只筛既有base标记对应的69个，不重新定义类划分。

P0交付 `binding.json`、`parity.json`、依赖缺失清单（如有）、实际环境与一次真实head调用日志。代码正确性通过只称P0完成，不称科研假设通过。

## 6. P1/P2：诊断、有限选模及固定复验

### 6.1 P1 数据、划分与严格位置留出

P1使用原622个TRAIN base对象/24家族、69个DEV base对象/4家族，4条件，主前缀8；少于8实际视角时保持父前缀截取语义，不填造观测。只用父输入/非标签元数据拟合，不调用 `targets()` 取membership。

每个条件独立读取 `inputs.npz['site_xyz'][condition]` 和 `site_available[condition]`。以至少一个valid观测且site_available为真的site作为诊断位置集合。跨度最大轴决定排序主轴，跨度并列选X、Y、Z顺序；之后X/Y/Z/site索引稳定破平局，用 `np.array_split(sorted_sites,4)` 形成四个连续空间块。每位置的所有视图同fold。位置数量不足或holdout无有效评价对时记录UNSUPPORTED，不换分块寻找好结果。

每次3块拟合、1块留出。评价边只在该holdout块有≥4个共同有效位置的视图对上定义。拟合图仍至少8共同位置成边。用原全部valid定义active views，fit不足导致断连时fold不支持；不得用评价位置补连接。

**双线性供体清除必做，不能把“是否有元数据”留成自由选择。** 原代码已提供UV及grid manifest。对每视角用原 `normalized_grid` 获得g，再按 `p=(g+1)*[W_raw,H_raw]/2-0.5` 得到raw-grid连续坐标。供体为实际有非零插值系数且在边界内的最多4个单元，按 `align_corners=False, padding_mode=zeros`。只要某拟合site在任一视角与该视角的任一holdout site共享供体，就从所有视角的fit_sites中移除该site。清除后的同一fit集合用于所有边与对照，不存在绕另一条边使用留出位置的路径。

UV/grid缺失时该对象严格诊断为 `MISSING_DIAGNOSTIC_METADATA`，P2仍可使用七张量继续；不静默切换成宽松诊断作为主证据。深层感受野仍可能重叠，因此称空间留出诊断，不称独立数据集。

### 6.2 P1 对照与汇总

主估计M1，参照为零bias、C1（只用清除后的fit位置）、以及5个固定种子 `[101,211,307,401,503]` 的独立支持对照。

独立支持对照：保持M1每条边的视图对和共同数 n；在两端各自清除后的valid fit位置中，不放回独立抽取n个位置，以独立固定排列形成差分，执行同Huber/KKT；不改变边数/边权/gauge。随机发生器使用 SHA-256 派生的 `(object.key,condition,fold,v,u,null_seed)` 稳定种子，不使用Python进程随机hash。若两端可用集合恰与共同集合相同，集合破坏不可实现，标记 `DEGENERATE_SUPPORT_NULL`；结果保留，但不拿它证明对应关系无用或有用。配对顺序shuffle等价性只做合成单测，不再在所有对象上额外跑一轮。

留出分子N为所有可评价 `(fold,pair,site)` 上的 `||真实差分-预测bias差||²` 之和；分母D为同支持真实差分的平方和。对每对象×条件先累计N/D，再计算 `R=1-N/D`，D≤1e-20标UNDEFINED；不裁剪负R。每家族对其defined对象R等权平均，最终对defined家族等权平均。成对方法对照必须用共同defined的fold/pair/site支持，另报每方法自身覆盖，不能以不同分母直接排序。保存原始N/D/支持计数以便重算。

`mechanism_signal` 的工作定义：clean条件有≥50 TRAIN对象/8家族、≥20 DEV对象/3家族可评价，且TRAIN/DEV的家族均值R都>0。它不是显著性检验、不是因果结论。C1及独立支持对照更好或退化时明确报告。**P1不作为P2或简单对照进图的硬门槛**；即使信号不足，仍完成有限P2，不为了维护叙事忽略可用性能结果。

### 6.3 P2 seed17 的唯一选模规则

DEV为69个base原件、四条件、原完整200类竞争。基础温度0.07只用于原CE/NLL记录，不调整。先执行FC8、原FG2的两个种子基线；对seed17执行C1/C2/M1的3个非零lambda，零lambda引用原FG2。每方法在4个lambda中选1个；无新训练，也没有新checkpoint。

继承的定义必须精确：原件i的 `a_i=0.5*correct(clean)+(correct(truncate)+correct(append)+correct(truncate_append))/6`；A为各家族a均值的等权均值；M为出现的base类别a均值的等权均值；C为各家族clean准确率的等权均值。净修正是四条件普通记录的wrong→right减right→wrong，配对参照必须是**同种子原FG2**，不是FC8。

每方法lambda按 `(A降序,M降序,C降序,lambda升序)`，浮点比较容差1e-10。CE/NLL不加入本轮排序；不得调用旧choose或伪造`step>=250`。lambda=0可胜出。原始候选为0/0.125/0.25/0.5，禁止为每场景、类别、条件或种子另选强度。

简单对照冠军 `C_STAR` 在C1/C2各自最佳lambda之间，按同A/M/C排序；再比更小lambda，最后方法顺序C1在前。**在读seed29候选结果前写入 `selection_seed17.json` 并锁定M1、C1、C2各自lambda与C_STAR身份。**

### 6.4 P2 seed29 与升级条件

seed29只执行C1/C2/M1各自已锁lambda，使用固定 `S29_FG2_SURFACE` 权重；不重新搜lambda，不更换checkpoint，不根据seed29改C_STAR。

`pass(f,s)` 定义为相对同seed原FG2：

- lambda>0；ΔA≥0.005−1e-10；ΔM≥−0.005−1e-10；ΔC≥−0.005−1e-10。
- 普通净修正数>0。
- 校正造成的数值回退条件数/全部276个条件记录≤0.01；几何不支持回退另记，不混入数值回退率。

`pass_both(f)=pass(f,17) AND pass(f,29)`。这里用的是fraction，报告乘100写百分点；门槛是本轮有限计算的决策，不是CVPR录用标准。

P3集合唯一规则：

```python
if pass_both(M1):
    expanded = [M1, C_STAR]  # 即使C_STAR不达阈值，也保留必要的地图对照
elif pass_both(C_STAR):
    expanded = [C_STAR]      # 性能优化支线；不能当成M1机制胜利
else:
    expanded = []
```

若C_STAR的lambda=0，其map输出引用同seed原FG2，不新增物理推理/评分，但保留逻辑对照。不得使用非冠军简单方法的seed29结果临时替换C_STAR。P1信号是否存在单列，可影响科学解释，不能在上述表之外偷偷改激活集合。

### 6.5 前缀与作用量

锁定后，无论P3是否激活，完成DEV前缀2/4：原FG2、M1已锁lambda、C_STAR已锁lambda，两个种子，四条件。不再搜强度；loader、求解、对照只能读取这个前缀。不存在的视角不补齐。

P2记录每对象×条件×方法的：active视图、支持/图连通、lambda、回退原因、bias范数、校正前后group raw范数差、group phi差、最终embedding差、top1、GT margin、普通top1-top2 gap、修正/误伤、原权重实际作用量。完整768/1536维中间张量不全部写Git；保存紧凑向量/完整预测需要的数据与统计。默认不保留所有a矩阵，只为每family按key首个对象保留审计样例，其余保存范数/支持统计，避免产物爆炸。

## 7. P3：固定实例支持的完整地图复验

### 7.1 动态覆盖和预测锁

场景顺序从父配置完整继承：Replica8的8场景及CF18的18 captures，不重新抽子集。CF18仅7个物理家族，不等同完整ScanNet200 validation。

新方法最多M1与C_STAR，各2种子，最多104条**逻辑新方法场景记录**、8个新方法cohort池；完全相同payload可按严格identity复用物理评分。旧基线最多5个：REF_D2、REF_G1、FC8、FG2 seed17、FG2 seed29，最多130条引用和10个引用池。逻辑总数根据actual methods×scenes计算，不要求等于旧286/22。

P3所有激活方法/种子/场景的预测必须完成并锁定后才评分。若部分场景依赖缺失，记录部分结果，不拿缩小的cohort冒充完整pool，不重新选择方法。缺失场景允许按同manifest恢复补完。

### 7.2 完整数值路径，不绕开N/Q融合

复用父 `frozen_fc_routing.binding.load_scene`（只是数值读取，不是bind）。对 LR regression manifest 的 eligible_owners读取原condition0、prefix8观测，获得新FG2向量；map文本从每场景context的FC_text读取，核对valid_ids与p.valid_ids顺序一致，不把TRAIN的200类矩阵硬套到Replica场景。

每个eligible owner：只替换 `source['F']` 为新余弦分数，N/Q原分数和p.temperatures原样；严格执行：

```python
base = pool(source, p.temperatures, ('N', 'Q', 'F'))
final = simple_dependence(source, p.temperatures, base, 'PE_D2_GROUPED')
label = p.valid_ids[int(final.argmax())]
```

然后调用原 `build_payload(g1,d2,raw,relabels,eligible,nearest,matched,...)`。不改eligible名单、不重组几何、不删低支持owner、不更改原恢复对象/背景类别；确认owner_ids数组及geometry身份与G1一致。新类别导致原rank函数输出变化是合法现象，不得强行冻结旧类别rank，也不能用模型置信度另行排序。

依次记录“新F向量改变→F top1改变→融合后label改变→最终map指标改变”，因为仅改善F不一定改变N/Q/F最终输出。没有label/rank变化时按原评分identity复用，不重复计算出一个虚假新结果。

### 7.3 评价、基线核对与指标

复用原 `partition_evaluator`、`validate_overlaps`、`official_view`、`released_pool_with_classes`、`fraction_metrics` 等数值函数；自己实现动态方法循环和覆盖检查。不能直接运行含 `logical_records==286,len(methods)==11` 的旧evaluate主程序，也不能用猴子补丁伪造这些数值。

在**同一次正式地图评分**中，以每cohort顺序首场景额外运行原FG2两种子中的seed17基线，共2次基线物理核验，确认现有scorer和原历史结果相符；这2次单列为工程验证，不挤占104条科学候选上限。其它原基线在payload/rank/scorer身份一致时引用。无额外多轮scorer pilot。

核心结果APall/AP50/AP25/mIoU/mAcc；官方pooled AP不能由场景AP平均替代。分家族变化作为异质性描述，不冒充官方pooled AP。ScanNet base/novel按原class split，在完整200类竞争下报告；Replica不强行套用未经映射的160/40划分，写NOT_APPLICABLE。没有GT类别的per-class项保留原scorer空值语义，禁止自动填0改变均值。

新方法的地图结果不得用于重选lambda或seed。定义机械标签 `MAP_GAIN_VS_FG2(f)`：该方法已覆盖两cohort×两seed，四个单元的ΔAPall都>1e-10，ΔmIoU都≥−0.005−1e-10。此标签不代表全面优于旧系统或显著性，只用于报告一致分类；所有其它混合结果逐项列明，不被“未达标”隐藏。

`STRUCTURAL_ADVANTAGE_REGRESSION`仅在M1与C_STAR都完整评价、M1四单元APall均严格高于C_STAR且mIoU差均≥−0.005时标true；同时报告P1。未满足时禁止宣称M1独立优于简单对照。REF_G1/REF_D2仍完整对照，不能只展示较弱FC8。

### 7.4 成本

分开记录cache读取、CPU float64求解、CPU/GPU复制、phi/readout、payload构造、scorer和整体wall time；嵌套计时不得相加充当端到端。报告实际新RGB编码0、新训练0、新增phi调用实际数、GPU峰值与主机内存。缓存回放速度不称端到端FPS或在线30FPS。无合成延迟代替真实延迟。

## 8. 最小测试与恢复：足够，不膨胀

仅8个测试组，一次完整执行；修bug后只重跑受影响组及一个最小真实回归样例。不跑全仓pytest、无限fuzz、多GPU压力、全磁盘hash、漏洞扫描、陌生环境安装矩阵或大量伪边界组合。

1. 合成求解基础：已知相对偏移、零均值、缺失支持、断连回退；可复用随包12小例，不再复制成多套重复测试。
2. 数学退化：均衡池化抵消、差分均值shuffle等价；必须接受正确的零变化。
3. 七键及不可变输入：lambda0、无支持、标签sidecar内容变化不影响forward、无效padding不进入估计。
4. value-only真实接入：2 TRAIN对象，父head对照、真实phi、quality/alpha/null保持、非零lambda可执行。
5. 前缀/条件与位置隔离：2/4不访问后续视图、四条件独立site_xyz、fold供体清除不泄漏。
6. 选择状态机：lambda0胜出、A/M/C并列、C_STAR固定、seed29不重选、M1胜出但simple不达门槛仍进入必要对照、两者失败仍report/publish。
7. 恢复与覆盖：中断一个小样例后resume不重复已完成计算；动态scene覆盖与alias不重计物理运行；不触发旧训练。
8. 发布工件：本任务有限清单、相对链接存在、没有原始数据/基础模型/凭证、两次普通push协议的本地单测；最终用实际Git结果验收，不能以模拟push冒充发布。

历史父代码不改动。若发现必须修复的通用父bug，先在新命名空间写最薄兼容层，明确记录差异并对照；不得让父numerical kernels在新实验中静默改变。

恢复按 `phase input_identity = source/spec/父inputs/head/method/lambda/prefix` 判定。只有相同身份且输出完整的单元可复用。写盘用临时文件后原子替换；保留一次失败的错误日志即可，不创建无穷嵌套审计目录。科学参数修改必须新版本，不覆盖已有结果。工程源码修复时，仅失效受影响的下游计算，记录修复前后身份，不抹去已曝光结果。

## 9. 报告、交接和 GitHub 发布——本轮必须执行

### 9.1 必需交付

Git docs目录：本包原指令/源码绑定/审核说明、新 `RESULTS.md`、`HANDOFF.md`、`CLAIMS.md`、`REVIEW.md`。

Git artifacts目录：新配置、依赖小清单、parity/diagnostic summary、seed17全lambda表、seed29固定复验表、prefix表、activation、model registry、真实资源计数、实际阶段退出记录、适用时的map/pool/per-class紧凑数据、artifact_index。

主表只强制Markdown+CSV+JSON三种，不要求额外LaTeX、PPT或PDF。核心五表：基线核对；诊断及覆盖；选模与种子复验；完整map（或精确未运行原因）；成本和范围。必须保存所有有限lambda候选，不能只上传赢家。

`HANDOFF.md` 至少写清：BASE及本任务分支、任务结论与状态、真实运行阶段、锁定方法/强度和两个权重出处、依赖路径和读取链、环境/GPU锁、从缓存复现的实际CLI、从工件重算表格的方法、结果文件位置、结果支持/不支持的主张、失败恢复点及下一步建议。不写“权重已训练”或“线上已部署”。

### 9.2 小工件与大数据边界

本轮不产生新小头权重。`selected_model_registry.json` 描述 `BASE + parent head key/SHA + lambda + solver/schema/code identity`，足以定义部署候选；不要伪造一个训练checkpoint。允许引用BASE中的两份已发布小头。

上传新代码、配置、交接和足以复查结果的小工件；不上传原RGB-D、GT图、scan、稠密网格缓存、基础FC/text权重、运行环境、优化器、凭证。大中间结果保留OUT并登记path/bytes/SHA/再生成命令。重要结果不只留“本地路径”：需要可公开的紧凑预测/评分充分统计与表格。

以每文件40 MiB、本任务本次新增Git工件总量150 MiB为发布设计预算。超预算时使用gzip/精简非必要中间张量；不截断正式方法/场景/指标，不自动引入Git LFS或Release工作流。单个超过GitHub 100 MiB限制的blob绝不进入commit；这些大小限制来源与项目预算须区分。通常本轮表格/读出向量远低于预算，先测实际大小再处理。

### 9.3 正常双提交发布，解决“提交包含自己SHA”的循环

用户已经要求上传本仓库；正常branch push已授权，不需再次确认。按以下顺序：

1. 结束所有科学worker，控制进程记录真实返回码及实际完成/未激活/受阻范围；写报告和紧凑工件。记录 `scientific_source_digest`（只含生产代码与配置）及artifact_index。不存在新训练仍可有完整科学负结果。
2. 只stage新模块、入口、test、config及本任务docs/artifacts。运行一次 `git diff --check` 和有限新增文件大小/路径检查；不stage整个工作区、不扫描全仓密钥库。
3. 正常创建结果提交 `C1`。commit内不要预写自己的未来SHA；交接可用相对路径和scientific_source_digest。
4. 执行 `git push -u origin HEAD:refs/heads/<actual_branch>`；核对 `git ls-remote origin refs/heads/<actual_branch>` 返回的完整SHA等于C1。禁止force push、rewrite远端历史。
5. 在Git工件目录写 `PUBLICATION_RECEIPT.json`，只证明**上一条结果提交C1**的实际push、远端SHA、code/spec/artifact身份；将C1 permalink填入HANDOFF。再做一个仅包含发布收据/交接更新的文档提交 `C2`，正常push并核对远端SHA=C2。此文件不能声称自己已经证明C2。
6. `OUT/publication/final.json` 在C2远端核对后写入最终C2、本地/远端一致、C1收据位置和退出信息。它是外部终态证明，不再为把C2写进自身而无限提交。最终回答给用户C2及C1链接，Git中C2已经包含C1的真实发布收据。

若已有真实同任务C1/C2且远端匹配，resume复用，不产生无意义第三、第四条提交。正常网络错误允许一次重试；权限/身份配置缺失或远端冲突则保留本地提交、输出具体失败，状态 `PUBLICATION_BLOCKED`，不能声称上传成功。不修改全局代理/凭证，不借别人的Git身份提交。远端有无关更新时不强推，改用前述最小可用分支后缀并重新记录实际branch。

即使P1/P2无收益或P3未激活，也执行发布；其结果是“代码及负结果已发布”，不是“省略必要实验后宣称成功”。依赖缺失时同样发布完成的代码/受阻报告，但 `execution_status` 不得标COMPLETE。

### 9.4 终态与退出码

`run_summary.json`必须有三个独立维度，不用一个STOP字符串混淆：

- `execution_status`: `COMPLETE` / `BLOCKED_DEPENDENCY` / `PARTIAL_ENGINEERING`。
- `science_status`: `UNASSESSED_DEPENDENCY`、`NO_HELDOUT_SIGNAL`、`SIGNAL_ONLY_NO_READOUT_GAIN`、`READOUT_GAIN_MAP_UNCONFIRMED`、`MIXED_MAP_RESULT`、`ROUTINE_CONTROL_GAIN_ONLY`、`REGRESSION_MAP_GAIN`；保存P1/pass_both/MAP_GAIN等布尔事实及全部数字。机械排序：P2及复验未完整时为UNASSESSED，但仍保存已测阶段事实；P2完整后，完整map任一方法满足MAP_GAIN且M1不满足则ROUTINE_CONTROL，M1满足则REGRESSION_MAP_GAIN，完整map无者满足则MIXED；map缺失或不完整而pass_both为真则READOUT_GAIN_MAP_UNCONFIRMED；否则依P1有无信号分别SIGNAL_ONLY/NO_HELDOUT。依赖/发布受阻不得抹掉已完成的P1/P2数值。
- `publication_status`: `PUSH_VERIFIED` / `PUBLICATION_BLOCKED`。

负结果且全部条件阶段按规则完成、发布已核实，进程可退出0；未解决依赖/工程或发布失败返回非零。未激活P3是`NOT_ACTIVATED_BY_FIXED_RULE`，不是缺失实验。先前已激活却没有完整跑完的P3不能伪装成未激活。

`all` 必须以try/finally安排report和publish；科学失败不吞掉异常，也不阻止已完成工作的交接。实际`all`进程最终返回码由调用它的父shell/Codex记录；commit中的worker退出记录只声称已观察到的worker退出，不能提前伪造尚未退出的all进程证明。

## 10. 最终审核与任务安排

按以下顺序落地：

| 阶段 | 工作 | 进入条件 | 完成产物 |
|---|---|---|---|
| T0 | 读本地适用规则；建立worktree、binding、注册表、环境 | BASE/仓库可访问 | 实际依赖和实现计划 |
| T1 | solver＋薄包装器＋CLI，有限8组测试 | T0 | 可调用的新模块，真实parity |
| T2 | P1严格空间留出诊断 | 原训练缓存可读 | 诊断/覆盖/对照；不提前判性能失败 |
| T3 | P2 seed17全有限候选和锁定 | DEV可读、parity通过 | 唯一lambda与C_STAR |
| T4 | seed29固定复验、2/4前缀 | T3锁存在 | 升级布尔表、前缀结果 |
| T5 | P3所有预测→全局锁→评分 | expanded非空且依赖可用 | 动态完整map/pools/成本 |
| T6 | 数据与实现审核，报告、交接 | 无条件，含负结果/阻塞 | REVIEW/RESULTS/HANDOFF/CLAIMS |
| T7 | C1/C2正常Git发布及外部收据 | Git权限可用 | 已核对远端SHA和最终链接 |

审核必须核对：表格由正式JSON生成；方法/seed/条件覆盖正确；seed29及map未重选；父几何和owner未变；未新增训练/编码；地图N/Q/F链保留；引用与新算区分；假设未写成发现；费用计数不重复；发布未预写未来SHA。

发现错误先修复相应实现/文档再发布，重算只限受影响下游。禁止无限追加“再审核一次”的循环；正常情况做一次成品审核，修复后一次针对性复核。

最终给用户的回答必须包含实际任务结论、核心指标、完成/未运行原因、真实训练/编码计数、Git分支与C1/C2完整SHA、HANDOFF和RESULTS固定提交链接、剩余局限。不要只给“全部测试通过”或“已完成开发”，也不要承诺未执行的后台工作。

**现在执行T0–T7。新颖性是待检验主张；正确完成开发、有限选模、真实结果与正常发布是本任务的完成目标。**
