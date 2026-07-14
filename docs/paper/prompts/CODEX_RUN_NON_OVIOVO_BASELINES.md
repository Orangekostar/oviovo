# Codex Prompt: Run Non-OVIOVO Baselines

你现在的唯一目标是：在不运行、不调参、不修改 OVIOVO 算法的前提下，优先跑出 AAAI 论文主表中所有非 OVIOVO baseline 的可复现结果，并将结果通过 JSON provenance 自动接入论文表格。

全程内联执行，不调用任何 skill，不启动 subagent。不要只输出计划；持续执行到所有可行 baseline 完成，或每个不可行项都有可复现的阻塞证据。一个 baseline 失败后继续处理其他 baseline，不要整体停住。

## 1. 权威路径

- OVIOVO 主工程：`/home/ww/.config/superpowers/worktrees/oviovo/tsdf-contamination-visibility-gates`
- 当前分支：`feat/current-state-tree`
- 参考工作区：`/home/ww/oviovo_aaai_workspace`
- 仓库版本清单：`/home/ww/oviovo_aaai_workspace/manifests/repository_snapshot.tsv`
- baseline 执行审计：`/home/ww/oviovo_aaai_workspace/03_analysis/BASELINE_EXECUTION_AUDIT.md`
- baseline 能力矩阵：`/home/ww/oviovo_aaai_workspace/03_analysis/BASELINE_MATRIX.md`
- 环境状态：`/home/ww/oviovo_aaai_workspace/05_environments/environment_status.tsv`
- 安装报告：`/home/ww/oviovo_aaai_workspace/05_environments/INSTALLATION_REPORT.md`
- benchmark 协议：`/home/ww/oviovo_aaai_workspace/04_framework/BENCHMARK_FRAMEWORK.md`
- 中立评测接口：`<OVIOVO>/src/evaluation/`
- 表格模板：`<OVIOVO>/docs/paper/benchmark_tables.md`
- token registry：`<OVIOVO>/docs/paper/benchmark_tokens.tsv`
- 表格生成器：`<OVIOVO>/tools/benchmark_tables.py`
- 大型运行输出根目录：`/home/ww/oviovo_baseline_runs`
- 容器/编译工作目录：`/home/ww/oviovo_baseline_builds`
- 公共数据和权重目录：`/home/ww/oviovo_benchmark_assets`

先完整读取上述清单、审计、协议、环境状态和表格 registry，再执行命令。分析文档不能替代真实源码检查：运行每个项目之前，必须打开其 README、入口脚本、配置和 evaluator。

## 2. 本次精确范围

只处理主表的非 OVIOVO 行：

### Table 1: Static Mapping Quality

- `T1_OPENFUSION_*`
- `T1_OVIMAP_*`
- `T1_CONCEPTGRAPHS_*`
- `T1_DUALMAP_*`

数据和指标：

- Replica-8 compatibility：`room0, room1, room2, office0, office1, office2, office3, office4`。
- Replica-7 held-out：排除开发场景 `room0`，其余七个场景做 macro average。
- ScanNet200-5：运行前提交固定五场景 manifest；不得看结果后换场景。
- Runtime semantic mIoU/mAcc/f-mIoU、class-agnostic AP25/AP50、Geometry F@5cm。
- OpenFusion 没有原生 entity AP 的单元保持 `N/A`，不得填零或用语义点聚类伪造。

### Table 2: Dynamic Current-Map Quality

- `T2_OVIMAP_FROZEN_*`
- `T2_CONCEPTGRAPHS_FROZEN_*`
- `T2_DUALMAP_*`
- `T2_PANOPTIC_SHARED_*`
- `T2_KHRONOS_OPEN_*`
- `T2_KHRONOS_ORACLE_*`

数据和指标：

- 官方 TESSE-CD Apartment 与 Office 序列。
- 官方 Object F1、Dynamic-object F1、Change F1。
- 公共指标 Current mIoU、Ghost Rate、Background Recovery F@5cm、Recovery Frames。
- frozen baseline 只处理 pre-change 数据；介入后禁止更新地图。
- `Panoptic Mapping + shared masks` 必须标记 `composed`。
- `Khronos (GT semantics)` 必须标记 `oracle`，不得参加最佳结果排名。
- `Khronos (open-set)` 运行时不得读取 GT semantic、instance、change 或 visibility 标注。

### Table 4: Online Efficiency and Memory

- `T4_OVIMAP_*`
- `T4_CONCEPTGRAPHS_*`
- `T4_DUALMAP_*`
- `T4_KHRONOS_*`

统一报告：frontend、backend、maintenance、total seconds/frame、Hz、finalization、query p50/p95、peak GPU GB、peak RAM GB、final map MB、evaluation/I/O seconds。

只比较本机同硬件实测值。论文中的硬件数字不得混入排序列。

### 明确排除

- 不运行任何 `T1/T2/T4_OVIOVO*` token。
- 不运行 Table 3；它是 OVIOVO 消融。
- 本轮不跑补充表 S1-S3，除非主表全部完成且仍有足够时间。
- 不修改 `src/modules/`、`src/pipelines/`、OVIOVO 配置或模型权重。

## 3. 不可违反的规则

1. `02_repos/` 下全部参考仓库只读。开始和结束运行：

   ```bash
   cd /home/ww/oviovo_aaai_workspace
   VERIFY_ENVIRONMENTS=0 bash scripts/verify_workspace.sh
   ```

2. 禁止从论文 PDF 抄数字，禁止用旧协议数字冒充本协议结果，禁止以 `--help` 或 import smoke 冒充完成运行。
3. 运行时禁止 GT semantic、instance、change、visibility 输入；GT 只能进入 evaluator。Khronos oracle 是唯一显式例外，且结果必须隔离标记。
4. 在线方法禁止未来帧。验证方法：截断到 frame `t` 后，追加未来帧不得改变 `t` 时刻快照。
5. 不得修改 reference checkout 来修路径。使用外部 wrapper、环境变量、只读 bind mount 或复制到 `oviovo_baseline_builds` 的带许可证构建副本。
6. `NO-LICENSE` 项目只能运行和引用官方代码，禁止复制、改写、vendoring 源码。尤其是 OpenFusion。
7. 不得把 native、composed、frozen、offline、oracle 混为同一方法。
8. 不得手工修改 Markdown/LaTeX 数字。所有数字必须来自 JSON 文件和 JSON pointer。
9. 任何 threshold、label alias、场景 manifest 在测试集运行前冻结。不得使用 test annotation 调参。
10. 既有用户未跟踪文件不得删除、暂存或覆盖。禁止 `git reset --hard`、`git checkout --`。
11. 未经明确要求不要 push。允许在 `feat/current-state-tree` 上创建小提交；每个提交只包含 adapter、测试、manifest 或结果摘要。

## 4. 已知本机资源

- Ubuntu 24.04；Docker：`/usr/bin/docker`；tmux：`/usr/bin/tmux`。
- 3 张 NVIDIA A40 46 GB：GPU `0,1,2`。
- Replica RGB-D：`/home/ww/vv/dataset/Replica`。
- Replica 原始 GT mesh：`/home/ww/vv/dataset/Replica-Dataset/Replica_original`。
- 已有八场景：`room0, room1, room2, office0, office1, office2, office3, office4`。
- DualMap 环境：`/home/ww/miniconda3/envs/oviovo-dualmap`。
- stmetrics 环境：`/home/ww/miniconda3/envs/oviovo-stmetrics`。
- 已有 DualMap 权重：
  - `/home/ww/vv/paper2/DualMap/model/yolov8l-world.pt`
  - `/home/ww/vv/paper2/DualMap/model/mobile_sam.pt`
  - `/home/ww/vv/paper2/DualMap/model/FastSAM-s.pt`
- 已有 CropFormer 权重：`/home/ww/vv/paper2/Entity/checkpoints/CropFormer_hornet_3x_03823a.pth`。
- 本机当前未发现 TESSE-CD/ROS2 bags 或完整 ScanNet 数据。

允许从官方公开地址下载缺失数据/权重到 `oviovo_benchmark_assets`，但必须先检查磁盘空间、记录来源 URL、license、SHA256 和下载日期。需要账号、点击同意、私有 token 或人工接受协议时，不得绕过；记录 blocker 后继续其他任务。

## 5. 已有结果的使用边界

以下已有结果只能用于 smoke、adapter fixture 和开发回归，不能直接成为 held-out headline：

- DualMap room0：`/home/ww/vv/paper2/DualMap/output/benchmark_20260615_existing_methods/`
- OVI-MAP room0：`/home/ww/vv/paper2/OVI-MAP/output/benchmark_20260615_ovimap/`
- ConceptGraphs room0：`/home/ww/vv/lifelongmap_benchmack/concept-graphs/outputs/benchmark_room0_s10_200f/`
- 历史说明：`<OVIOVO>/docs/superpowers/reports/2026-06-15-paper-benchmark-table.md`

先用这些真实文件写 adapter fixture 并复核已有开发指标；正式 VERIFIED headline 必须来自冻结的完整场景 manifest 和本次可追踪运行。

## 6. 执行顺序

### Phase A: Preflight 与协议冻结

1. 检查 OVIOVO 和参考工作区 `git status`，保存到运行根目录。
2. 记录 GPU、CPU、RAM、驱动、Docker、Conda、磁盘空间。
3. 清点 Replica、ScanNet200、TESSE-CD、权重与已有输出。
4. 创建并提交：
   - `configs/evaluation/manifests/replica8.json`
   - `configs/evaluation/manifests/replica7_heldout.json`
   - `configs/evaluation/manifests/scannet200_5.json`，仅在官方数据可用后冻结。
   - `configs/evaluation/manifests/tesse_cd.json`，仅在官方数据可用后冻结。
5. manifest 必须记录 scene、frame range、stride、vocabulary、aliases、depth scale、pose source、split role。
6. 写 `preflight.json` 和 `baseline_completion.tsv`，状态仅允许 `PENDING/RUNNING/VERIFIED/BLOCKED/N/A`。

### Phase B: 共用 adapter 与结果契约

只允许在 OVIOVO 中新增或修改：

- `src/evaluation/baselines/`
- `scripts/evaluation/`
- `tests/evaluation/`
- `configs/evaluation/manifests/`
- `tools/import_benchmark_results.py`
- `docs/paper/results/baselines/`
- 必要的 `tools/benchmark_tables.py` 和对应测试，但不得改变表格行列协议。

每个 baseline 先用已有 room0 输出做一个最小 fixture。adapter 必须输出中立 `MapSnapshot`、runtime JSON 和方法元数据。测试至少验证：

- schema 和数组单位；
- runtime label 来自方法输出，不从 GT 回填；
- entity ID 稳定性声明；
- current/history scope；
- frozen map 在 intervention 后没有 update；
- composed/offline/oracle 名称；
- evaluator 对完美、空、stale、ID-switch fixture 的响应。

### Phase C: 按可行性运行

严格顺序：

1. **DualMap**：环境和三项权重已存在，最先跑。
2. **ConceptGraphs**：先验证已有 room0 输出；在独立 Conda 环境补官方运行依赖，不与 OVIOVO 环境合并。
3. **OVI-MAP**：使用 Ubuntu/ROS Noetic Docker 或已有可复现容器；reference checkout 只读。
4. **Khronos**：使用官方 ROS2 Jazzy/devcontainer 路径；依赖安装到外部 workspace。
5. **OpenFusion**：只使用官方 Docker/README 路径；不得复制无许可证源码。
6. **Panoptic Mapping + shared masks**：在独立 ROS1 容器运行，明确记录共享 mask 来源，因此标 composed。

每个方法执行四级 gate：

1. import/CLI smoke；
2. 1 帧或最小合法输入 smoke；
3. Replica `room0` 200-frame stride-10 开发回归；
4. 冻结 manifest 的完整正式运行。

前一级失败不得启动该方法的长运行，但必须记录失败命令、exit code、日志和最小下一步。然后继续下一个 baseline。

长运行使用 tmux，状态文件必须包含 `started_at/finished_at/exit_status/commit/config_hash/output_dir`。最多每张 GPU 一个长任务：

- GPU 0：DualMap；
- GPU 1：ConceptGraphs/OVI-MAP Python frontend；
- GPU 2：OpenFusion/Panoptic frontend；
- Khronos/ROS 任务优先 CPU，若启用 GPU 则显式占用空闲设备。

不得为了并行让两个任务写同一目录、同一 cache 或同一 result manifest。

### Phase D: 动态与 frozen 协议

TESSE-CD 可用后：

1. 先验证 timestamp、pose、intervention annotation 对齐。
2. OVI-MAP/ConceptGraphs 只用 pre-change 部分生成一个 frozen snapshot。
3. intervention 后只重复查询/评估该 snapshot，禁止继续调用 mapper update。
4. DualMap 运行 native dynamic 模式，不能使用 local-map-only 配置冒充动态模式。
5. Khronos 分别运行 open-set 和 GT-semantics oracle 配置，输出目录严格隔离。
6. 所有方法通过同一 common evaluator 计算 Current mIoU、Ghost Rate、BG F@5cm、Recovery Frames。
7. 官方 TESSE/Khronos 指标使用官方 evaluator；common 指标使用 OVIOVO 中立 evaluator。两者不能混称。

### Phase E: 效率测量

1. 固定同一机器、输入帧、stride、分辨率和 warm-up。
2. 用单独进程采集 wall time、peak RSS、GPU peak memory；不要从日志估算 peak。
3. initialization、frontend、backend、maintenance、finalization、evaluation/I/O 分开。
4. 每个 deterministic 方法运行一次；有随机性的配置用三个固定 seed。
5. ROS/container startup 不计入 online seconds/frame，但必须单独记录 initialization。

### Phase F: 自动写入论文结果

每次 VERIFIED 运行生成一个小型、可提交 JSON 摘要到：

`<OVIOVO>/docs/paper/results/baselines/<method>/<dataset>/<run_id>/result.json`

摘要必须包含：

- method/display label/mode；
- upstream commit；
- adapter commit；
- environment 或 container digest；
- 完整命令；
- config path 和 SHA256；
- weights URL/path/SHA256；
- dataset manifest path/SHA256；
- pose source；
- hardware；
- seed；
- raw output/log path及 SHA256；
- metrics；
- runtime breakdown；
- protocol deviations；
- status=`VERIFIED`。

实现 `tools/import_benchmark_results.py`：

1. 输入 result manifest 和 `benchmark_tokens.tsv`。
2. 只允许更新本提示词列出的非 OVIOVO token。
3. 从 `result.json` 的 JSON pointer 读取值，不接受命令行直接传数字。
4. 更新 registry 的 `source_json/json_pointer/status/note`，保留全部 OVIOVO token 为 `UNFILLED`。
5. 生成派生文件 `benchmark_tables_baselines.md` 和 `benchmark_tables_baselines.tex`：VERIFIED baseline 显示格式化数值，未完成和 OVIOVO 单元保留 token，N/A 显示 `--`。
6. canonical `benchmark_tables.md/.tex` 仍作为无数字模板，不手工编辑。
7. importer 必须拒绝：缺文件、无效 JSON pointer、NaN/Inf、精度不符、方法/数据集不匹配、OVIOVO token、重复 token、非 VERIFIED 结果。

## 7. 完成标准

每个 baseline 行最终只能是以下之一：

- `VERIFIED`：官方代码真实运行，协议兼容，结果 JSON、命令、commit、配置、资产 hash 完整。
- `BLOCKED`：保存真实失败命令、exit code、日志、缺失数据/环境和下一步；registry 保持 `UNFILLED`。
- `N/A`：方法接口原生不能产生该指标，并有明确说明。

最终必须交付：

1. `baseline_completion.tsv`：覆盖 Table 1/2/4 所有非 OVIOVO 行。
2. 每个 VERIFIED 单元可由 `source_json + json_pointer` 重新读取。
3. `benchmark_tables_baselines.md/.tex`。
4. 一份 `BASELINE_RUN_REPORT.md`，仅报告真实完成、真实阻塞、协议偏差和可复制命令。
5. 聚焦测试、registry 验证、表格生成验证全部通过。
6. `VERIFY_ENVIRONMENTS=0 bash scripts/verify_workspace.sh` 最终 PASS，19 个 reference repo 全部 clean。
7. OVIOVO mapping/pipeline 文件无修改，OVIOVO 相关 token 全部未运行、未填值。

## 8. 工作方式与汇报

- 每 30-60 秒给出一句简短状态，说明正在运行什么、产物路径和阻塞原因。
- 不要因为某个 ROS/container baseline 困难而等待用户；先继续独立任务。
- 只有在需要登录、license 接受、私有凭证、付费资源或可能破坏现有数据时才停止并询问。
- 不得用“应该能跑”“已安装”代替执行证据。
- 每个阶段结束只汇报：改动文件、实际命令、退出码、结果路径、VERIFIED/BLOCKED 数量、下一阻塞项。

现在开始执行 Phase A。不要再次写方案，不要运行 OVIOVO。
