# OVI-MAP module validation handoff

本次交付覆盖冻结的 ScanNet 最小模块验证研究。完整[发布包](../../../artifacts/static_ovmap/module_validation_v1/release-20260923-native-v10/)与[冻结和数据锁](../../../artifacts/static_ovmap/module_validation_v1/finalization-20260923-native-v10/source_manifest.json)分别归档。历史 Room0、native-v8/v9 与早期缺数据的交付仅保留作溯源；本页和 native-v10 发布包是当前结论。

- 实现：COMPLETE。S 与 Q 已训练并测量；G_ORIGINAL/G_AGREEMENT 已测量，G_QUALITY 为 BLOCKED_PARTITION_TARGET_SUPPORT。
- 科学结果：NO_NET_GAIN；最终候选 N0。组合和 B100/400 曲线未满足冻结触发条件。
- CONFIRM：NOT_REQUIRED_NO_RETAINED_CANDIDATE。scene0553_00、scene0064_00 只有已授权原始数据，没有导出图像、转换标注或产生实验行。
- 发布：生成报告时为 NOT_CHECKED_BY_REPORT；最终推送结果以仓库外 publication-20260923.json 的本地/远端 SHA 比对为准。

## 结果与边界

SELECT 两场景 N0 平均 released uAP = 0.2574074074074074（25.740741%），mIoU = 0.35334443201828547（35.334443%）。全部分场景数值、定义分母、对象匹配增损、有效性转换和成本在 [结果报告](MODULE_VALIDATION_RESULTS.md) 与发布包 scientific_evidence.json。

S 的 CAL 教师为 S_WOW_VOTE，S_SIMPLE / S_NO_CONTEXT / S_PAIRED 阈值分别为 0.3 / 0.1 / 0.2。三个小头均实际训练。S_PAIRED 没有通过冻结的精度/非劣化规则，保留 N0。

G 仅有 5 / 355 个 FIT 分组的目标可区分，涉及 2 场景，低于 20 分组 / 4 场景的冻结训练门槛。因此未训练 G_QUALITY。G_AGREEMENT 在 SELECT 改动了 scene0626_00 的 1,320 个点，但未改善指标，保留 G_ORIGINAL。G 的新组件语义读取和点数排序与 N0 不同；N0→G_ORIGINAL 桥接变化不作为几何模块收益。

Q 的 CAL 比较策略锁定为 Q_COMBINE。SELECT 平均 uAP 从 27.676768% 升至 Q_GAIN 的 32.028620%，mIoU 从 30.668959% 升至 43.621312%；但 scene0445_00 uAP 下降 3.333333 个百分点，未通过逐场景非负门槛。Q_GAIN 平均实际付费请求 200，比较策略为 198.5，不能声称相同实际成本下占优。B100/400 曲线、组合与 holdout 均按预定规则跳过，没有据此调参或扩展数据。

这只是 8 FIT / 2 CAL / 2 SELECT 的冻结小规模研究，未建立总体泛化或系统实时性结论；基础模型预训练重叠未知。下一项研究动作：仅用已有 FIT/CAL 日志分析 Q 的跨场景效用偏移，再为新假设制定独立协议；本轮不重选或重训。

## 身份与实际路径

工作树：`/home/ww/crove/ovimap-module-validation`；分支：`research/ovimap-module-validation-v1`。

实际 S/G/Q 预测生成代码：`1d83aec5e3c5220b0a5397e1bc3f720dfcbac3b9`。最终冻结/报告代码：`b6ab45b8dadf6672d2bcc2a6cfaed9bf41d60a0c`。后续交付提交归档结果、审计脚本和文档；其 SHA 在外部发布回执，避免自引用。

上游 OVI-MAP：`f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424`；独立补丁工作树：`/home/ww/crove/ovimap-module-validation-upstream`；编译依赖复用 `/home/ww/oviovo_baseline_builds/ovimap-ubuntu24-native/OVI-MAP`。跟踪补丁 SHA-256：`b8ef35aef2490b563d2cca1451baef6f5afad2963e071016bcb837aee3e935b5`。

实际加载扩展：`/mnt/shared/ww/ovimap-module-validation-v1/tooling/repairs-20260922/native-query-v10/lib/consistent_gsm.cpython-311-x86_64-linux-gnu.so`；SHA-256：`4ecf3debd102382c2a2ca489e045be10091b029fb7ed965a4c3c6e2f5dc47f57`。相邻 `receipt.json` 记录编译环境和真实两帧验证。native-v10 修复只读 raycaster membership；RGB 修复显式声明 HWC，包含窄裁剪的真实预处理验证。

根目录为 `/mnt/shared/ww/ovimap-module-validation-v1`。数据在 `data/scannet`；模型在 `models/{siglip2-large-patch16-384,WOW-Seg,WOW-Seg-Meta,all-MiniLM-L6-v2}`；native SigLIP 在 `/home/ww/vv/paper2/model_cache/google_siglip-large-patch16-384`。实际模型配置、权重和代码文件哈希见发布包依赖清单及各模型回执。14 个预选原始场景已齐备，12 个开发场景已导出；用户提供的 `/home/ww/getscannet.py` 保持不变。

原生缓存在 `scannet_runtime_v1`，科学结果在 `scannet_study_v1`，完整公共流程在 `attempt_011`。大数组、图像、基础模型权重留在外部，external_artifacts.json 记录真实路径、字节数和 SHA-256；这不表示这些大文件已上传 Git。

公共 `all` 进程在所有阶段回执完成后，于重复刷新进度的校验阶段终止（退出码 143）。随后独立核对了 8 个阶段的状态、依赖摘要和 20 个主要输出哈希，更新最终进度，再调用原有 `export_release` 函数导出。完整记录见发布包中的 [completed_phase_verification.json](../../../artifacts/static_ovmap/module_validation_v1/release-20260923-native-v10/completed_phase_verification.json)；实际导出命令收录于最终审计。

## 实测成本

以下仅计当前 native-v10 与修复后 RGB 适配器的有效运行，退役调试运行另存历史目录。attempt_011 是结果复核与交付运行；其阶段墙钟包含共享存储校验，不计作新的独立实验或再次推理成本。

- CropFormer 前端：12 个开发场景，3,337.062770 秒。
- 原生 capture：12 场景，2,400 槽位，2,305 有效帧、95 无效位姿；11,866.383228 秒。
- S：每模型 1,903 请求；native 17,127 个物体/背景裁剪，SigLIP2 11,418 个裁剪，WOW 1,894 次成功生成、9 次技术不可用。请求计时分别 658.524564 / 448.146843 / 1,395.818540 秒。
- G：4,035 个逻辑请求 / 24,210 个逻辑裁剪；3,326 次唯一物理请求 / 19,956 裁剪，物理推理 706.156557 秒。
- Q：7,369 个逻辑请求 / 44,214 个逻辑裁剪，失败 0；6,634 次物理请求 / 39,804 裁剪，735 次缓存命中，物理推理 1,421.880463 秒。
- 三个 S 小头与 Q_GAIN 共训练 1.521769 秒；4 个 checkpoint 共 131,528 字节。G_QUALITY 没有训练；未训练任何基础模型。

capture 墙钟、S 请求计时和 G/Q 物理推理计时范围不同，不能直接相加作为端到端耗时。CropFormer、加载耗时与细项保留在 scientific_evidence.json。

## 复现

在上述工作树执行，沿用已锁定环境和配置。仅使用 GPU 2 的现有全局锁；不改变其他任务。

```bash
cd /home/ww/crove/ovimap-module-validation
export OVIMAP_DATA_ROOTS=/mnt/shared/ww/ovimap-module-validation-v1/data/scannet
export OVIMAP_VERIFY_HASH_CACHE=1
export PYTHONPATH=/home/ww/crove/ovimap-module-validation/scripts/evaluation/verification_cache:/home/ww/crove/ovimap-module-validation
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
PY=/home/ww/miniconda3/envs/ovimap-map/bin/python
"$PY" scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_011/resolved_config.json \
  --phase all --output-root /mnt/shared/ww/ovimap-module-validation-v1
```

上述命令用于科学阶段复核，本轮全部阶段已经完成。resolved-config 复用科学配置和回执；提交身份变化时公共入口会建立新 attempt，不会重新选择科学参数。哈希缓存为显式启用、进程内有效，首次读取真实文件，复用前检查元数据；新近修改文件重读校验。

原导出器只压缩大于 256 KiB 的 JSON，首次完整导出为 143,325,566 字节，超过 100 MiB，因此未创建发布目录。本轮保留冻结代码，通过仓库外 256 MiB 临时中间包及独立 [compact_release.py](../../../artifacts/static_ovmap/module_validation_v1/finalization-20260923-native-v10/compact_release.py) 无损压缩 12,929 个小型 JSON，最终发布包为 96,099,584 字节（91.65 MiB），仍严格执行 100 MiB 上限。主要结果文件和 4 个权重保持原始字节；压缩文件的原始来源、变换类型、解压后 SHA-256 均在 export_manifest.json。中间清单另以 gzip 归档；首次失败、中间导出和最终压缩命令均见 scoped_final_audit.json。已存在的目标目录不会被覆盖；复现发布需使用新的目标路径，不能仅向原导出器传入 `--export-root`。

叶任务入口：
```bash
"$PY" scripts/evaluation/run_ovimap_scannet_capture.py --config configs/evaluation/ovimap_module_scannet_runtime.json
for branch in semantic geometry query selection; do
  "$PY" "scripts/evaluation/run_ovimap_scannet_${branch}.py" \
    --config configs/evaluation/ovimap_module_scannet_study.json --phase all
done
"$PY" scripts/evaluation/run_ovimap_module_study.py \
  --spec docs/paper/static_ovmap/module_validation_v1/spec/PROTOCOL_SPEC.json \
  --resolved-config /mnt/shared/ww/ovimap-module-validation-v1/attempt_011/resolved_config.json \
  --phase confirm --output-root /mnt/shared/ww/ovimap-module-validation-v1
```

公共 confirm 入口会读取冻结的 N0 决策并返回 NOT_REQUIRED，不打开 holdout。不要通过直接调用叶任务绕过该门槛。

验证记录见 [scoped_final_audit.json](../../../artifacts/static_ovmap/module_validation_v1/finalization-20260923-native-v10/scoped_final_audit.json)：199 项原有模块测试通过；报告修订 3 项、哈希缓存 8 项和启用缓存后的相关回执测试 40 项通过（有重叠，不累计成独立测试总数）。新增无损压缩 3 项测试通过。实际核验覆盖 104 个 S/G/Q 预测、24 条 Q 轨迹、116 条去重评估行、34 组均值、421 个冻结源文件及全部 16,430 个发布文件；来源哈希、变换内容、解压字节和 4 个小头权重均核对通过。
