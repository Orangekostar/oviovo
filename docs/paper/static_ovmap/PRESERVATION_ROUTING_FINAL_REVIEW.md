# 提示词完成核对

核对结论：代码实现与本轮适用实验满足原提示词；精度门槛未通过，属于完成的负结果路径。未触发的条件实验没有冒充实测结果。最终 Git 发布状态由外部 `publication/final.json` 认证，提交内不写自己的未来 SHA。

| 要求 | 核对结果 | 证据 |
| --- | --- | --- |
| 新 worktree、指定基线/分支、父目录只读 | 通过；基线 6767eb90c2fd6999621b356b87269c012821713a | source_binding.json |
| 原提示词与固定配置不变 | 16 个文件与 ZIP 逐字节一致；运行 JSON 与原协议逐字节一致 | preservation_routing_v1/、configs/static_ovmap/preservation_routing_v1.json |
| P0 父实验分析 | 18 组历史 TRAIN/DEV 结果、旧损失可用性及抵消诊断保留；没有重训旧方法 | parent_audit/receipt.json、table2_parent_audit |
| P1 固定数据和提名计划 | TRAIN 622 个 base-positive 原对象/74 类；24/4 家族和 160/40 类划分沿用；prefix 2/4/8 教师与 H2 名单预锁定 | prepare.json、teachers/manifest.json、H2_plan.json |
| 数值、梯度和输出接口 | 12 组测试通过；两个真实 CUDA 对象、20 步丢弃工程训练及两场景真实官方评分通过 | validation/final_tests.json、engineering/receipt.json、engineering/two_scene.json |
| R0–R3 匹配训练 | 同 seed17 初始化和抽样，各 2,000 步，共 8,000；五个固定训练检查点；step0 不参与提名 | validation/R_execution_checks.json、training/seed17/ |
| 曲线与恢复 | 8,000 条组件损失/有效分母完整；selected/last 的模型、优化器、LR、sampler、Python/NumPy/Torch/CUDA RNG 已检查 | training/、validation/R_execution_checks.json |
| 固定门槛与选择 | 四个 R 均未通过；最高 DEV A 为 R1 36.38%，FC8 为 37.29%；不按时间或 H 选模 | R_nomination_seed17.json、selection.json |
| 提名后诊断 | FC8 和四个选中头：TRAIN 785、DEV 87、旧 H 101 个原对象，各四个相关条件；旧 H 已曝光，不是独立确认 | diagnostics/receipt.json、recognition/、table3_robustness_compute |
| 条件分支 | seed29、G 训练、H2/SAM2 获取及新全图评分按固定失败门槛未触发；对应空值标注 NOT_TRIGGERED | predictor_lock.json、table1_whole_map、table1_whole_map_repeat |
| 真实权重发布 | 四个小训练头及所需冻结初始化，共享参考去重；严格加载后八个真实 DEV 样本结果逐位一致；FC/text 保持外部依赖 | export/weights_registry.json、export/roundtrip.json |
| 表格与四份报告 | 三个表格系列具备 CSV/JSON/Markdown/LaTeX；另有五行 DEV 扰动、四行训练耗时和参数主表；负结果、单 seed 与条件分支界限已披露 | PRESERVATION_ROUTING_RESULTS/SELECTION/CLAIMS/HANDOFF.md |
| 完整 CLI 与失败记录 | 最终完整 all/resume 实际退出 0；报告封装错误已修复；旧未知退出与已观察失败分别保留；不伪造丢失的阶段计时 | execution_observed.json、execution_last.json、history/ |
| 发布内容与 Git 校验 | 小权重、紧凑评分/嵌入/曲线入 Git；不含原始扫描、GT 图、图像掩码、dense grid 或基础模型权重；正常 push 后核对完整 SHA | 外部 publication/final.json |

[机器核对记录](../../../artifacts/static_ovmap/preservation_routing_v1/validation/final_prompt_review.json) · [实验结果](PRESERVATION_ROUTING_RESULTS.md) · [实际完整 CLI 退出](../../../artifacts/static_ovmap/preservation_routing_v1/execution_observed.json)

G/H2/SAM2/新全图分支的代码已实现并接受相应数值/接口检查，本轮没有进行这些分支的完整科学实测。当前证据不支持精度成功、重复验证成功或新几何收益。
