# 交接与复现

固定九方法 × 26 场景已完成，覆盖 234/234 场景行及 18/18 官方完整池化结果。目标门槛：未达到；统一研究推荐 `EV01_G1_V2`，部署 `N0_UNCHANGED`。

Replica-8 与 ScanNet-CF18 均为已曝光队列；CF18 包含 18 次采集、7 个物理场景家族。结果属于回顾性研究选择，不是独立验证。

实现冻结提交 `c012c552b10946e5754b7852650450f6a06766ad`。唯一最终结果库为 `artifacts/static_ovmap/evidence_exploration_v1/study_result_store.json.gz`，身份 `23206dac743a1b667b0643ba1ac7627d202e779cedacc5c8b933d94308210084`。所有表格单元格由该库生成，来源见 cell_provenance.json。

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_evidence_exploration.py --phase all --resume
```

共享逻辑输出根 `/mnt/shared/ww/ovimap-evidence-exploration-v1/attempt_001`；物理位置 `/home/ww/ovimap-evidence-exploration-v1-storage/attempt_001`。共享盘耗尽后仅迁移本任务输出，逐文件 SHA256 校验，保留原路径符号链接和部分失败文件；原输入身份与成功结果继续复用。

ENOSPC 失败叶的墙钟未记录，故为 null；成功评分与模型调用成本保留。首次准备字段修复前的额外 1 次 CPU 投射另计。后续诊断补充只处理锁定预测，新增 GPU/官方评分为 0。

权重、完整 owner/depth 图、FC dense cache、ScanNet RGB/mesh 不进入 Git；精确依赖与重构入口见 large_artifact_dependencies.json.gz。AnyUp 固定 351807a9 与 paper 检查点 9d035c0f…，严格载入；源码上游未修改，新增流式 wrapper 保留 CC-BY-4.0 署名。

模型/文本/类别顺序、G1/候选、D2 incumbent、官方 rank 定义均固定。GPU 2 单进程；评分 3 个 CPU 进程、BLAS ≤4、raycast 4。真实 office1/scene0011_00 pilots、focused tests、冷计时后预测一致性和逐项 requirement review 为验收证据。最终 GitHub full-SHA 发布收据位于提交之外的 output_root/publication/final.json。
