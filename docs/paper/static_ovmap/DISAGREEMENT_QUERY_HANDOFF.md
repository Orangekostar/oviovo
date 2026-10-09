# Disagreement query handoff

固定筛选完成：10 条方法路径、4 个已曝光场景、40 条完整输出结果、20 个有序双场景池。三个预定扩展对比均失败，按协议保留 DQ01_G1，部署 N0_UNCHANGED。完整 8/18 场景回归 NOT_RUN_NO_SCREEN_SIGNAL；严格与材料目标未在完整回归中验证。

代码、参数和原始任务包在本分支；父实验只读。绑定实际 234 行/18 池父结果，映射 SU00_D2/SU01_G1 与 IR 血缘。

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_disagreement_query.py --spec configs/static_ovmap/disagreement_query_v1.json --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-disagreement-query-v1/attempt_001 --phase all --resume
```

可单独使用 bind/diagnose/screen/expand/time/report/verify。expand 与 time 尊重已冻结条件，不接受手动强制扩展。新增算子与输入变更须先归档受影响的新任务后代；当前实现拒绝跨身份命中，原实验不会被修改。

[紧凑结果](../../../artifacts/static_ovmap/disagreement_query_v1/result_store.json)、[可恢复清单](../../../artifacts/static_ovmap/disagreement_query_v1/bundles/manifest.json)、[完成矩阵](../../../artifacts/static_ovmap/disagreement_query_v1/completion_matrix.md)。bundle 包含小型 FULL/子区域 mask、特征/余弦、物理支持、损失无损概率、诊断、评分事件和完整 owner 类别/rank 注册表。原始 RGB-D、完整固定几何/owner 数组、模型权重与 dense 特征留在共享存储，内容身份和位置在清单及 source_binding 中。

用 bundles.unpack_json 读取 gzip JSON；bundles.restore_payload(record, baseline) 由指定 G1/D2 owner 数组恢复完整输出，并核验 prediction_key/record_key。verify 阶段实际从 Git 工作树内 bundle 恢复全部 40 条输出和 512 条逐对象更新。

物理代表样本最多 4096，属于确定性面积求积近似；不做容差焊接。FC 完整词表/FP32/池化/温度固定。科学采集实际使用 GPU 2、一处 FC 工作者、每工作者 4 CPU 线程；评估最多 3 进程。正分支回归与冷测处理器未在本次负筛选结果上执行，不能作为已验证性能。

权威外部目录：`/mnt/shared/capacity/node101/ww/ovimap-disagreement-query-v1/attempt_001`。真实全 CLI 日志与退出码见 `execution/all_resume.commands.json`；正常 push 后完整本地/远端 SHA 与结果身份见外部 `publication/final.json`。后者不写入自引用提交。
