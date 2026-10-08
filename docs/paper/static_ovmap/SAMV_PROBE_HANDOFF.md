# 交接

完整代码/资源配置以外部 freeze.json 为准，基座 a95c24d990cea57b14bb537e9acca95b95659bda，研究分支 research/ovimap-samv-local-probe-v1。全局窗口 6，SAM-V/SAM2独立环境；原始FC保持FP32。

```bash
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_ovimap_samv_local_probe.py --spec configs/static_ovmap/samv_local_probe_v1.json --parent-root /mnt/shared/ww/ovimap-source-preserving-update-v1/attempt_001 --output-root /mnt/shared/ww/ovimap-samv-local-probe-v1/attempt_001 --phase all --resume
```

持久存储：/mnt/shared/capacity/node101/ww/ovimap-samv-local-probe-v1/attempt_001。输入、稠密地图、RGB/depth与模型权重留在共享盘，只发布实际紧凑 mask、改动行、类决策、指标与 trace。公开引用见 artifacts/static_ovmap/samv_local_probe_v1/source_references.json。

指定完整 CLI 已实际运行两次并以0退出，最后终态为2026-10-08T13:46:19.824017+00:00；完整resume复用成功叶子，分割进程调用总数仍为9。终态、日志和baseline parity实际文件已收录在 [紧凑证据](../../../artifacts/static_ovmap/samv_local_probe_v1/receipts)。普通push之后的本地/远程完整SHA只在外部 `publication/final.json` 记录，不在提交中自引用未来SHA。

模型失败调用数：2，详细真实日志/已知耗时/未知null见 costs.json。两次导入问题已纠正，成功的同身份分割叶子在resume时复用；新增前向不被计作缓存命中。部署仍为N0。

原始科学冻结 `1d9eca0a5ac600270bdfb812dd1c15aceb0eba95` 在新AP与剩余科学推理之前；后续 `6d3280531150afc0ce0b2c9d87dcb2f55abc6b82` 只补报告显示/数字溯源、CPU与head计数、CLI真实无证据构造校验。模型、规划、lifting/readout/输出/scorer、配置、资产和窗口均未改；对应新模型调用与AP调用为0。freeze_initial和明确amendment均保留。

来源到实际函数的映射，以及原包指令0–11、契约A–I、评测1–6的主代理逐项核对见 [最终核对](../../../artifacts/static_ovmap/samv_local_probe_v1/FINAL_REQUIREMENT_REVIEW.md) 和对应JSON。表格PDF与四场景真实mask渲染均已人工查看；该证据完成了报告生成时暂记为pending的显示检查。

全部64个模型查询、28条完整地图评测、14个固定池和8个额外计时已完成；最终研究选模G1，状态 COMPLETE_NO_PILOT_GAIN，没有自动扩展到Replica8/CF18。当前无未解决的模型或评分执行依赖；独立完整地图延迟与在线FPS不是本探针测量项。正式窗口设备为NVIDIA A40；SAM-V用torch2.3.1+cu121，SAM2用torch2.5.1+cu124，遵守各自作者环境要求。
