# Replica8 + ScanNet15 受控比较

主表见 [COMPARISON.md](COMPARISON.md)，可读数值见 [results.json](results.json) 和 [main.csv](main.csv)；消融见 [ablation.csv](ablation.csv)。数值均来自实际预测，表格显示百分数。

范围参考 [OVI-MAP 论文的实验设置与表格](https://arxiv.org/html/2603.26541v1#S11)。Replica 使用 office0–4、room0–2；ScanNet 使用论文 ConceptFusion18 与既有验证集的交集，共15场景。排除 scene0168_00/01/02，因为它们属于 [ScanNet 官方训练集](https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_train.txt)。场景名单在新增预测评分前固定，未按结果筛选。

| 数据集 | 固定场景 |
|---|---|
| Replica8 | office0, office1, office2, office3, office4, room0, room1, room2 |
| ScanNet15 | scene0011_00, scene0011_01, scene0050_00, scene0050_01, scene0050_02, scene0084_00, scene0084_01, scene0084_02, scene0231_00, scene0231_01, scene0231_02, scene0378_00, scene0378_01, scene0378_02, scene0518_00 |

IoU/ACC 使用各方法同一批 AP 实例预测，投影到完整原始顶点序。重叠按原 AP 置信度排序，相同置信度取原始较早列；不使用 GT 决定归属。未覆盖的有效 GT 点仍计错。先累加场景混淆矩阵，再在有 GT 的类别上计算 mIoU/mAcc；oAcc 为总体点准确率。Ours 的 seed17/29 分别整体评分后取均值。

这是共同 AP 对象类别口径（Replica48 / ScanNet198），与论文 Replica51 / ScanNet200 完整语义口径不同。AP25、AP50、APall 使用原官方实例评测逻辑，ScanNet15 从相同15场景的实例匹配重新整体池化；未沿用312场景的 AP，也未对场景 AP 求平均。Replica8 的12个方法/种子 AP 池与既有冻结结果逐项完全一致。

复用既有 RGB-D、模型和已完成预测；本轮只补 OpenMask3D、MaskClustering 的15个 ScanNet 场景。新增推理使用原作者代码及参数、原有200帧均匀采样输入，保留原 RGB/深度、相机和点序；为 OpenMask3D 补取唯一缺失的 ScanNet200 验证权重，SHA256 为 `6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534`。不进行训练。

Open3DIS 的 ScanNet 预测来自作者归档，其输入是否同为200帧尚未证明，因此本表不能声称所有方法输入视图完全匹配。既有 GT 接触与 Replica 训练角色限制继续适用，见 [角色说明](../benchmark_completion_v2/replica_role_extension_v1.md)；本轮场景过滤不消除这些历史限制。推理硬件与其他进程共享，本轮不据此比较端到端运行速度。

完整执行证据、原始点混淆矩阵和实例匹配保存在共享目录 `attempt_001/semantic_metrics_v1/controlled`，其哈希与实际退出记录汇总见 [EVIDENCE.json](EVIDENCE.json)。中断、失败和未执行的排程不计入结果；只接受实际退出为0、输出文件哈希通过校验的逻辑阶段。

仅重新汇总已冻结结果：

```bash
cd /home/ww/crove/ovmap-benchmark-completion-v2/worktree
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/compute_controlled_comparison.py \
  --root /mnt/shared/capacity/node101/ww/ovmap-benchmark-completion-v2/attempt_001 \
  --out /mnt/shared/capacity/node101/ww/ovmap-benchmark-completion-v2/attempt_001/semantic_metrics_v1/controlled \
  --phase pool
```
