# 显存与可比耗时

同机、同场景、同计时边界的 G1/G3：各 8 场景 × 2 次正式冷调用，包含必要文件输出。

| Method | Calls | Seconds_scene | Allocated_GiB | Reserved_GiB | FC_images_scene |
| --- | --- | --- | --- | --- | --- |
| G1_SELECTED | 16 | 19.89 | 1.89 | 2.46 | 4.50 |
| G3_SELECTED | 16 | 20.94 | 1.89 | 2.21 | 11.50 |

G3 比 G1 慢 5.31%；峰值 allocated 显存相同。reserved 是缓存分配器预留峰值，受分配历史影响，不能将该差值解释为模型节省显存。两列只覆盖 FC 恢复进程，未包含公共 Native/SigLIP/CropFormer 系统显存。

此前独立配对优化实验：G1 44.448 → 18.853 秒/场景，加速 2.36 倍，耗时减少 57.58%；峰值 allocated 均为 1.89 GiB。该实验与上表的插桩实验分开，不能混用其分母。

OVI-MAP 原表与我们的模块对照，单位 ms/处理帧；n 为每 n 个物理输入帧处理一次：

| Module | OVIMAP_ms | G1_ms | G3_ms | n |
| --- | --- | --- | --- | --- |
| RGB segmentation | 964.80 | 1474.20 | 1474.20 | 30 |
| Depth segmentation | 88.40 | 1488.41 | 1488.41 | 30 |
| 2D--3D association | 76.30 | 3676.87 | 3676.87 | 30 |
| View selection | 140.80 | 52.22 | 52.03 | 10 |
| Semantic extraction | 131.30 | 219.72 | 194.48 | 10 |

OVI-MAP：RTX3090 + i7-12700K；我们：A40 + Xeon Silver 4314。[原论文表 8](https://arxiv.org/html/2603.26541v1)为作者报告；其表 7/8 未报告峰值显存，参数量不能代替峰值显存。跨硬件且关联包装的计时范围不同，上表只用于定位差距，不形成公平的端到端加速比。视角与语义行是离线摊销成本，语义更新 n=10 仍是预算方案；RGB/深度/关联是实际 n=30 离线序列计时。

当前主要差距在深度分割与 2D–3D 关联。深度+关联同一执行器每个处理帧合计 5165.28 ms，在 n=30、30 FPS 输入下只有 1000 ms 预算；完整系统不满足该均值预算。没有实测连通的在线流水线。

新证据探索的同边界四方法冷计时及显存：

| Method | Calls | Seconds_scene | Allocated_GiB | Reserved_GiB | FC_inputs_total | AnyUp_QK_total | Added_projection_total |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EV01_G1_V2 | 16 | 22.71 | 1.90 | 2.21 | 72 | 0 | 0 |
| EV07_COMBINATION | 16 | 24.20 | 4.35 | 6.65 | 72 | 70 | 72 |
| EV03_CONTRAST | 16 | 22.69 | 1.90 | 2.21 | 72 | 0 | 72 |
| EV06_OWNER_ANYUP | 16 | 25.49 | 4.33 | 6.65 | 72 | 70 | 72 |

该四方法表包含驻留 FC+AnyUp、重算 G1/额外几何/FC/AnyUp/预测 payload；收据和预测一致性校验在计时外。与前面的包含文件导出旧 G1/G3 表保持独立。
