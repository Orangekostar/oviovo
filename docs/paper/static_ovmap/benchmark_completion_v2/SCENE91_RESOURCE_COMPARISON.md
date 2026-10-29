# 第91场景显存与作业耗时对比

本轮固定管线的整场进程显存采样峰值为 **27.332 GiB**（27988 MiB），来自CropFormer阶段。场景为ScanNet `scene0608_02`，物理GPU0/NVIDIA A40；CPU为Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz。这是共享建图、F及N/Q产出并生成四方法×两种子结果的真实场景作业成本。

## 本机实测

| 作业 | 墙钟 s/场景作业 | 进程显存采样峰值 GiB | 采样数 | 有GPU进程的采样数 |
| --- | ---: | ---: | ---: | ---: |
| cropformer | 290.88 | 27.332 | 288 | 282 |
| native_map | 1534.47 | 0.000 | 1523 | 0 |
| F_MV1000 | 155.94 | 2.377 | 153 | 144 |
| native_query | 416.32 | 3.016 | 410 | 396 |

墙钟包含模型加载、CPU处理与并发环境影响，分别对应完整作业；不将四项墙钟之和命名为端到端延迟。native_map显式禁用CUDA，该行0表示未观测到该作业的GPU进程占用，不表示CPU内存为0。F作业覆盖多视角特征，N/Q作业包含200次实际查询尝试及完整帧屏障回放，均不是原论文五个在线模块的一一对应计时。

整场共2742条NVML采样，查询错误0；目标间隔1秒，观测最大间隔1.360871秒。每条记录按同一场景worker及其子进程的PID/start与GPU UUID归属，求同时占用之和后取最大值。独立复算检查原始日志SHA/字节数、四作业真实退出0、进程归属子集、各阶段覆盖及观察器/监督器终态。历史进程父子边未由复算器重新构建。

该数值为进程显存的采样下界；它包含CUDA上下文等进程占用，不能与PyTorch allocated/reserved直接混用，也不宣称捕获了采样间隙中的瞬时最大值。独立复算证明`cb2dfe6e31b54558ab9d61795ec9cd780022530fcdc04fb30b4b93fa8c65251e`，真实退出0`6cf62cde1bb6c14fd962e36e51d004356348c5c8510b751386f986b5665429f0`。

## OVI-MAP表8参考

以下保留作者原表的Run-Time/Frame、Thread、Skipped Frames形式。原表在RTX3090+i7-12700K上测量；最后一列仅为按30FPS输入及原表n推导的目标处理频率。[原论文表8及Runtime节](https://arxiv.org/html/2603.26541v1#S8)。

| 作者模块 | ms/处理帧 | 线程编号 | Skipped Frames（原列值n） | 目标处理Hz @30FPS输入 |
| --- | ---: | ---: | ---: | ---: |
| RGB segmentation | 964.8 | 1 | 30 | 1.0 |
| Depth segmentation | 88.4 | 2 | 30 | 1.0 |
| 2D--3D association | 76.3 | 2 | 30 | 1.0 |
| View selection | 140.8 | 2 | 10 | 3.0 |
| Semantic extraction | 131.3 | 3 | 10 | 3.0 |

原文用每n帧处理一次解释其列值，因此这里不再额外加1。线程2由深度分割、2D–3D关联及视角选择共用，视角选择/特征提取的n=10；这些合并单元格已在缓存PDF第12页实际看图核对。本轮本机管线使用固定200帧采样及按场景的串行作业，未实测同样的在线线程调度/固定抽帧n，因此不填造本机的对应在线频率或跨硬件加速倍数。

| 系统/范围 | 显存GiB | 测量类型 |
| --- | ---: | --- |
| OVI-MAP原论文表7/8 | 未报告 | 参数量不能替代实测显存 |
| 本轮完整场景进程采样 | 27.332 | NVML约1秒采样下界，A40 |

原论文没有该显存数值，当前不计算显存优势倍数。本机数据为一场景实测，不证明完整benchmark效率优势或已达到在线30FPS。既有G1/G3等单工作者的PyTorch显存/模块测时保留在[RESOURCE_COMPARISON.md](../RESOURCE_COMPARISON.md)，其测量边界另列。

作者参考PDF SHA256 `abbb5e569218a0aecbe9155069f15a14e8044ef7d9474737b5a459dc7a8d3fc8`，已核对实际字节；对应原始对比JSON为`artifacts/static_ovmap/resource_comparison_20261009/comparison.json`。逐行作业identity、精确值、引用来源及本文SHA见`review/G4_scene91_resource_comparison.json`。本次整理新增模型推理和训练均为0；公开47场景索引及已审核82场景主表不因本表而更新。
