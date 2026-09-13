# 0913_1 静态 OVI-MAP 开发交接

状态：**可执行实验与 Replica8 全部评测/汇总已完成；仍有明确的局部 BLOCKED 项，不声明原提示所有要求全部满足。**

工作树 `/home/ww/crove/ovimap-static`，本地/远端分支
`research/ovimap-static-benchmark-v1`，远端 `Orangekostar/oviovo`。
基底 `d5c0688bc662f8e65455cb9c62909de87c941b43`；最终提交 SHA 在终端交接报告，
不为把自身 SHA 写入文件而制造循环提交。

## 实际完成范围

| 实验 | 状态及范围 | 结果入口 |
|---|---|---|
| B0、S1a/b/c、随机/质量/all-views 对照 | COMPLETE，历史 Room0 开发 | [S1_RESULTS.md](S1_RESULTS.md) |
| S2 fallback | COMPLETE，Room0；无新增 TP，关闭 | [R3_RESULTS.md](R3_RESULTS.md) |
| G1 merge、G2 merge+split | COMPLETE，Room0；负/混合收益，关闭 | [R4_RESULTS.md](R4_RESULTS.md) |
| D0 ROI、D1 GLA-CLIP、D2 AnyUp、D3 稀疏净化 | COMPLETE，Room0；均不及原生基线，关闭 | [R5_RESULTS.md](R5_RESULTS.md) |
| T0 SpaCeFormer、T1 融合 | COMPLETE，Room0，额外三维预训练独立轨 | [R6_RESULTS.md](R6_RESULTS.md) |
| B1 重建原生、冻结 C1、完整查询池消融 | COMPLETE，8 场景 × 12 条件逐场景评测；8/7场景汇总均完成 | `artifacts/static_ovmap/replica8_full_history` |
| Q0/Q1 难例复核 | NOT_RUN，原文可选，没有实现或新增查询 | 条件 registry 明确关闭 |
| ScanNet18 | BLOCKED，实际 RGB-D/pose/GT 输入与帧表未绑定 | `scannet18_asset_inventory.json` |
| 原包 reference/test_static_core.py | BLOCKED，输入包无该文件 | [REQUIREMENTS_AUDIT.md](REQUIREMENTS_AUDIT.md) |

失败的前缀历史回放、原评测加载错误、SpaCeFormer OOM 尝试及重复运行差异均保留。
失败尝试不记为成功，也不按 GT 最高分替换主运行。

## 结果的归因边界

- 历史 Room0：B0 mIoU 0.332857 / semantic AP 0.154316；S1a 0.362175 / 0.186723。
  只换固定 K=8 选择，面积融合、encoder、query 和 geometry 不变；不是 Replica8 指标。
- 初始重建 Room0：B1 mIoU 0.333373 / AP 0.153614；S1a 0.335586 / 0.155800。
  原生版本及查询/几何变化必须单列，不能把跨版本差值归于方法。
- C1 在扩展场景前已冻结为 retained-top10 池的 S1a；完整池另列 `_FULL`，
  不因扩展评测更换最终组合，不宣称多个模块增益可以相加。
- R5 改用独立 CLIP 空间，显式 ROI 编码对照；不是原 SigLIP 空间中的选择收益。
- T1 semantic AP 0.205160，但原始面积置信度下 mIoU 0.218774；
  类内面积归一化改变跨类别重叠分配后 mIoU 0.351151，AP 不变。
  两种口径均保留；额外训练和不稳定重复运行不属于无额外训练的主轨。
- 发布版 semantic AP、发布版类别无关 mP/mR、canonical 类别无关 AP 分别命名。
  论文类别无关 AP 来源未核实，该列 null，不能用 scene-macro canonical AP 顶替。

## 运行命令与大产物

实际逐场景 argv、cwd、阶段耗时与退出状态见
`artifacts/static_ovmap/replica8_full_history/<scene>/scene_pipeline.json`；
native mapper/front-end/geometry 命令见同目录 `native_mapping_manifest.json`。
各开发实验的完整命令及环境见 R3–R6 对应结果文档和收据。

本轮使用的文本缓存为
`/mnt/shared/ww/ovimap-static-20260913/text/replica51_rebuilt.npz`；
feature space `sha256:f26a6f0eafda897bcfb5ac2b2319f8747a02425c7e39017bec9454317d3f1713`。
历史与新构建的 source identity 不相同，不混合两者特征。
外部 OVI 源版本、dirty 移植源 hash、模型权重和 200 个输入帧的绑定见 native manifest；
GLA-CLIP/AnyUp/SpaCeFormer 权重和环境版本见 R5/R6 报告。

大产物根目录 `/mnt/shared/ww/ovimap-static-20260913/`：
`native_replica8`、`native_full_history`、`evaluated_full_replica8`。
历史开发大产物位于 `/home/ww/oviovo_baseline_runs/20260913_static_ovmap/`。
这些是本机资产路径，不是公开下载链接；复现需已有相应模型和数据访问条件。
仓库只保存小结果、命令、大小/校验和及来源凭据，不上传原始数据、模型或巨型 mesh。

## 验证、成本与未解决项

8 个场景逐一通过原始读出/实例 mask/GT 配对和独立严格 5 cm 投影核验，
再计算全部原生几何分区指标。相关单元与真实 smoke 的命令和结果见阶段报告。
新聚合器的 pooled confusion 单元测试及两场景真实发布版 AP 验证通过；
完整 8 场景聚合 CLI exit 0，全部6份汇总产物已归档。

前端、几何、mapping+VLM+export 按 native 阶段时间报告；三次 mapping 补跑复用
前端/几何，复用阶段增量成本为零，原时间戳仅作为来源记录。
未独立测量的 VLM、导出和原生 peak GPU 显存为 null，不补零或估算。
当前是共享机器场景并行运行，不作隔离同机端到端 FPS 或加速比结论。

主表、额外7场景诊断、消融、成本、逐场景与选择诊断见 [REPLICA8_RESULTS.md](REPLICA8_RESULTS.md)。
最终远端 SHA 由终端 `git ls-remote` 与本地 HEAD 比对后报告。
局部限制：ScanNet 输入与原包参考测试缺失；历史丢弃查询未恢复；论文类别无关 AP
来源、历史二进制身份和 SpaCeFormer 预训练排除/重复性未完全核实。
没有证据支持“所有基准稳定超越 OVI-MAP”。
