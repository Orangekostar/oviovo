# 外部基线调度与论文来源复核

2026-10-24：主代理复核8份冻结生产源码、3份原ZIP合同、4个固定作者提交及23份入口文件；重核既有端口证据中的133份作者来源描述。生产源码、原测试及本次2280份基线中的既有科学文件保持不变。本轮只检查源码与已有结果，不运行外部模型、AP评分或训练。

主审：`review/G4_external_dispatch_source_contract_primary_review.json`（`d1d0f7d4ab9926e32b9875c8e9af18b214249ff461240cc608ffd3545459c1db`）。实际子进程退出0：`bootstrap/external_dispatch_source_primary_actual_exit.json`（`3dcea715aa675a40c0a71bb06aab59ddaf319b40a3995092c6f291ba0262e4f9`）；日志SHA：`d2f330beebeaa6ab303c21670955422f3c162f6c11f57527d8a5d4acb9640285`。2280份稳定文件和93份冻结文件的SHA/mtime前后均一致。

## 论文数值与变体

三篇已下载原论文的PDF SHA与下载收据一致，三次文本提取实际退出0。下列5行共15项AP逐项核对；百分数仅用于显示，结果记录使用0–1分数。

| 方法 | 基准 | APall | AP50 | AP25 | 当前完整变体可用作者参考 |
|---|---|---:|---:|---:|---|
| OpenMask3D | ScanNet200 | 15.4 | 19.9 | 23.1 | 是 |
| OpenMask3D | Replica48 | 13.1 | 18.4 | 24.2 | 否，checkpoint变体不兼容 |
| Open3DIS | ScanNet200 | 23.7 | 29.4 | 32.8 | 是，2D+3D |
| Open3DIS | Replica48 | 18.5 | 24.5 | 28.2 | 是，2D+3D |
| MaskClustering | ScanNet200 | 12.0 | 23.3 | 30.1 | 是，semantic列 |

来源为[OpenMask3D](https://arxiv.org/pdf/2306.13631)表1/2（PDF第7/8页）、[Open3DIS](https://openaccess.thecvf.com/content/CVPR2024/papers/Nguyen_Open3DIS_Open-Vocabulary_3D_Instance_Segmentation_with_2D_Mask_Guidance_CVPR_2024_paper.pdf)表2/4（第7页）、[MaskClustering](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_MaskClustering_View_Consensus_based_Mask_Graph_Clustering_for_Open-Vocabulary_3D_CVPR_2024_paper.pdf)表2的semantic列（第6页）。主审记录各PDF SHA、具体页码、数值及匹配行摘要，不导出原论文全文。

所有作者记录均明确`locally_executed=false`、`re_scored_on_host=false`。OpenMask3D原Replica表使用ScanNet200训练的Mask3D；任务计划的任意场景checkpoint端口不能继承该值。该历史数值保留在参考记录，主表对应格维持缺失。

## 调度与主表入口

实际CLI的baselines阶段调用`baselines.run`；`external.py`中的早期`recipes`函数不在该调度路径中。后者仍提供固定提交及格式转换函数，不能据早期通用recipe推断当前资产已齐备。

四个仓库HEAD均等于指定pin，跟踪文件无相对HEAD改动；23份实际入口还直接与对应Git blob比较原字节。MaskClustering保留完整CropFormer/图聚类/CLIP-H14，OpenMask3D保留Mask3D/SAM-ViT-H/CLIP-L14@336，Open3DIS固定ISBNet/Grounded-SAM/细化后第二轮3D输出，OVI明确标注共同评测协议port。

重放原`external_cell`及其依赖函数的AST，使用已审核82场景归档中的真实result_store，24个外部T1格与归档表完全一致。完整本地pool优先；兼容作者参考带“†”；局部本地pool不能替代完整主表，变体不兼容的参考保持缺格。此检查不重算任何AP，也不审核当前可变表格或扩大82场景报告范围。

## 必须保留的执行边界

Open3DIS固定提交的`configs/scannet200.yaml`为p2d/p3d/refined全True；原`configs/replica.yaml`为True/False/True。因此原Replica默认配置属于2D分支，不能直接作为所选2D+3D完整变体的本地命令。若未来进行本地执行，需显式启用3D分支并提供真实ISBNet proposals、deep features和superpoints，按作者顺序完成2D、3D、细化、第二次3D。当前仅使用兼容作者报告值，没有把该默认配置计为本地全流程成功。

源码检查确认独立mask/重叠与原置信度的格式转换、官方match充分统计池化、缺依赖与运行错误区分，以及Replica端口首个真实G3通过后才继续后续GPU场景的入口门。既有GPU算子和合成格式检查只保留各自范围，不能替代正式外部Replica G3；此处未重新运行这些检查。

当前recipes仍是82场景报告快照，逐场景核验范围由独立主审决定。完整ScanNet/Replica覆盖、正式统计、最终17章审核、C1/C2及实际终态恢复仍待完成。

最终有限来源清单须显式加入这次的review脚本、execute脚本、实际退出JSON、实际日志和主审JSON；既有有限来源清单及历史候选不改写。
