# 实际源码阅读与适配边界

已在基线 `30a6e07eb2b2fe6f69dc9bfa8405df171ff17f9f` 本机阅读下列生产代码，而非只读执行包。新工作区 `/mnt/shared/ww/ovimap-evidence-exploration-v1/worktree`；父实验只读。

| 源码/符号 | 输入 → 输出 | 本轮用法/变化 |
|---|---|---|
| runtime_parity.binding import_metrics/bind_reference | v2收据/历史源 → actual reference/172行14池 | 消费实际 reference/metric store；新建窄binder，不重跑历史冻结 |
| runtime_parity.runner execution_config/load_common/CommonInputs | 原context/capture/NQF → 几何、Native support、完整类别、导出projection | 复用只读loader；预测器只传RGB/depth/geometry/text子集，评测projection不用于触发/选择 |
| runtime_parity.views build_views/scientific_key/CallLoader | 固定capture/raw/candidates → 几何排序G1/G3请求、mask、RGB | 科学采集复用G1；冷测重新R2，scientific_key只核对B1，不冒充AnyUp键 |
| runtime_parity.kernels ExactProjector | 全mesh+camera+depth → 保守ROI射线与原mask | 保持R2与全场景遮挡；额外几何不把ROI当完整owner图 |
| cvpr_compact.projected_views FullSceneProjector.project_frame | 预测mesh/camera/FP32 depth → 全owner raster、可见mask | 仅已选G1帧补充其他owner支持；原mask必须一致 |
| cvpr_compact.area_fallback pool_region/region_vector | raw dense+signed mask → 原pool或empty area fallback、原head向量 | B1及coarse控制保持原路径，不全局改area pooling |
| runtime_parity.session RuntimeSession.encode_exact | 请求/RGB/mask+resident FC → FP32向量与计数 | 作为精度与cold参考；新增采集worker允许有身份的dense cache |
| a7_evidence_upgrade.region_adapter image_tensor/signed_mask/operator_nodes | uint8 RGB/bool mask → 800/1333/pad32、原FC算子 | 像素几何和冻结head不变；AnyUp另做ImageNet guidance |
| cvpr_compact.region_worker classify_regions/FCSession | unit视觉vector/text → 无条件argmax与可用性 | B1行为参考；新decision独立记录accepted，禁止借available隐藏defer |
| cvpr_compact.outputs construct_output/build_method_outputs | 固定Native与D2、recovered_labels → support稳定、重算rank的locked payload | 用construct_output低层接口，不伪造旧cosine-argmax contract |
| cvpr_compact.evaluation/recovery_wave2.evaluation | locked payload+expanded registry → released场景/全池化/类指标 | 新protocol adapter，保持数学阈值、忽略、ties、全部GT |
| released_trace.trace_released_matches | 原released匹配器+matches → 实际TP/FP/ignore事件及PR/FN parity | 仅锁定预测后诊断，不作预测输入 |

父报告 RESULTS/SELECTION/CLAIMS、reference_binding、imported_metrics 已阅读；actual v2 region/prediction/official evaluator receipts 已检查结构。父 FC interpreter 实为 `oviovo-radseg`（不是控制器环境），GPU UUID 为 `GPU-41ba4cb6-edfe-ed30-c047-a19d9f23d0a8`。

AnyUp 已读取固定提交的 hubconf/model/chunked_attention/attention_masking：原值V不投影，Q/K经过RMSNorm，平均头 attention 带原window mask。checkpoint的3540612字节与固定SHA256已核验；现有torch2.4环境可import，无需升级环境。上游保持未修改，重采样适配器另写。
