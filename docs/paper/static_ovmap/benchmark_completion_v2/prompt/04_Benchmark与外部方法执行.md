# 标准benchmark、完整baseline与实际资源执行

本文件给出有限而确定的执行路线。工作目标是补真实结果，不是仅写一个feasibility表。运行许可/官方数据和模型确实缺失时，保存可运行adapter及具体缺项；不把依赖缺失包装成实验已完成。

## 1. 正式范围与历史范围分开

主目标：ScanNet200官方validation（预期312scene，实例合法类别取官方列表）与Open3DIS/OpenMask3D使用的Replica48类8scene。只读这些作者/官方release的split与映射，不自己抽12scene替代。

锁定完整scene list、各方法所需输入、任务标签空间、GT顶点序、ignore/stuff/min-region/IoU/输出confidence。论文中AP称谓与原评价一致：AP一般IoU .50:.95、AP50、AP25；精确运行列表为准。CF18和旧Replica51仍作为旧范围，绝不是第三个标准benchmark。

ScanNet类200/实例198、Replica48/旧51的差别需要在protocol report中写出实际列表交集/差集，不以常识假定。主文可不写工程细节，但不能在不同类数的分数之间直接宣布超越。

## 2. 方法固定与官方入口

以下仓库/文件在本次已读取；执行前先固定确切commit和权重。已有本地同身份源码优先；外部README若更新只更改安装兼容/路径，不无声修改方法。

### 2.1 OVI-MAP（直接近邻）

官方库：`OVI-MAP/OVI-MAP`；原版位于 `origin_impl`，main为后续重构。优先绑定父OVMap工件实际original implementation与官方origin_impl；若二者不同，则外部baseline在独立checkout跑，不能把本地A0自动叫official full method。

必需资产：CropFormer_hornet_3x_03823a预测、对应RGB-D/pose、原native SigLIP模型及文本、ROS/voxblox reconstruction运行环境。官方README入口：`scripts/panoptic_mapping_.py`（main重构示例还有`run_async.sh`）、`scripts/utils/mesh_postprocess_utils.py`/module、`scripts/eval_inst_seg.py`。origin_impl具体参数由其CLI读取，禁止把main示例的`Nyu40`当Replica48配置照抄。

开发adapter工作：固定原论文观测计划与原方法超参；读取其真实输出mask/语义；将预测映射到共同benchmark坐标/类别，然后调用共同evaluator。作者原51类只能提供另列paper result；Replica48需按固定文本与映射重跑语义/评分。几何不因类别变化重建，确实缺native mapping输入时只新建该scene一次。

**已核实勘误：作者README说明所有实验voxel是0.01m，论文0.1是笔误。**移除旧材料以该笔误声称不公平的理由，保留真正的数据/来源/监督/类别差异。

### 2.2 OpenMask3D

官方：`OpenMask3D/openmask3d`（NeurIPS2023，不能写CVPR2023）。

先看`README.md`和`run_openmask3d_scannet200_eval.sh`。ScanNet完整评测使用官方ScanNet200训练的class-agnostic Mask3D checkpoint及segments；`run_openmask3d_single_scene.sh`对单scene直接点云方式有不同mask结果，不能用demo输出声称复现full-val。

依赖：官方Mask3D checkpoint、SAM ViT-H、CLIP、MinkowskiEngine等；不改已有用户环境。已有完整masks与features最优先复用，输出分别为 `(num_points,num_masks)`和`(num_masks,feature_dim)`。

ScanNet正确脚本运行完整class-agnostic提案、SAM辅助裁剪、CLIP多视图特征与eval。Replica数据适配使用作者任意scene checkpoint及其48类协议。输入intrinsics/pose、point z-up右手系、depth_scale按实际数据元信息填写；README的1000/6553.5只对应特定编码，不能覆盖已经是米的深度。

### 2.3 Open3DIS

官方：`VinAIResearch/Open3DIS`。本轮主变体固定 **ISBNet＋Grounded-SAM＋refinement**，不因3D依赖不方便退化成2D-only却仍称同一行。

优先阅读`docs/DATA.md`作者的 `Scannet200_OpenVocab_ISBNet-GSAM`完整结果下载项，若获得相同point/ID/version的全val预测，直接共同evaluator评分。一个示例processed scene不等于312个scene。只下载有权限资源，不签协议或冒用凭证。

若需完整推理，`docs/RUN.md`要求的顺序必须保留：
```
sh scripts/grounding_2d.sh <locked-config>
sh scripts/generate_3d_inst.sh <locked-config>
sh scripts/refine_grounding_feat.sh <locked-config>
sh scripts/generate_3d_inst.sh <locked-config>   # refine后再次完成3D输出
```
`sh scripts/eval.sh`实际进入`open3dis/evaluation/eval.py`，检查硬编码数据路径及CUDA=0，改为独立config/继承设备，不改数值处理。ISBNet的proposal与深层特征必须真实存在。docs中默认stride5与论文stride设置可能不同，以选定作者paper配置/预计算预测身份为准，完整记录，不混用配置拼数值。

Replica使用DATA.md作者提供的8scene预处理数据、GT、提案及特征，固定48类标签映射。作者结果链接不可访问时保存具体URL和错误，继续可运行同类任务；不得以另一个模型占位。

### 2.4 MaskClustering

官方：`PKU-EPIC/MaskClustering`。完整流程依赖CropFormer及OpenCLIP ViT-H/14 LAION2B。读取`preprocess/scannet/process_val.py`、`prepare_gt.py`并通过单独配置提供真实路径；`python run.py --config scannet`完成2D masks、聚类、CLIP特征与评价。

官方README列ScanNet/ScanNet++/Matterport，未给Replica正式入口。ScanNet照原完整配置；Replica如需主表该格，开发dataset reader以目标RGBD/pose/PLY和48类名字输入，聚类/特征/超参不变，命名为 **MaskClustering (Replica dataset port)**，不能说作者本来发布了该结果。

官方3090耗时表给ScanNet总13.5GPUh；这只用于识别旧“总12h”限制不合理，不是新host耗时保证。现有缓存相同CropFormer预测可以共用；不同模型checkpoint、帧计划或mask设置则不能共享成同一提案。

## 3. 完整实现与有限port的区别

只改变绝对路径、设备、I/O封装、数据坐标/语义ID的合法映射，是已记录的部署或dataset port。删除模型层、少跑refinement、改分辨率、换SAM/CropFormer、替换提案、缩小场景而挂原名，是改变方法，必须另命名或明确未复现。

官方已测数字可引用：满足相同split、合法类别、评价域/阈值和完整方法配置时标author-reported并给出处。不能把OVI51类8.50填到Replica48，也不能复制Open3DIS全val结果当CF18本地测量。无需为了“有日志”而重算所有已兼容官方预测。

## 4. 标准Ours适配必须做什么

新namespace实现 `prepare_native_scene`：根据原OVI入口生成label-free instance partition及真实capture/deferredmetadata，提供03 SceneBundle；调用E20/E21同一native/Q_GAIN来源数值，F采用已锁定的FC区域向量＋MV1000。

这不是原G1恢复完整重建，也不是Open3DIS提案上拼的另一种方法。若选择接入其他proposal会改变方法身份，本轮不默认开启。T1/T2全局四格只共享上述native预测支持。

投影到共同GT坐标只用几何，保留真实缺失区域。所有U四格用相同top1/constant1导出；baseline保留原score。全方法比较允许原骨干/提案不同，S1清楚列信息与训练来源；不要声称各方法算力或观测完全一样。

反向声明限制：即使标准evaluator运行了全val，曾用该val标签梯度训练的MV/Q仍不能列成干净benchmark主结论。protocol审查一旦发现此情况，输出该真实值到exposed面板、缺少清洁权重明确登记；本轮不自动启动再训练。

## 5. 任务调度、数据与预算

发布数据和模型下载是科学依赖，不是安全测试。优先复用相同身份的本地数据或作者预测；无许可/登录权限时不能绕过。下载大文件先读取清单与磁盘容量，流式分scene，保留至少20GiB及下一scene安全空间。

既有GPU单worker；用户实际配额优先。没有用户配额时执行冻结有限scope，不虚设一个12h上限然后声称任务可全做完。单pipeline首个正式scene测时后记录预测总消耗范围；长任务持续保存per-scene进度，不重复完成job。

baseline与Ours的特征不兼容时允许各自必要编码；**这里的补实验任务不同于旧零编码归因任务**。新网络训练仍为0。只为构建baseline原方法所需的官方预训练推理安装独立环境，不升级用户现有环境。

## 6. 验证与发布结果等级

每个benchmark×method记录：full_scene_coverage、labels/point_order/evaluator兼容、所有learned组件训练角色、author-prediction/source-code身份。任何一个必需条件未解决，该标准格不能有“同协议完成”标记。

主表对每个方法不强求两个benchmark格全有，但Ours未完成对应benchmark时不得写在该benchmark优于其他法。未知/有缺口与数值负结果分开；最终摘要的实证范围只能由这些状态决定。
