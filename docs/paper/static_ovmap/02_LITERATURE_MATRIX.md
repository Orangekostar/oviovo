# 2026 论文—代码—OVI-MAP迁移矩阵
核验日期：2026-09-13。这里按正式会议年份计数，部分预印本首次出现于2024/2025年。清单共21项，另将OVI-MAP作为目标基线。
“代码公开”只表示存在可检查的实现，不等于所有权重、数据、完整复现步骤都齐备，也不表示已在本环境复现。没有把仅有占位README的MV3DIS、OpenVoxel、LightSplat算入此清单。许可证各异，“源码公开”不等同全部采用OSI宽松许可证。
阅读深度：覆盖论文方法及相关实验/限制、官方仓库结构和复现入口；对直接决定实施的OVI-MAP导出/VLM/评测、用户仓库协议/静态锚点、GLA-CLIP前向、AnyUp加载接口和SpaCeFormer接口进行了源代码级检查。没有逐行审计全部21个仓库。
## P01 · GeoPurify — ICLR 2026
论文：https://arxiv.org/html/2510.02186v1

仓库：https://github.com/tj12323/GeoPurify

**方法要点：** 几何教师净化跨视角视觉语言特征；原方法包含稀疏三维学生训练。

**已核验实现入口：** `run/train.sh; run/val.sh; config/geopurify_scannet.yaml`

**公开状态：** 代码、配置和模型下载入口公开；未在本环境加载权重。

**迁移判断：** 借鉴几何约束的残差净化，不先重训整个学生。

**不能越过的比较边界：** 原文的训练场景挑选利用语义标注统计；不能称整条流程完全不接触三维标注。ScanNet20结果不能直接对比ScanNet200。

**优先级：** 中：M3

## P02 · EmbodiedSplat — CVPR 2026
论文：https://arxiv.org/html/2603.04254v1

仓库：https://github.com/0nandon/EmbodiedSplat

**方法要点：** 在线稀疏系数场、全局CLIP码本，以及几何感知语义分支。

**已核验实现入口：** `src/; config/experiment/; modules/`

**公开状态：** 代码和权重入口公开；官方区分incremental和online配置。

**迁移判断：** 在特征存储成为瓶颈时，借用码本/稀疏系数，不替换已知RGB-D几何。

**不能越过的比较边界：** 论文几何为3DGS；默认配置的参考帧选择与严格在线设置要区分。其FPS不是本系统端到端FPS。

**优先级：** 中：M4

## P03 · SpaCeFormer — ICML 2026
论文：https://arxiv.org/html/2604.20395v1

仓库：https://github.com/NVlabs/WarpConvNet/tree/main/warpconvnet/models/spaceformer

**方法要点：** 无外部proposal的三维实例查询解码，以及大规模多视角伪标注训练。

**已核验实现入口：** `space_former_seg.py; build_spaceformer; load_spaceformer_checkpoint`

**公开状态：** 三维网络代码和checkpoint入口已公开；标签匹配/NMS是下游步骤，不在该目录内完整提供。

**迁移判断：** 作为额外三维训练轨道的独立强基线和补充proposal来源。

**不能越过的比较边界：** 输出clip_feat维度1152、语义空间为SigLIP2；不能和OVI的1024维SigLIP向量直接平均。输入必须是原生重建点，不是GT mesh。

**优先级：** 高：T3D

## P04 · Ov3R — CVPR 2026 Highlight
论文：https://arxiv.org/html/2507.22052v1

仓库：https://github.com/ZoranGong/Ov3R

**方法要点：** 联合几何与开放词汇表示的前馈三维重建。

**已核验实现入口：** `clip3r/; recon.py; train.py; evaluation/`

**公开状态：** 重建/训练代码公开；README的部分2D/3D开放词汇推理发布项仍需逐项核实。

**迁移判断：** 用于比较联合特征学习思路；当前不把它列为现成语义导出器。

**不能越过的比较边界：** 本任务已有RGB-D与位姿，重学几何未必划算；不能把部分代码公开写成整条任务可直接复现。

**优先级：** 低：对照

## P05 · LegoOcc — CVPR 2026 Oral
论文：https://arxiv.org/html/2602.22667v2

仓库：https://github.com/JuIvyy/LegoOcc

**方法要点：** 语言高斯占据表示、占据聚合与渐进温度调节。

**已核验实现入口：** `src/legoocc/; docs/train_eval.md; config/`

**公开状态：** 源码与训练/评测说明公开。

**迁移判断：** 参考不确定性加权思想；不把预测占据当观测表面。

**不能越过的比较边界：** 单目占据与RGB-D表面实例不是同一输出，几何监督和不可见空间预测均须披露。

**优先级：** 低：不主集成

## P06 · EvObj — CVPR 2026
论文：https://arxiv.org/html/2605.13152v1

仓库：https://github.com/vLAR-group/EvObj

**方法要点：** 从合成对象先验学习判别和补全的迭代对象中心表示。

**已核验实现入口：** `discerning_module/; completion_module/; scannet/train_scannet_vae.py`

**公开状态：** 判别、补全及部分数据集训练代码公开。

**迁移判断：** 用于解释whole-object先验的价值；不直接补出未观测几何。

**不能越过的比较边界：** 并非无需训练；公开的真实ScanNet流程有类别适用限制，不能当51/200类通用即插即用模型。

**优先级：** 低：不主集成

## P07 · CompetitorFormer — CVPR 2026
论文：https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.html

仓库：https://github.com/DuanchuWang/CompetitorFormer

**方法要点：** 显式查询竞争，缓解重复查询和实例碎片。

**已核验实现入口：** `competitorformer/; configs/; tools/`

**公开状态：** 模型源码目录公开；README训练/评估命令和权重表仍有TODO。

**迁移判断：** 借用竞争/互斥思想设计三维候选冲突约束；不直接启动全网重训。

**不能越过的比较边界：** CVF收录及摘要已核验；终稿全文下载未完成，不用未核实性能数字作为决策依据。源码公开不等于权重/完整复现流程齐备。

**优先级：** 中：M2概念

## P08 · LEGO — ECCV 2026
论文：https://arxiv.org/html/2608.10057v1

仓库：https://github.com/WHU-USI3DV/LEGO

**方法要点：** 结合三维尺度、共视和分层聚类组织多粒度区域。

**已核验实现入口：** `lego run; clustering/label_matrix.npz; cluster_tree.json`

**公开状态：** 源码、命令行流水线和输出结构公开。

**迁移判断：** 借用part/object层级与共视关系，约束最终导出whole-object粒度。

**不能越过的比较边界：** 3DGS/离线优化与原生TSDF不同；CC BY-NC-SA许可需遵守。不能用GT为每个实例挑最有利层级。

**优先级：** 高：M2

## P09 · SAM 3 — ICLR 2026
论文：https://arxiv.org/html/2511.16719v1

仓库：https://github.com/facebookresearch/sam3

**方法要点：** 文本或视觉概念提示的分割与跟踪。

**已核验实现入口：** `sam3/model_builder.py; sam3/model/sam3_image_processor.py`

**公开状态：** 推理、训练和模型访问入口公开；checkpoint需申请授权。

**迁移判断：** 仅在高不确定对象或视觉提示分割中做受预算约束的候选补充。

**不能越过的比较边界：** 建图时输入Replica-51/ScanNet200类别列表是query-conditioned设置，须单独报告。SAM3.1是仓库后续更新，不能与原论文模型混称。

**优先级：** 中：M5可选

## P10 · INSID3 — CVPR 2026 Oral
论文：https://arxiv.org/html/2603.28480v1

仓库：https://github.com/visinf/INSID3

**方法要点：** 利用冻结DINOv3特征进行参考实例分割与位置偏差修正。

**已核验实现入口：** `官方仓库模型/推理入口；接入前锁定具体commit与调用接口`

**公开状态：** 源码及使用入口公开；没有在本环境执行模型。

**迁移判断：** 用高可信历史掩码作为自生成参考，检验跨视角传播能否补充遗漏。

**不能越过的比较边界：** 原任务使用参考标注；把自生成mask作为参考是本项目的新适配，不是原论文已证实结论。DINO特征不等于文本对齐特征。

**优先级：** 中：M5可选

## P11 · GLA-CLIP — CVPR 2026
论文：https://arxiv.org/html/2603.23030v1

仓库：https://github.com/2btlFe/GLA-CLIP

**方法要点：** 跨窗口语义一致性和局部定位改进。

**已核验实现入口：** `gla_clip_segmentor.py: GLA_CLIPSegmentation.forward_feature; open_clip/`

**公开状态：** 真实segmentor与修改后的open_clip源码公开；关键前向接口已读。

**迁移判断：** 作为稠密语言特征分支的优先候选之一；抽取分类前特征，解耦查询。

**不能越过的比较边界：** 完整segmentor初始化读取类别词表，需拆开feature extraction与text scoring；不能直接将它的预测掩码当query-independent几何。

**优先级：** 高：M3

## P12 · SynCLIP — CVPR 2026
论文：https://arxiv.org/html/2607.11008v1

仓库：https://github.com/Justlovesmile/SynCLIP

**方法要点：** 通过同义词空间一致性和细化增强区域语义表示。

**已核验实现入口：** `src/; scripts/dist_SynCLIP_eva_vitb16_coco.sh; F-ViT/`

**公开状态：** 训练代码公开；本次核验到的README未提供明确最终SynCLIP权重直链。

**迁移判断：** 首先只做冻结的同义词文本集成对照；原方法训练作为后续备选。

**不能越过的比较边界：** 廉价的prompt ensemble并不等同复现SynCLIP；缺最终权重时不得使用基座权重冒充。

**优先级：** 中：M1对照

## P13 · SPAR — CVPR 2026
论文：https://arxiv.org/html/2604.02252v1

仓库：https://github.com/naomikombol/SPAR

**方法要点：** 将多裁剪稠密特征能力蒸馏为单次前向表示。

**已核验实现入口：** `generate_embeddings.py; inference_demo.py; lightning_segmentor.py; train.py`

**公开状态：** 源码、嵌入生成与推理入口以及预训练权重表公开；MaskCLIP更新版与论文Legacy权重应分开。

**迁移判断：** 若稠密分支确实涨点但太慢，优先考虑共享帧特征/蒸馏压缩。

**不能越过的比较边界：** 学生训练与直接使用冻结VLM不同；源任务的提升不能平移为OVI-MAP提升。

**优先级：** 高：M3/M4

## P14 · PCA-Seg — CVPR 2026
论文：https://arxiv.org/html/2603.17520v1

仓库：https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg

**方法要点：** 并行空间/语义代价聚合与自适应组合。

**已核验实现入口：** `cat_seg/; configs/; plain_train_net.py; train_net.py; eval.sh`

**公开状态：** 模型训练、评测代码和配置公开。

**迁移判断：** 作为语义教师备选；用于分析局部与上下文语义为何冲突。

**不能越过的比较边界：** 有训练成本；语义分割、部件分割与三维实例AP不同。不能直接搬用其类别监督而不披露。

**优先级：** 中：M3备选

## P15 · VIP — ICML 2026
论文：https://arxiv.org/html/2605.12325v1

仓库：https://github.com/MiSsU-HH/VIP

**方法要点：** 基于dino.txt的词汇/视觉引导和显著性聚合。

**已核验实现入口：** `dinosegmentor.py; eval_seg.py; configs/; dist_test.sh（README中的eval.py与目录需核对）`

**公开状态：** 评测源码和配置公开。

**迁移判断：** 作为GLA-CLIP之外的单个稠密语义候选；在固定别名规则下检验尾类识别。

**不能越过的比较边界：** dino.txt具备文本对齐，不应与普通DINO混为一谈；不同模型采用各自文本编码器，只融合校准分数。

**优先级：** 高：M3备选

## P16 · WOW-Seg — ICLR 2026
论文：https://arxiv.org/html/2605.16903v1

仓库：https://github.com/AAwcAA/WOW-Seg-Meta

**方法要点：** Mask2Token区域语义识别与防止实例间干扰的级联注意力。

**已核验实现入口：** `wow_eval/single_mask_infer.py; internvl/; demo/`

**公开状态：** 源码和模型入口公开；官方ICLR收录已核对。

**迁移判断：** 对少量难分类物体的最佳视图进行区域命名，并映射到冻结评测词表。

**不能越过的比较边界：** 原模型训练用GT mask及2D标注，推理时才可换预测mask；自由文本命名不能用测试GT手工映射。

**优先级：** 高：M5可选

## P17 · OVRCOAT — CVPR 2026
论文：https://arxiv.org/html/2603.21386v1

仓库：https://github.com/nickormushev/OVRCOAT

**方法要点：** 开放词汇区域对齐与CLIP条件的objectness调整。

**已核验实现入口：** `ovrcoat/; configs/coco/panoptic-segmentation/; train_net.py`

**公开状态：** 源码、训练配置和使用入口公开。

**迁移判断：** 借用开放类别objectness校准，评估是否减少真实小物体被过滤。

**不能越过的比较边界：** 训练监督、类别词表与输出粒度须独立记录；不能把新类分数调整当无成本纯几何修改。

**优先级：** 中：M5/置信度

## P18 · X2SAM — ECCV 2026
论文：https://arxiv.org/html/2605.00891v1

仓库：https://github.com/wanghao9610/X2SAM

**方法要点：** 统一多类分割任务的语言模型与Mask Memory。

**已核验实现入口：** `x2sam/x2sam/demo/app.py; x2sam/x2sam/configs/`

**公开状态：** 训练、评测与demo代码公开。

**迁移判断：** 作为高成本难例分割备选；不是第一轮默认前端。

**不能越过的比较边界：** 语言条件和额外计算预算必须计入；默认高容量模型不保证在线速度。

**优先级：** 低：M5备选

## P19 · CoSMo3D — CVPR 2026 Oral
论文：https://arxiv.org/html/2603.01205v1

仓库：https://github.com/JinLi998/CoSMo3D

**方法要点：** LLM引导的规范空间建模与三维语义部件分割。

**已核验实现入口：** `app/segment/eval_benchmark.py; release_module/training/; model/`

**公开状态：** 源码及模型入口公开。

**迁移判断：** 借鉴规范化/粒度思想；用于判断部件强模型为何未必能提升whole-object AP。

**不能越过的比较边界：** 对象部件任务与室内场景实例任务不一致，当前不做主干替换。

**优先级：** 低：不主集成

## P20 · AnyUp — ICLR 2026 Oral
论文：https://arxiv.org/html/2510.12764v1

仓库：https://github.com/wimmerth/anyup

**方法要点：** RGB引导、保持原特征语义空间的通用特征上采样。

**已核验实现入口：** `hubconf.py: anyup / anyup_multi_backbone; anyup/model.py`

**公开状态：** 核心加载接口、官方用法、权重入口公开且已核对。

**迁移判断：** 用于稠密语义支路的边界/小目标特征池化，先和双线性插值做同输入消融。

**不能越过的比较边界：** 输入图像需ImageNet归一化；NATTEN和原论文版本窗口略不同。上采样不创造新语义知识，必须计入显存与延迟。

**优先级：** 高：M3

## P21 · Rewis3D — CVPR 2026
论文：https://arxiv.org/html/2603.06374v1

仓库：https://github.com/Rewis3d-MPI/Rewis3d

**方法要点：** 利用重建促进弱监督语义学习。

**已核验实现入口：** `Rewis3d_Model/; Rewis3d_Reconstruction/`

**公开状态：** 模型与重建源码公开。

**迁移判断：** 作为2D–3D一致性训练的补充研究，不放入零训练主方案。

**不能越过的比较边界：** 包含弱监督信号，主要任务并非原生OVI-MAP实例建图；仅作拓展对照，不拿其分数直接排名。

**优先级：** 低：后续研究

## 不计入21项的直接相关候选

- MV3DIS：官方仓库 https://github.com/zybjn/MV3DIS 在本次检查时为占位性发布，没有核验到实际方法实现。
- OpenVoxel：项目页 https://peterjohnsonhuang.github.io/openvoxel-pages/ 的代码公开状态不足以支持直接复用。
- LightSplat： https://github.com/vision3d-lab/lightsplat 的方法代码发布不完整；不把预计算特征/数据当完整实现。
- GeoGuide、OVSeg3R：本次没有闭合“正式论文—官方可运行仓库”的完整链条，因此不算入20篇以上公开代码清单。
- CVPR Findings/Workshop论文、EntitySAM等2025论文可以做额外参考，不在本清单中冒充2026主会论文。

## 会议收录补充核验入口

- GeoPurify（ICLR）：https://proceedings.iclr.cc/paper_files/paper/2026/hash/039bc8e424e1fc196b4203555b54ebc4-Abstract-Conference.html
- SAM 3（ICLR）：https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html
- WOW-Seg（ICLR）：https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc4fbc2c77d2150c4e61e0fca6c2e95a-Abstract-Conference.html
- AnyUp：https://wimmerth.github.io/anyup/
- SpaCeFormer：https://nvlabs.github.io/SpaCeFormer/
- 其余会议/Oral/Highlight信息以各条官方仓库或项目页声明为依据；它不是对论文质量的统一量化评分。
