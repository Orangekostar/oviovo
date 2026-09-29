# OVI-MAP / A7：2026论文驱动的逐项模块实验路线

日期：2026-09-29。状态：研究与实验设计，**未执行新实验、未下载/加载新权重、未修改远端仓库**。

## 1. 目标与现有锚点

源码锚点：`Orangekostar/oviovo@1074eb746808ca6882b30ccd827167474bb7eb8f`。保持A7_COS_REFIT为AP主参照，A7_COS_FIXED为mIoU侧的并列参照，N0始终保留。两个A7都来源于已观察的历史实验，不视为独立盲选冠军。

Replica官方当前类别排名＋全数据集池化，单位%：

| 方法 | APall | AP25 | AP50 | mIoU | mAcc |
|---|---:|---:|---:|---:|---:|
| N0 | 8.685 | 34.585 | 21.477 | 27.261 | 32.695 |
| M2_CAL | 8.910 | 34.030 | 20.175 | 26.792 | 33.887 |
| A7_COS_FIXED | 9.187 | 35.516 | 21.078 | 27.899 | 33.969 |
| A7_COS_REFIT | 9.218 | 35.201 | 21.236 | 27.720 | 33.811 |

以上为已有结果，不是本次预测。[结果源](https://github.com/Orangekostar/oviovo/blob/1074eb746808ca6882b30ccd827167474bb7eb8f/artifacts/static_ovmap/m2_reviewer_study_v1/attempt_001/table_A_core.md)。

当前最需处理的不是增加温度参数，而是：目标区域不纯、少数视图污染、相同错误被重复支持、现成图像编码器的区域识别限制，以及部分实例形状本身不合格。后者不能通过改类修好。

## 2. 文献筛选边界

共24篇2026主会论文，按正式会议年份，而非首次arXiv上传年份。20篇是本轮新的候选/参考，4篇用于复核已经试过的路线。未用workshop凑数。已核对会议来源及方法说明；对首批候选补读关键代码。**没有声称24个项目全部可运行，也没有声称完整复现24篇论文。**

状态“代码/权重入口已见”不是本机安装成功；空仓库、将来开源、仅论文均不放进立即执行队列。下表链接可能同时包含论文和作者项目；正式执行应固定当前commit和checkpoint，不使用未核实的第三方同名仓库。


| ID | 论文 | 会议 | 公开状态与研究用途 |
|---|---|---|---|
| P01 | **SAM 3: Segment Anything with Concepts**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html) · [官方仓库](https://github.com/facebookresearch/sam3) | ICLR 2026 | 官方代码与受许可权重入口；本次未加载。概念条件分割与视觉提示；分别用于类别空间核验、二维ROI修复。不可无条件当成类无关CropFormer替代品。 |
| P02 | **Mitigating Objectness Bias and Region-to-Text Misalignment for Open-Vocabulary Panoptic Segmentation (OVRCOAT)**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Kormushev_Mitigating_Objectness_Bias_and_Region-to-Text_Misalignment_for_Open-Vocabulary_Panoptic_Segmentation_CVPR_2026_paper.html) · [官方仓库](https://github.com/nickormushev/OVRCOAT) | CVPR 2026 | 代码与权重链接已见；重点函数已读，未部署。将物体存在性与区域识别分开；首轮只移植外部mask区域分类，不将COAT强行套进固定owner分数。 |
| P03 | **Zoo3D: Zero-Shot 3D Object Detection at Scene Level**<br>[论文/一手来源](https://arxiv.org/html/2511.20253v1) · [官方仓库](https://github.com/col14m/Zoo3D) | CVPR 2026 | Zoo3D_0代码已公开，run.py与semantics/open_voc.py已读。几何投影/遮挡筛选→合适视图→SAM区域修复→语言特征。其输出是3D框，不能直接冒充实例mask方法。 |
| P04 | **ReLaGS: Relational Language Gaussian Splatting**<br>[论文/一手来源](https://arxiv.org/html/2603.17605v1) · [官方仓库](https://github.com/dfki-av/ReLaGS) | CVPR 2026 | 官方代码已公开；ROFA公式与默认值已核对。对象多视图异常特征过滤。默认3σ在当前3/8/10视图限制下不能直接发挥作用；需明确小样本适配。 |
| P05 | **Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning**<br>[论文/一手来源](https://valeoai.github.io/posts/cvpr-2026) · [官方仓库](https://github.com/valeoai/Franca) | CVPR 2026 | 官方代码与骨干/RASA权重已见；视觉SSL模型。位置偏差解耦与多粒度视觉描述。用于区域纯度/对应关系，不是有现成文本塔的SigLIP替代品。 |
| P06 | **ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/66178beae8f12fcd48699de95acc1152-Abstract-Conference.html) · [官方仓库](https://github.com/pavelsuma/ELViS) | ICLR 2026 | 代码、demo与预训练头入口已见；描述子提取指南仍不完整。局部描述子对应及最优传输式匹配，用于同一物体视图质量/重复性判断，不输出类别。 |
| P07 | **Object-Centric Refinement for Enhanced Zero-Shot Segmentation (OC-ZSS)**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7db2ffcbfd0bd361d47b7fa612bd2ba2-Abstract-Conference.html) | ICLR 2026 | 主会论文已核实；本次未确认可直接使用的官方推理权重。SSL对象提示和对象—patch双阶段交互；需要真正的训练/权重，不等于手工平均patch。 |
| P08 | **Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Direct_Segmentation_without_Logits_Optimization_for_Training-Free_Open-Vocabulary_Semantic_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 主会论文与方法HTML已见；官方可执行代码未确认。由分布差异直接构造分割表征；作为稠密特征读出后备，不宣称改温度就是复现。 |
| P09 | **ViTPrompt: Training-Free Prompt Refinement with Visual Tokens for Open-Vocabulary Detection**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Qin_ViTPrompt_Training-Free_Prompt_Refinement_with_Visual_Tokens_for_Open-Vocabulary_Detection_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；官方代码/权重未确认。初次检测的视觉实例token辅助第二次检测。高置信伪标签可能自强化；不能直接给SigLIP插token。 |
| P10 | **SRA-Det: Learning Omni-Grained Open-Vocabulary Detection Beyond Category Names**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_SRA-Det_Learning_Omni-Grained_Open-Vocabulary_Detection_Beyond_Category_Names_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；官方代码/权重未确认。多语义属性检索与软AND匹配，启发细粒度类别对照；提示模板实验不是SRA-Det复现。 |
| P11 | **Bilateral Information-aware Test-time Adaptation for Vision-Language Models (BITTA)**<br>[论文/一手来源](https://scholars.hkbu.edu.hk/en/publications/bilateral-information-aware-test-time-adaptation-for-vision-langu/) · [官方仓库](https://github.com/super-jw/BITTA) | ICLR 2026 | 作者/机构主会确认，官方代码已见；需要测试时优化。同时利用不同熵输入，避免只对低熵错误过拟合。放在后期、按场景重置并计费。 |
| P12 | **Improving Calibration in Test-Time Prompt Tuning for Vision-Language Models via Data-Free Flatness-Aware Prompt Pretraining (FPP)**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Jang_Improving_Calibration_in_Test-Time_Prompt_Tuning_for_Vision-Language_Models_via_CVPR_2026_paper.html) · [官方仓库](https://github.com/YonseiML/fpp) | CVPR 2026 | 主会论文给出官方代码；未确认当前权重兼容性。平坦区域提示初始化。比较同骨干TPT与TPT+FPP；不能说比冻结A7没有额外成本。 |
| P13 | **SoC: Semantic Orthogonal Calibration for Test-Time Prompt Tuning**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Fillioux_SoC_Semantic_Orthogonal_Calibration_for_Test-Time_Prompt_Tuning_CVPR_2026_paper.html) · [官方仓库](https://github.com/leofillioux/SoC) | CVPR 2026 | 主会论文给出官方代码；需梯度优化。平滑原型分离，保留近义类别结构；不是每个来源增加一个温度。 |
| P14 | **CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.html) · [官方仓库](https://github.com/DuanchuWang/CompetitorFormer) | CVPR 2026 | 官方训练/推理代码已读；权重与训练场景需执行前绑定。解码器内查询竞争。作为重型3D候选对照；不把修改NMS称为复现。 |
| P15 | **OnlinePG: Online Open-Vocabulary Panoptic Mapping with 3D Gaussian Splatting**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Zhai_OnlinePG_Online_Open-Vocabulary_Panoptic_Mapping_with_3D_Gaussian_Splatting_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；本次未确认可用官方仓库。滑窗局部实例、双向局部—全局匹配。仅在候选形状确有缺口时研究后端接入。 |
| P16 | **GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Tao_GeoGuide_Hierarchical_Geometric_Guidance_for_Open-Vocabulary_3D_Semantic_Segmentation_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；官方代码/权重未确认。超点、整实例与实例间几何语义约束。需要学习，不能退化成已失败的全图平滑。 |
| P17 | **LightSplat: Fast and Memory-Efficient Open-Vocabulary 3D Scene Understanding in Five Seconds**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Bang_LightSplat_Fast_and_Memory-Efficient_Open-Vocabulary_3D_Scene_Understanding_in_Five_CVPR_2026_paper.html) · [官方仓库](https://github.com/vision3d-lab/lightsplat) | CVPR 2026 | 检查时官方仓库仍标代码待发布。语义索引压缩和相关mask聚类；用于效率后备，不直接当成熟可跑模型。 |
| P18 | **Ov3R: Open-Vocabulary Semantic 3D Reconstruction from RGB Videos**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Gong_Ov3R_Open-Vocabulary_Semantic_3D_Reconstruction_from_RGB_Videos_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；本次未核实官方完整运行包。语义融入RGB几何重建；与当前RGB-D/已知位姿设定不同，作为整系统对照而非小插件。 |
| P19 | **EmbodiedSplat: Online Feed-Forward Semantic 3DGS for Open-Vocabulary 3D Scene Understanding**<br>[论文/一手来源](https://openaccess.thecvf.com/content/CVPR2026/html/Lee_EmbodiedSplat_Online_Feed-Forward_Semantic_3DGS_for_Open-Vocabulary_3D_Scene_Understanding_CVPR_2026_paper.html) | CVPR 2026 | 主会论文已核实；摘要称将公开代码，当前可运行状态未确认。稀疏系数/全局语言码本和3D几何特征；不与ICCV2025同名导航论文混淆。 |
| P20 | **Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/aaa0ac4253da75faf9b0dc0dda062612-Abstract-Conference.html) | ICLR 2026 | 主会论文已核实；代码/权重未确认。对象感知稀疏窗口与记忆检索；仅在SAM2修复有用后研究效率，不是精度首选。 |
| P21 | **AnyUp: Universal Feature Upsampling**<br>[论文/一手来源](https://wimmerth.github.io/anyup/) | ICLR 2026 | 官方项目确认ICLR2026；本项目已测试过相关分支。只在新的强稠密特征证明分辨率受限后复测，不重跑旧弱GLA分支。 |
| P22 | **GeoPurify: A Data-Efficient Geometric Distillation Framework for Open-Vocabulary 3D Segmentation**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/039bc8e424e1fc196b4203555b54ebc4-Abstract-Conference.html) · [官方仓库](https://github.com/tj12323/GeoPurify) | ICLR 2026 | 作者公开代码/检查点；以前轻量适配不等于完整方法。学生亲和网络与几何教师蒸馏，后期需学习条件；不将手工pooling负结果外推到完整方法。 |
| P23 | **Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation (GLA-CLIP)**<br>[论文/一手来源](https://github.com/2btlFe/GLA-CLIP) | CVPR 2026 | 官方代码已检索；本项目相关稠密分支已负收益。全球/局部语义对齐；作为已测对照，不重复直接全局替换。 |
| P24 | **WOW-Seg: A Word-Free Open World Segmentation Model**<br>[论文/一手来源](https://proceedings.iclr.cc/paper_files/paper/2026/hash/fc4fbc2c77d2150c4e61e0fca6c2e95a-Abstract-Conference.html) · [官方仓库](https://github.com/AAwcAA/WOW-Seg-Meta) | ICLR 2026 | 官方代码/权重，且用户项目已有实测。区域生成式识别。现有跨场景排序反转，暂不重跑原样方案。 |

## 3. 实验队列：不是同时安装24个模型

| 顺序 | 试验 | 实际改动 | 必须对照 | 成本/关键判据 |
|---|---|---|---|---|
| C0 | 容量替换强对照 | 当前S2-L→官方SigLIP2-SO400M；只替换静态S2来源 | S2-L直接/融合、SO400M直接/融合；同目标/视图/crop、各自图文空间 | 无需训练；新增该模型区域编码；模型容量是否比复杂模块更划算；不计为2026新算法贡献 |
| E01 | 固定轨迹的稳健多视图读出 | 先只替换Q来源的最终聚合，不反馈查询控制器；用已有保留视图 | 原area均值、等权均值、无硬阈值的几何中位数；N0/S2固定 | 缓存CPU；无新视觉推理；检验坏视图污染；少视图/零方差显式处理，不使用无法触发的3σ |
| E02 | 只修语义ROI | 同一S2视图、bbox与3D owner；原union mask、纯global mask、SAM2视觉框修复mask逐项比较 | 原ROI；mask-only改变；后续才单独测试bbox改变，不能同时改视图数 | 新增区域crop编码；SAM修复有单独分割成本；语义输入是否含邻物/背景；本轮不改3D mask，不等同于前端替换 |
| E03 | 区域对齐模型替换 | OVRCOAT out_of_vocab_classification接受投影区域，替换S2来源而非堆第四票 | 现有S2；同骨干未区域微调的FC-CLIP区域读出；OVR区域读出；各自直接与融合 | 公开权重推理；本轮不训练基础模型；检验训练过的区域—文本对齐；原始logits/余弦/概率身份清晰且独立校准 |
| E04 | SAM3目标类别空间核验 | 对全部预测分歧对象，在固定视图上验证预先冻结的候选类别集合 | A7；同类别候选集的A7；SAM3空间支持直接读出；验证后组合另立消融 | 图像编码按帧共享，文本提示/解码按次计费；比较返回mask与目标区域的空间匹配，不把画面中任意同类存在当作本对象证据 |
| E05 | 局部描述子驱动的视图质量 | Franca/DINO对照或ELViS对同一owner内视图作局部对应；只调整语义视图权重 | 全局余弦相似、局部描述子对应；同区域/同视图数/同最终类别源 | 若局部tokens不在缓存，必须计新增特征前向；视觉一致性不等于类别正确；先不在编码前查询中偷用未付费描述子 |
| E06 | 细粒度文本证据 | 全部类别固定名词/结构/外观描述库；先只改变S2文本读出 | 原名词；固定属性描述；名词+属性组合；图像完全冻结 | 文本编码；无新增图像推理；不是临场按GT补同义词或上下文先验强改类；模板方案仅是SRA思想适配 |
| E07 | 对象中心稠密特征 | OC-ZSS/解析分割作为后备区域来源；需先有官方可用实现或充分实现依据 | 同骨干普通mask pooling与对象中心表示；语义读出其他条件相同 | 训练/权重条件待绑定，现阶段不启动大训练；若原始稠密特征已负收益，不用upsampling包装为成功 |
| E08 | 受控测试时适配 | 仅在一个区域分支中依次试官方TPT+FPP、TPT+SoC或BITTA | 同官方兼容骨干冻结版、普通TPT、单一新增机制；不一次叠三者 | 梯度/增强/状态存储均计费，逐场景重置；方法是transductive适配；无测试标签不等于无成本或无场景状态 |
| E09 | 完整二维前端替换 | CropFormer→OVRCOAT实体/panoptic mask；保持RGB-D、位姿、深度融合和TSDF算法 | CropFormer+原语义；新前端+原语义；CropFormer+A7；新前端+A7的2×2 | 新二维分割与建图/全部语义来源需重绑；先看类别无关AP与边界，再看语义AP；不能给新owner直接复制旧特征 |
| E10 | 二维边界修复进入几何 | 基于CropFormer视觉框/掩码提示SAM3，每个原候选先只返回一个改良mask | CropFormer原mask；视觉修复mask；类别文本提示另算一组 | 必须重新建图；不承诺提高召回；视觉提示修边与概念条件新对象检出是两个任务，拒绝混合计算 |
| E11 | 完整三维候选对照 | CompetitorFormer用当前预测点云/颜色/法向生成候选，再按同样区域语义读出 | 独立完整模型对照优先；不先做候选并集刷候选AP | 较高；需绑定训练集/权重/监督条件；训练集重叠、闭集类别过滤、GT mesh输入要明确，不能伪装无监督公平替换 |
| E12 | 几何学习与局部全局关联 | 有明确形状错误后再测试GeoPurify亲和、GeoGuide整物体约束或OnlinePG双向关联 | 固定候选集比较旧一致性与新关系信号；一次只试一个机制 | 高；独立训练信号与完整输入是前提；不重复LOCAL细块抢owner，不让硬IoU阈值压平所有训练目标 |

## 4. 四个最重要的接入陷阱

### 4.1 ReLaGS的3σ不能照搬到少视图缓存

ReLaGS对每视图特征计算与其他视图的平均余弦，再z-score过滤，论文默认阈值为3。当前S2最多3视图，原生读取最多8，Q最终保留最多10。对在同一n个观测中计算的总体标准差，任意标准化值满足绝对值不超过`sqrt(n-1)`。因此n<=10时严格`z < -3`不可能成立；用样本标准差上界更小。即便接入完全正确，也可能必然退化为原平均。

E01不复制这个默认阈值。先对固定的已支付、最终保留视图做参数很少的小样本稳健聚合，如单位向量的几何中位数；n=1保持原特征，n=2不宣称可辨别异常。对照应区分area平均与等权平均，以免把去掉面积权重当成稳健机制贡献。此适配是新实验，不称为完整ReLaGS复现。Q控制器、请求序列和谱系丢弃必须冻结，不将新读出反馈到过去的查询状态。

### 4.2 Zoo3D不是“把CropFormer换掉就赢”

公开Zoo3D_0仍使用CropFormer，并用投影框提示SAM2生成更合适的区域，随后使用CLIP。其默认图像尺寸、5视图、骨干和三裁剪都与当前A7不同。仅复用`FastCLIPFeatures.get_mask_crops()`中的投影框/区域修复思想；不要同时搬入新的骨干、5视图、不同尺寸和原论文bbox评测。

E02首轮锁定原3视图和bbox、同一SigLIP2、同一三维mask，仅替换语义黑背景mask。mask改动通过新缓存身份记录。后续SAM视觉修复不等于三维owner已经变好。几何受益必须另做E10重新建图。

### 4.3 OVRCOAT已有可接的外部mask分类入口

`ovrcoat/ovrcoat.py::out_of_vocab_classification(masks, clip_features, text_classifier, num_templates)`会将mask插值到稠密特征尺寸，进行区域pooling、视觉投影和文本分类。它不需要先采用该模型的全部panoptic输出才能研究区域分类。

先为当前OVI投影mask建立明确的padding/resize/mask值域适配；从分类logits或归一化区域特征处导出完整类别证据。返回值本身含softmax概率，**不能把该概率伪装成余弦再次随意softmax**。模型自己的文本编码、模板和预训练监督要记录。首轮只替换S2分支，不加第四来源、不改N0/Q。保留同骨干未区域微调的FC-CLIP对照，区别“换架构”与“区域训练”的作用。

### 4.4 SAM3是区域/概念证据，不是现成可靠性概率

锁定SAM3论文模型而非静默升级SAM3.1。权重可能需许可申请；申请未获批就是该叶任务的实际前提，不影响E01–E03。

只在由预测本身定义的分歧对象集合做首轮空间核验；候选类别来自已有来源，包含原生类别，执行前固定生成规则。相同类别在图像其他位置被检测到，不代表当前对象属于该类；应核对SAM mask与指定目标区域的对应。正确信息不在候选集时，记录该机制的能力上限。

先比较原始SAM建议的纠正/损坏，再研究采用方式，避免又加门槛让所有修改关闭。不要以GT挑选要调用SAM的目标，也不要把缺失检测直接当作物体不存在。全部未干预对象仍进入整图评价。

## 5. 当前仓库的实际接入点

| 现有代码（1074eb7） | 当前行为 | 拟议接入 |
|---|---|---|
| `m2_reviewer_study/core.py` | A7仅替换N0 score bundle；先锁定预测再评价 | 保留只改一个来源的结构；新方法用独立ID/输出目录 |
| `module_validation/semantic_study.py::prepare_semantic_manifest()` | 128目标上限、3视图，基于可见像素排序，并证明segment谱系 | 复用真实目标和请求；不能按owner数字直接重用新几何 |
| `semantic_study.py::direct_readouts()` | model白名单仅native/siglip2/wow，返回完整scores与fallback标志 | 新模型需要真正适配器；不能只改一个配置名称 |
| `module_validation/region_evidence.py::native_crops()` | 原生区域六裁剪 | ROI-only消融的固定裁剪基准 |
| `composition_study/visual_requests.py::VisualRequestLoader` | 模型/crop/mask/图像身份分离及真实付费请求 | 新区域或新模型必须新缓存身份，保留逻辑/物理费用 |
| `m2_reviewer_study/scores.py` / `composition_study/object_evidence.py` | 读取完整分数、真实可用性与最后保留观测 | 稳健聚合读取原特征，不从概率反演embedding |
| `m2_reviewer_study/fusion.py::fuse()` | 已支持子集来源及独立温度 | 新区域来源与A7做配对，不默认加票 |
| `m2_reviewer_study/evaluation.py::SceneEvaluator,pool()` | 双排名和原官方跨场景调用 | 官方当前类排名+池化为主；冻结排名只辅助归因 |
| 上游`panoptic_mapping_.py`与`frameToSegmentsCropFormer()` | 实例mask+深度→片段→TSDF/关联 | E09/E10是真正前端改动，需全新捕获和配对语义 |

这些是已存在接口或已读源码中的行为；试验包装器/新模型适配器仍是待开发，不是本次已实现成果。

## 6. 执行顺序与最小验收

1. 不重跑旧560条记录，不做全仓安全审计。复用已锁定的N0/A7结果与缓存。
2. 每次只激活一个试验族。首批E01→E02→E03→E04；C0作为必要的容量强对照，不能将容量收益冒充模块创新。
3. 先输出目标→输入→原始建议→最终输出的漏斗。如果没有实际干预，要解释是数学退化、不可用输入还是模型没有新信息；不能把一行相同分数当成充分检验。
4. 语义试验不改三维mask、owner或源域实例评分；官方导出仍按最终类别计算其规定分数。新模型/聚合若改变分数分布，参数只在允许开发数据确定，并保留同样标量校准预算。
5. 每种新来源保留直接读出与替换S2后的融合两行。单路较差但有独有纠错，仍可评价组合；不得只输出有利行。
6. 共享模型的重复观测不自动是独立证据。只有相同模型、处理器、图像、mask、bbox和精度的确切操作才可以扣除缓存重复成本。
7. 现有ScanNet/Replica全是已暴露研究数据。用作开发和历史回归无需阻塞，但不得称新盲测。新增数据或明确独立公开split用于最终冻结确认；新候选的预训练数据重叠也需记录。
8. 主要报告APall、AP50、mIoU的变化向量、实际纠正与损坏、来源缺失、最差场景及成本。不要规定“每个场景每个指标都不降”才能继续，也不要仅用均值掩盖严重损失。
9. 两个模块有有效或互补证据后，只做`base / base+A / base+B / base+A+B`的2×2组合；不遍历全部幂集。组合失败保留归因，不临场强改类别规则。
10. 每个试验结束提交相关代码、配置、小结果、`TRIAL_RESULTS.md`与`TRIAL_HANDOFF.md`到用户仓库任务分支，普通push后核对远端SHA。此文是研究路线，不是已获得结果或已执行推送的声明。

## 7. 第一批必须交的表

- 性能：N0/A7_REFIT/A7_FIXED及当前候选；主列为官方池化，附场景差值。
- 干预：处理对象/视图数、真实可用、原始建议、最终修改及技术回退。
- 纠错：相同可诊断对象集的corrected/harmed、匹配增加/损失、仍未利用的正确信息。
- 成本：各模型新前向、裁剪/提示、完整方法所需逻辑操作、峰值显存/记录到的时间；缺失计时标缺失。

## 8. 我对收益的预测方式

不以其他论文的AP增幅外推本项目。E01/E02优先针对坏视图与ROI污染；E03针对区域—文本错位，E04提供具空间证据的不同来源；E09–E12才针对实例几何。这些是待检验的作用方向，不承诺某模块涨若干百分点。不能保证精度排序与现成模型规模排序一致。

明确暂缓：原样重做WOW、GLA+AnyUp；继续只改三个温度；把Franca当作文本对齐模型；把未发布的LightSplat安排成可即刻部署；为少数视图直接使用3σ；混合不同模型的latent向量；同时重写Gaussian地图和语义主干。

## 9. 关键代码证据链接

- [本地core](https://github.com/Orangekostar/oviovo/blob/1074eb746808ca6882b30ccd827167474bb7eb8f/src/static_ovmap/m2_reviewer_study/core.py)
- [本地semantic_study](https://github.com/Orangekostar/oviovo/blob/1074eb746808ca6882b30ccd827167474bb7eb8f/src/static_ovmap/module_validation/semantic_study.py)
- [本地visual_requests](https://github.com/Orangekostar/oviovo/blob/1074eb746808ca6882b30ccd827167474bb7eb8f/src/static_ovmap/composition_study/visual_requests.py)
- [官方OVI主流程](https://github.com/OVI-MAP/OVI-MAP/blob/f8f7bcd0ca8228f6b8b4064f2e29dcee3a502424/scripts/panoptic_mapping_.py)
- [OVRCOAT分类](https://github.com/nickormushev/OVRCOAT/blob/main/ovrcoat/ovrcoat.py)
- [Zoo3D区域处理](https://github.com/col14m/Zoo3D/blob/main/Zoo3D_0/semantics/open_voc.py)
- [SAM3官方接口与许可](https://github.com/facebookresearch/sam3)
- [SigLIP2-SO400M官方强对照](https://huggingface.co/google/siglip2-so400m-patch14-384)

第三方仓库main链接记录本次阅读位置，正式执行必须冻结执行当日的SHA；本文件没有声称这些main不会变化。
