# 外部对照范围

EXTERNAL_FULL_REPRODUCTION=NOT_RUN。

MV_ATTN是在固定region输入上训练的256维、4层受控多视图attention模型，使用本轮父CE目标。[CAMFusion v1](https://arxiv.org/html/2604.12551v1)采用不同的自注意力/跨视图记忆交互、1024维8层模型与额外多视图对比目标。本轮仅检验受控学习式融合，不能把结果当作原版CAMFusion复现。

B_PFE_ADAPTED仅在当前对象采样位置做可见性区域特征聚合。[Open3DIS v3 §3.3](https://arxiv.org/html/2312.10671v3)还包含跨提案的点特征累积、原多尺度裁剪与提案流程；[作者仓库](https://github.com/VinAIResearch/Open3DIS)的完整方法未在本轮运行。不能据此表述超越两篇原论文的完整系统。原始设置复现需要单独协议与预算。
