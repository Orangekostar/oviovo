# External controls

EXTERNAL_FULL_METHOD_REPRODUCTION=NOT_RUN。110候选是本项目固定分数混合/偏置/角度对照，不是WiSE-FT、CAMFusion或Mask-Adapter复现。

[WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html)采用模型权重组合；本轮MIX组合原冻结FC和MV输出分数，未运行其完整训练/评估。

[GZSL校准](https://arxiv.org/abs/1605.04253)研究已见/未见类竞争校准；gamma类偏置是强简单控制，不能作为广泛新颖性主张。

[Mask-Adapter官方仓库](https://github.com/hustvl/MaskAdapter)是mask特征适配相关近邻；本轮没有执行其外部协议。有限FISTA/convex evidence定义尚未激活真实B实验；归一化后的向量一般不在原欧氏凸包，不声称精确交集投影。公开来源包括CVF检索记录、[WiSE-FT原作者预印本](https://arxiv.org/abs/2109.01903)、GZSL原论文和Mask-Adapter官方仓库；CVF直接抓取出现超时/403，没有把这些失败写成读过PDF全文。不是穷尽最新文献查新或CVPR录用判断。
