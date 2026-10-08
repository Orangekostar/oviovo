# 证据与主张边界

{
  "COMBINATION_EQUALS_COMPONENT": false,
  "SEGMENTOR_SUPPORT_SIGNAL": false,
  "SEMANTIC_MASK_SIGNAL": false
}

SAM-V 是外部预训练分割器。本实验只检验固定提示、实测深度观察、共同 lifter 下的模块集成，不构成新几何感知解码器，也不证明对所有多视角方法更优。

结构与语义主张独立于选模目标。SV04 对照 SV03 使用同一双源双视角成功域；SV05 固定复制 SV02 分区和 SV04 标签，等同某组件时不声称协同增益。未获取新2D GT，不把前插入 panoptic 预测作为GT。

四个已曝光场景只提供开发证据，不是独立确认。两场景池类别覆盖有限，MATERIAL并非统计显著性。仅窗口计时可比较，完整地图延迟及在线FPS未测。
