# 共同MV模型与区域输入源码复核

2026-10-24：复核共同模型锁与region-only输入组件，读取冻结源码、权重运输文件和原G2缓存身份。本轮没有加载模型或重做forward，新增图像编码/训练均为0。旧分区导出、完整新benchmark和其他G2/G3/G4要求按其独立证据验收。

| 合同要求 | 核查结果 |
| --- | --- |
| matched1000真实权重 | 只使用MV_ATTN:seed17/29:matched1000显式profile；原加载器核对运输SHA、解包内容SHA和实际state的method/seed/step，严格加载state_dict，eval并关闭梯度。当前两份运输文件SHA与模型锁及公开manifest一致。 |
| 原模型与anchor | 保留原MV attention、768维表示及推理η=0.5。b仍由原FrozenFCBase对有效FC region向量求均值再归一化。新场景worker核对两seed的b完全一致。 |
| 七字段区域接口 | 保持fc_vectors/view_valid/frame_ids及其顺序，V≤8；local_raw/unit/metadata分别为[V,0,1536]/[V,0,768]/[V,0,8]，local_valid为bool[V,0]。原MV数值embedding不读取局部数值。 |
| 原始缓存身份 | 按原namespace、object input SHA、views与feature identity查找SparseInputs；模型锁引用的两份真实完整输入及四份b/z缓存数组SHA/字节数核对一致。 |
| 原G2实测 | office0:1与scene0011_00:1各自两seed，共4组full/region-only比较，b/z误差均0、top1一致；原FC text dtype为float64。此前实际为8次head forward与2次静态anchor调用，本轮只读复核。 |
| 精度与scope | method-lock及新场景worker先设置TF32关闭、math SDP等原FP32配置。共同模型锁仍需要各benchmark独立词表/映射/学习数据角色审核。 |

逐段复核`registry.py`、`region_inputs.py`以及`runner._model_lock`和`fc_worker.run`的相关调用路径；同时阅读原`MVReadout`、`FrozenFCBase`及权重解包/加载实现。三个继承源码与固定BASE Git blob逐字节一致；9份相关源码/合同的SHA及mtime封存。1960份稳定文件及93份冻结文件SHA/mtime保持，运行源码未改写。

共同模型锁：`69592d6c35139cdd27f25ff16cfed1b0d1d5e3c0e659c59c77bca77ecfda6eae`；公开权重manifest：`8c3da1a224aa36e80c7075dae661e8af7913f07f95d94c47ec8931df859176c6`。本次证明：`$OUT/review/G4_model_region_input_source_primary_review.json`，`8f2d148d1f9d03c40e9d0eaa099a1f6f6bf7125d438d3d9f01720447ac0bd5cd`。

范围限于共同模型与区域输入组件。原clean8V condition及模型等价性不直接证明benchmark未曝光或完整方法的独立确认；完整协议、实验、正式统计与最终17章审核继续按合同完成。
