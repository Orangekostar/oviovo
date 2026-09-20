# Cached SF-to-OVI semantic transfer handoff

**工程 COMPLETE；科学 COMPLETE_NO_NET_GAIN；六条件均完成。** 固定OVI形状的缓存SF标签确有可用信息，但当前对应与支持门控均未带来净收益。两份缓存不是独立场景或盲验证；额外3D预训练仍存在，训练场景排除状态UNVERIFIED。

## 实测结论

- O-only/旧E1：uAP20.644%，mIoU36.556%。S-A两run：uAP15.367%，mIoU31.544%；S-B：uAP18.213%，mIoU31.718%；S-C主/重复：uAP18.213%，mIoU31.718%/31.720%。AP50/AP25/mAcc与类别子集进出见[结果表](SF_OVI_SEMANTIC_RESULTS.md)。
- S-A各提出并采用8次改类；strict >.5唯一几何对应中0次纠正、4次损坏。旧门槛曾阻止这4次有害建议，其中3次余弦优势不足、1次视图不一致；未发现旧门槛阻止该阈值下的有效纠正。
- S-B提出/采用24/25次改类；S-C采用23/23次。每个run的S-B/C均实际新增4个匹配（10004、15000、17000、33000），丢失10个（4000、6000、9006、19002、19005、19006、25000、39005、48000、51000）。S-C拒绝的1/2个建议在该阈值下无明确对应，没有阻止上述10次损坏。
- 主缓存receiver10，对应GT24000：合格供体中存在正确类，但最大IoU选中的query175为类14。这个路由漏选发生在预测完成后的诊断，未反馈修改供体。
- 主/重复的U00新增7/6、最佳记录OVI形状足够但原类错5/4、无足够OVI形状2/2全部复现；同时检查所有baseline misses。不能将这些对象数当作AP贡献或上界。

## 身份与执行范围

起点与审查锚点均为 `dc7aa67a9175cf0fe278ec7f62f5b6fbc979ead3`。独立工作树 `/home/ww/crove/ovimap-sf-to-ovi-semantics`，任务分支 `research/ovimap-sf-to-ovi-semantics-v1`。未修改旧OD/T1算法、评估器或旧产物。最终验收使用用户提供的九文件压缩包，并核对其中的 `SOURCES.md`、`SOURCE_BINDINGS.json`、协议、来源审查及原因账本；执行输入仍按固定提交的源码路径、收据及真实缓存绑定。

[配置](../../../configs/evaluation/ovimap_sf_ovi_semantics_v1.json)在新评估前冻结。双向源域覆盖≥0.5以整数比较实现，非IoU≥0.5；全部OVI接收者、全部非空SF供体先按几何建集合，包括同类供体。按IoU降序、原query ID升序、canonical ID升序选供体；没有GT、类别分歧筛选、反转best_owner_id或额外一对一约束。

S-A只用E1标签、准确baseline mask hash及原valid_ids顺序匹配的历史聚合余弦；采用pre-gate argmax，保留实际无输入状态。S-C完全复用S-B供体建议，仅以`IoU*T0.scores`加权同类支持≥2/3采用；T0分数的objectness×class probability约定已验证，没有重复乘概率。仅mask+原类完全相同者合并为max质量组，没有近似NMS。单正权重供体能通过；支持量不表示独立确认或校准概率。

[输入绑定](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/input_binding.json)记录消费的O-only/U00、两份T0、receipt、源掩码、坐标、原生文本顺序和保存语义行的实际SHA。SF query/原T0行/类/掩码及两run返回坐标逐项核对；OVI canonical owners与64个输出掩码精确对应。没有读取GT/cause ledger用于预测，没有模型、图像、GPU、raw logits或新渲染访问。

[预测执行身份](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/prediction_manifest.json)、[评估身份](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/evaluation_manifest.json)与[最终源码验证](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/execution_receipts.json)分开保存。最后修复了非有限缓存分数应弃权而非JSON序列化报错的边界情况；128个真实receiver记录在修复前后逐项一致，[对照证据](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/native_input_fix_parity.json)。最终审查又从冻结receiver账本确定性补齐六份条件清单的 `proposed/accepted/changed` receiver集合；集合计数与原漏斗逐项一致，未改变标签、身份键、科学阈值或六条件评测结果。Markdown生成时修复了历史子集计数为int、新计数为list的兼容问题，仅重新生成报告。

## 评估、成本与验证

六条条件记录中有5份新payload真实调用发布版评估器；重复S-A与主S-A的owner/masks/active labels/rank strings/protocol完整键一致，显式复用1次。历史O-only/旧E1的几何、掩码文件、类别及原分数字符串已核验，标为REUSED_REFERENCE。活动OVI掩码本来就是disjoint unique masks，未重复评估两列。原始发布版阈值、strict比较、100点门槛、48实例类/51语义类、FN/FP/duplicate/ignored与no-GT规则保持不变。所有未覆盖源点保持owner0/semantic0。

历史掩码通过相对路径引用，发布版打印“outside of prediction path”提示后继续读取；这不是跳过掩码。逐mask相等检查、实际预测记录及trace的AP/PR/FN精确parity均通过，未修改发布版代码来隐藏提示。

CPU线程8，源点chunk65536。预测总18.87s（含资产核对），两run对应与决策1.01s/0.75s；峰值预测RSS约777MB。评估与分阶段时间见[cost_summary](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/cost_summary.json)，未单独测量的summary耗时为null。**新增图像/文本/2D/3D推理、映射和训练均为0**。这是缓存后处理成本，不是端到端速度或模型推理成本。

9项定向测试通过，包括目标分离/顺序/门控前后标签、query对齐、双向覆盖、重复供体/单供体/零质量、条件清单receiver集合、几何排名不变、类别子集与真实发布版事件；真实缓存smoke完成。最终验证覆盖六份标签→语义点数组、固定owner/ranks、完整receiver账本及released trace parity。

实际命令（工作树内）：

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_static_sf_ovi_semantics.py --config configs/evaluation/ovimap_sf_ovi_semantics_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/evaluate_static_sf_ovi_semantics.py --config configs/evaluation/ovimap_sf_ovi_semantics_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/summarize_static_sf_ovi_semantics.py --config configs/evaluation/ovimap_sf_ovi_semantics_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/verify_static_sf_ovi_semantics.py --config configs/evaluation/ovimap_sf_ovi_semantics_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/test_cached_semantic_transfer.py tests/evaluation/test_static_t1_evaluator_trace.py
```

解释器为既有ovimap-map，Python3.11、NumPy1.26.4。重建时复制冻结配置，仅替换`new_output_root`为新目录，依次运行上述入口；禁止覆盖既有预测/评估目录。不需要模型权重、网络安装或GPU。

## 交付与需求核对

[小结果目录](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/)含JSON/CSV、128接收者决策、1152条receiver×threshold×condition正确性记录、全部备选供体及pair矩阵、支持门控交叉表、原生owner跨run比较、全阈值真实对象/事件、headroom及成本/测试/执行收据。大型语义数组及继承资产留在 `/mnt/shared/ww/ovimap-sf-to-ovi-semantics-v1/` 和绑定的旧目录；[路径/大小/SHA清单](../../../artifacts/static_ovmap/sf_ovi_semantics_v1/large_artifacts.json)不代表大文件上传。

| 要求 | 权威产物/校验 | 状态 |
|---|---|---|
| WP0冻结输入/正确T0与E1身份 | input_binding、frozen_config | 完成 |
| WP1 pre-gate建议、全供体反向对应 | receiver_decisions、reverse_correspondence、donor_registry | 完成 |
| WP2三规则×两run、固定几何排名 | 六condition manifests、execution_receipts | 完成 |
| WP3原发布版评估、明确复用与子集转换 | performance、metric_protocol、evaluation_manifest | 完成：5实评+1精确复用 |
| WP4四张表、正确性/门控/路由/对象/全misses | RESULTS、correctness、routing、headroom、events | 完成 |
| WP5 scoped代码、测试、报告、普通push | 本分支；最终回复给完整commit及远端SHA | 提交发布后核验 |

**唯一下一行动：**单独预注册并检验供体对应/选择的修订，依据是主缓存GT24000已有合格正确供体却被当前IoU规则漏选。仍应覆盖全部接收者并报告已有正确实例的损失；本次不执行该后续实验，不新增门槛或更换语义教师。
