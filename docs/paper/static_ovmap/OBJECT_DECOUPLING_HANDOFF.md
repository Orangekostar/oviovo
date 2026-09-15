# Object/semantic decoupling v1 handoff

**软件与实验：COMPLETE；科学结论：COMPLETE_NO_NET_GAIN；跨场景有效性：未建立。** 两份 Room0 FP32 缓存的 8 个条件已实际评估。Unique AP 全部为 20.644%；主缓存 E2 的 mIoU +0.3814 个百分点，E3 将其抵消。E1 和 E4 均无指标收益。strict >.5 下相对 O-only 新增/丢失对象均为0，U00额外7/6个对象均未保留。不是模型或场景泛化成功。

## 代码与输入身份

- 实际起点：`b8d360be5e6c6a4a5f81d370589dc997b97ebd03`，独立工作树 `/home/ww/crove/ovimap-object-semantic-decoupling`，分支 `research/ovimap-object-semantic-decoupling-v1`；原工作树及旧实验未修改。
- [冻结配置](../../../configs/evaluation/ovimap_object_decoupling_v1.json) 在任何新 GT 评估前写入。预测规则与阈值只用此版本；两份 Room0 历史结果已参与设计，不视作盲验证。
- [预测执行源码散列](../../../artifacts/static_ovmap/object_decoupling_v1/prediction_manifest.json)、[输入绑定](../../../artifacts/static_ovmap/object_decoupling_v1/input_binding.json)、[最终代码及数值校验](../../../artifacts/static_ovmap/object_decoupling_v1/final_verification.json)。补充的 [T0/processor文件确认](../../../artifacts/static_ovmap/object_decoupling_v1/final_asset_confirmation.json)记录实际消费文件。最终 SHA 在提交后的交付回复中给出，不递归写入本文件。
- 执行后提取等价 query-ID 查表/实例子集函数、补齐65536点分块计数和消费资产确认、增加精确评估缓存键及验证工具。分块版重新运行两份几何缓存，全部动作、活动ID和owner逐项相等，见[分块验证](../../../artifacts/static_ovmap/object_decoupling_v1/chunk_verification.json)。53个目标由保存的159个视图特征重算，类决策相同；聚合余弦跨进程最大观察误差 `2.78e-17`，验证容差 `1e-14`，float32视图余弦容差 `1e-7`。未再次调用图像模型。
- 缓存键补强后重跑8个评估单元，所有 AP/语义/独立几何指标与首轮完全相等；首轮输出保存在外部 `evaluation_initial/`。当前结果来自带完整身份键的评估器。这里是软件校验，不是新超参数版本或16个独立实验。
- 共享坐标逐行精确一致；候选、源掩码、原 U00、RGB 与对应历史收据散列核验。共享32帧正实例、深度一致像素缓存直接复用，未重建原45万原子图。ID0观测未知且弃权，不称作完整可见像素。
- 模型/processor：`/home/ww/vv/paper2/model_cache/google_siglip-large-patch16-384`，FP32，`cuda:0`，严格 `local_files_only=True`。模型 SHA256 `ab804d2c2c631159f65afa1de27787c5dba90cbfa189f26404de1944d50a18a2`；processor `f59da2f87c3cd079bd4f8f3037e81b277c60c498e279a8020331f67a5a3157e8`。
- 1024D 原生51类文本及4个 canonical 短语，不使用SF softmax作为余弦门槛。保留原六裁剪、排他切片和RGB顺序；[上游 MIT 许可](licenses/OVI_MAP_MIT.txt)随适配实现保留。

## 方法边界与产物

与 LOCAL 不同，本任务显式授权有界图像编码、完整候选退役、未知所有权、E1/E3改类和E4最终源点排名。旧方法的全覆盖、固定类别/排名检查没有削弱；所有改动均为新文件。E2仅读取几何与SF内部 objectness，未读GT或新语义。小范围SF重叠以原 objectness、canonical ID消歧，两个run均未使用原始logits。

E1严格保留O-only所有权；E3保留E2全部掩码和活动注册表；E4保留E3掩码、标签与语义点数组，只把唯一新增实例分数91改为78（源点计数，非评估域33点）。零归属坐标不删除。主缓存仅接受SF query20/canonical99的ADD，91个源点保留78，33个评估点均为wall-plug；E3改成blanket后33点全错。实例不足100点，不能计为新TP。重复缓存无接受动作。未出现实例类别子集进出或99–101点门槛穿越。

[四张结果表](OBJECT_DECOUPLING_RESULTS.md)包含完整正负对照。[机器结果目录](../../../artifacts/static_ovmap/object_decoupling_v1/)包含1944条全对象/阈值记录、43条评估侧原因记录、所有接受/拒绝动作、全部语义目标及分数、源/最终IoU和范围、事件标记、区域整数混淆差与排名诊断。geometry_archive在编码及GT评估前保存。大数组、模型、图像、初轮/最终评估掩码均留在 `/mnt/shared/ww/ovimap-object-semantic-decoupling-v1/`；[清单](../../../artifacts/static_ovmap/object_decoupling_v1/large_artifacts.json)记录实际路径、字节数与SHA256，**不代表大文件上传GitHub**。

历史AT_O_AREA/U00/U11与FILL/LOCAL/SPATIAL来自绑定版本，明确为REUSED_REFERENCE；新的活动重叠候选AP和Unique AP均实际重算，未冒用U00 AP。发布版 strict阈值、100点、48实例类/51语义类、ignored与无GT空值均保持原实现。无持久缓存盲目续跑：预测/评估入口要求新输出目录；已保存payload由独立验证器核验。

## 实际命令与成本

环境：`ovimap-map`，Python3.11.15、NumPy1.26.4、torch2.1.1、transformers4.49.0，CPU线程8。在任务工作树执行：

```bash
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_static_object_decoupling.py --config configs/evaluation/ovimap_object_decoupling_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/evaluate_static_object_decoupling.py --config configs/evaluation/ovimap_object_decoupling_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/summarize_static_object_decoupling.py --config configs/evaluation/ovimap_object_decoupling_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/verify_static_object_decoupling.py --config configs/evaluation/ovimap_object_decoupling_v1.json
/home/ww/miniconda3/envs/ovimap-map/bin/python -m pytest -q tests/test_object_arbitration.py tests/test_object_decoupling_semantics.py tests/evaluation/test_static_projected_masks.py tests/evaluation/test_static_projected_instance_metrics.py tests/evaluation/test_static_released_loader.py
```

已有目录不可覆盖；重新构建时复制配置并仅更换`new_output_root`。验证器比较本次保存的`evaluation_initial/`与最终`evaluation/`；从零重建单轮时可直接使用评估器自身隔离断言。15项定向测试通过，实际六裁剪smoke与8条件评估通过。未运行全仓库测试。

联合目标为27/26，视图请求81/78；实际前向81/3，第二run精确复用75个输入。总84个encoder batch、504张crop，远低于4608上限；新2D/3D分割、映射、训练均为0。模型加载1.65s、裁剪/processor6.21s、编码14.31s、完整预测112.89s；峰值GPU分配2.80GB、RSS4.57GB。复用投影无新投影计算；文本读取与像素掩码组装未单独计时，包含在总预测时间。历史mapping/SF成本未纳入增量成本，不是端到端基线。

## 需求核对

| 要求 | 实现/证据 | 状态 |
|---|---|---|
| WP0绑定/冻结、两份distinct FP32、CPU/GPU预算 | config、input_binding、prediction_manifest | 完成 |
| WP1原因拆分，无GT预测 | cause_ledger；诊断脚本独立于预测 | 完成 |
| WP2三类完整对象动作、整组回滚、1:1组评分 | object_arbitration、两run actions账本 | 完成；1/0动作接受 |
| WP3原生离线六裁剪、最终掩码、全余弦和门槛 | candidate_semantics、semantics、real_crop_smoke | 完成；无O-only改类 |
| WP4只改源域最终面积排名 | final_instance_ranking、ranking_eligibility_examples | 完成；无AP收益 |
| WP5八条件、发布版真实事件、原区域pred0闭合 | performance、all_object_outcomes、regional_deltas、final_verification | 完成 |
| WP6四表、完整账本、许可、外部大文件清单、分支发布 | RESULTS、本handoff、small_artifacts | 本地交付完成；远端SHA以最终回复核验 |

局限：单个开发场景的两个预测缓存；相关2D教师可能重复合并错误，原生SigLIP重询可能重复错误；稀疏像素与有限视图也属于本干预。候选union AP不是disjoint AP上界；未将E4排名变化解释成边界改善，未计算不成立的析因交互。

**唯一下一实验：**冻结相同目标掩码与视图预算，单独比较一个独立语义教师与本次原生SigLIP重询，禁止任何类别反馈改变几何；检验本次失败是否来自相关语义错误。该实验尚未运行，也未在本任务下载或引入新模型。
