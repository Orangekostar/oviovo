# OVMap CVPR补实验：源码绑定执行包 v2

主入口：`00_给Codex的最终执行指令.md`。同时读取01–04，数值/枚举以02、算子语义以03为准。**不要与旧v1追加拼接**。

本包针对已经选定的“多源融合＋锚定校准”方法和benchmark-first表格，不重新研究新的门控或凸包模型。正文三张表是：T1外部baseline×benchmark，T2完整系统MF×AC，T3校准消融。旧26场景的eligible替换不再冒充全局四格。

文件：
- 00：17节执行合同、阶段依赖、模型选择、范围与发布。
- 01及sources.json：25项项目源码、6项外部官方资料、4项CCFA规范依据；来源中区分本次重读与此前有效上下文复用。
- 02：冻结配置，两个种子都用MV matched1000、165候选/模型/seed、完整benchmark有限矩阵。
- 03：缓存键、CAP/MIX/NMIX、defined/H、全owner SceneBundle、缺源回退与评分。
- 04：OVI-MAP、OpenMask3D、Open3DIS、MaskClustering完整运行与标准协议适配。
- 05：实质审核、旧包缺陷修复、运行时阻塞动作。
- reference_ops.py / check_contract.py：可执行CPU参考与合同检查。
- 06：实际CPU检查结果，不代表服务器实验结果。
- 07：运行资产位置示例；null按真实依赖状态处理，不能当已存在。

没有执行新训练、GPU、官方benchmark、baseline或GitHub发布。本次交付的是可执行工作设计与数值参考；Codex负责按00完成开发、真实运行和C1/C2发布。需要原数据、预训练资产及现有运行环境的阶段必须核验真实可用性。

运行本包廉价检查：
```bash
python check_contract.py
```

Codex需开发的新入口不是当前已有文件：
```bash
PYTHONPATH=src:. "$PY_CONTROL" scripts/evaluation/run_ovmap_benchmark_completion_v2.py \
  --semantic-root "$SEMANTIC_ROOT" --structure-root "$STRUCTURE_ROOT" \
  --output-root "$OUT" --path-map "$PATH_MAP" --asset-config "$ASSET_CONFIG" \
  --phase all --resume
```
