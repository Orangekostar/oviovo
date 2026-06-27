# Per-Object Ghost 标注 —— 设计文档

日期: 2026-06-27
状态: 设计完成，待实现

---

## 1. 问题

当前 frame_metrics 的 `ghost_object_count` 只记录本帧 ghost 数量，不区分是哪个 object 被 ghost 了。导致评估时无法过滤——场景自带瞬态 proposal 产生的 ghost 和真实 DELETE 事件产生的 ghost 混淆。

## 2. 方案

### 2.1 dynamic_maintenance.py

`last_moved_object_ids` 从 `List[int]` 改为 `List[Tuple[int, str]]`。
`last_ghost_object_ids` 新增 `List[Tuple[int, str]]`。

在 DISAPPEARED 和 MOVED 判决点，append `(obj.object_id, label)`：
- label 从 `obj.semantic_memory.label_hypotheses[0][0]` 或 `obj.debug["anchor_semantics"]["canonical_label"]` 取

### 2.2 main_pipeline.py

`ghost_object_count: int` 改为 `ghost_objects: List[Tuple[int, str]]`

### 2.3 eval_dynamic_benchmark.py

DELETE 检测：只统计 label 匹配 "lamp" / "indoor-plant" / "cushion" 的 ghost。
MOVE 检测：只统计 label 匹配 "bowl" / "box" 的 moved。
ADD 检测：只统计 label 匹配 "cup" / "bleach" / "tennis" 的 new_candidate。

### 2.4 实验矩阵不变

同一数据集，tsdf vs time(10/30/60) 对比。
