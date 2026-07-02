# Association Label Bonus 设计

日期: 2026-06-28
状态: 设计完成，待实现

---

## 1. 问题

Association 评分只有空间项（voxel_vote + centroid + bbox + geometry），没有语义项。YOLO 产了标签但 association 不用。同标签的 bowl 跨帧无法关联 → 1000+ 个 object 碎片。

## 2. 方案

`_compute_score` 加 label_match 项：

```python
total = voxel_score + centroid_score + bbox_score + geo_score + label_bonus
```

- `label_bonus = 0.15` if patch.label == object.label else 0
- `w_voxel = 0.45` (从 0.5 降 0.05，总 weight 保持 1.0)

**不改 Rule C**：geometry 仍然占 0.85，label 只占 0.15。geometry 主导，label 辅助 break tie。

## 3. 改动范围

- `src/modules/association.py:_compute_score` — 加 ~8 行
- `configs/default.yaml` — 加 `label_match_weight: 0.15`

## 4. 预期效果

- 同 label patch 匹配成功 → object 不再碎片化 → 1000 → ~50
- gate 的 `allowed_owner_id` 不再为 None → gate 正常工作
- DELETE/MOVE 检测恢复
