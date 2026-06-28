# Conflict-Aware Surface Owner Gate 设计

日期: 2026-06-28
状态: 设计完成，待实现

---

## 1. 问题

优化后的 surface_owner_gate 过于严格：voxel owner 与 patch owner 不一致时一律拒绝。静态场景最优，动态场景把变化信号也滤掉了。

## 2. 方案：证据冲突 = 变化信号

```
voxel owner ≠ patch owner 时，不直接拒绝，判断：

IF patch 持续出现到此体素（hit_count >= 2）
   AND voxel 旧 owner 的 TSDF support 正在下降:
   → ACCEPT（旧 owner 衰退 + 新证据积累 = 真实变化）

ELSE:
   → REJECT（孤立噪声或 flickering）
```

## 3. 实现

在 `object_update.py` 的 `_filter_patch_by_surface_owner` 中，foreign_owner 分支加：

```python
if owner_id != patch_target_id:
    # Check if old owner's support is declining and new patch is persistent
    old_owner_support = tsdf_module.summarize_instance_support(volume, owner_id)
    support_declining = old_owner_support["owned_voxel_count"] < old_owner_support.get("peak_voxel_count", 0) * 0.5
    
    # Track how many times this voxel has seen this new patch
    conflict_key = (vk, patch_target_id)
    conflict_hits[conflict_key] = conflict_hits.get(conflict_key, 0) + 1
    
    if conflict_hits[conflict_key] >= 2 and support_declining:
        continue  # ACCEPT — real change detected
    
    decision_foreign_owner_mask[idx] = True  # REJECT
```

## 4. 特性

- 静态场景：owner support 稳定 → 永不误开门 ✅
- 动态变化：owner support 下降 + 新 patch ≥ 2 次 → 自动开门 ✅
- 相机抖动：短暂遮挡不造成 support 下降 → 关门 ✅
