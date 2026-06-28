# Cross-Frame Voxel Label Consistency 设计

日期: 2026-06-28
状态: 设计完成，待实现

---

## 1. 问题

YOLO 产标签 → association 匹配同标签 → 仍创建 11 个 "lamp" object。因为跨视角 space 不重叠，association 无法合并已存在的碎片化 object。

## 2. 文献依据

OVI-MAP (CVPR 2026) / Miao et al. (IROS 2024): 每个 voxel 投语义标签票，多数者胜。同 label 的 voxel 区域合并为一个 instance。

## 3. 方案

### 3.1 VoxelOwnerSupport 加 label_votes

```python
@dataclass  
class VoxelOwnerSupport:
    support: Dict[int, float]     # 已有: instance_id → 累积投票
    label_votes: Dict[str, float] # 新增: semantic_label → 累积投票
    
    @property
    def dominant_label(self) -> str:
        return max(self.label_votes, key=self.label_votes.get) if self.label_votes else ""
```

### 3.2 integrate_patch 加 label voting

patch 集成时同步写 label_votes。

### 3.3 dynamic_maintenance 加 label-merge

每 check_interval 帧执行: 同 dominant_label 的 ACTIVE object → centroid < 2.0m + 至少一个被当前帧观测 → 合并。

## 4. 改动范围

- `data_structures.py`: VoxelOwnerSupport +1 field +1 property
- `tsdf_instance_map.py`: integrate_patch +2 lines
- `dynamic_maintenance.py`: +2 methods (~50 lines)

## 5. 预期

- lamp: 11 → 1, bowl: 3 → 1, 总 object 数 471 → ~50
- DELETE 能检测到 lamp 消失 (ownership_ratio 可计算)
