# Maintenance Mode Switch + AAAI 评估方案

日期: 2026-06-27
状态: 设计完成，待实现

---

## 1. 目标

在同一个 V4 动态数据集上产出 TSDF 驱动 vs 时间信号 baseline 的定量对比结果。

---

## 2. 改动

### 2.1 dynamic_maintenance.py 加 mode 开关

```python
class DynamicMaintenanceModule:
    def __init__(self, config):
        self.mode = config.get("mode", "tsdf")  # "tsdf" | "time"
        ...

    def process(self, state):
        if state.frame_count % self.check_interval != 0:
            return state

        if self.mode == "time":
            return self._process_time_based(state)
        return self._process_tsdf(state)  # 当前逻辑
```

**time 模式逻辑**（精简版，不需要状态机）：

```python
def _process_time_based(self, state):
    max_inactive = self.config.get("ghost_max_inactive_frames", 30)
    for obj in state.objects.values():
        if obj.state in (ObjectState.REMOVED, ObjectState.GHOST):
            continue
        frames_since = state.frame_count - obj.last_seen_frame
        if frames_since > max_inactive:
            obj.state = ObjectState.GHOST
    return state
```

`ghost_max_inactive_frames` 配三个值：10/30/60，显示帧率敏感性。

### 2.2 配置

```yaml
# default.yaml 新增
dynamic_maintenance:
  mode: "tsdf"  # "tsdf" or "time"

# apt0_dynamic_benchmark.yaml YOLO 路径修复
anchor_frontend:
  enabled: true
  model_path: "/home/ww/vv/paper2/yolov8s-world.pt"
```

### 2.3 实验矩阵

| Run | Mode | YOLO | ghost_max_inactive | 产出 |
|-----|------|------|--------------------|--------|
| A | tsdf | on | — | tsdf.jsonl |
| B | time | off | 10 | time10.jsonl |
| C | time | off | 30 | time30.jsonl |
| D | time | off | 60 | time60.jsonl |

### 2.4 评估对比表格式

| 方法 | DELETE | MOVE | ADD | 误判 |
|------|--------|------|-----|------|
| TSDF (ours) | 19 | 6 | ? | ? |
| Time (10f) | ? | ? | ? | ? |
| Time (30f) | ? | ? | ? | ? |
| Time (60f) | ? | ? | ? | ? |

---

## 3. 不变更范围

- pipeline 核心逻辑不变
- TSDF 三级判决逻辑不变
- 数据集不变
