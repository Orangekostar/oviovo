# 动态 Benchmark V3 设计文档

日期: 2026-06-26
状态: 设计完成，待实现

---

## 1. 目标

用两段式静态场景拼接替代帧级事件插入，在 apt_0 上验证 TSDF 驱动的三级判决（DISAPPEARED / MOVED / NEW OBJECT）。

核心改进：不再要求在渲染过程中精确控制每个物体的变更帧号，而是分两次独立渲染"变化前"和"变化后"场景，在帧边界处自然产生所有变化。

---

## 2. 方案总览

```
同一轨迹 traj.txt, 两次独立 Habitat 启动:

Run A (frame    0-1199): apt_0 + 8 个 YCB 物体（4 DELETE + 4 MOVE-旧位置）
Run B (frame 1200-2399): apt_0 + 8 个 YCB 物体（4 MOVE-新位置 + 4 ADD）

拼接 → 2400 帧连续序列
         │
         ├─ frame    0-1199: 静态 baseline
         └─ frame 1200-2399: 变化后持续观测
```

GT：**所有 12 个变化发生在 frame 1200**。检测窗口：[1250, 1400]。

---

## 3. 物体配置

| 物体 | Run A 位置 | Run B 位置 | 判决类型 | GT object_id |
|------|-----------|-----------|---------|-------------|
| bowl_01 | (前) | 不存在 | DISAPPEARED | run_a_id:1-4 |
| bowl_02 | (前) | 不存在 | DISAPPEARED | |
| cup_01 | (前) | 不存在 | DISAPPEARED | |
| cup_02 | (前) | 不存在 | DISAPPEARED | |
| chair_ycb_01 | 旧位置 | 新位置 | MOVED | run_a_id:5-8 |
| chair_ycb_02 | 旧位置 | 新位置 | MOVED | |
| donut_01 | 旧位置 | 新位置 | MOVED | |
| donut_02 | 旧位置 | 新位置 | MOVED | |
| sphere_01 | 不存在 | (前) | NEW OBJECT | run_b_id:1-4 |
| sphere_02 | 不存在 | (前) | NEW OBJECT | |
| apple_01 | 不存在 | (前) | NEW OBJECT | |
| apple_02 | 不存在 | (前) | NEW OBJECT | |

所有物体放置位置使用 `--initial-in-front-distance` 和 `--initial-in-front-spacing` 放在摄像机正前方，确保可见性。

---

## 4. 数据生成

### 4.1 轨迹

复用已有轨迹：`/home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt`

格式：每行 16 个空格分隔浮点数，4×4 矩阵行主序。2400 行。已验证在 apt_0 场景上可用。

### 4.2 脚本

修改 `/home/ww/vv/habitat-data-collector/scripts/20260227/replay_room0_traj_with_multi_objects.py`：
- 添加 `--objects-dir` 指向 YCB：`/home/ww/vv/habitat_data/objects/ycb`
- 添加 `--object-plan` JSON 文件指定要放置的物体列表
- 移除硬编码的 room0 特定逻辑
- 移除 `--align-sim3`（不需要轨迹对齐）

### 4.3 Run A 命令

```bash
python replay_apt0_traj.py \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --scene-path .../apt_0.scene_instance.json \
  --scene-dataset-config .../replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --object-plan run_a_objects.json \
  --output outputs/dynamic_benchmark_v3/run_a \
  --frames 1200 --width 640 --height 480 --seed 7
```

### 4.4 Run B 命令

```bash
python replay_apt0_traj.py \
  --traj-file /home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt \
  --scene-path .../apt_0.scene_instance.json \
  --scene-dataset-config .../replicaCAD.scene_dataset_config.json \
  --objects-dir /home/ww/vv/habitat_data/objects/ycb \
  --object-plan run_b_objects.json \
  --output outputs/dynamic_benchmark_v3/run_b \
  --frames 1200 --width 640 --height 480 --seed 7
```

### 4.5 合并

```bash
# 拼接 Run A + Run B，重命名帧号连续
# 前 1200 帧: run_a/rgb/frame_XXXXXX.jpg → merged/rgb/frame_XXXXXX.jpg
# 后 1200 帧: run_b/rgb/frame_XXXXXX.jpg → merged/rgb/frame_{i+1200:06d}.jpg
# depth 同理
# 拷贝 traj.txt（完整 2400 帧）到 merged/
```

### 4.6 event_log.json

```json
{
  "scene": "apt_0",
  "total_frames": 2400,
  "transition_frame": 1200,
  "detection_window": [1250, 1400],
  "expected": {
    "disappeared": {
      "count": 4,
      "labels": ["bowl", "bowl", "cup", "cup"]
    },
    "moved": {
      "count": 4,
      "labels": ["chair", "chair", "donut", "donut"]
    },
    "new_object": {
      "count": 4,
      "labels": ["sphere", "sphere", "apple", "apple"]
    }
  }
}
```

---

## 5. Pipeline 评估

### 5.1 frame_metrics.jsonl 字段

已有字段（Task 4 实现）：`ghost_object_count`, `moved_this_frame`, `new_candidate_this_frame`

### 5.2 评估逻辑

```python
# 简化版 eval:
# 在 detection_window [1250, 1400] 内统计:
detected_disappeared = ghost_count_in_window > 0
detected_moved = len(moved_in_window) >= 1
detected_new = len(new_candidates_in_window) >= 1
```

### 5.3 评估脚本

复用已有 `scripts/eval_dynamic_benchmark.py`，适配新的 event_log.json 格式。

---

## 6. 与被弃用方案的对比

| | V1（帧级事件） | V3（两段拼接） |
|------|------|------|
| GT 精度 | 每个事件精确帧号 | 所有事件在同一帧 |
| 相机轨迹 | 随机导航，不可复现 | 固定 traj，可复现 |
| 渲染次数 | 1 次 2400 帧 | 2 次 1200 帧（可并行） |
| 物体可见性 | safe_sample 随机视角 | 已有轨迹验证过 |
| 复杂度 | 需 event plan + frame-triggered 逻辑 | 只需两个 object_plan JSON |

---

## 7. 不变更范围

- OVIOVO pipeline 核心逻辑不改
- TSDF maintenance 判定逻辑不改
- eval_dynamic_benchmark.py 仅适配新 event_log 格式
- ReplicaCAD / YCB 数据路径不变
