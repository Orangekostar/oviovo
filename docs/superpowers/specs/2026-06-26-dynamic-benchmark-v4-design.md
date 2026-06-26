# 动态 Benchmark V4 设计文档

日期: 2026-06-26
状态: 设计完成，待实现

---

## 1. 目标

基于 `replica_dynamic_object_orbit_capture.py` 的逻辑，在 apt_0 场景上生成带帧级 GT 标注的 2400 帧动态 benchmark 数据集。验证 TSDF 驱动的三级判决（DISAPPEARED / MOVED / NEW OBJECT）。

---

## 2. 时间线

全部变更帧号为 10 的整数倍。800 帧一个阶段，共 3 阶段：

```
Phase 1   frame   0- 799  8 个 YCB 物体全部可见
Phase 2   frame 800-1599  DELETE×4 (bowl×2,mug×2) + MOVE×4 (pitcher×2,brick×2 → 新位置)
Phase 3   frame1600-2399  ADD×4 (banana×2,apple×2)

GT 标注:
- frame 800:  DELETE ×4 + MOVE ×4
- frame 1600: ADD ×4
```

检测窗口：[850, 1000] for DELETE/MOVE, [1650, 1800] for ADD。

---

## 3. 物体摆放方案

### 场景家具（apt_0 已提取）

| 家具 | 位置 |
|------|------|
| table_01 | (0.41, 0.66, -0.17) |
| table_02 | (0.90, 0.61, 2.41) |
| table_03 | (1.99, 0.33, 5.82) |
| sofa | (3.91, 0.40, 5.27) |
| table_04 | (4.20, 0.44, 6.62) |

### 物体配置

| 物体 | 数量 | YCB 模板 | 位置 | 操作 |
|------|------|----------|------|------|
| bowl | 2 | `024_bowl` | table_02 桌面 (+0.06m) | DELETE @ 800 |
| mug | 2 | `025_mug` | sofa 扶手 (+0.40m) | DELETE @ 800 |
| pitcher | 2 | `019_pitcher_base` | NavMesh 采样地面 | MOVE @ 800 → 新 NavMesh 点 |
| foam_brick | 2 | `061_foam_brick` | NavMesh 采样地面 | MOVE @ 800 → 新 NavMesh 点 |
| banana | 2 | `011_banana` | table_04 桌面 (+0.06m) | ADD @ 1600 |
| apple | 2 | `013_apple` | table_01 桌面 (+0.06m) | ADD @ 1600 |

防悬空：
- 桌面上物体: `table_y + 0.06m + object_y_lift`
- 地面上物体: `NavMesh.snap_point()` 获取合法地板坐标

---

## 4. 相机轨迹

复用已有轨迹：
`/home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt`

Camera-to-World 格式，Y-up 坐标系。每帧相机位姿经 NavMesh 校验：
- `snap_point()` 吸附到最近合法导航点
- `distance_to_closest_obstacle()` 检查 clearance > 0.25m
- 超过 `max_snap_dist=0.5m` 的位姿→跳过（防止穿墙）
- Y = `snapped_floor_y + sensor_height`（1.5m 人眼高度）

参数: `--traj-file` + `--traj-pose-format c2w` + `--traj-max-snap-dist 0.5`

---

## 5. 生成流程

一次 Habitat 启动，顺序执行：

```
Phase 1: 放置 8 个物体 → capture_from_traj(frames=0..799, phase="before")
Phase 2: rigid_mgr.remove_object_by_id() ×4 (bowl,mug)
         obj.translation = new_pos ×4 (pitcher,brick → 新 NavMesh 点)
         → capture_from_traj(frames=800..1599, phase="mid")
Phase 3: rig_mgr.add_object_by_template_handle() ×4 (banana,apple)
         → capture_from_traj(frames=1600..2399, phase="after")
```

输出结构:
```
outputs/dynamic_benchmark_v4/
├── rgb/frame_000000.jpg ... frame_XXXXXX.jpg
├── depth/depth_000000.png ...
├── traj.txt (拷贝自原始)
├── camera_log.json (每帧状态: snap 距离, clearance)
└── event_log.json (GT 标注)
```

### event_log.json

```json
{
  "scene": "apt_0",
  "total_frames": 2400,
  "events": [
    {"frame": 800,  "type": "DELETE", "count": 4, "labels": ["bowl","bowl","mug","mug"]},
    {"frame": 800,  "type": "MOVE",   "count": 4, "labels": ["pitcher_base","pitcher_base","foam_brick","foam_brick"]},
    {"frame": 1600, "type": "ADD",    "count": 4, "labels": ["banana","banana","apple","apple"]}
  ],
  "detection_windows": {
    "delete": [850, 1000],
    "move":   [850, 1000],
    "add":    [1650, 1800]
  }
}
```

---

## 6. Pipeline 评估

Pipeline 产出 `frame_metrics.jsonl`（已有 `ghost_object_count`, `moved_this_frame`, `new_candidate_this_frame`）。

评估脚本对比 event_log.json：
- 统计 ghost>0 的帧数（在 delete 窗口内 vs 窗口外误判）
- 统计 moved 事件（在 move 窗口内）
- 统计 new_candidate 事件（在 add 窗口内）

---

## 7. 依赖

- Habitat conda env: `/home/ww/miniconda3/envs/habitat/`
- NVIDIA EGL: `LD_LIBRARY_PATH=/home/ww/tmp/oviovo-pr/nvidia-egl/lib`
- YCB: `/home/ww/vv/habitat_data/objects/ycb/`
- ReplicaCAD: `/home/ww/vv/habitat_data/scene_datasets/replica_cad/`
- Traj: `/home/ww/vv/dataset_editable/replica_12changes_apt_0_2400f_20260306_061703/traj.txt`
- 脚本: 修改 `replica_dynamic_object_orbit_capture.py`
