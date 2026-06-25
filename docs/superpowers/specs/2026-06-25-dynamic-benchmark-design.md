# 动态 Benchmark 设计文档

日期: 2026-06-25
状态: 设计完成，待实现

---

## 1. 目标

**定性验证**——TSDF 驱动的三种判决（DISAPPEARED / MOVED / NEW OBJECT）在 ReplicaCAD 真实渲染数据上正确触发。

不做定量对比（vs 时间信号方法），不覆盖全部 91 个场景。用 `apt_0` 12 变更事件验证三种判决类型均可被 pipeline 正确检测。

---

## 2. 方案总览

```
Scene:      apt_0 (ReplicaCAD)
Trajectory: 复用已有 2400 帧轨迹
变更:       12 事件，每 150 帧触发，4 DELETE + 4 MOVE + 4 ADD

Timeline:
Frame     0  ████ static baseline (150 frames)
Frame   150  DELETE #1 (lamp_02)
Frame   300  DELETE #2 (lamp_01)
Frame   450  DELETE #3 (chair_01)
Frame   600  DELETE #4 (chair_05_1)
Frame   750  MOVE   #1 (chair_05_0)
Frame   900  MOVE   #2 (table_02)
Frame  1050  MOVE   #3 (chair_04)
Frame  1200  MOVE   #4 (table_04)
Frame  1350  ADD    #1 (bowl)
Frame  1500  ADD    #2 (bowl)
Frame  1650  ADD    #3 (cup)
Frame  1800  ADD    #4 (cup)
Frame  1800  ████ observation buffer (600 frames)
Frame  2400  END
```

---

## 3. 两阶段数据 Pipeline

### Stage 1: Habitat 渲染（只跑一次）

```
event_plan.json ──→ replica_dynamic_rgbd_generator
                     │
                     ├─ apt_0 场景 + 物体操作
                     ├─ 已有 2400 帧轨迹
                     ├─ 每 150 帧执行一个事件
                     └─ 输出: rgb/ + depth/ + traj.txt
                               + changes.json
                               + event_log.json (GT 标注)
```

**event_log.json 格式**（Stage 1 产出，Stage 2 消费）：

```json
{
  "scene": "apt_0",
  "total_frames": 2400,
  "events": [
    {
      "event_id": 1,
      "frame": 150,
      "type": "DELETE",
      "object_handle": "frl_apartment_lamp_02_:0000",
      "old_position": [0.65, 1.14, 2.18],
      "expected_judgment": "DISAPPEARED",
      "detection_window": [200, 350]
    }
  ]
}
```

### Stage 2: Pipeline 评估（可重复跑）

```
rgb/depth/traj ──→ OVIOVO Pipeline
                    │
                    ├─ SAM2 proposals
                    ├─ TSDF integration
                    ├─ Dynamic Maintenance (TSDF 驱动)
                    └─ 输出: frame_metrics.jsonl
                              + event_log.json 对比
                              → benchmark_report.json
```

---

## 4. 事件设计

事件复用 apt_0 已有 12 变更的 object handle 和位置，增加帧号：

| Event | Frame | Type | Handle | Expected Judgment |
|-------|-------|------|--------|-------------------|
| 1 | 150 | DELETE | lamp_02_:0000 | DISAPPEARED |
| 2 | 300 | DELETE | lamp_01_:0000 | DISAPPEARED |
| 3 | 450 | DELETE | chair_01_:0000 | DISAPPEARED |
| 4 | 600 | DELETE | chair_05_:0001 | DISAPPEARED |
| 5 | 750 | MOVE | chair_05_:0000 | MOVED |
| 6 | 900 | MOVE | table_02_:0000 | MOVED |
| 7 | 1050 | MOVE | chair_04_:0000 | MOVED |
| 8 | 1200 | MOVE | table_04_:0000 | MOVED |
| 9 | 1350 | ADD | bowl_06_:0001 | NEW OBJECT |
| 10 | 1500 | ADD | bowl_03_:0000 | NEW OBJECT |
| 11 | 1650 | ADD | cup_05_:0000 | NEW OBJECT |
| 12 | 1800 | ADD | cup_02_:0001 | NEW OBJECT |

检测窗口：事件发生后 50-200 帧（给 SAM2 + TSDF 积累足够的观测窗口）。

---

## 5. Pipeline 改动

### 5.1 frame_metrics.jsonl 新增字段

在 `main_pipeline.py` 的 `last_frame_debug` 中新增：

```python
"ghost_object_count": int(
    sum(1 for o in state.objects.values() if o.state == ObjectState.GHOST)
),
"moved_this_frame": [
    int(oid) for oid in ...  # 本帧被判 MOVED 的 object_id
],
"new_candidate_this_frame": [
    int(pid) for pid in ...  # 本帧被提升的 candidate provisional_id
],
```

### 5.2 Stage 1 脚本

修改 `habitat-data-collector/scripts/20260227/replica_dynamic_rgbd_generator.py`：
- 接受 `--event-plan` JSON 文件参数
- 在指定帧执行增删移物体操作
- 导出 `event_log.json`（含帧号和 GT 标注）

### 5.3 评估脚本

新增 `scripts/eval_dynamic_benchmark.py`（~80 行）：
- 输入: `event_log.json` + `frame_metrics.jsonl`
- 输出: `benchmark_report.json` + 终端可读报告

---

## 6. 评估指标与报告

| 指标 | 计算方式 |
|------|----------|
| 检测正确率 | GT 事件在窗口内被正确判决 / 12 |
| 误判数 | 非变更帧产生虚假 GHOST/MOVED/NEW 的次数 |
| 平均检测延迟 | 检测帧号 - 事件发生帧号 |

**输出报告格式**：

```
Dynamic Benchmark Report — apt_0 2400f 12 events
══════════════════════════════════════════════════
Event  Type    Frame  Detected  Delay   Result
────── ──────  ─────  ────────  ─────   ──────
  1    DELETE    150    Frame 200  50f    ✅
  2    DELETE    300    Frame 350  50f    ✅
  3    DELETE    450    Frame --    --     ❌ (未检测)
  ...
──────────────────────────────────────────────────
Recall:  11/12 (91.7%)
False +: 0
Avg delay: 47 frames
```

---

## 7. 边界情况

- 物体删除后 TSDF 所有权自然衰减需要时间——检测窗口下限 50 帧给衰减缓冲
- MOVE 事件后相机可能已离开新位置——需要保证轨迹在新旧位置都有覆盖
- ADD 事件后 SAM2 需要几帧才能稳定产生该物体的 proposal——窗口下限 50 帧

---

## 8. 不变更范围

- OVIOVO pipeline 核心逻辑不改（仅 frame_metrics 加 3 个字段）
- TSDF maintenance 判定逻辑不改
- habitat-data-collector 架构不改（仅加 event_log 导出 + event plan 参数）
