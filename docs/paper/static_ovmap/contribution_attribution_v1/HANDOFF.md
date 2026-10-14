# OVMap contribution attribution handoff

REPO: /home/ww/crove/ovmap-contribution-attribution-v1/worktree
OUT: /home/ww/crove/ovmap-contribution-attribution-v1/attempt_001
Branch: research/ovmap-contribution-attribution-v1
Control Python: /home/ww/miniconda3/envs/ovimap-map/bin/python
FC Python: /home/ww/miniconda3/envs/oviovo-radseg/bin/python
GPU: 0; inherited queued lock /mnt/shared/ww/ovimap-module-validation-v1/.visual-gpu-0.lock

```bash
cd /home/ww/crove/ovmap-contribution-attribution-v1/worktree
/home/ww/miniconda3/envs/ovimap-map/bin/python -u scripts/evaluation/run_ovmap_contribution_attribution_v1.py --spec configs/static_ovmap/contribution_attribution_v1.json --parent-root /mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001 --output-root /home/ww/crove/ovmap-contribution-attribution-v1/attempt_001 --gpu 0 --phase all --resume
```

普通权重锁：[0.0, 0.5, 0.5]；selection_role=FUSION_DEV。
完成/缺失状态：{'execution_status': 'COMPLETE', 'evidence_status': 'ATTRIBUTION_COMPLETE_EXPOSED_ONLY', 'publication_status': 'NOT_ATTEMPTED_SNAPSHOT', 'submission_evidence_complete': False, 'missing': []}
科学子进程已观察退出：0。历史失败见stage_exits，最终成功阶段另列；没有预写父all退出。
Frozen heads出自FG_BASE bec4176d9a6784a89ea293a9803e5cf6fab178ec，没有新模型权重。完整matches仍在OUT；Git PR充分统计可复核AP，bootstrap需要完整matches，不以精简字段冒充可重做几何匹配。

发布收据边界：C1发布后写PUBLICATION_RECEIPT.json；C2和最终all真实退出只写OUT外部。当前报告快照不预写未来SHA。

表格重建：使用相同binding和冻结输出运行 --phase report；不会调用head或评分。数据角色、直接输入、实际hash和机器结果见artifact_index和experiment_ledger。部署N0_UNCHANGED，confirmation=NOT_RUN_BY_CONTRACT。

## C1 publication verified

C1: 49857195f32123e9726e2643c670300288aa817b

[RESULTS](https://github.com/Orangekostar/oviovo/blob/49857195f32123e9726e2643c670300288aa817b/docs/paper/static_ovmap/contribution_attribution_v1/RESULTS.md) · [HANDOFF](https://github.com/Orangekostar/oviovo/blob/49857195f32123e9726e2643c670300288aa817b/docs/paper/static_ovmap/contribution_attribution_v1/HANDOFF.md)

C1普通push及完整remote SHA核验已完成。此收据不预言C2；C2真实核验记录在OUT/publication/final.json。
