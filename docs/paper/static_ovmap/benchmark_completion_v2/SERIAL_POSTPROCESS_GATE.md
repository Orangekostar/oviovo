# 原后处理的写入进程等待门

2026-10-24 16:51 UTC当前状态：NFS恢复后，旧等待程序2811638及其监督器已终止，原等待程序真实退出1；下文为中断前的历史运行记录。新的本机优先监督器5484/start64581752与等待程序5485/start64581758已接管同一个原后处理3705706/start43719637，后处理仍为无子进程的T状态。新等待程序未发送CONT；双卡协调器4177429/start64497744继续运行。恢复主审真实退出0，见[NFS_RECOVERY.md](NFS_RECOVERY.md)。

新门同时等待全部原/新生产者结束，并验证完整312场景、并行实际0、原流控制器实际-15、新辅助实际0或20及93份冻结文件，才恢复同一个原后处理。不得手动恢复原控制器或后处理，不得重跑旧one-shot。旧`serial_postprocess_gate/actual_exit.json`中的退出1是已封存历史，当前新门状态位于`bootstrap/nfs_recovery_20261024/gate/progress.json`；其真实终态将先写本机`recovery/nfs_outage_20261024_0645/roles/gate/actual_exit.json`，再同步共享输出。运行中尚无实际退出回执属正常。

当前只读检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. $LOCAL_ASSET/python3 - <<'CHECK'
from pathlib import Path
from static_ovmap.benchmark_completion_v2.common import verified
root=Path('$OUT')
local=Path('$LOCAL_ASSET/gate')
launch=verified(local/'launch.json')
for name in ('supervisor','child'):
    pin=launch[name]
    stat=Path('/proc')/str(pin['pid'])/'stat'
    if stat.exists():
        row=stat.read_text().rsplit(') ',1)[1].split()
        print(name,pin['pid'],row[0],row[19], 'SAME_START',row[19]==pin['start'])
    else:
        print(name,pin['pid'],'TERMINAL_OR_ABSENT')
p=root/'bootstrap/nfs_recovery_20261024/gate/progress.json'
d=verified(p)
print('progress',d['status'],d['actual_signals_sent'],d.get('live_producers'))
p=local/'actual_exit.json'
print('actual_exit',verified(p) if p.exists() else '尚未观测到真实退出')
CHECK
```

以下保留原门的设计、测试和历史挂接证据。

原冻结postprocess只有一个等待条件alive(marker)，marker指向旧流式监督进程3631603；它不显式等待独立OVI评分写入进程。双卡队列在完成312个自有场景后会退役旧控制器，两个时点不构成OVI已结束的保证。本次通过任务内等待程序补齐顺序，保留原后处理PID及五个阶段，原科学源码和调度源码均未修改。

实际运行：系统Python3.12.3；监督进程2811637/start58088229，等待进程2811638/start58088236。等待源码SHA `13fdaaec978ad5ec5c709ab86c916bb661ed3a4d63f0c91c0034046c6b861897`，监督源码SHA `1d57969cbdbc9fb8b50747f2e135677d7253bcad18bec6d9d5e2706747b970f1`，位置均为OUT/bootstrap/serial_postprocess_gate。当前等待进程持有指向原后处理3705706/start43719637的内核pidfd，原后处理无子进程、内核状态T；仅发送过SIGSTOP。原控制器3631604/start43301917也按既有双卡接管方案保持T。两个T状态均为有意等待，不能直接手动CONT或重启。

等待条件为旧流式控制器及监督器、双卡协调器及监督器、独立OVI评分器、元数据审计及监督器、容量监测及监督器共9个已固定PID/start的进程全部终止。然后核对完整312场景scan结果、真实协调器退出0、原控制器实际退出-15及原命令，复核93份冻结文件，再通过保留的pidfd恢复同一个原后处理。旧OVI/审计的部分退出仍保留，缺失末尾缓存交给原baselines处理；不在旧写入者仍存活时另起作者预测。

7项控制测试通过，包含真实隔离子进程的pidfd暂停/恢复、陈旧PID拒绝及有活跃写入者时不放行。第一次RED是新增等待模块尚不存在的7项错误，不是科学实验或历史竞争重现。旧Conda环境缺少pidfd API的失败与一次入口身份抄写漏一位的预检失败均保留；未改动任何真实入口身份。校正后同样7项测试及真实只读预检通过。最终测试收据 `bootstrap/serial_postprocess_gate/green_system_v2.json` 身份 `82709a8414296bb4f91ad4ae6a3dc92e28b6ce17c85c96e58a344f5153817907`，预检实际退出0 `preflight_retry_actual_exit.json` 身份 `e19f39577ed7180ffa2a2b14bf47926baf09c2005f83c6e42cd3b3178cb95059`。

入口主审 `review/G4_serial_postprocess_gate_entry_primary_review.json` 身份 `47e49581dd5b42c35d7bb11b896527074ea46d6570e5f053c47f49a23a4bcbee`，主审实际退出0身份 `f823408e04ba011bfe53ff8012dd0ec30fdb938509a8a748b93df0ea7112a8b6`。实际挂接核验读取/proc确认监督器、等待程序、原后处理的PID/start、argv、子进程和内核pidfd目标；主审 `review/G4_serial_postprocess_gate_launch_primary_review.json` 身份 `c8c63aef2b3a8499a5af66b72c4d93445b204476ef26a4628710705d907221cd`，实际退出0身份 `47ac07fe699b664ec2da0b4f56dc85bf84aa129193a5252a18c42e4160ed068c`。双卡协调器仍正常运行，未暂停科学worker。

目前仅证明等待程序已挂接和原后处理实际暂停。完整队列尚未结束，真实SIGCONT、原五阶段最终退出和等待程序最终退出都尚未发生，不据此宣称最终问题已全部解决。若等待程序失败，它记录真实退出并保留已暂停的后处理，由主代理核对原因后恢复；不自动重新派发或回填成功。

只读检查（在本任务worktree）：

```bash
PYTHONPATH=src:. $LOCAL_ASSET/python3 - <<'CHECK'
from pathlib import Path
from static_ovmap.benchmark_completion_v2.common import verified
p=Path('$OUT/bootstrap/serial_postprocess_gate')
for name in ('child.json','progress.json','actual_exit.json'):
    f=p/name
    print(name, verified(f) if f.exists() else '尚未结束')
CHECK
```

1640份保护文件及93份冻结文件的SHA和修改时间、公开47场景索引SHA均核对不变。最终C1前的有限执行来源清单必须补入本等待程序及真实最终收据，并等待它、监督器与原后处理都结束；v8补丁仍未应用，完整提示词终审和C1/C2仍待执行。

另有协议第91场景的有限只读显存观察器，见[MEMORY_PROBE91.md](MEMORY_PROBE91.md)。它只读作业与NVML，不写科学结果；九个生产进程的原放行条件保持不变。最终发布证据封存前，还需该观察器及其监督器退出并核验真实收据。不要把其入口通过当成完整显存测量完成。


2026-10-24：额外只读的第91项显存观察器2914903/start58374979及监督器2914902/start58374973已实际退出0并内核终止，独立复算及资源表主审完成，见MEMORY_PROBE91.md。它们的完整来源/终态可纳入最终有限清单；原9个生产进程的等待及实际312场景放行条件继续按既有门控执行。

2026-10-24源码顺序复核：原后处理仅调用benchmark_score/statistics/baselines/tables/review，最终记录`publication_not_attempted=true`；发布仍需独立的主代理逐项审核。后处理退出后，再集成发布v8与资源身份候选、重建报告，并让最终主审绑定新整体producer及最终result_store，之后执行C1/C2及实际终态恢复。当前terminal、发布事务/final与primary_final均不存在，未提前升级任何最终完成状态；详见[RESOURCE_RUNTIME_METADATA.md](RESOURCE_RUNTIME_METADATA.md)。
