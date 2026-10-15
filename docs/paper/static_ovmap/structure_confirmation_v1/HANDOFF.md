# FG2结构验证与确认交接

当前状态：COMPLETE。

```bash
cd "/home/ww/crove/fg2-structure-confirmation-v1/worktree"
/home/ww/miniconda3/envs/ovimap-map/bin/python scripts/evaluation/run_fg2_structure_confirmation_v1.py --output-root "/home/ww/crove/fg2-structure-confirmation-v1/attempt_001" --phase all --resume
```

--parent-root仅代表FG_ROOT；继承parent_root与lr_source_root语义未改。实际FG_ROOT=`/mnt/shared/capacity/node101/ww/ovimap-frozen-fc-routing-v1/attempt_001`，LR_ROOT=`/mnt/shared/capacity/node101/ww/ovimap-learned-object-readout-v1/attempt_001`。原数据与训练缓存只读；新数据和断点保存在独立OUT。小头/无损cosines索引位于artifacts/static_ovmap/structure_confirmation_v1，architecture.json与weights/manifest.json提供结构、参数与加载接口。

SAM子进程通过本任务sam_runtime.py隔离nvJitLink，不修改原环境。完整终态续跑验证数据/源码与远端身份后不启动GPU worker、不新增head或提交。发布收据生成后，本文件追加C1永久链接；C2精确SHA保存在外部OUT/publication/final.json，避免自引用提交。

加载公开model-only小头并验证原权重字节（在仓库根目录执行，不启动GPU）：

```bash
PYTHONPATH=src:. /home/ww/miniconda3/envs/ovimap-map/bin/python - <<'PY'
from pathlib import Path
from static_ovmap.structure_confirmation_v1.common import verified
from static_ovmap.structure_confirmation_v1.export import load_export
folder = Path("artifacts/static_ovmap/structure_confirmation_v1")
manifest = verified(folder / "weights/manifest.json")
key = manifest["profiles"]["S_LL:seed17:selected"]
head, state = load_export(folder, key, device="cpu")
print(state["method"], state["seed"], state["step"], key)
PY
```

该加载接口恢复小头；实际前向仍需锁定FC的phi及精确七张量输入。基础FC权重和原始ScanNet数据未上传。

C1：`94d0cab99bd55dfb46b9a401f9a560b9a78832e5`

[RESULTS](https://github.com/Orangekostar/oviovo/blob/94d0cab99bd55dfb46b9a401f9a560b9a78832e5/docs/paper/static_ovmap/structure_confirmation_v1/RESULTS.md) · [HANDOFF](https://github.com/Orangekostar/oviovo/blob/94d0cab99bd55dfb46b9a401f9a560b9a78832e5/docs/paper/static_ovmap/structure_confirmation_v1/HANDOFF.md)
