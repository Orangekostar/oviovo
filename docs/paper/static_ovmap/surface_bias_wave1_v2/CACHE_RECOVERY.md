# 缓存恢复与完成状态：2026-10-14

原缓存仍在node101。共享盘根目录可访问，但本机容量子挂载未启动，
两个实验目录的软链接因此指向空挂载点。启动日志确认：嵌套automount
要求先挂载网络父目录，同时又要求早于local-fs.target，产生启动依赖循环。

已启动既有node101挂载，并备份本机fstab后，移除五项容量子挂载的
`x-systemd.automount`，改由remote-fs.target启动。生成的挂载单元校验通过；
未进行实际重启测试。配置备份：
`/var/backups/ovimap-cache-recovery-20261014/fstab.before`。
详细本机恢复收据保留在OUT的`storage_recovery/20261014.json`。

两份关键父清单与Git归档逐字节一致。没有重建、重新编码或替换缓存。
正式preflight缺失依赖为0；真实FC parity通过，8项测试通过、0项跳过。
原`all --resume`进程实际返回0。

机制诊断覆盖622个TRAIN、69个DEV原件的四种条件，共2,764条记录。
clean条件M1的家族均值R：TRAIN为-0.171644，DEV为-0.132550，
可评价覆盖分别为177个对象/24家族及29个对象/4家族，未满足正信号要求。
同支持的对照见[配对诊断表](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/tables/table2_paired_clean.md)。

| 方法 | 锁定lambda | seed17的A增量（百分点） | seed29的A增量（百分点） | seed17/29净校正数 |
| --- | ---: | ---: | ---: | ---: |
| C1：视图中心化 | 0.125 | +0.1042 | +0.9280 | +1 / -1 |
| C2：位置Huber共识 | 0 | 0 | 0 | 0 / 0 |
| M1：对应图校正 | 0 | 0 | 0 | 0 / 0 |

A沿用父实验的家族等权、四条件加权定义。三个方法均未达到固定的
双种子门槛；已完成2/4视角检查，地图阶段为`NOT_ACTIVATED_BY_FIXED_RULE`。
科研状态为`NO_HELDOUT_SIGNAL`，部署保持N0。本轮新增优化器更新、
RGB编码、SAM提议均为0；存在对缓存特征的GPU重放和phi调用。

正式数值结果首先发布于C1
`7c5aa3cd00f29c2308e8d78d8ae02f117054bd05`，对应worker源码摘要为
`7e981eb15271e3fde205289d331745605dab733aaf787660dc4eac1cc608a0cc`。
随后只修订报告模板中的旧受阻文字，并更新文档、审核和恢复记录。
主代理核对：其它生产文件及配置逐字节不变，reporting.py仅字符串常量改变；
没有重算数值阶段。worker退出记录保留实际生产源码摘要。

主代理独立重算176项N/D、覆盖、配对支持和家族均值，全部与正式JSON一致；
同时核对12个seed17候选、3个固定seed29复验及12条前缀结果。
见[正式审核收据](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/primary_review.json)
和[恢复摘要](../../../../artifacts/static_ovmap/surface_bias_wave1_v2/cache_recovery.json)。

结论限于已曝光场景与父实验的GT实例提议。此次校正未验证有效，
没有新增整图结果，也不支持OVI-MAP整体超越或在线30 FPS主张。
