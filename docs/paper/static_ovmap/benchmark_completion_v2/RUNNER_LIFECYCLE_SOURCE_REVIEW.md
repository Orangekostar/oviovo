# 调度、恢复与发布接入预检（未结项）

2026-10-26，主代理逐段审核 `binding`、`common`、`runner`、`lifecycle`、`runtime`、`work_state`、`refresh`。这些源码和对应三份测试仍与既有97项CPU验证收据一致；本次未重复运行测试、模型或AP。

- 输入限定于显式参数、已锁定路径和声明的父依赖；冻结配置、GPU冲突及恢复时参数变化均检查。
- 子进程保存真实退出和最多两次同身份尝试；失败及缺少必要输出的GPU作业保留费用。独立阶段继续收集结果，部分运行返回20。
- 存在可运行任务时禁止结项。负收益与执行失败分开；缺依赖保持 `PARTIAL_CLOSED`，科学主张独立判定。
- 未改变的终态 `all --resume` 先核验源码、参数和收据，再决定返回，早于科学导入及状态写入。显式资产刷新另走有历史保全的受影响阶段。

源码主审：`$OUT/bootstrap/reboot_20261026/runner_lifecycle_source_primary_review.json`，identity `bf557e3b330d903e71d942450bf2a707ff185093fd4a814f728b2bf82649d7bf`；实际退出0收据 `6bf1f286291b8cc7c7d4b0e92e7dcb02b9bb014793f62df3a01e21ae1db7d0b1`。既有终态CLI测试为合成输入，不能替代本任务发布后的真实验证。本次没有创建 `terminal.json`、最终主审或发布事务。

发布候选v8已有无需 `gh` 的公开HTTPS读取路径。本机实际调用候选原函数，在父提交 `9bbe040570ecbaec854e6d8aa94ba82a3a1afa2a` 读取 `src/static_ovmap/semantic_update_evidence_v1/calibration.py`，3326字节与本地SHA一致。证明 `publication_public_HTTP_transport_review.json` identity `62aceaf926b4aa10b71f449a6ab0ea8dd962d7b74e86948f9e2a7595a50ccc3f`，实际退出0 `401f67f5438f5e191a4095b6e99b0ad83b4582c4e3565aff3eb6168ede28a180`。

该网络验证只证明已有公开文件可回读；尚未验证实际push或C1新增数值结果。候选仍未应用，完整场景、正式统计、最终17章审核、C1/C2及发布后真实终态验证继续等待各自前置完成。
