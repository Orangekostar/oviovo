# OVI-MAP 三模块验证：最终 Codex 执行包

版本：2026-09-20-v1。

## 使用方式

将本目录整体提供给目标服务器上的 Codex，要求读取 **`CODEX_FINAL_EXECUTION_EN.md`** 并执行。该文件是00–05六份英文规格的逐字串接，单文件版与分文件版完全等价，不是两套方案。执行常量来自 `PROTOCOL_SPEC.json`；源码依据见 `SOURCES.md`；交付前审核见 `CODE_REVIEW_AND_AUDIT_ZH.md`。

建议启动语句：

```text
Read CODEX_FINAL_EXECUTION_EN.md and execute the complete scoped study in
Orangekostar/oviovo. Use PROTOCOL_SPEC.json to produce a resolved runtime
configuration; do not send the specification directly to historical runners.
Implement all three modules, run the independent experiments supported by
actual assets, select models/modules only on the specified development splits,
and complete the frozen confirmation when eligible. Commit and push the scoped
code, configuration, small trained heads, actual results and both Markdown
reports, then verify the remote SHA. Do not stop at another plan, run unrelated
safety regressions, or report missing experiments as completed.
```

## 本轮到底执行什么

| 模块 | 首版实际接入 | 实验目标 |
|---|---|---|
| S 语义 | 原生ROI，SigLIP/SigLIP2/WOW；固定建议集合的KEEP/REPLACE小模型 | 区分新语义信息、额外监督、对象/背景证据的收益 |
| G 几何 | 原生数值片段/表面导出；完整划分侧挂模块 | 固定候选集合，比较旧一致性和学习到的完整性评分 |
| Q 查询 | 当前帧原生地图快照；视觉编码前的查询排序 | 同预算上限下比较combine、面积、旧不确定性和预测收益 |

G首版是静态精炼导出，不声称已实现在线可逆所有权。Q首版是因果回放和预算内排序，不搬入完整RL。WOW/LEGO等名字不是我们的新增贡献。原生主干和历史结果不覆盖。

## 数据与工作量边界

默认12个合法已取得的独立ScanNet训练场景家族：8FIT/2CAL/2SELECT；另2个未用于本研究选择的验证家族作确认。每次新原生捕获至多200个计划帧，独立捕获上限14；必要的历史Room0核对重放至多1次，单列。没有独立数据就明确阻塞学习验证，不用Room0帧/crop冒充独立场景。

每场景语义至多128个目标×3视图；全部地图实例仍被评估，预算外目标留原判断并披露覆盖率。几何至多64组×8完整假设。Q主预算200次六crop原生查询，符合条件才补100/400预算曲线。只跑预定消融，最多2种逐步组合，不跑3模块巨大析因矩阵。

本包不是已经可运行的算法仓库，也没有附带模型权重。它指定Codex必须开发的接口、算法、训练/选择/评测与推送步骤。当前生成端未运行新算法、未验证服务器数据读取、未推送GitHub。

## 产物

- `docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md`
- `docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md`
- `artifacts/static_ovmap/module_validation_v1/` 的真实小结果与训练/选模记录
- `third_party_patches/ovimap/module_validation_v1/` 的官方代码补丁
- 任务分支 `research/ovimap-module-validation-v1`，实际推送后核验SHA

负结果也必须发布；只有代码完成而关键实验未测时，必须分别报告代码完成和实验阻塞，不能写成整项科学验证完成。任务包完整性检查与算法测试是不同事情。
