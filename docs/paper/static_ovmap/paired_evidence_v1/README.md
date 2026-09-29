# OVI-MAP paired evidence study — Codex package

本包生成于2026-09-29，基于已核对的 `Orangekostar/oviovo@9c35333088551a9d4a67937936d4d9c347b2b207`。

将整个目录交给Codex，以 `CODEX_FINAL_EXECUTION_EN.md` 和 `PROTOCOL_SPEC.json` 执行开发、实验、选模、报告及GitHub发布。不要直接重跑旧wave-1的`all`来代替新任务。

| 文件 | 用途 |
|---|---|
| CODEX_FINAL_EXECUTION_EN.md | 唯一主执行指令，包含数学定义、真实接入、执行顺序、缺失处理及发布 |
| PROTOCOL_SPEC.json | 固定方法、场景、参数、预算、选择、记录数量 |
| CODE_REVIEW_AND_AUDIT_ZH.md | 中文方案、源码依据、修正内容与验收边界 |
| MATH_AND_EDGE_CASES.md | 数学性质、推导、假设、退化情况、哪些不能宣称 |
| math_reference.py | 无数据、无网络的合成算子参考检查；不是生产实验实现 |
| MATH_CHECK_RESULTS.json | 本次已运行的合成数值检查，零benchmark评价 |
| SOURCES.md | 固定提交源码链接及相关数学前作 |
| PACKAGE_AUDIT.json | 指令包与数值检查的实际审核记录 |
| FILE_MANIFEST.json | 本包文件字节数及SHA256 |

新任务分支：`research/ovimap-paired-evidence-v1`。

不新增视觉推理，不改变几何。先做配对残差和依赖融合；条件信息查询、E06、新模型或几何模块不在本次授权内。新方法失败不意味着任务失败，但必须交付真实可执行部分和实测结果。

本包不是已开发完成或已跑出新指标的证明。当前只完成源码复核、执行规格及合成数学检查；生产适配、场景实验和推送由Codex执行。
