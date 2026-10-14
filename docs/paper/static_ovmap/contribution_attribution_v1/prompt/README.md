# OVMap 贡献归因：Codex执行包

将整个目录交给能够读取原仓库和服务器缓存的 Codex，入口为 **00_给Codex的最终执行指令.md**。不要混用旧版方案中的可选解释；本包主文件给出数值与状态合同，02为一致的机器配置。

## 文件

- `00_给Codex的最终执行指令.md`：开发、数据绑定、12主配置、权重选择、归因/统计/定性、提名与发布的完整命令。
- `01_源码证据与任务绑定.md`、`source_evidence_manifest.json`：35项固定来源、读取范围、函数与判断。
- `02_冻结执行配置.json`：固定角色、场景、方法、选择规则与资源上限；复制为仓库新config。
- `03_交付前审核报告.md`：已修正的歧义及当前真实审核范围。
- `04_合同自检.py`、`contract_checks.json`：不依赖模型或网络的合同检查；仅证明定义/算术/文件关系，不证明真实实验。
- `PACKAGE_MANIFEST.json`：本包文件bytes/SHA256，不包含自身以避免循环。

运行包级检查：

```bash
python 04_合同自检.py --package-root . --output contract_checks.json
```

在Codex实现并检查新入口后，使用真实变量运行：

```bash
"$PY_CTRL" -u scripts/evaluation/run_ovmap_contribution_attribution_v1.py   --spec configs/static_ovmap/contribution_attribution_v1.json   --parent-root "$FG_ROOT" --output-root "$OUT" --gpu "$GPU"   --phase all --resume
```

上述仓库入口是**待开发的契约**，不是本包提供了已接入仓库的可执行分析器。主指令明确缺依赖分支和源路径解析。

**本轮不新增训练、RGB特征、SAM或地图构建。**通过原缓存完成分析和有限选择；后续12家族确认只登记，不自动运行。不改变N0部署。用户已要求正常GitHub研究分支发布，使用C1结果提交、C2收据提交，最终状态留OUT。

本包交付不是科研结论：当前未运行新的服务器分析、未复算真实指标、未创建或推送新研究分支。
