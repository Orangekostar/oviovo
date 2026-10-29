# 200场景完整分类分数实测

2026-10-26：前200个ScanNet场景已完成1600份方法/种子分类矩阵，共123504个owner行，其中113680行有定义、9824行未定义。行数跨方法和种子累计。当前汇总身份为`c542552de344a825e845d793a773b682d00ccda6bd76c5e19551acff9a53ee3e`。

| 范围 | 矩阵 | owner行 | 有定义 | 未定义 |
|---|---:|---:|---:|---:|
| 原190场景 | 1520 | 119072 | 109640 | 9432 |
| 新增191–200项 | 80 | 4432 | 4040 | 392 |
| 当前200场景 | 1600 | 123504 | 113680 | 9824 |

新增场景为`scene0338_00`、`scene0338_01`、`scene0338_02`、`scene0378_00`、`scene0378_01`、`scene0378_02`、`scene0553_00`、`scene0553_01`、`scene0553_02`、`scene0660_00`，共554个独立owner。四项科学作业、三项有序元数据核验及合并主审均实际退出0；准备步骤固定前200份预测及评分收据，后台第201项及以后的收据不进入本次汇总。

| 单次执行步骤 | 实际退出 | 进程墙钟（秒） | 实际退出收据身份 |
|---|---:|---:|---|
| 准备及归档 | 0 | 31.4673 | `346f2d513ece092bcf5226da18a18bd3fda2f4a8a3242851efdd09e5049669f9` |
| 分类数值校验 | 0 | 86.2682 | `a57dfc86d62f7faaa176915ebabc9fa602ac0988fcaf14e225abd8f468140abf` |
| 主审及新基线 | 0 | 35.9935 | `ebced36ce57c30b3248274039ac2a82787e7f8b121510feae568ff6f63cdf76e` |

新增4432行的886400个类别位置完成独立数值校验，绝对和相对容差均为`1e-12`；既有标签、置信度和导出标志一致，原owner及未定义行保留。数值循环、稳定softmax和访问限制三个AST单元与既有入口一致；使用原始共享角度CAP函数。数值进程未访问GT、几何或模型worker，未导入torch/tensorflow。7497份数值输入及既有证据保持SHA/修改时间，新增模型推理、训练更新及AP重算均为0。50份GT元数据由主审单独盲哈希，不解码GT标签。

旧1520条矩阵描述和3040份矩阵文件保持SHA/修改时间，3201份输出只读复用。旧190汇总归档到`review/history/class_scores_190scenes_4ea374b5c6d51864621128a20c9036f2125967dc05b40457c25d74cf1d722892.json`，SHA为`9818bbfa013a40327f56565c22be51638b2bc57a5ad8262945925ca6c4f33120`，修改时间`1792963589985306685`纳秒，与原件一致且inode独立。原3400份稳定基线中仅分类汇总路径授权替换，其余3399份旧路径不变；新增160份文件后，新基线共3560份。93份生产/测试/元数据冻结文件保持不变。

可核验记录位于本轮输出根目录：

- `review/G4_200scene_class_score_preparation_primary_review.json`：`90d4764da58929db523d2a9de8a23594c26ca6e55113451ebfc4294abc4c3ed0`。
- `review/G4_full_scores_scenes191_to200_actual.json`：`17608fb6867a2f4fa50c620316679d9ad638011156dfdfc904e6069f66b61cb1`。
- `review/G4_full_scores_scenes191_to200_primary_review.json`：`74f47fc3b8117c401ee4fde8c0ff494d1ce0822226df2c66ec030ca71a26daeb`。
- `bootstrap/before200_report_primary_review.json`：`acb2fae03644ee38386feaddcba556151439d2c935e6f03c3061327f2b0c6817`，作为后续逐场景主审基线。
- 三步实际退出收据及日志分别保存在`bootstrap/prepare_class_scores200_actual*`、`bootstrap/full_scores_scenes191_to200_actual*`及`bootstrap/full_scores_scenes191_to200_primary_actual*`。

[准备入口](CLASSIFICATION_SCORES200_ENTRY.md)和交接身份`e746f86a3758a94b2ec9f4593161989f93430cad4df766715032bbb9a70283d4`保留创建时状态；140/150/160/170/180/190/200入口各17份支持文件保持原SHA/修改时间。最终有限来源清单须纳入本批脚本、差异、入口、三步记录、日志/实际退出收据、新基线、旧汇总归档及本文。

逐场景主审200/312、1600组；配对池82、条件诊断84、公开47和Replica0/8分别保持原范围。GPU0/GPU2继续后续科学队列。完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。
