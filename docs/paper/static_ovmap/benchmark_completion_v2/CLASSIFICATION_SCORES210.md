# 210场景完整分类分数实测

2026-10-26：前210个ScanNet场景已完成1680份方法/种子分类矩阵，共128600个owner行，其中118328行有定义、10272行未定义。行数跨方法和种子累计。当前汇总身份为`22065e8dc9c8524c8acf615fd7efefdb298a4e1145d54911ce6613a3c2341646`。

| 范围 | 矩阵 | owner行 | 有定义 | 未定义 |
|---|---:|---:|---:|---:|
| 原200场景 | 1600 | 123504 | 113680 | 9824 |
| 新增201–210项 | 80 | 5096 | 4648 | 448 |
| 当前210场景 | 1680 | 128600 | 118328 | 10272 |

新增场景为`scene0334_00`、`scene0334_01`、`scene0334_02`、`scene0527_00`、`scene0663_00`、`scene0663_01`、`scene0663_02`、`scene0664_00`、`scene0664_01`、`scene0664_02`，共637个独立owner。四项科学作业、三项有序元数据核验及合并主审均实际退出0；准备步骤固定前210份预测及评分收据，后台第211项及以后的收据不进入本次汇总。208项建图合法跳过677/688帧，计划200帧、capture198帧，三项新核验退出0；该差异保留，见[NATIVE_SKIP_RECOVERY.md](NATIVE_SKIP_RECOVERY.md)，未将缺失帧写成完成。

| 单次执行步骤 | 实际退出 | 进程墙钟（秒） | 实际退出收据身份 |
|---|---:|---:|---|
| 准备及归档 | 0 | 64.2845 | `468025ddcc38be522fb33e611de95bcb725767f7b271e752bc85b04815901fa6` |
| 分类数值校验 | 0 | 95.4814 | `a44f3f3f763bb68e8c4b074f029dba98f4bee2a54792d505c51f3f5fc3e1b6ea` |
| 主审及新基线 | 0 | 21.4666 | `c88525f22a0d55014cf87dc533542bef04060147cbaae087a3e6905299438b14` |

新增5096行的1019200个类别位置完成独立数值校验，绝对和相对容差均为`1e-12`；既有标签、置信度和导出标志一致，原owner及未定义行保留。数值循环、稳定softmax和访问限制三个AST单元与既有入口一致；使用原始共享角度CAP函数。数值进程未访问GT、几何或模型worker，未导入torch/tensorflow。7867份数值输入及既有证据保持SHA/修改时间，新增模型推理、训练更新及AP重算均为0。50份GT元数据由主审单独盲哈希，不解码GT标签。

旧1600条矩阵描述和3200份矩阵文件保持SHA/修改时间，3361份输出只读复用。旧200汇总归档到`review/history/class_scores_200scenes_c542552de344a825e845d793a773b682d00ccda6bd76c5e19551acff9a53ee3e.json`，SHA为`800b9b3513b2b230e698436f6a71706303fe09412f7db6cbdd649a83e0e81fc9`，修改时间`1792975648295725300`纳秒，与原件一致且inode独立。原3560份稳定基线中仅分类汇总路径授权替换，其余3559份旧路径不变；新增160份文件后，新基线共3720份。93份生产/测试/元数据冻结文件保持不变；跳帧恢复904份历史文件/7份源码、gate64份和有限来源198份支持路径均保持SHA/mtime。

可核验记录位于本轮输出根目录：

- `review/G4_210scene_class_score_preparation_primary_review.json`：`e6bac9f47468596286bdee747b7c3420e30d39c90325d1c309ae3036b8a49a60`。
- `review/G4_full_scores_scenes201_to210_actual.json`：`3edffca8d3bd9b3baa0ea179a2e2240b102895f9351cb976e6543dc491f1c5a3`。
- `review/G4_full_scores_scenes201_to210_primary_review.json`：`6b19131eab84950e4721c4f7116e8c47efdf9e85da302a7249b24ec6072a2226`。
- `bootstrap/before210_report_primary_review.json`：`b465bdea0a9cb609f7431d65d2791030a969f4f1f100b2e60228fd10244f8169`，作为后续逐场景主审基线。
- 三步实际退出收据及日志分别保存在`bootstrap/prepare_class_scores210_actual*`、`bootstrap/full_scores_scenes201_to210_actual*`及`bootstrap/full_scores_scenes201_to210_primary_actual*`。

[准备入口](CLASSIFICATION_SCORES210_ENTRY.md)和交接身份`c3e6551292865b2a1c8abaefd2b32472b2698b6dfa6d8c88876377af2341b93c`保留创建时状态；140/150/160/170/180/190/200/210入口各17份支持文件保持原SHA/修改时间。最终有限来源清单须纳入本批脚本、差异、入口、三步记录、日志/实际退出收据、新基线、旧汇总归档及本文。

逐场景主审210/312、1680组；配对池82、条件诊断84、公开47和Replica0/8分别保持原范围。GPU0/GPU2继续后续科学队列。gate适配已审核并启动，尚未真实放行；完整任务、正式统计、最终来源清单/17章审核与C1/C2尚未完成。
