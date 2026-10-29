# 90场景完整分类分数验收

当前本地完整分类分数覆盖90个ScanNet场景、720份200类矩阵，共53880个owner-method行：49712行有定义、4168行无定义。此处为分类分数缓存范围；已审核配对池化表仍为共同82场景，来源/条件诊断84场景，公开包仍47场景/45场景表格。完整312场景及Replica8场景尚未完成。

本次仅新增协议第88–90项的24份矩阵、2072个owner-method行和414400个类别分数位置，其中1800行有定义、272行无定义。范围固定为前90份预测收据，运行中的后续场景不自动纳入。

| 场景 | owner | F可用 | MF关闭unknown | MF开启unknown |
| --- | ---: | ---: | ---: | ---: |
| scene0432_01 | 21 | 14 | 7 | 1 |
| scene0608_00 | 117 | 98 | 19 | 10 |
| scene0608_01 | 121 | 100 | 21 | 10 |

每场景包含U00/U10/U01/U11与种子17/29的8份矩阵。数值复算沿用原稳定softmax、原共享angle_cap和逐owner独立循环，三段代码经AST核对一致；新矩阵在1e-12绝对/相对容差内一致。标签、confidence、exported、owner及类别轴与原冻结预测完全一致，无定义行保留NaN分数、defined=false和label=0。

此次缓存物化未访问GT标签或原始几何、未启动模型/训练/评分worker。50份既有GT收据只在准备及主审进程中盲核对SHA/mtime；原数值进程的访问守卫保留。数值进程核对3511份输入及既有证据不变，1392份原矩阵文件SHA/mtime不变；重复准备1441份当前文件及汇总保持原SHA/mtime。

| 实际步骤 | 子进程退出 | 收据identity |
| --- | ---: | --- |
| 准备 | 0 | `ec3a77656dbb4401e0bbfb57880b0d84a9d317ec76851edab1411b7d505550a6` |
| 数值复算 | 0 | `fbeca4860d37e25004ff429db715a8ef051965630bc0f2ad2e23190b00e075f0` |
| 主审 | 0 | `6bec5020668c2af9a16072c147936a5afca3f7fe9fda12805e7ab450a10af357` |

准备证明：`review/G4_90scene_class_score_preparation_primary_review.json`，`633d43fdd3d578ea14843dd6c1c5833c7086dd5157ad2e6e377f15ca1e876bc4`。

数值证明：`review/G4_full_scores_scenes88_to90_actual.json`，`0a9f234267e63338bbb6ad3ea3d08b11cf44a7bb8f64fe274bb8df37d2b56d73`。

主审证明：`review/G4_full_scores_scenes88_to90_primary_review.json`，`5c886488d6ce9798fb01eb87961ddbbaa5b30c6d990d54b5a7acf90af0682df1`。主审核对全部720份矩阵的manifest、数组SHA/字节数、原预测来源、完整owner域及方法/种子键；旧696条汇总记录原样保留。主审未重算AP。

当前汇总：`publication/class_scores.json`，`60f905115d22de97c636b10f76a553157a5d6ddc892857da8e9e8dfdd940f7b2`。

旧87场景汇总`24db9257e1249dd909c58ed62711cc2a0858328f1d7b8b2755f601500509a66d`已普通复制归档至`review/history/class_scores_87scenes_24db9257e1249dd909c58ed62711cc2a0858328f1d7b8b2755f601500509a66d.json`，SHA、mtime与旧1752份基线一致，归档使用独立inode。旧1752份基线文件保留为历史；唯一授权变更的既有输出为分类汇总87→90，其余1751份旧路径及93份原冻结文件保持SHA/mtime。新增48份矩阵文件并验收后生成新1800份稳定基线：`bootstrap/before90_report_primary_review.json`，`9eec8642bf8eeabed68d8bde05589a5085e63be30b235ca7ecdc72c3ccbcc0cd`。后续接续检查使用此新基线，不覆盖原历史记录。

执行源码为`bootstrap/prepare_class_scores90.py`、`validate_full_scores_scenes88_to90.py`、`review_full_scores88_to90.py`、`execute_class_scores90.py`，沿用已通过3项控制的`snapshot_score_scope.py`；精确源码SHA、实际命令和日志SHA见准备及真实退出收据。它们是一次性证据入口，不重跑已完成入口。

双卡科学队列继续，本次矩阵物化新增GPU推理0。显存观察器仍等待第91项整场终态；本次数值证据不证明效率、显著性、完整benchmark优势或在线30FPS。完整任务的来源包刷新、最终提示词逐项审核及C1/C2发布尚未完成。
