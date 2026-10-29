# 87场景完整分类分数验收

2026-10-23：本地完整分类分数已通过实际数值验证和主审，覆盖协议前87个ScanNet场景、696份200类矩阵，合计51808个owner-method行，其中47912有定义、3896显式无定义。配对池化报告仍为已审核的82场景，来源/条件诊断84场景，公开包47场景；此次没有重新计算AP、训练或模型推理。

第81–87项新增56份矩阵、8328个owner-method行和1665600个类别分数，其中7884行有定义、444行无定义。两种子、四方法的完整新矩阵均按原始共享angle_cap、稳定softmax及等权N/Q/F分组独立重建，容差为atol=rtol=1e-12；原owner、类别轴、标签、置信度、导出标志和mask身份保持一致，MF关闭时N/Q访问为0。

后台可能继续生成后续场景，因此此次在进程内固定前87份预测收据。原数值生产模块未修改；只限制该次调用的收据枚举，调用结束即恢复Path。3项控制测试覆盖晚到场景排除、异常恢复和无效范围拒绝，真实退出0，收据为`bootstrap/snapshot_score_scope_green_actual_exit.json`（`f02e47c78c31d18b949c39b55306998678e3c3fe1db5284ccd5c8b5556c39ee3`）。

首次数值入口在最终文件保护检查中失败：新增的保护基线包含50份GT元数据收据，其哈希读取被原有GT访问保护拒绝。失败实际退出1（`25ab88a3d7f0a528b0360afd31c44ba79649a08695df4fef63a94b9c972671e5`）及精确源码、日志、准备收据和当时已生成的汇总均保留，10份文件普通复制归档到`failures/class_scores87_guard_scope/archive.json`（`8280363a78b399bd9a38a4a39ff875fa180753b74210388ac75ca83b84b0a69b`）；该失败未被当作数值成功。

恢复入口保留同一访问保护和三段数值/保护AST，将这50份元数据的盲哈希核验放在数值程序之外。数值程序仍禁止GT、原始几何、模型和子作业访问，全部3429份纳入数值保护的文件不变。独立数值循环重新实际执行并通过，原640份矩阵的1280个文件及已生成的1393份当前工件均保持SHA/mtime。

| 证据 | 身份 |
|---|---|
| 当前`publication/class_scores.json` | `24db9257e1249dd909c58ed62711cc2a0858328f1d7b8b2755f601500509a66d` |
| 恢复准备`review/G4_87scene_class_score_recovery_preparation_primary_review.json` | `9a6748e2392716ea4e35ee46223aae855bbd82ae020c2d86bf6895e257a8a4ee` |
| 数值验证`review/G4_full_scores_scenes81_to87_recovery_actual.json` | `32ab38feb429735d7756158dc58514379ee2443e01b4712136aa864507cd87ab` |
| 数值真实退出0`bootstrap/full_scores_scenes81_to87_recovery_actual_exit.json` | `3ddb6ab17b87826abb647194aabf1c62539e3e5f5b0adaf9274b3935dcb5c50d` |
| 主审`review/G4_full_scores_scenes81_to87_primary_review.json` | `059b94266d72149c246e0a4d3e885cc50103cfc9605e77774abcf05bb81340e9` |
| 主审真实退出0`bootstrap/full_scores_scenes81_to87_recovery_primary_actual_exit.json` | `fb6d751f1991263a5e10bd5cb52c69cd9b761490a889c27cb608c47d4ec369e2` |
| 新1752文件基线`bootstrap/before87_report_primary_review.json` | `76ee918e4ca6b25a9d691ecb62108612cf68533aabaf1b33e44d152d7265a60d` |

旧80场景汇总已普通复制归档，保留原字节和mtime。原1640份保护基线作为历史记录不改写：其中1639个路径继续不变，唯一明确授权变化为当前分类汇总。新基线包含这1639个路径、当前汇总及112份新增manifest/数组，共1752份；后续主审应使用新基线。93份原冻结文件继续保持SHA/mtime，正在运行的源码与原测试均未修改。

可用以下命令核对已完成收据，不要重新运行一次性入口：

```bash
OUT=$OUT
cat "$OUT/bootstrap/full_scores_scenes81_to87_recovery_actual_exit.json"
cat "$OUT/bootstrap/full_scores_scenes81_to87_recovery_primary_actual_exit.json"
```

GPU0/GPU2仍各负责一个场景，CPU建图阶段可暂时没有CUDA进程。原后处理等待门继续等待9个固定生产进程和完整312场景的真实退出证据，见[SERIAL_POSTPROCESS_GATE.md](SERIAL_POSTPROCESS_GATE.md)。最终有限发布来源清单还须纳入本批失败归档、恢复源码、真实退出收据及新基线；当前134份隔离来源包仍是其原有范围，尚未更新C1/C2或最终完成状态。
