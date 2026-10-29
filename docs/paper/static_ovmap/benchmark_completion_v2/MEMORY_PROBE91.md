# 一个完整场景的显存采样

2026-10-24当前：第91项完整显存采样及独立复算已完成，观察器/监督器已内核终止且实际退出0。共2742条NVML采样，错误0、最大间隔1.360871秒，整场进程显存采样峰值27.332 GiB（下界）。资源表生成及逐项主审也实际退出0，见[SCENE91_RESOURCE_COMPARISON.md](SCENE91_RESOURCE_COMPARISON.md)。下文保留准备与运行中的历史记录。

2026-10-23：针对协议第91项scene0608_02，已在四个科学作业及场景worker启动之前挂接只读采样器，入口实际退出0、主审实际退出0。入口检查时采样器状态为WAITING_FOR_SCENE91_DISPATCH，尚无完成的显存结果；不能将启动成功或控制测试值写成实测显存。

源代码与运行收据位于`$OUT/bootstrap/scene91_vram_probe/`。采样器、监督器均只依赖标准库，使用nvidia-smi读取NVML进程显存；不导入模型、不发信号、不改变科学输入、原代码、线程设置或双卡调度。固定跟踪这一个正式场景，不追加推理。当前GPU0/GPU2任务继续运行。

每秒采样实际worker及其递归子进程，读取所有线程创建的子进程；以PID、start、命令、父子关系和实际GPU UUID确认归属。查询前后PID身份不同的行拒绝归属，其他场景或其他GPU的进程不计入。记录分割、CPU建图、F、N/Q的采样峰值，以及同一时刻该场景全部进程显存之和的最大值。显存单位为MiB。

这属于**进程显存采样峰值**，是瞬时峰值的下界；不是PyTorch分配器峰值。报告还将给出实际最大采样间隔、采样数和查询错误。四个原作业须DONE、真实退出0，全部阶段覆盖及GPU阶段有效样本须齐备，协调器还须观察到worker真实退出0，才能接收完整测量；不完整观测保留原始数据并返回partial码20。作业墙钟沿用原收据，含CPU、加载和等待，不作为独占模块延迟或在线FPS。

8项控制测试涵盖进程/GPU隔离、PID复用、后台线程子进程、真实独立控制进程的生命周期、无效/重复NVML行、部分覆盖拒绝、密封格式兼容及晚启动拒绝。初版通过6项；随后两个新增边界分别先复现失败再修复，最终8项通过。原始失败日志、收据及两份早期源码/测试快照均保留；这些是控制测试，不是真实显存数据。

| 入口证据 | 身份 |
|---|---|
| 最终控制测试`green_v3.json`，实际退出0 | `57c7c7afc377e0e3d6ae602c989cdce66019054dacf323e2949ab7daa01354ec` |
| `entry.json` | `8a3b2866d6cdab4f625c74eeeafc29fcb6a8304314de32e3583f2909f90f39a3` |
| `launch_actual_exit.json`，实际退出0 | `364b49aa94d4a9bc74aa1205a41a84fe19339294d0d8e68dd5fccb97186cad86` |
| `review/G4_scene91_vram_probe_entry_primary_review.json` | `1b30fd44e1c2c3b3a50db776c0460777ffb58fb2a6d6f16caf5fc52808e8a2c1` |
| `entry_primary_actual_exit.json`，实际退出0 | `7e8edd7823cfc5095a822491bddd472884f2cecfa6d74845bf86b20169d13741` |

已固定采样器2914903/start58374979，监督器2914902/start58374973；最终源码SHA为`86fb6a2db672c2081fb84e7dd5c857e84e9c2b43d64b666e645346554f19851b`。入口核验93份原冻结文件和1752份稳定输入的SHA/mtime不变。

只读查看状态：

```bash
OUT=$OUT
cat "$OUT/bootstrap/scene91_vram_probe/progress.json"
```

完成后核对`receipt.json`与监督器`actual_exit.json`，从`samples.jsonl`独立重算峰值并核对四份作业、worker及硬件身份，再填效率表。本批测量不自动生成OVI-MAP的同口径显存，也不证明30FPS。最终有限发布清单须加入本批源码、控制/实际收据、失败版本、原始采样日志及独立重算证据；目前134份历史来源包未更新。此观察器须完成并保留真实终态收据后才封存最终发布证据。

独立复算入口`bootstrap/scene91_vram_probe/independent_review.py`现已准备。它用自己的聚合逻辑检查每条原始采样：GPU身份、PID/start、阶段进程是场景进程的子集、同时占用之和、严格递增的单调时钟、阶段/整体采样数与峰值、原始日志SHA及字节数。原作业四阶段与worker的真实退出、采样器/监督器内核终态、源码绑定也须一致。历史父子进程边不会由复算器重建；聚合独立于采样器函数。

4项控制测试通过，包含同时占用之和与阶段峰值之和的区别、错误归属/无效记录拒绝、汇总值不能覆盖原始记录，以及未完成观测须先拒绝。控制收据`review_green.json`身份`21002fb50ad8fc0552c3fae305a816d7fb4e6eb357fa5adb16ddfe6992e45c09`。针对当前真实存活的采样器执行原入口时，仅尝试读取缺失的`actual_exit.json`后即按预期拒绝；该被检入口未读取其他收据、未hash原始采样、未写输出、未创建worker，1752份保护路径及93份原文件SHA/mtime不变。准备证明`review/G4_scene91_memory_review_preparation_primary_review.json`身份`b8fabb6756c68b9ffbae41bbc60069e9a1dfc0c7e54847e5b786d36c352d1ebc`，准备检查真实退出0`review_preparation_actual_exit.json`身份`0175d83bd1cc8784309d73aad045810c47d89ec5bcb0f0f2f3bc07f1ec9d9e87`。这证明等待条件有效，不是完整显存测量通过。

待场景结束且观察器/监督器实际退出0后，才实际运行该入口；成功输出`review/G4_scene91_sampled_memory_primary_review.json`，再单独记录其真实子进程退出，并按该证明填效率表。测量进行期间的进度峰值保持阶段观测含义，不用于最终整场比较。


后续分类缓存的87→90授权汇总转换已完成，原87汇总及1752份基线保留为历史，当前后续主审基线为1800份，见CLASSIFICATION_SCORES90.md。显存入口和pending拒绝证明中的1752份为当时真实检查范围；本次转换未更改显存采样源码、原93份冻结文件或任何场景科学输入。完整显存验收仍须等待观察器真实结束。


最终真实退出记录入口`bootstrap/scene91_vram_probe/execute_independent_review.py`已准备（SHA256 `6bdd916431c91f478efdcb2299b4ef2323240b60d8188074da965889ae776b31`）；它先要求观察器`actual_exit.json`为DONE/0，再运行独立复算，并将真实退出与日志记录为`review_actual_exit.json`及`review_actual.log`，绑定观察器终态收据。已核对语法及与既有实际退出包装器的差异；尚未执行完整验收。最终有限来源清单须纳入该入口及其实际执行收据。


2026-10-24终态：观察器收据`0e1e9b8c6d78441babdcb1d305841c5df92a0cf3161370a362b8c997043d1334`，真实退出0`959b117f7faec9d5cb9252fa1915658b35111b14d72e1736e2006c433f8d3e50`；独立复算`cb2dfe6e31b54558ab9d61795ec9cd780022530fcdc04fb30b4b93fa8c65251e`，真实退出0`6cf62cde1bb6c14fd962e36e51d004356348c5c8510b751386f986b5665429f0`。观察器2914903/start58374979及监督器2914902/start58374973均已内核终止。资源表`review/G4_scene91_resource_comparison.json`（`d9f0846451af13f677ca33684b89fb70a53abba600fd423b8fcccdbc20118990`），4行本机值和5行作者参考的逐项主审`fd3f0d9665b59909af8515c7087a2df1a06d1f252073a75aff4b68ce1dc6065a`，真实退出0`e3207dfeb8f183b8d27c4a75e32192af31d02cf9a5bb03caa283777f7de37f33`。原93份及当前1800份输入SHA/mtime不变。该证明覆盖一场景NVML采样下界与作业墙钟，不升级为在线FPS、独占模块延迟或跨硬件加速/显存优势。


2026-10-24：闭合原始采样已普通复制为`bootstrap/scene91_vram_probe/samples_closed.log`，与原`samples.jsonl`保持相同SHA/mtime及706946字节，2742行逐条解析成功，使用独立inode且原文件保留。导出证明`raw_samples_export.json`（`a62fecc1e7ed9478dc28cb65289febf104007aee2142815618f8b712baf30083`），真实退出0`trace_export_actual_exit.json`（`ec58169d1545acf15259daa698c167bfb37681991ded43995e3bd16d1bfa79c6`），共同内容SHA256 `411f1fab6ab51cb3e769dc5c021c276782c7b1237212aa765be6c910d4bc82cb`。93份冻结文件及1800份当前保护输入SHA/mtime不变。本步只准备有限来源包的可选日志输入，当前134份历史清单及公开47索引未更新。最终刷新应显式纳入preserve_closed_trace.py、execute_trace_export.py、导出证明、实际日志/退出及samples_closed.log。
