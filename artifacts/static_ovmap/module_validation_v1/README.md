# Current module-validation delivery

当前交付采用 12 个独立开发场景的 native-v10 捕获。实现 COMPLETE；S/Q 与 G 的原始/一致性对照已测量，G_QUALITY 因可区分训练目标不足而阻断。冻结最终候选为 N0，科学结果 NO_NET_GAIN，CONFIRM 为 NOT_REQUIRED_NO_RETAINED_CANDIDATE。

当前入口：

- [结果报告](../../../docs/paper/static_ovmap/MODULE_VALIDATION_RESULTS.md)：五张主表，116 条分场景评估行与 34 组均值。
- [交接文档](../../../docs/paper/static_ovmap/MODULE_VALIDATION_HANDOFF.md)：代码/扩展身份、成本、数据模型路径、复现命令与结论边界。
- [生成发布包](release-20260923-native-v10/)：公共 attempt_011 的报告、方法矩阵、实际评估/事件/计费记录和四个已训练小头；大数组与基础模型权重仅登记外部路径、字节数和哈希。
- [最终冻结与审计](finalization-20260923-native-v10/)：数据获取锁、冻结配置/选择回执、来源哈希、可复跑审计脚本与最终审计记录。
- [原生捕获验证](native_v10_capture_validation.json)：2,305 个有效帧与 22,307 个候选请求的技术验证，单独区别于科学收益结论。

实际 S/G/Q 生成代码：`1d83aec5e3c5220b0a5397e1bc3f720dfcbac3b9`。最终冻结/报告代码：`b6ab45b8dadf6672d2bcc2a6cfaed9bf41d60a0c`。

报告数值审计与全部 16,430 个发布文件的来源/哈希核验已通过。发布包为 96,099,584 字节（91.65 MiB）；小型 JSON 采用无损 gzip，清单记录解压后的哈希，4 个小头权重保持原始字节。最终 GitHub 推送状态、完整 SHA 和包含代码/文档的总交付大小，以仓库外 `/mnt/shared/ww/ovimap-module-validation-v1/publication-20260923.json` 为准。

除上述当前入口外，根目录旧文件和 `repairs_20260922/` 保留作历史交付证据。旧材料中的“独立场景不可用”“完整驱动尚未实现”不再描述当前工程；native-v8/v9 和修复前 RGB 结果也不参与当前科学结论。旧边界测试不能替代本轮真实实验，当前负结果也不应被旧的总体完成声明改写。
