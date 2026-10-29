# node7 Replica 与完整 Open3DIS 当前进展

本地原始 Replica 已复用，未重新下载整套 Replica。已补齐 GroundingDINO 和 ISBNet 作者权重，原有普通 SAM-H 的 594 个张量名称、形状全部匹配作者普通构造器；普通 SAM 与当前源码 HQ 构造器的差异仍须显式处理，不能把普通权重称为完整 HQ。

独立环境离线复制实际退出 0，Python 3.10.20、Torch 1.12.1+cu113、NumPy 1.23.5 可导入。独立 CUDA 11.3.122 编译器及 GCC 9.3 已实际安装；完整依赖和作者 CUDA 自定义算子尚未完成。CPU Segmentator 使用 ISBNet 安装说明指定的 Karbo123 源码，提交 4c6126551685166c6c300551e9ad63db988928c4，C++ 算法及 Python 包装函数不改；首次控制环境缺 pkg_resources 的失败已保留，独立环境第二次编译实际退出 0。

真实 office0 输入准备 v3 实际退出 0：589517 个原顶点按原序保留，588759 个四边形完整转换成 1177518 个三角面，逐面索引核验通过，生成 1248 个真实 superpoints。RGB 采用作者 float64 rgb/127.5-1.0；作者 test 两元组读取和固定 0.35*pi 旋转函数 AST 不改。生成期间 GT 张量读取为 0，14 项测试通过。格式转换规则来自 node7 现有 Habitat 的 Magnum StanfordImporter 源码，副本和 SHA256 已固定。v1 仅接受三角面而退出 1；v2 因 Open3D 失败后仍返回部分网格而被单独拒绝，虽实际退出 0，其旧输出明确禁止用于推理。两次旧记录和全部输出均保留。

原 GT-free Open3DIS reader 的真实 office0 400 帧验证已完成。本轮输入和 superpoints 仅是准备产物，没有生成真实 ISBNet 提案或 dense features，没有执行完整 Grounded-SAM→3D→refinement→3D 流程，也没有完整 Open3DIS 首场景正式 G3。后续完整 source bridge 仍须明确作者一致的配置、真实 confidence、最终输出分支及隔离 tracker，并保留 ISBNet 和 refinement。

原六个科学队列继续运行；最新准确场景/单元覆盖以 reader_and_recovery_progress_review_v2.json 为准。仅 U00/U10 可评分，14 个训练角色仍未解决；U01/U11、完整 Replica bootstrap、池化结果和终稿发布尚未完成。完整任务保持未完成，禁止将部分 Replica 或准备成功当作完整复现或发布完成。
