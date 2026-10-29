# Node7 Replica 接入（2026-10-29）

用户指出的 node7（本机 node107）确有完整原始 Replica。8 个正式场景的原始 semantic mesh 与已保留作者评测网格，全部 XYZ 点序、RGB 字节逐项相等。发现证明为 `G4_node7_original_Replica_discovery.json`。此前“原始 Replica 缺失”的判断已撤回。

已有 OpenIns3D 与 OpenYOLO3D 发布的逐点评测 GT，8 个场景两份文件字节相同。原始 Replica、作者网格和已有 OpenScene 坐标域已对齐。`replica_3d` 是程序使用的预处理格式；无需用户另找同名数据集。

当前接入只新增作者网格路径、派生四元组路径，以及将旧 RGB-D 路径映射到已核验修复副本。协议、模型、校准和完成的 ScanNet 数值不变。派生文件明确记录来源，不冒称原作者四元组原始字节。

首轮 `office0` 在旧目录的 `frame000693.jpg` 复现 JPEG 损坏；同名修复副本可完整解码。失败调用退出 1、日志和 69 张已生成分割保留，110.214 秒 GPU worker 成本单列。通过既有 `path_map` 接口修正数据位置，200 个原选中 RGB-D 帧全部解码后，首场景连接 v2 已启动。

首场景连接 v2 已实际退出 0。该调用先完成原建图、F/MV 和 N/Q，冻结全部可用预测后才读取数值 GT；转换类别编号时保留全部 589517 个点和实例后缀。U00/U10 两个 seed 的 4 条真实预测，均与原 Open3DIS、原 OpenYOLO3D 评分器核对一致；每条比较全部 480 个类别/IoU AP 项，最大差异为 0。真实 G3 已通过，不等于 Replica 全部实验完成。

首场景原建图实际耗时 2251.085 秒。OpenMask3D 的 masks/features/query 与 MaskClustering 的 frontend/cluster/features/query、两种完整方法的官方评分及作者评分核对，均已实际退出 0。OVI-MAP 首场景也已实际退出 0，仅复用原始 N 特征和作者六位小数面积置信度，没有使用 F/MV/Q 分数或新增 GPU 编码。三个对比方法均核对全部 480 个类别/IoU AP 项，最大差异为 0。共享文本特征复用已通过实际原始计划核对，保留旧成功收据且没有新增模型调用；复用边界的 9 项测试检查输入变化和输出损坏会被拒绝。

MaskClustering 首次特征提取因缺少 `libnvrtc.so` 名称链接退出 -6。环境内已有匹配 PyTorch 的 CUDA 11.8 库，仅新增指向原库的相对链接，未改库二进制、模型或参数。第二次重试实际退出 0；首轮失败日志与 160.468 秒成本仍留在同一原始任务的 attempts 中，没有重置两次上限。

其余 7 个标准场景已进入三个原始生产者队列：GPU0 负责 office1/office4/room2，GPU1 负责 office2/room0，GPU2 负责 office3/room1。全局 binding 不变，各队列单独记录 GPU 覆盖。同一 F 或 N/Q 缓存只允许一个同阶段写入者，CPU 评分最多并发 3 个；真实进程锁的 3 项检查通过。队列只调用原数值生产者，仍按每场景预测冻结、派生 GT、完整分母评分的顺序执行。

OpenMask3D、MaskClustering 的其余 7 个场景完整推理队列已分别按 GPU1、GPU2 启动，与原生队列共用原有逐阶段 GPU 锁。推理仅延后四元组文件存在性这一评分依赖；模型、几何或其他实际推理依赖缺失仍阻塞，不伪造 GT。推理先冻结原方法预测，再等原生场景的真实 GT 转换与评分记录，调用完整原评分流程。两方法串行处理各自场景，共享 RGB-D 布局的创建使用场景锁。边界的 14 项测试通过，新通用核对入口在三个真实首场景预测上均重现 480 项零差异。

OVI-MAP 其余场景的 CPU 队列已启动，等待相应原生 N 数据完成后复用评分。队列进程运行不等于 GPU 正在计算；原建图使用 CPU，GPU 按各阶段实际子进程占用。上述后续队列尚未全部完成，不能计入已完成场景。

office1 已完成当前可用 U00/U10 两 seed 的预测与评分，并写入原生场景完成记录；OpenMask3D、MaskClustering 的 office1 预测也已冻结，后续推理仍在运行。office2、office3 原生建图实际退出 0，不代表各自完整场景评分已完成。

office3 首次 N/Q 实际退出 1：原子任务在共享 GPU2 被 MaskClustering 占用时使用非阻塞锁，直接报资源冲突。该失败收据、10.241 秒成本和旧队列的实际退出均保留。新增独立资源边界只将同一 GPU 锁改为等待，仍执行原 N/Q 数值函数；4 项真实锁竞争和两次总重试上限测试通过。GPU2 恢复队列位于 `native_queue_GPU2_resource_wait_v1_actual`，复用原成功阶段，并将原失败与新调用合计限制为两次；启动不视为恢复完成。旧 `native_queue_GPU2_actual` 是已关闭失败记录，不能作为实时活动队列。

截至本轮复核，office3 N/Q 恢复调用已实际退出 0，保留首次失败与第二次成功这两次调用；office0–office3 的 U00/U10 两 seed 共 16/64 条标准行完成原评分及逐项作者核对。其余 U01/U11 仍未生成。OpenMask3D、MaskClustering 的 office1、office2 预测均已冻结，正在推理 office3；冻结预测不计为已评分场景。OVI-MAP 已完成 office0–office2 评分，旧后续队列因 office3 原生队列终止而实际退出 20；其两份成功后续场景结果经文件 hash 核验原样复用，新的 CPU 恢复队列位于 `OVI_MAP_remaining_queue_recovery_v1_actual`。等待边界的 8 项检查验证了明确恢复所有者、真实进程 start、已退出状态、已完成场景优先和旧失败收据保留。

完整 Open3DIS 的 GT-free reader 已实现并在 office0 全部 400 个 stride-5 帧上实际 CPU 验证通过，保留 589517 个作者顶点原序，所有 pose 与原轨迹逐项相等，验证期间未调用 torch.load。完整方法的真实 confidence、模型资产和入口 source bridge 仍未完成，详见本地 `Open3DIS_Replica_full_status.md`；reader 通过不等于完整 Open3DIS 推理或正式 G3 已完成。

当前标准预测仍受 14 项训练 base 角色映射限制，可先产生 U00/U10 两 seed。语义角色不会由 GT 或 AP 推定。此前 0/64 的 Replica 统计和 129 项剩余任务是接入前快照，不能当作此后实时队列。

旧发布进程在 C1 前停止；当前主审核保持未就绪。完成新场景验证后才更新覆盖、表格和发布来源。ScanNet 的 312 场景、1000 draw 原统计和已关闭的五节点诊断不重跑。

实际入口与收据位于本地任务 `recovery/takeover_20261026/node7_replica_connection_20261029`。首场景实际退出收据身份为 `810095720fce2356c117f173ad030ca7e01d4326b5674d13bf2b1878ceafaeca`，连接记录为 `6495fa1e889743bb2fadef9429a93700c782f3f4488d2d1d790a150d7c8bf8e3`。后续查看 `native_queue_GPU{0,1,2}_actual`、`{OpenMask3D,MaskClustering}_remaining_queue_actual`、`OVI_MAP_remaining_queue_actual` 的实际退出和同步收据，以及逐场景完成记录；进程启动、GPU 占用或中间文件不能代替评分完成。
