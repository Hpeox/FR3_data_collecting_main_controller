# MainController

MainController 是本项目的数据采集主控包。它作为 ROS2 `ament_python` 包构建和运行，同时在普通 Python 控制循环中管理 FT300S、XenseTacSensor、RealSense、rosbag2 recorder 和 ZMQ 遥测接收。

## 运行前提

- 在 ROS2 Python 环境下运行；如果当前 shell 处于 conda 环境中，先执行 `conda deactivate`。
- 系统 Python 需要可导入 `rclpy`、`rosbag2_interfaces`、`rosidl_runtime_py`、
  `sensor_msgs`、`realsense2_camera_msgs`、`zmq`、`numpy`、`matplotlib`。
- ZMQ telemetry 是必需输入，默认连接地址为 `tcp://127.0.0.1:6000`。
- FT300S 和 Xense 由主控启动对应 conda 环境进程；RealSense 和 rosbag2 由主控启动 ROS2 launch。
- 启动阶段会先等待 FT300S / XenseTacSensor 完成 UDS 连接并发送 `INIT_READY`，
  再启动 RealSense camera launch 和 rosbag2 recorder，避免 Xense SDK 的 V4L camera
  index 初始化与 RealSense 节点扫描同时发生。

## 构建与启动

在仓库根目录执行：

```bash
conda deactivate
cd MainController
colcon build --packages-select main_controller
source install/setup.bash
ros2 run main_controller main_controller -- \
  --task-name 16mm-peg-in-hole
```

直接保存到代码仓库下的 `runtime_sessions/` 和 `runtime_frames/`：

```bash
ros2 run main_controller main_controller -- \
  --task-name gear-insert-big2small \
  --repo-root /home/robot/Desktop/gello-deploy \
  --zmq-connect tcp://192.168.10.37:6000 \
  --xense-sdk-version 2.0.1 \
  --sensor-flush-timeout-s 300 \
  --alignment-base realsense:bundle
```

保存到 `/data/external/runtime/runtime_sessions/` 和
`/data/external/runtime/runtime_frames/`：

```bash
ros2 run main_controller main_controller -- \
  --task-name 16mm-peg-in-hole \
  --repo-root /home/robot/Desktop/gello-deploy \
  --runtime-root /data/external/runtime \
  --zmq-connect tcp://192.168.10.37:6000 \
  --xense-sdk-version 2.0.1 \
  --sensor-flush-timeout-s 300 \
  --alignment-base realsense:bundle
```

其他参数可用：

```bash
ros2 run main_controller main_controller -- --help
```

`colcon build` 时会校验并记录当前集成仓库根目录作为 build-time
repo-root hint。若 install tree 被移动，或在其他路径 / 机器运行，请通过
`--repo-root PATH` 显式指定仓库根。MainController 作为集成仓库的一部分构建，
不支持脱离同级 `FT300S`、`XenseTacSensor`、`RealSense` 模块独立 build。
`--repo-root` 只表示集成代码仓库根目录。`--runtime-root` 表示运行数据根目录，
并固定派生 `<runtime-root>/runtime_sessions` 和
`<runtime-root>/runtime_frames`；未指定时默认使用 `repo-root`。
`--task-name` 是必选参数，只允许 ASCII 字母、数字、`.`、`_`、`-`，必须以字母或
数字开头且不能包含 `..`。主控启动时从
`<repo-root>/TaskInstruction/<task-name>.json` 加载并验证 instruction 和权重；
每个新 demo 创建时随机抽取一次，pause/resume 不会重新抽样。文件格式和自动权重规则
见 `TaskInstruction/README.md`。
`--xense-sdk-version` 使用 SDK 版本语义，允许 `1.x`、`2.0` 或 `2.0.1`，
默认 `2.0.1`；主控内部映射为 `1.x -> Xense310`、`2.0 -> xense2_bak`、
`2.0.1 -> xense2` 的 conda 环境启动 XenseTacSensor。

## 交互命令

启动完成并进入 `WAIT_START` 后，终端可输入：

- `s`：开始新 demo，或从暂停状态恢复采集。
- `p`：暂停当前 demo；会暂停传感器和 rosbag2 recorder。
- `d`：结束并保存当前 demo；主控会同时发送两个传感器 `DEMO_DONE_REQ` 并调用
  rosbag2 `stop`，随后等待传感器 flush ACK / `saved_file` 和 rosbag stop 结果。
  传感器 flush 默认有有限 timeout。采集保存完成后，
  主控会自动生成对齐配置、索引和报告，并从保存的 `zmq_telemetry.npz` 生成 gripper
  预览图；对齐和绘图结束前不能开始下一次采集。操作者可用
  `--sensor-flush-timeout-s none` 或 `--sensor-flush-timeout-s unbounded`
  显式切到无界等待；这是为现场确实可能超长 flush 的传感器保留的预期模式，
  选择该模式即接受主控会一直等待对应 ACK / disconnect / ERROR 的行为。
- `x`：丢弃当前 demo。
- `q`：退出主控；如果正在 finalizing，会等待保存和自动对齐流程结束后退出。若正在
  active demo，会用 `STOP_REQ` 尝试 flush sensor、停止 rosbag、写
  `status: "failed"` manifest 并保存已有主控侧 `.npz`。

初始化在启动后自动执行，不需要额外交互命令。

`done` 表示 required sensors finished、rosbag stopped successfully 且 required
post-checks passed；`discarded` 表示用户 `x` 发起的 discard transaction 成功完成；
`failed` 表示系统或 command transaction 未成功。`PAUSE_REQ`、rosbag `pause`、
`DEMO_DONE_REQ`、finish-time rosbag `stop` 或用户 `DEMO_DISCARD_REQ` 中任一 required
operation 返回 `ERROR`、超时或抛错，都会在 manifest 中记录 `failure_stage`、
`failure_reason` 和 per-operation command result。pause/discard/failed finish 会进入
`ERROR -> STOPPING -> STOPPED`，避免主控状态与物理 sensor 状态不一致。
active demo 中的 `q`、ZMQ receiver fatal、RealSense metadata fatal、UDS 非命令期
disconnect 或 required subprocess unexpected exit 会写 `status: "failed"` manifest，
保存已有主控侧 `.npz`，但不发送 `DEMO_DONE_REQ` / `DEMO_DISCARD_REQ`，也不运行自动
timestamp alignment；异步 fatal 随后进入 `ERROR -> STOPPING -> STOPPED`。
`STOP_REQ` ACK 中的 `saved_file` 是 optional diagnostic output，缺失时对应
`sensor_paths` 为 `None`。该字段按正式协议只表示 basename / filename；MainController
runtime 内部解析为 runtime root 下的 `runtime_frames/<saved_file>`，manifest 中只写
runtime-root 相对 `sensor_paths`。UDS peer disconnect 会唤醒 pending ACK wait 并把对应
command 标记为 `uds_disconnected`；有限 flush timeout 会记录 `ack_timeout` 和 timeout 秒数。
当 `sensor_flush_timeout_s` 显式配置为 `none` / `unbounded` 时，不产生
`ack_timeout`，等待只会被 ACK、对应 sensor `ERROR`、UDS disconnect 或进程停止唤醒。
时间戳对齐结果不复用采集 `status`，而是写入独立的 `manifest.alignment.status`；
自动对齐失败不会把采集 `done` 改写为 `failed`。

`s` 的 start/resume 流程由 MainController 作为多传感器事务 owner 协调。FT300S
和 XenseTacSensor 必须全部 ACK `START_REQ`，且 rosbag `record` / `resume`
必须成功后才进入 `COLLECTING`。如果任一 required sensor 返回 `ERROR`、超时，或
rosbag record/resume 失败，MainController 会写入 `status: "failed"` 的轻量 manifest。
新 demo start 失败时，回滚目标是已 ACK start 的 sensor；paused resume 失败时，回滚
目标是所有已经持有 paused demo context 的 required sensor。若 rollback 全部确认，
主控清空当前 demo context 并回到 `WAIT_START`；若任一 rollback target 无法确认
`DEMO_DISCARD_REQ`，或 rosbag cleanup stop 失败，manifest 会记录
`rollback_unconfirmed_sensors`，随后进入 `ERROR -> STOPPING -> STOPPED`。
`status: "discarded"` 只表示用户 `x` 命令成功完成。

## 输出

输出目录固定为 `<runtime-root>/runtime_sessions/`；默认 runtime root 为代码仓库根目录：

- `controller_events_run_YYYYmmdd_HHMMSS.jsonl`：主控状态、命令、告警、错误和保存记录。
- `process_logs/run_YYYYmmdd_HHMMSS/`：FT300S、Xense、RealSense、rosbag2 子进程日志。
- `demos/demo_YYYYmmdd_HHMMSS/`：单次 demo 的数据。
- `*.npz`：主控侧缓存的结构化数据，如 ZMQ、RealSense metadata、UDS frame 记录。
  `realsense_metadata.npz` 包含 metadata JSON 中的 `clock_domain`；如果单帧 metadata
  缺少该字段，会保存为空值并在 log/report 中告警，不会导致采集失败。
- `manifest.json`：demo 保存摘要、`run_id`、`xense_sdk_version`、`task_name`、`language_instruction`、相对 demo 目录的 `.npz` / `rosbag_uri` 路径、相对 runtime root 的 `sensor_paths`（例如 `runtime_frames/<saved_file>`）、`frame_counts`、本 demo 丢帧统计、RealSense image readiness / rosbag post-check、`xense_tactile_postcheck` / `xense_tactile_preview` 和本 demo RealSense 重启记录。任务字段顺序为 `xense_sdk_version`、`task_name`、`language_instruction`；instruction 是该 demo 创建时的抽样结果。用户成功 discard 会写 lightweight manifest，`status: "discarded"` 且 `npz` 为空，不保存高频 `.npz`。active-demo abort 会写 `status: "failed"` 并保存已有 `.npz`。
- `demos/demo_YYYYmmdd_HHMMSS/aligned/`：主控自动对齐输出目录，默认包含 `alignment_config.json`、`aligned_index.npz`、`aligned_manifest.json` 和 `alignment_report.md`。自动对齐不生成 `aligned_numeric.npz` 等实际训练数据文件。
- `/tmp/main_controller/gripper.png`：最近一个成功完成 demo 的 gripper 预览图。command
  和反馈序列分别使用各自过滤后的局部 sample index；下一次成功 demo 会覆盖该文件。
  如果 Xense ACK 中提供了 `/tmp/main_controller/xense_tactile_preview/*.npz`，同一张图还会
  以两个 tactile subplot 绘制两路 `force_resultant`，并根据 preview 内的 `edge_warning`
  标红对应 sensor。preview 是临时绘图输入，plot 子进程使用后会删除；preview 缺失、
  写入失败或读取失败时仍会生成 gripper-only PNG。绘图由独立 Python 子进程同步完成，默认 timeout 为 30 秒，可通过
  `--gripper-plot-timeout-s` 调整。绘图失败只会在终端和 controller event log 中告警，
  不会把已完成采集的 `done` 状态改为 `failed`。

MainController 自动对齐使用显式 `--alignment-base`，默认 `realsense:bundle`。
支持 `realsense:bundle`、`realsense:<topic>`、`xense:pair`、`robot` 和 `grid`。
`realsense:bundle` 以多相机 visual bundle 作为目标时间轴；只有
`realsense:<topic>` 会让 RealSense image streams 走 per-stream scalar matching。
`xense:pair` 使用同一 raw row 的 `max(timestamp_ns_0, timestamp_ns_1)`，并保证
`xense_0_*` 和 `xense_1_*` 投影自同一个 source row。可用
`--alignment-start-trim-s` / `--alignment-end-trim-s` 裁掉全局 overlap 窗口首尾样本。

需要独立重跑或调参时，可在仓库根目录使用 `tools/align_demo_timestamps_v3.py`：

```bash
python tools/align_demo_timestamps_v3.py \
  --demo-dir runtime_sessions/demos/demo_YYYYmmdd_HHMMSS \
  --repo-root . \
  --base realsense:bundle \
  --mode causal \
  --start-trim-s 1.0
```

`aligned_index.npz` 保存 `t_ns`、`segment_id`、`sample_valid`、各 stream 的
`<stream>_index/time_ns/delta_ns/valid`，以及 bundle / pair 诊断字段；topic metadata
保存在 `alignment_config.json` / `aligned_manifest.json`，不再作为 per-sample array
写入 `aligned_index.npz`，RealSense `frame_number` 也不再作为对齐输出字段。对齐索引中
FT300S 字段使用 `ft300s_*` key，报告中显示为 `FT300S`；Xense 两路触觉传感器分别输出
`xense_0_*` 和 `xense_1_*`。

TODO: materialize 实际训练数据暂不作为当前可用命令提供；需要先确认数据集具体组织格式。

## 监控与故障处理

主控会持续读取 ZMQ，即使当前处于暂停状态也不会停止 drain，避免远端队列溢出。采集时会监控 FT300S、Xense、ZMQ 和 RealSense metadata 的 frame id / seq 连续性与帧间隔；发现不连续或间隔显著变大时，会同时打印到终端并写入 log。

RealSense metadata topic 负责实时 timestamp 和丢帧监控。主控进入 `WAIT_START` 前会检查 required image topics 的 readiness baseline；开始或恢复 rosbag recording 前会再次检查。formal 模式默认要求 `cam1` 到 `cam4` 的 color `image_raw` 和 `aligned_depth_to_color/image_raw` 共 8 个 topic，`debug_degraded` 模式必须显式配置子集。demo 完成后，主控使用当前 demo 的实际 rosbag URI 做 required image topic metadata post-check；缺失 topic、类型错误、零帧或 count skew 超阈值会把 manifest 写成 `status: "failed"`、记录 `failure_stage: "realsense_rosbag_postcheck"`，并进入 `ERROR -> STOPPING -> STOPPED`。count skew 阈值按当前 rosbag 中单个 required topic 实际帧数的百分比计算，默认 `--realsense-rosbag-count-skew-limit-percent 0.5`。如果 RealSense launch 输出中出现 `Hardware Error` 或 `Depth stream start failure`，主控会在 collecting 状态下先切换到暂停，只有自动暂停成功且主控仍处于可恢复状态时才重启 RealSense launch。
readiness timeout 可通过 `--realsense-image-ready-timeout-s` 调整。需要临时降级诊断时，可使用 `--realsense-capture-mode debug_degraded` 并重复传入 `--realsense-debug-image-topic <topic>` 指定 required 子集；正式采集应保持默认 `formal`。

Xense tactile post-check 在 Xense 进程收到 `DEMO_DONE_REQ` 后基于内存中的
`frames_data` 计算，不由 MainController 回读完整 `.npy`。每路 sensor 使用
`mean(norm(force_resultant_6d, axis=1)) <= 0.1` 判定 zero-force；有且仅有一个
sensor zero-force 时，demo 写为 `status: "failed"`、`failure_stage: "xense_tactile_postcheck"`，并保持现有 `ERROR -> STOPPING -> STOPPED` 行为。两个
sensor 都 zero-force 或都非 zero-force 不会触发 failure。edge warning 使用首/尾各
15 点的 per-channel `mean(abs())`，默认阈值为 `0.5`；warning 会写入
`xense_tactile_postcheck` 并用于 plot 标红，但不会改变 demo `done` 状态。

## 测试

回到仓库根目录执行：

```bash
conda deactivate
python -m pytest MainController/src/main_controller/test/test_maincontroller_core.py -q
```

mock 集成测试：

```bash
conda deactivate
python -m pytest MainController/src/main_controller/test/test_maincontroller_mock_runtime.py -q
```

该测试不需要真实 FT300S 或 Xense 硬件：FT300S 使用 100 Hz UDS mock，Xense 使用 30 Hz UDS mock，ZMQ 使用本进程 endpoint，rosbag2 service 和 RealSense metadata subscriber 使用 fake 对象。测试会覆盖 `s -> p -> s -> d`、`s -> d -> s -> d`、`s -> x -> s -> d`、paused resume rollback、rosbag pause/stop transaction failure、UDS finalization timeout / disconnect、暂停和 demo 间隙的 ZMQ 持续 drain、4 相机 / 8 metadata streams、`.npz`/manifest 保存，以及 RealSense fatal error 自动暂停和重启逻辑。

## 硬件验收边界

mock 测试只能证明 formal mode 会按配置要求 4 相机 / 8 image topics，并在缺失 topic 时 fail closed；不能证明现场物理 RealSense 均在线。最终硬件验收需要在真实四相机环境运行 formal capture，确认 `cam1` 到 `cam4` 的 color `image_raw` 和 `aligned_depth_to_color/image_raw` 均 ready、被 rosbag 记录，并通过 demo 完成后的 metadata post-check。
