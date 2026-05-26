# MainController

MainController 是本项目的数据采集主控包。它作为 ROS2 `ament_python` 包构建和运行，同时在普通 Python 控制循环中管理 FT300S、XenseTacSensor、RealSense、rosbag2 recorder 和 ZMQ 遥测接收。

## 运行前提

- 在 ROS2 Python 环境下运行；如果当前 shell 处于 conda 环境中，先执行 `conda deactivate`。
- 系统 Python 需要可导入 `rclpy`、`rosbag2_interfaces`、`rosidl_runtime_py`、
  `sensor_msgs`、`realsense2_camera_msgs`、`zmq`、`numpy`。
- ZMQ telemetry 是必需输入，默认连接地址为 `tcp://127.0.0.1:6000`。
- FT300S 和 Xense 由主控启动对应 conda 环境进程；RealSense 和 rosbag2 由主控启动 ROS2 launch。

## 构建与启动

在仓库根目录执行：

```bash
conda deactivate
cd MainController
colcon build --packages-select main_controller
source install/setup.bash
ros2 run main_controller main_controller
```

常用参数：

```bash
ros2 run main_controller main_controller -- \
  --repo-root .. \
  --zmq-connect tcp://127.0.0.1:6000 \
  --output-dir ../runtime_sessions \
  --sensor-flush-timeout-s 300 \
  --alignment-base-source realsense
```

其他参数可用：

```bash
ros2 run main_controller main_controller -- --help
```

`colcon build` 时会校验并记录当前集成仓库根目录作为 build-time
repo-root hint。若 install tree 被移动，或在其他路径 / 机器运行，请通过
`--repo-root PATH` 显式指定仓库根。MainController 作为集成仓库的一部分构建，
不支持脱离同级 `FT300S`、`XenseTacSensor`、`RealSense` 模块独立 build。

## 交互命令

启动完成并进入 `WAIT_START` 后，终端可输入：

- `s`：开始新 demo，或从暂停状态恢复采集。
- `p`：暂停当前 demo；会暂停传感器和 rosbag2 recorder。
- `d`：结束并保存当前 demo；主控会同时发送两个传感器 `DEMO_DONE_REQ` 并调用
  rosbag2 `stop`，随后等待传感器 flush ACK / `saved_file` 和 rosbag stop 结果。
  传感器 flush 默认有有限 timeout。采集保存完成后，
  主控会自动生成对齐配置、索引和报告；对齐结束前不能开始下一次采集。操作者可用
  `--sensor-flush-timeout-s none` 或 `--sensor-flush-timeout-s unbounded`
  显式切到无界等待；这是为现场确实可能超长 flush 的传感器保留的预期模式，
  选择该模式即接受主控会一直等待对应 ACK / disconnect / ERROR 的行为。
- `x`：丢弃当前 demo。
- `q`：退出主控；如果正在 finalizing，会等待保存和自动对齐流程结束后退出。

初始化在启动后自动执行，不需要额外交互命令。

`done` 表示 required sensors finished、rosbag stopped successfully 且 required
post-checks passed；`discarded` 表示用户 `x` 发起的 discard transaction 成功完成；
`failed` 表示系统或 command transaction 未成功。`PAUSE_REQ`、rosbag `pause`、
`DEMO_DONE_REQ`、finish-time rosbag `stop` 或用户 `DEMO_DISCARD_REQ` 中任一 required
operation 返回 `ERROR`、超时或抛错，都会在 manifest 中记录 `failure_stage`、
`failure_reason` 和 per-operation command result。pause/discard/failed finish 会进入
`ERROR -> STOPPING -> STOPPED`，避免主控状态与物理 sensor 状态不一致。
UDS peer disconnect 会唤醒 pending ACK wait 并把对应 command 标记为
`uds_disconnected`；有限 flush timeout 会记录 `ack_timeout` 和 timeout 秒数。
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

默认输出目录为仓库根目录下的 `runtime_sessions/`：

- `controller_events_run_YYYYmmdd_HHMMSS.jsonl`：主控状态、命令、告警、错误和保存记录。
- `process_logs/run_YYYYmmdd_HHMMSS/`：FT300S、Xense、RealSense、rosbag2 子进程日志。
- `demos/demo_YYYYmmdd_HHMMSS/`：单次 demo 的数据。
- `*.npz`：主控侧缓存的结构化数据，如 ZMQ、RealSense metadata、UDS frame 记录。
  `realsense_metadata.npz` 包含 metadata JSON 中的 `clock_domain`；如果单帧 metadata
  缺少该字段，会保存为空值并在 log/report 中告警，不会导致采集失败。
- `manifest.json`：demo 保存摘要、`run_id`、相对 demo 目录的 `.npz` / `rosbag_uri` 路径、相对仓库根的 `sensor_paths`、`frame_counts`、丢帧统计、RealSense image readiness / rosbag post-check 和 RealSense 重启记录。用户成功 discard 会写 lightweight manifest，`status: "discarded"` 且 `npz` 为空，不保存高频 `.npz`。
- `demos/demo_YYYYmmdd_HHMMSS/aligned/`：主控自动对齐输出目录，默认包含 `alignment_config.json`、`aligned_index.npz`、`aligned_manifest.json` 和 `alignment_report.md`。自动对齐不生成 `aligned_numeric.npz` 等实际训练数据文件。

需要独立重跑或调参时，可在仓库根目录使用 `tools/align_demo_timestamps.py`：

```bash
python tools/align_demo_timestamps.py \
  --demo-dir runtime_sessions/demos/demo_YYYYmmdd_HHMMSS \
  --repo-root . \
  --alignment-base-source realsense \
  --mode causal \
  --start-trim-s 1.0
```

对齐基准可以通过 `--alignment-base-source realsense|xense` 选择；手动工具也支持
更精确的 `--base realsense:<topic>|xense:0|robot|grid`，且 `--base` 优先级更高。
选择 Xense 作为基准时使用 `timestamp_ns_0`。对齐索引中 FT300S 字段使用
`ft300s_*` key，报告中显示为 `FT300S`；Xense 两路触觉传感器分别输出
`xense_0_*` 和 `xense_1_*`。

TODO: materialize 实际训练数据暂不作为当前可用命令提供；需要先确认数据集具体组织格式。

## 监控与故障处理

主控会持续读取 ZMQ，即使当前处于暂停状态也不会停止 drain，避免远端队列溢出。采集时会监控 FT300S、Xense、ZMQ 和 RealSense metadata 的 frame id / seq 连续性与帧间隔；发现不连续或间隔显著变大时，会同时打印到终端并写入 log。

RealSense metadata topic 负责实时 timestamp 和丢帧监控。开始或恢复 rosbag recording 前，主控会短暂检查 required image topics 的 readiness baseline；formal 模式默认要求 `cam1` 到 `cam4` 的 color `image_raw` 和 `aligned_depth_to_color/image_raw` 共 8 个 topic，`debug_degraded` 模式必须显式配置子集。demo 完成后，主控使用当前 demo 的实际 rosbag URI 做 required image topic metadata post-check；缺失 topic、类型错误、零帧或 count skew 超阈值会把 manifest 写成 `status: "failed"`、记录 `failure_stage: "realsense_rosbag_postcheck"`，并进入 `ERROR -> STOPPING -> STOPPED`。如果 RealSense launch 输出中出现 `Hardware Error` 或 `Depth stream start failure`，主控会在 collecting 状态下先切换到暂停，再重启 RealSense launch。

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
