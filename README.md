# MainController

MainController 是本项目的数据采集主控包。它作为 ROS2 `ament_python` 包构建和运行，同时在普通 Python 控制循环中管理 FT300S、XenseTacSensor、RealSense、rosbag2 recorder 和 ZMQ 遥测接收。

## 运行前提

- 在 ROS2 Python 环境下运行；如果当前 shell 处于 conda 环境中，先执行 `conda deactivate`。
- 系统 Python 需要可导入 `rclpy`、`rosbag2_interfaces`、`realsense2_camera_msgs`、`zmq`、`numpy`。
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
  --zmq-connect tcp://127.0.0.1:6000 \
  --output-dir ../runtime_sessions \
  --sensor-flush-timeout-s 300
```

其他参数可用：

```bash
ros2 run main_controller main_controller -- --help
```

## 交互命令

启动完成并进入 `WAIT_START` 后，终端可输入：

- `s`：开始新 demo，或从暂停状态恢复采集。
- `p`：暂停当前 demo；会暂停传感器和 rosbag2 recorder。
- `d`：结束并保存当前 demo；传感器 flush 默认有有限 timeout，可用
  `--sensor-flush-timeout-s none` 显式切到无界等待。
- `x`：丢弃当前 demo。
- `q`：退出主控；如果正在 finalizing，会等待保存结束后退出。

初始化在启动后自动执行，不需要额外交互命令。

`done` 表示 required sensors finished、rosbag stopped successfully 且 required
post-checks passed；`discarded` 表示用户 `x` 发起的 discard transaction 成功完成；
`failed` 表示系统或 command transaction 未成功。`PAUSE_REQ`、rosbag `pause`、
`DEMO_DONE_REQ`、finish-time rosbag `stop` 或用户 `DEMO_DISCARD_REQ` 中任一 required
operation 返回 `ERROR`、超时或抛错，都会在 manifest 中记录 `failure_stage`、
`failure_reason` 和 per-operation command result。pause/discard/failed finish 会进入
`ERROR -> STOPPING -> STOPPED`，避免主控状态与物理 sensor 状态不一致。
UDS peer disconnect 会唤醒 pending ACK wait 并把对应 command 标记为
`uds_disconnected`；flush timeout 会记录 `ack_timeout` 和 timeout 秒数。

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

默认输出目录为仓库根目录下的 `runtime_sessions/session_YYYYmmdd_HHMMSS/`：

- `controller_events.jsonl`：主控状态、命令、告警、错误和保存记录。
- `process_logs/`：FT300S、Xense、RealSense、rosbag2 子进程日志。
- `demos/demo_YYYYmmdd_HHMMSS/`：单次 demo 的数据。
- `*.npz`：主控侧缓存的结构化数据，如 ZMQ、RealSense metadata、UDS frame 记录。
- `manifest.json`：demo 保存摘要、rosbag 路径、传感器保存文件、`frame_counts`、丢帧统计、RealSense image readiness / rosbag post-check 和 RealSense 重启记录。用户成功 discard 会写 lightweight manifest，`status: "discarded"` 且 `npz` 为空，不保存高频 `.npz`。

## 监控与故障处理

主控会持续读取 ZMQ，即使当前处于暂停状态也不会停止 drain，避免远端队列溢出。采集时会监控 FT300S、Xense、ZMQ 和 RealSense metadata 的 frame id / seq 连续性与帧间隔；发现不连续或间隔显著变大时，会同时打印到终端并写入 log。

RealSense metadata topic 负责实时 timestamp 和丢帧监控。开始或恢复 rosbag recording 前，主控会短暂检查 required image topics 的 readiness baseline；formal 模式默认要求 `cam1` 到 `cam4` 的 color `image_raw` 和 `aligned_depth_to_color/image_raw` 共 8 个 topic，`debug_degraded` 模式必须显式配置子集。demo 完成后，主控使用当前 demo 的实际 rosbag URI 做 required image topic metadata post-check；缺失 topic、类型错误、零帧或 count skew 超阈值会把 manifest 写成 `status: "failed"`。如果 RealSense launch 输出中出现 `Hardware Error` 或 `Depth stream start failure`，主控会在 collecting 状态下先切换到暂停，再重启 RealSense launch。

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
