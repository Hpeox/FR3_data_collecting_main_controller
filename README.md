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
colcon build --packages-select MainController
source install/setup.bash
ros2 run MainController main_controller
```

常用参数：

```bash
ros2 run MainController main_controller -- \
  --zmq-connect tcp://127.0.0.1:6000 \
  --output-dir ../runtime_sessions
```

其他参数可用：

```bash
ros2 run MainController main_controller -- --help
```

## 交互命令

启动完成并进入 `WAIT_START` 后，终端可输入：

- `s`：开始新 demo，或从暂停状态恢复采集。
- `p`：暂停当前 demo；会暂停传感器和 rosbag2 recorder。
- `d`：结束并保存当前 demo；传感器 flush 无硬超时。
- `x`：丢弃当前 demo。
- `q`：退出主控；如果正在 finalizing，会等待保存结束后退出。

初始化在启动后自动执行，不需要额外交互命令。

`done` 表示所有 required finish 操作完成；`discarded` 表示用户 `x` 发起的 discard
transaction 成功完成；`failed` 表示系统或 command transaction 未成功。`PAUSE_REQ`、
`DEMO_DONE_REQ` 或用户 `DEMO_DISCARD_REQ` 中任一 required sensor 返回 `ERROR` 或超时，
都会在 manifest 中记录 `failure_stage`、`failure_reason` 和 per-sensor command result。
pause/discard 失败会进入 `ERROR -> STOPPING -> STOPPED`，避免主控状态与物理 sensor 状态
不一致。

`s` 的 start/resume 流程由 MainController 作为多传感器事务 owner 协调。FT300S
和 XenseTacSensor 必须全部 ACK `START_REQ`，且 rosbag `record` / `resume`
必须成功后才进入 `COLLECTING`。如果任一 required sensor 返回 `ERROR`、超时，或
rosbag record/resume 失败，MainController 会对已 ACK start 的 sensor 发送
`DEMO_DISCARD_REQ` 回滚，写入 `status: "failed"` 的轻量 manifest，并清空当前
demo context。`status: "discarded"` 只表示用户 `x` 命令成功完成。

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
python -m pytest MainController/src/MainController/test/test_maincontroller_core.py -q
```

mock 集成测试：

```bash
conda deactivate
python -m pytest MainController/src/MainController/test/test_maincontroller_mock_runtime.py -q
```

该测试不需要真实 FT300S 或 Xense 硬件：FT300S 使用 100 Hz UDS mock，Xense 使用 30 Hz UDS mock，ZMQ 使用本进程 endpoint，rosbag2 service 和 RealSense metadata subscriber 使用 fake 对象。测试会覆盖 `s -> p -> s -> d`、`s -> d -> s -> d`、`s -> x -> s -> d`、暂停和 demo 间隙的 ZMQ 持续 drain、4 相机 / 8 metadata streams、`.npz`/manifest 保存，以及 RealSense fatal error 自动暂停和重启逻辑。
