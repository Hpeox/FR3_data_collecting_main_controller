"""Smoke test the native rosbag2 recorder launch and service lifecycle.

Usage:
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 MainController/src/main_controller/test/smoke_rosbag2_recorder_lifecycle.py
"""

from __future__ import annotations

import subprocess
import tempfile
import time
import signal
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
LAUNCH_FILE = REPO_ROOT / "RealSense" / "launch" / "rosbag2_recorder.launch.py"
PARAMS_FILE = REPO_ROOT / "RealSense" / "config" / "recorder_params.yaml"
QOS_FILE = REPO_ROOT / "RealSense" / "config" / "qos_override.yaml"


def _call_service(executor, client, request, timeout_s: float):
    future = client.call_async(request)
    executor.spin_until_future_complete(future, timeout_sec=timeout_s)
    if not future.done():
        raise TimeoutError(f"service call timed out: {client.srv_name}")
    exception = future.exception()
    if exception is not None:
        raise RuntimeError(f"service call failed: {client.srv_name}: {exception}") from exception
    result = future.result()
    if result is None:
        raise RuntimeError(f"service call failed: {client.srv_name}")
    return result


def main() -> None:
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/maincontroller_ros_logs")
    Path(os.environ["ROS_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    if not LAUNCH_FILE.exists():
        raise RuntimeError(f"missing canonical recorder launch file: {LAUNCH_FILE}")
    if not PARAMS_FILE.exists():
        raise RuntimeError(f"missing recorder params file: {PARAMS_FILE}")
    if not QOS_FILE.exists():
        raise RuntimeError(f"missing recorder QoS override file: {QOS_FILE}")

    try:
        import rclpy
        from rosbag2_interfaces.srv import Pause, Record, Resume, Stop
    except Exception as exc:
        raise RuntimeError(f"ROS 2 Jazzy Python dependencies are unavailable: {exc}") from exc

    with tempfile.TemporaryDirectory(prefix="maincontroller_rosbag_smoke_") as tmp:
        tmp_path = Path(tmp)
        bootstrap_uri = tmp_path / "bootstrap"
        real_uri = tmp_path / "real_demo_rosbag"
        log_path = tmp_path / "rosbag2_recorder_launch.log"
        with log_path.open("w", encoding="utf-8") as log_fp:
            process = subprocess.Popen(
                [
                    "ros2",
                    "launch",
                    str(LAUNCH_FILE),
                    f"bootstrap_uri:={bootstrap_uri}",
                ],
                cwd=REPO_ROOT,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                text=True,
            )
            rclpy.init(args=None)
            node = rclpy.create_node("maincontroller_rosbag_smoke")
            executor = rclpy.executors.SingleThreadedExecutor()
            executor.add_node(node)
            try:
                clients = {
                    "record": node.create_client(Record, "/rosbag2_recorder/record"),
                    "resume": node.create_client(Resume, "/rosbag2_recorder/resume"),
                    "pause": node.create_client(Pause, "/rosbag2_recorder/pause"),
                    "stop": node.create_client(Stop, "/rosbag2_recorder/stop"),
                }
                deadline = time.monotonic() + 20.0
                for name, client in clients.items():
                    remaining = max(0.1, deadline - time.monotonic())
                    if not client.wait_for_service(timeout_sec=remaining):
                        raise RuntimeError(f"service did not become ready: {name}")

                _call_service(executor, clients["stop"], Stop.Request(), 10.0)

                record_request = Record.Request()
                record_request.uri = str(real_uri)
                record_result = _call_service(
                    executor,
                    clients["record"],
                    record_request,
                    10.0,
                )
                if getattr(record_result, "return_code", 0) != 0:
                    raise RuntimeError(
                        "record service failed: "
                        f"return_code={record_result.return_code}: "
                        f"{getattr(record_result, 'error_string', '')}"
                    )

                _call_service(executor, clients["resume"], Resume.Request(), 10.0)
                _call_service(executor, clients["pause"], Pause.Request(), 10.0)
                _call_service(executor, clients["resume"], Resume.Request(), 10.0)
                _call_service(executor, clients["stop"], Stop.Request(), 10.0)
            finally:
                executor.remove_node(node)
                executor.shutdown()
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10.0)
        print(
            "rosbag2 recorder smoke passed: "
            f"launch={LAUNCH_FILE}, params={PARAMS_FILE}, qos={QOS_FILE}"
        )


if __name__ == "__main__":
    main()
