"""ROS2 service wrapper for rosbag2 recorder control."""

from __future__ import annotations

from pathlib import Path


class RosbagControl:
    """Synchronous rosbag2 service client wrapper."""

    def __init__(self, node_name: str = 'main_controller_rosbag_control'):
        try:
            import rclpy
            from rosbag2_interfaces.srv import Pause, Record, Resume, Stop
        except Exception as exc:  # pragma: no cover - depends on ROS environment.
            raise RuntimeError(f'ROS2 rosbag service dependencies are unavailable: {exc}') from exc

        self._rclpy = rclpy
        self._Record = Record
        self._Resume = Resume
        self._Pause = Pause
        self._Stop = Stop
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.record_client = self.node.create_client(Record, '/rosbag2_recorder/record')
        self.resume_client = self.node.create_client(Resume, '/rosbag2_recorder/resume')
        self.pause_client = self.node.create_client(Pause, '/rosbag2_recorder/pause')
        self.stop_client = self.node.create_client(Stop, '/rosbag2_recorder/stop')

    def wait_ready(self, timeout_s: float) -> bool:
        """Wait until all rosbag2 services are available."""
        clients = (self.record_client, self.resume_client, self.pause_client, self.stop_client)
        return all(client.wait_for_service(timeout_sec=timeout_s) for client in clients)

    def record(self, uri: Path, timeout_s: float = 15.0) -> None:
        """Call /record with a bag URI."""
        request = self._Record.Request()
        request.uri = str(uri)
        self._call(self.record_client, request, timeout_s)

    def resume(self, timeout_s: float = 15.0) -> None:
        """Call /resume."""
        self._call(self.resume_client, self._Resume.Request(), timeout_s)

    def pause(self, timeout_s: float = 15.0) -> None:
        """Call /pause."""
        self._call(self.pause_client, self._Pause.Request(), timeout_s)

    def stop(self, timeout_s: float = 15.0) -> None:
        """Call /stop."""
        self._call(self.stop_client, self._Stop.Request(), timeout_s)

    def close(self) -> None:
        """Destroy the ROS node."""
        self.node.destroy_node()

    def _call(self, client, request, timeout_s: float) -> None:
        future = client.call_async(request)
        self._rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout_s)
        if not future.done():
            raise TimeoutError(f'rosbag2 service call timed out: {client.srv_name}')
        result = future.result()
        if result is None:
            raise RuntimeError(f'rosbag2 service call failed: {client.srv_name}')
