# Inference MainController

The inference MainController owns one persistent FR3 inference session and a
sequence of explicitly operated rollouts. It is a separate ROS 2 entrypoint
from the existing collection MainController and does not change the collection
workflow.

The controller starts and owns the workstation-side FT300S, XenseTacSensor,
RealSense, rosbag2 recorder, raw FGT1 telemetry receiver, and one persistent
`lerobot-rollout --strategy.type=controlled` worker. LeRobot owns the policy,
`Robot::FR3`, SensorHub, observation production, and reset target selection.
MainController does not launch or supervise SensorHub or NUC processes.

## Runtime prerequisites

- Run the entrypoint with the system ROS 2 Python environment. If the shell is
  currently inside a conda environment, run `conda deactivate` first.
- Build from the integration repository containing sibling `FT300S`,
  `XenseTacSensor`, `RealSense`, and `LeRobotFR3` directories.
- The system Python environment must provide the MainController ROS 2 and
  Python dependencies, including `rclpy`, `rosbag2_interfaces`, `zmq`, and
  `numpy`.
- The following conda environments must already exist:
  - `modbus314` for FT300S;
  - `Xense310`, `xense2_bak`, or `xense2`, selected by
    `--xense-sdk-version`;
  - `lerobot-fr3-312` for the persistent LeRobot worker.
- The policy path must be readable by the `lerobot-fr3-312` environment.
- The NUC robot command and telemetry endpoints must be reachable from the
  workstation. MainController does not start or recover the NUC runtime.
- The default LeRobot control socket is
  `/run/user/<uid>/lerobot_controlled.sock`. Its parent directory must exist and
  be writable by the current user.

No FR3/FCI or physical sensor operation should be started until the normal
site-specific robot, NUC, network, camera, and emergency-stop checks have been
completed.

## Build

From the integration repository root:

```bash
conda deactivate
cd MainController
colcon build --packages-select main_controller
source install/setup.bash
```

If the install tree is moved or the build-time repository hint no longer
matches the checkout, pass `--repo-root` explicitly when starting the
controller.

## Start an inference session

Example using the repository-local runtime directories:

```bash
ros2 run main_controller inference_main_controller -- \
  --repo-root /home/robot/Desktop/gello-deploy \
  --policy-path "/home/robot/Desktop/gello-deploy/LeRobotFR3/outputs/acmt_dp/peg/real/seed42/pretrained_model" \
  --task "insert the peg into the hole" \
  --zmq-connect tcp://192.168.1.37:6000 \
  --robot-command-endpoint tcp://192.168.1.37:6001 \
  --robot-telemetry-endpoint tcp://192.168.1.37:6000 \
  --xense-sdk-version 2.0.1
```

To keep runtime artifacts outside the source checkout:

```bash
ros2 run main_controller inference_main_controller -- \
  --repo-root /home/robot/Desktop/gello-deploy \
  --runtime-root /data/external/runtime \
  --policy-path /path/to/policy \
  --task "insert the peg into the hole" \
  --zmq-connect tcp://192.168.1.37:6000 \
  --robot-command-endpoint tcp://192.168.1.37:6001 \
  --robot-telemetry-endpoint tcp://192.168.1.37:6000
```

Show the current command-line interface with:

```bash
ros2 run main_controller inference_main_controller -- --help
```

### Command-line options

| Option | Required | Default | Meaning |
| --- | --- | --- | --- |
| `--policy-path` | Yes | None | Policy path passed to the persistent LeRobot worker. |
| `--task` | Yes | None | Literal task instruction passed to LeRobot. |
| `--repo-root` | No | Build-time/current checkout discovery | Integration repository root. |
| `--runtime-root` | No | `repo-root` | Root containing `runtime_sessions/` and `runtime_frames/`. |
| `--control-socket-path` | No | `/run/user/<uid>/lerobot_controlled.sock` | Controlled worker UDS path. |
| `--zmq-connect` | No | `tcp://192.168.1.37:6000` | MainController direct FGT1 telemetry subscription. |
| `--robot-command-endpoint` | No | `tcp://192.168.1.37:6001` | FR3 command endpoint passed to LeRobot. |
| `--robot-telemetry-endpoint` | No | `tcp://192.168.1.37:6000` | FR3 telemetry endpoint passed to LeRobot/SensorHub. |
| `--aligned-stall-timeout-s` | No | `0.075` | RUNNING aligned-sequence stall threshold. |
| `--lerobot-aligned-max-age-ms` | No | `100` | LeRobot stale-observation fallback threshold; it must remain longer than the MainController stall threshold. |
| `--xense-sdk-version` | No | `2.0.1` | Xense runtime selection: `1.x`, `2.0`, or `2.0.1`. |

The fixed v1 aligned-observation SHM name is
`/fr3_aligned_observation`. MainController reads only its version-pinned,
read-only 320-byte `FR3OBS2` global header; it does not copy observation
payloads or import LeRobot runtime modules for this health check.

## Startup sequence

Startup is session-scoped:

1. Start the direct FGT1 receiver and the FT300S/Xense UDS clients.
2. Start the managed FT300S, XenseTacSensor, RealSense, and rosbag2 recorder
   processes.
3. Wait for both sensor UDS connections and `INIT_READY`, then verify rosbag2
   recorder services.
4. Start one persistent controlled LeRobot worker and connect its UDS control
   channel.
5. Wait for LeRobot `READY` in `WAIT_INITIALIZE`, map the aligned SHM header,
   require `ready=1`, and enter `WAIT_START`.

Resource registration and startup are synchronized with fatal teardown. Once
a session-fatal request begins, startup stops scheduling resources. Anything
already started remains registered for the one-shot teardown path.

An aligned-header `fatal` value is diagnostic in MainController. It is not a
transfer of SensorHub fault ownership and is not proof that the complete
system is healthy. SensorHub fatal propagation remains owned by the existing
LeRobot/FR3 health path.

## Operator commands and lifecycle

Commands are validated against the state in which they are received. Invalid
commands are rejected immediately and are not deferred.

| Input | Valid state | Result |
| --- | --- | --- |
| `i` or `initialize` | `WAIT_START` | Send `INITIALIZE`; successful `INITIALIZED` completion enters `READY`. |
| `s` or `start` | `READY` | Start recording, send `START`, and enter `RUNNING` after `STARTED`. |
| `d` or `stop` | `RUNNING` | Send `STOP`, finish the recording as `done`, and return to `WAIT_START`. |
| `a` or `abort` | `RUNNING` | Send `ABORT`, finish the recording as `failed`, and return to `WAIT_START`. |
| `q` or `shutdown` | `WAIT_START`, `READY`, or `RUNNING` | Perform normal graceful session `SHUTDOWN`. |

A normal repeated session is:

```text
WAIT_START
  -> INITIALIZE -> READY
  -> START -> RUNNING
  -> STOP -> WAIT_START
```

The LeRobot process, policy/model, `Robot::FR3`, and SensorHub remain
persistent across these rollout cycles. MainController never supplies a
`q_reset`; reset target selection and the concrete reset handshake stay in the
FR3/LeRobot/NUC path.

### Normal shutdown

Explicit `q`/`shutdown` is the only graceful termination path. From `RUNNING`,
MainController first sends `STOP` and completes recording finalization, then
sends `SHUTDOWN` so LeRobot may perform its normal return-home behavior.

`shutdown` is rejected during blocking lifecycle operations such as
`INITIALIZE`; it is not queued for later. The operator may request it again
after the current operation settles. If LeRobot rejects `SHUTDOWN`, the
controller returns to `WAIT_START` only when the termination intent is still
`SHUTDOWN`. A concurrent promotion to `FAIL_STOP` is never cleared by the old
negative-ACK path.

### Ctrl-C and fatal termination

`KeyboardInterrupt`/SIGINT is a system-style `FAIL_STOP`, not a shortcut for
graceful `SHUTDOWN`. Ctrl-C therefore does not request a new return-home motion.

Required process failure, LeRobot/control failure, recording failure, sensor
runtime error/disconnect, or an unrecoverable MainController error also
establishes `FAIL_STOP`. MainController retransmits `FAIL_STOP` with increasing
wire sequences until delivery/fatal teardown/process exit is confirmed, waits
for bounded worker exit, and then stops remaining workstation resources. No
managed process is transparently restarted.

Do not use Ctrl-C when the intended operation is a normal graceful shutdown;
enter `q` instead.

## Rollout health and recovery classification

The following events are rollout-recoverable while `RUNNING`:

- explicit operator `abort`;
- robot FGT1 `JUMP_HOLD=1`;
- aligned `latest_sequence` failing to advance beyond
  `--aligned-stall-timeout-s`.

They cause `ABORT`, a `failed` rollout recording, and return to `WAIT_START`
without restarting LeRobot.

Temporary loss of NUC robot/gripper telemetry is not independently timed out
by MainController. If the loss causes the aligned sequence to stall during
`RUNNING`, the active rollout follows the recoverable aligned-stall path.

The v1 external implementation may not surface a SensorHub fatal through
LeRobot while LeRobot is blocked in an idle control phase. RUNNING/FR3
operations observe it through the normal LeRobot health path. This is an
accepted v1 external limitation, not a MainController implementation or
release blocker. MainController intentionally adds no SensorHub polling,
supervisor, readiness gate, or fatal propagation protocol.

## Recording and output

Each session creates:

```text
<runtime-root>/runtime_sessions/inference_<timestamp>_<pid>_<time_ns>/
  controller_events.jsonl
  session_manifest.json
  process_logs/
    ft300.log
    xense.log
    realsense.log
    rosbag_recorder.log
    lerobot.log
  rollouts/
    rollout_0001/
      manifest.json
      ft300_timestamps.npz
      xense_timestamps.npz
      zmq_telemetry.npz
      rosbag/
```

- `controller_events.jsonl` records state transitions, admitted/rejected
  commands, process events, sensor commands, health observations, and
  termination events.
- `session_manifest.json` records `termination_mode`, `termination_reason`,
  and the rollout count.
- `ft300_timestamps.npz` and `xense_timestamps.npz` contain MainController-side
  frame/timestamp receipt records, not copied sensor payloads.
- `zmq_telemetry.npz` contains the directly received raw FGT1 frame fields and
  MainController receipt timestamps.
- `rosbag/` is the per-rollout rosbag URI.
- `manifest.json` records rollout status, timestamps, sensor command results,
  relative artifact paths, frame counts, and any failure details.

FT300S and XenseTacSensor receive `<runtime-root>/runtime_frames` as their
sensor-side save directory.

## Tests and conformance audit

From the integration repository root:

```bash
conda deactivate
/usr/bin/python3 -m pytest -q \
  MainController/src/main_controller/test/test_inference_protocol.py \
  MainController/src/main_controller/test/test_inference_controller.py \
  MainController/src/main_controller/test/test_maincontroller_core.py
```

Collection regression coverage remains separate and should also stay green:

```bash
conda deactivate
/usr/bin/python3 -m pytest -q \
  MainController/src/main_controller/test/test_maincontroller_mock_runtime.py
```

The constraint-by-constraint implementation and external ownership audit is
maintained in
[`docs/inference_main_controller_constraint_conformance.md`](docs/inference_main_controller_constraint_conformance.md).

These tests do not operate an FR3/FCI, NUC runtime, RealSense device, FT300S,
or Xense hardware. Physical deployment requires a separate authorized hardware
validation.
