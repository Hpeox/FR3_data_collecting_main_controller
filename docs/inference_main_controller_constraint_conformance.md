# Inference MainController Constraint Conformance

Normative source: `../../../../FR3 Inference MainController Design Constraints.md`

This matrix is maintained with the inference MainController implementation. `External` means the constraint is owned by the checked-out LeRobotFR3, SensorHub, or NUC implementation and is not changed by this package. `Blocked` identifies a concrete external contract gap rather than silently moving that ownership into MainController.

## Status summary

| Status | Meaning |
| --- | --- |
| Planned | MainController implementation or test evidence is still required. |
| Satisfied | Implemented in this package with cited evidence. |
| External satisfied | Verified in the current local external implementation. |
| External blocked | The current external implementation does not completely expose or enforce the required contract. |
| Not applicable | An explicitly permitted option is intentionally not implemented. |

## Conformance matrix

| Constraint | Owner | Status | Implementation and test evidence |
| --- | --- | --- | --- |
| 1. MainController owns inference session, rollout, recording/bookkeeping, required workstation processes, fault classification, and termination mode | MainController | Planned | Inference controller and focused tests pending. Existing collection controller in `main_controller/main.py` remains unchanged. |
| 1. MainController does not own SensorHub/NUC processes, policy state, or `q_reset` | Shared boundary | Planned | Inference process command will launch only persistent LeRobot; no SensorHub/NUC process entry or reset target belongs in its config. |
| 2. LeRobot, policy/model, `Robot::FR3`, and SensorHub persist across rollouts | MainController / LeRobot | Planned | One session-scoped managed LeRobot process; repeated-cycle test pending. External worker is loop-based in `LeRobotFR3/src/lerobot/rollout/strategies/controlled.py`. |
| 3. One ordinary transaction; ACK acceptance is not completion; accepted requests wait for completion STATUS; rejected requests expect no STATUS | MainController | Satisfied | `inference_protocol.ControlledClient.transact()` serializes ordinary operations and distinguishes ACK from completion STATUS; `test_inference_protocol.py` covers accepted, rejected, and concurrent callers. |
| 4. Validate inputs at receipt time; reject invalid input immediately; never defer it | MainController | Planned | Atomic input admission and no command backlog scheduler pending. |
| 5. LeRobot phase validation, negative ACK for new invalid requests, no ordinary queue, defensive drain only | LeRobot | External satisfied | `ControlledStrategy._validate_and_ack()` and `_transition_to()` in `LeRobotFR3/src/lerobot/rollout/strategies/controlled.py`; covered by `LeRobotFR3/tests/test_controlled_rollout.py`. |
| 6. `INITIALIZE` prepares the next rollout; MainController selects timing but never creates/passes `q_reset` | Shared boundary | Planned | MainController transaction pending. External `FR3.initialize_rollout()` owns randomization and target selection. |
| 6. NUC deterministically executes the concrete reset and does not randomize it | NUC | External satisfied | Current reverse server consumes the concrete `RESET_JOINT` target; reset randomization is absent from the NUC path. |
| 7. MainController uses LeRobot completion STATUS, never raw `RESETTING`, for `INITIALIZE` completion | MainController | Planned | Protocol transaction test pending. Direct FGT1 `RESETTING` is recorded only. |
| 7. FR3 lower path owns `RESETTING` handshake | LeRobot / SensorHub / NUC | External satisfied | `FR3._reset_joints()` waits for `RESETTING=1` then `0`; existing focused LeRobot tests cover the handshake. |
| 8. Repeated explicit `WAIT_START -> INITIALIZE -> READY -> START -> RUNNING -> WAIT_START` cycles; no automatic reset/start | MainController | Planned | Repeated-cycle and invalid-deferred-input tests pending. |
| 9. `WAIT_START` is not a NUC readiness guarantee; no v1 NUC readiness states/gate | MainController | Planned | State model and tests pending. |
| 10. SensorHub remains persistent and owned by `Robot::FR3`; temporary NUC telemetry loss is recoverable without restart/epoch/SHM recreation | LeRobot / SensorHub | External satisfied | `FR3.connect()` owns SensorHub; current `SensorHubRuntime` preserves its persistent readers/caches and stops aligned publication during unavailable NUC telemetry. |
| 11. SensorHub fatal surfaces through LeRobot and terminates the worker/session | LeRobot / SensorHub | External blocked | `FR3._check_health()` consumes SensorHub fatal during FR3 operations, but the controlled worker does not poll FR3 health while blocked in an idle control phase. No external code is changed here. MainController may observe aligned fatal diagnostically but does not assume SensorHub ownership. |
| 12. MainController uses aligned SHM `ready/latest_sequence/fatal` as the RUNNING health gate and does not copy full payload | MainController | Planned | `aligned_health.AlignedHealthReader` is implemented and tested as a 320-byte read-only mapping; RUNNING watchdog integration remains pending. |
| 12. Prefer a formal external header-only/read-only aligned SHM API | LeRobot / SensorHub | External blocked | Current external `AlignedObservationClient.read()` copies and decodes a full slot; no public header-only API exists. A version-pinned read-only header fallback will be isolated locally without importing policy/robot state. |
| 13. FGT1 fans out to SensorHub and MainController; MainController records raw telemetry and observes control flags without becoming a NUC recovery protocol | Shared boundary | Planned | Direct receiver already exists; flags, rollout recording, and `JUMP_HOLD` handling pending. ZMQ PUB/SUB relay supports independent subscribers. |
| 14. User abort, `JUMP_HOLD`, aligned stall, temporary NUC telemetry loss, and manually recoverable NUC failure remain rollout recoverable | MainController | Planned | Fault classifier and abort-recording tests pending. |
| 15. RUNNING aligned sequence stall causes ABORT; MainController threshold is shorter than LeRobot max-age threshold | MainController | Planned | `InferenceConfig` rejects equal/reversed thresholds; RUNNING watchdog integration remains pending. |
| 15. LeRobot max-age watchdog is armed only in phases that need fresh observations and disarmed between rollouts | LeRobot | External satisfied | Controlled strategy calls `robot.get_observation()` only inside `_run_rollout()` and returns to blocking control waits outside RUNNING. |
| 16. `JUMP_HOLD=1` while RUNNING causes immediate recoverable abort | MainController | Planned | FGT1 flag decode and controller test pending. |
| 17. Required process exits/fatals, LeRobot/session/policy/reset failure, and internal error are session fatal; no transparent process restart | MainController | Planned | Required-process supervision and FAIL_STOP mapping tests pending. |
| 18. Normal rosbag record/stop lifecycle is allowed; recorder process recovery restart is forbidden | MainController | Planned | Per-rollout rosbag control and process-exit test pending. |
| 19. SHUTDOWN is user-only, state-valid, non-deferred, and permits return-home; every system fault maps to no-new-motion FAIL_STOP | MainController / LeRobot | Planned | Termination classifier and tests pending. External worker separates `_graceful_shutdown()` from `_fatal_stop()`. |
| 20. FAIL_STOP retransmits with increasing sequence until positive ACK, fatal teardown STATUS, or worker exit; ACK is not teardown completion; bounded wait/force termination follows | MainController | Planned | `ControlledClient.send()` allocates a fresh monotonic sequence per send and multiplexes ACK/STATUS concurrently; termination-loop integration remains pending. |
| 21. No reset-preemption receiver requirement; FAIL_STOP may continue transmitting while synchronous reset blocks | Shared boundary | Planned | MainController retransmission design pending; external worker remains single-threaded. |
| 22. Existing LeRobot fatal/exit is sufficient; controller disconnect is fatal; no heartbeat | Shared boundary | Planned | MainController worker-exit/disconnect handling pending. External `ControlledUDSDisconnected` already makes loss of the sole client fatal. |
| 23. MainController owns recording; raw FGT1 remains direct; abort consistently fails/stops recording; recording fatal escalates | MainController | Planned | Rollout recorder and manifest tests pending. |
| 24. Preserve termination mode/reason and required SHUTDOWN/FAIL_STOP teardown ordering | MainController | Planned | Session manifest/event log and ordering tests pending. |
| 25. Explicit v1 non-goals remain absent | MainController | Planned | Final source audit and focused negative tests pending. |
| 26. Core invariants hold together | Shared boundary | Planned | Final constraint-by-constraint audit pending. |

## External evidence baseline

- Controlled lifecycle wire protocol: `LeRobotFR3/src/lerobot/rollout/control_uds.py`.
- Persistent controlled worker and lifecycle phase validation: `LeRobotFR3/src/lerobot/rollout/strategies/controlled.py`.
- FR3-owned SensorHub lifecycle and reset handshake: `LeRobotFR3/src/lerobot/robots/fr3/fr3.py`.
- Aligned observation ABI: `LeRobotFR3/src/lerobot/robots/fr3/sensorhub/aligned_shm.py`.
- Recoverable NUC telemetry behavior: `LeRobotFR3/src/lerobot/robots/fr3/sensorhub/runtime.py` and `cache.py`.
- NUC FGT1/reset flags and deterministic reset target consumption: `zmq_franka_gello/franka_gello_zmq/telemetry.py` and the reverse server path.
