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
| 1. MainController owns inference session, rollout, recording/bookkeeping, required workstation processes, fault classification, and termination mode | MainController | Satisfied | `inference_controller.InferenceMainController` owns these lifecycles. `test_inference_controller.py` covers repeated rollouts, recoverable abort, required-process fatal, and termination behavior. Existing collection controller in `main_controller/main.py` is unchanged. |
| 1. MainController does not own SensorHub/NUC processes, policy state, or `q_reset` | Shared boundary | Satisfied | `InferenceConfig.lerobot_command()` launches only the persistent worker and passes no reset target/randomization option. There is no SensorHub/NUC managed-process entry. |
| 2. LeRobot, policy/model, `Robot::FR3`, and SensorHub persist across rollouts | MainController / LeRobot | Satisfied | `_start_lerobot()` is session-scoped; rollout methods never restart it. `test_repeated_explicit_rollouts_keep_one_worker` covers two complete cycles through one fake worker. |
| 3. One ordinary transaction; ACK acceptance is not completion; accepted requests wait for completion STATUS; rejected requests expect no STATUS | MainController | Satisfied | `inference_protocol.ControlledClient.transact()` serializes ordinary operations and distinguishes ACK from completion STATUS; `test_inference_protocol.py` covers accepted, rejected, and concurrent callers. |
| 4. Validate inputs at receipt time; reject invalid input immediately; never defer it | MainController | Satisfied | `admit_user_action()` validates and reserves the transition under `state_lock` before handoff. Tests prove `START` and `SHUTDOWN` received during `INITIALIZING` are rejected and never queued/sent. |
| 5. LeRobot phase validation, negative ACK for new invalid requests, no ordinary queue, defensive drain only | LeRobot | External satisfied | `ControlledStrategy._validate_and_ack()` and `_transition_to()` in `LeRobotFR3/src/lerobot/rollout/strategies/controlled.py`; covered by `LeRobotFR3/tests/test_controlled_rollout.py`. |
| 6. `INITIALIZE` prepares the next rollout; MainController selects timing but never creates/passes `q_reset` | Shared boundary | Satisfied | `_initialize_rollout()` sends only `INITIALIZE`; `InferenceConfig` has no reset target. External `FR3.initialize_rollout()` owns randomization and target selection. |
| 6. NUC deterministically executes the concrete reset and does not randomize it | NUC | External satisfied | Current reverse server consumes the concrete `RESET_JOINT` target; reset randomization is absent from the NUC path. |
| 7. MainController uses LeRobot completion STATUS, never raw `RESETTING`, for `INITIALIZE` completion | MainController | Satisfied | `_initialize_rollout()` enters `READY` only after `INITIALIZED` STATUS. `test_raw_resetting_never_completes_initialize` proves FGT1 bit 0 does not change the lifecycle state. |
| 7. FR3 lower path owns `RESETTING` handshake | LeRobot / SensorHub / NUC | External satisfied | `FR3._reset_joints()` waits for `RESETTING=1` then `0`; existing focused LeRobot tests cover the handshake. |
| 8. Repeated explicit `WAIT_START -> INITIALIZE -> READY -> START -> RUNNING -> WAIT_START` cycles; no automatic reset/start | MainController | Satisfied | State transitions are command-driven; `test_repeated_explicit_rollouts_keep_one_worker` covers two explicit cycles and exact wire operation order. |
| 9. `WAIT_START` is not a NUC readiness guarantee; no v1 NUC readiness states/gate | MainController | Satisfied | `InferenceState` has no NUC readiness state and `admit_user_action('initialize')` is the explicit operator confirmation boundary. |
| 10. SensorHub remains persistent and owned by `Robot::FR3`; temporary NUC telemetry loss is recoverable without restart/epoch/SHM recreation | LeRobot / SensorHub | External satisfied | `FR3.connect()` owns SensorHub; current `SensorHubRuntime` preserves its persistent readers/caches and stops aligned publication during unavailable NUC telemetry. |
| 11. SensorHub fatal surfaces through LeRobot and terminates the worker/session | LeRobot / SensorHub | External blocked | `FR3._check_health()` consumes SensorHub fatal during FR3 operations, but the controlled worker does not poll FR3 health while blocked in an idle control phase. No external code is changed here. MainController may observe aligned fatal diagnostically but does not assume SensorHub ownership. |
| 12. MainController uses aligned SHM `ready/latest_sequence/fatal` as the RUNNING health gate and does not copy full payload | MainController | Satisfied | `AlignedHealthReader` maps only the 320-byte global header; startup checks `ready/fatal`, and `_watchdog_loop()` checks `ready/latest_sequence/fatal` only while RUNNING. Unit tests cover header reads and stall behavior. |
| 12. Prefer a formal external header-only/read-only aligned SHM API | LeRobot / SensorHub | External blocked | Current external `AlignedObservationClient.read()` copies and decodes a full slot; no public header-only API exists. A version-pinned read-only header fallback will be isolated locally without importing policy/robot state. |
| 13. FGT1 fans out to SensorHub and MainController; MainController records raw telemetry and observes control flags without becoming a NUC recovery protocol | Shared boundary | Satisfied | `ZmqTelemetryReceiver` is session-persistent; FGT1 `flags` are decoded and saved in rollout `zmq_telemetry.npz`. MainController only consumes `JUMP_HOLD` as a control flag and has no NUC control/recovery API. |
| 14. User abort, `JUMP_HOLD`, aligned stall, temporary NUC telemetry loss, and manually recoverable NUC failure remain rollout recoverable | MainController | Satisfied | `request_rollout_abort()` maps these RUNNING health outcomes to `ABORT -> failed recording -> WAIT_START` without setting a session termination mode. Absence of direct FGT1 is not independently timed out. |
| 15. RUNNING aligned sequence stall causes ABORT; MainController threshold is shorter than LeRobot max-age threshold | MainController | Satisfied | `_watchdog_loop()` admits recoverable ABORT on a stall; `InferenceConfig` rejects equal/reversed threshold ordering. `test_aligned_stall_requests_recoverable_abort` covers the path. |
| 15. LeRobot max-age watchdog is armed only in phases that need fresh observations and disarmed between rollouts | LeRobot | External satisfied | Controlled strategy calls `robot.get_observation()` only inside `_run_rollout()` and returns to blocking control waits outside RUNNING. |
| 16. `JUMP_HOLD=1` while RUNNING causes immediate recoverable abort | MainController | Satisfied | `_on_zmq_frame()` atomically reserves ABORT on robot flag bit 1. The focused test verifies failed recording and return to `WAIT_START`. |
| 17. Required process exits/fatals, LeRobot/session/policy/reset failure, and internal error are session fatal; no transparent process restart | MainController | Satisfied | All managed process callbacks and lifecycle failures call `request_fail_stop()`; no inference path calls `ManagedProcess.restart()`. Required-process exit classification is unit tested. |
| 18. Normal rosbag record/stop lifecycle is allowed; recorder process recovery restart is forbidden | MainController | Satisfied | `_begin_recording()`/`_finish_recording()` use recorder services per rollout. Recorder fatal/exit uses session `FAIL_STOP`; there is no recorder process restart. |
| 19. SHUTDOWN is user-only, state-valid, non-deferred, and permits return-home; every system fault maps to no-new-motion FAIL_STOP | MainController / LeRobot | Satisfied | Admission accepts user `shutdown` only in `WAIT_START`/`READY`; faults call `request_fail_stop()`. External worker separates `_graceful_shutdown()` return-home from `_fatal_stop()`. |
| 20. FAIL_STOP retransmits with increasing sequence until positive ACK, fatal teardown STATUS, or worker exit; ACK is not teardown completion; bounded wait/force termination follows | MainController | Satisfied | `_execute_fail_stop()` retries via fresh `ControlledClient.send()` sequences, stops delivery on the specified evidence, then `_wait_for_worker_exit()` enforces bounded exit/force-stop. The focused test requires three transmissions before ACK. |
| 21. No reset-preemption receiver requirement; FAIL_STOP may continue transmitting while synchronous reset blocks | Shared boundary | Satisfied | MainController response multiplexing and fail-stop thread do not modify LeRobot. The external worker remains single-threaded; periodic commands can wait in the transport until reset returns. |
| 22. Existing LeRobot fatal/exit is sufficient; controller disconnect is fatal; no heartbeat | Shared boundary | Satisfied | Control disconnect and worker exit are session fatal; no heartbeat was added. External `ControlledUDSDisconnected` already makes loss of the sole client fatal. |
| 23. MainController owns recording; raw FGT1 remains direct; abort consistently fails/stops recording; recording fatal escalates | MainController | Satisfied | `InferenceRolloutStore`, sensor commands, and rosbag services are MainController-owned. ABORT writes a failed manifest; cleanup errors raise into session `FAIL_STOP`. |
| 24. Preserve termination mode/reason and required SHUTDOWN/FAIL_STOP teardown ordering | MainController | Satisfied | `termination_mode`/`termination_reason` are persisted in `session_manifest.json`; both termination paths wait/force LeRobot before `_stop_runtime_resources()` stops its dependencies. |
| 25. Explicit v1 non-goals remain absent | MainController | Planned | Final source audit pending; focused tests already cover no deferred input, no reset target, and no required-process restart behavior. |
| 26. Core invariants hold together | Shared boundary | Planned | Final constraint-by-constraint audit pending. |

## External evidence baseline

- Controlled lifecycle wire protocol: `LeRobotFR3/src/lerobot/rollout/control_uds.py`.
- Persistent controlled worker and lifecycle phase validation: `LeRobotFR3/src/lerobot/rollout/strategies/controlled.py`.
- FR3-owned SensorHub lifecycle and reset handshake: `LeRobotFR3/src/lerobot/robots/fr3/fr3.py`.
- Aligned observation ABI: `LeRobotFR3/src/lerobot/robots/fr3/sensorhub/aligned_shm.py`.
- Recoverable NUC telemetry behavior: `LeRobotFR3/src/lerobot/robots/fr3/sensorhub/runtime.py` and `cache.py`.
- NUC FGT1/reset flags and deterministic reset target consumption: `zmq_franka_gello/franka_gello_zmq/telemetry.py` and the reverse server path.
