# Robot Manipulation Playground

> **Project:** Universal Pick & Place Framework (The Mark Series)
> **Core Tech:** MuJoCo 3.x, Pinocchio, OpenCV, Python

This repository contains a progressively evolving robotics framework designed to execute complex pick-and-place experiments with a Franka Panda arm using native MuJoCo simulation. 

---

## 🚀 How to Run

> **One-time setup per terminal session** — set the project root on `PYTHONPATH`:
> ```powershell
> $env:PYTHONPATH = "D:\Mujoco\MyLearningSpace"   # PowerShell
> ```

| Mark | Command |
|------|---------|
| M1 — Basic IK | `python src/tasks/m01_basic_ik/run.py` |
| M2 — Nullspace IK | `python src/tasks/m02_nullspace_ik/run.py` |
| M3 — Trajectory Planner | `python src/tasks/m03_trajectory_planner/run.py` |
| M4 — Sense, Plan & Recover | `python src/tasks/m04_sense_plan_recover/run.py` |
| M4 — Sabotage mode | `python src/tasks/m04_sense_plan_recover/run.py --sabotage` |
| M5 — RRT Motion Planning | `python src/tasks/m05_rrt_planning/run.py` |
| M6 — Keyboard Teleoperation | `python src/tasks/m06_keyboard_teleop/run.py` |
| M7 — Vision Pipeline | `python src/tasks/m07_machine_vision/run_vision.py --prompt "pick up the blue block"` |
| M7 — Segmentation Pipeline | `python src/tasks/m07_machine_vision/run_segmentation.py --color blue` |
| M7 — Debug mode | append `--debug` to either M7 command |
| M8 — Autonomous Avoidance | `python src/tasks/m08_autonomous_avoidance/run.py --color red` |
| M8 — Continuous loop mode | append `--loop` to run indefinitely after each pick+place |
| M8 — Debug mode | append `--debug` to show segmentation view during scan |
| M9–M14 | *(commands added as each mark is finalized)* |

---

## 🌟 Concept Evolution: The Mark Series

The framework has evolved significantly, progressing through different paradigms of robot control, machine vision, and motion planning.

### Mark 1 & 2: Inverse Kinematics (IK) & Dynamic Position Control
- **Concept:** Mapping task-space Cartesian goals to joint-space commands via the Jacobian pseudo-inverse.
- **Implementation:** The `IKController` computes joint velocities which are then integrated into target positions for MuJoCo's built-in PD Position Actuators. This ensures the arm moves using natural, finite motor torques rather than non-physical kinematic overrides.

### Mark 3: Trajectory Planning
- **Concept:** Replacing reactive IK chasing with pre-planned, time-scaled paths between waypoints so the end-effector always travels in a straight line through task space.
- **Implementation:** `TaskSpaceTrajectory` uses cubic time-scaling ($s = 3\alpha^2 - 2\alpha^3$) for smooth acceleration/deceleration at segment boundaries. Translation is decoupled from rotation — forced to follow a strict Euclidean straight line — eliminating the curvy arc that caused the gripper to clip the cube on retract in M1/M2. Features a real-time trail visualiser (green dots) and moving-target sphere (red) rendered directly in the MuJoCo viewer.

### Mark 4: Sense, Plan & Recover
- **Concept:** Dynamic "Sense & Plan" pipelines that react to the environment rather than blindly executing static waypoints, with closed-loop failure recovery.
- **Implementation:** `generate_task_states()` reads the cube's live position from `d.xpos` each time it is called, so waypoints are always grounded in reality. A dedicated "Verify Lift" state checks `is_grasped()` after every pick; on failure it re-scans the cube, resets the state machine, and retries (up to `MAX_RETRIES`). Includes a `--sabotage` CLI flag that teleports the cube mid-grasp to stress-test the recovery path.

### Mark 5: RRT Motion Planning (The Brain)
- **Concept:** Finding geometric, collision-free paths in joint-space using Rapidly-Exploring Random Trees (RRT) before execution.
- **Implementation:** The `RRT` planner injects "ghost states" into MuJoCo's native collision engine (`mj_kinematics`, `mj_collision`) to validate random samples in milliseconds. Features Goal Biasing and Path Shortcut Smoothing to generate perfect, sweeping Joint-Space trajectories.

### Mark 6: Teleoperation & Data Collection (Human-in-the-Loop)
- **Concept:** Allowing a human operator to drive the robot in real-time, bridging the gap between autonomous algorithms and imitation learning.
- **Implementation:** Custom `KeyboardTeleop` and `PS4Teleop` interfaces capture inputs (WASD/Arrow keys or PS4 analog sticks) and map them back to velocity targets for the `IKController`. Includes pipelines for continuous logging (`teleop_data_collection.py` and `ps4_data_collection.py`) to record observations and actions for future neural network training.

### Mark 7: Machine Vision (The Eyes)
- **Concept:** Granting the robot the ability to dynamically locate chaotic payloads instead of relying on hardcoded coordinates.
- **Implementation:** Extensive use of `mujoco.Renderer` to capture RGB/Depth/Segmentation data. We implemented two distinct perception systems (Computer Vision vs True Segmentation) capable of converting 2D pixels back into 3D world coordinates.
- 📖 **See deep-dive documentation:** [`src/perception/README.md`](src/perception/README.md)

### Mark 8: Autonomous Avoidance (The Full System)
- **Concept:** The crowning achievement. Grafting the Segmentation Perception pipeline (Eyes) directly into the RRT Motion Planner (Brain).
- **Implementation:** The robot executes a "Perception Stow", visually scans the scattered table, locates the target, and autonomously plans a heavily-arced RRT trajectory to snatch the cube from behind a 30cm glass wall without a single collision.

### Mark 9: YOLOv8 Object Detection (Real-Time AI)
- **Concept:** Fast, lightweight, bounding-box object detection for continuous "neural physics" tracking.
- **Implementation:** Integrated `ultralytics` YOLOv8. Replaced colored geometric blocks with high-fidelity Google Scanned Objects (Oranges, Bowls). The arm continuously sweeps the area with an angled camera while YOLO predicts coordinates.

### Mark 10: VR MoCap Teleoperation
- **Concept:** High-fidelity human data collection via spatial tracking.
- **Implementation:** Connected Oculus VR controller tracking to the IK Controller (`teleop_m10_mocap.py`). This allows 6D pose matching, moving beyond analog sticks to capture true human intent for neural network imitation learning.

### Mark 11: VLM Reasoning (Qwen-VL)
- **Concept:** Moving from rigid classes (YOLO) to semantic understanding. You can ask the robot to "pick up a healthy snack", and it will reason about the scene to find the orange.
- **Implementation:** Integrated the `Qwen-VL-Chat` Vision-Language Model. The model analyzes an RGB frame against a text prompt to output a bounding box, which is then dynamically converted into a grab-able physics component.

### Mark 12: Florence-2 (Advanced VLA Integration)
- **Concept:** The pinnacle of our semantic pipeline. Faster and more robust reasoning than Qwen.
- **Implementation:** Replaced the heavy Qwen model with Microsoft's `Florence-2-large`. Developed a robust State Machine that handles "Clenched Fist" self-collision protections, handles dynamic Z-heights based on the semantic target (e.g. grasping a tall bottle vs a flat box), and includes a Slip-Detection Recovery Protocol that automatically kicks in if the payload is dropped during RRT execution.

### Mark 13: Reactive Avoidance (APF)
- **Concept:** Moving away from static, blind global planners (RRT) towards local, real-time 500Hz responsive systems capable of dodging moving threats dynamically.
- **Implementation:** Implemented an **Artificial Potential Field (APF)** controller. Attractive forces pull the End-Effector towards the goal, while Repulsive forces from surrounding dynamic obstacles are translated into Joint-Space Velocities via Pinocchio Jacobians ($J^T_{link} F_{rep}$) and injected directly into the Operational Null-Space, forcing the entire arm to reflexively tuck away from danger natively out-of-the-box.

---

## 🏗️ Architectural Principles

### Separation of Concerns
Never mix robot definitions with task logic. The project follows a strict directory structure:

| Directory | Purpose |
|---|---|
| `robots/` | XMLs, URDFs, and a `robot.py` wrapper — **Hardware** |
| `controllers/` | Math and solvers (IK, APF, Grasping, Teleop) — **Low-Level Control** |
| `perception/` | Cameras, Segmentation, OpenCV — **Sensing** |
| `planners/` | RRT, Task/Joint Space Trajectories — **Foresight** |
| `tasks/` | State machines and viewer loops — **Execution** |

### Universal API
Every robot must expose the **exact same API** via `config.py` (e.g., `ACTIVE_JOINTS`, `ARM_DOF`, `Q_HOME`). This allows task scripts to be completely **robot-agnostic**.

## Deep Dive Documentation

For granular details on exactly how this framework overcomes complex robotic challenges, refer to our specialized domain readmes:

- 🏗️ **[Core Architecture & Physics Tuning](src/README.md)**: Deep dive into how we solved MuJoCo simulation bugs, achieved an industrial-strength grasp ("The Iron Grip"), and the critical mathematical transition from purely Kinematic Inverse-Jacobians to Trajectory-Planned Dynamic Position Control.
- 👁️ **[Machine Perception](src/perception/README.md)**: Details on generating ground-truth Segmentation Masks vs constructing Computer Vision RGB/HSV thresholding pipelines.
