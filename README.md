# Robot Manipulation Playground

> **Project:** Universal Pick & Place Framework (The Mark Series)
> **Core Tech:** MuJoCo 3.x, Pinocchio, OpenCV, Python

This repository contains a progressively evolving robotics framework designed to execute complex pick-and-place experiments with a Franka Panda arm using native MuJoCo simulation. 

---

## 🌟 Concept Evolution: The Mark Series

The framework has evolved significantly, progressing through different paradigms of robot control, machine vision, and motion planning.

### Mark 1 & 2: Inverse Kinematics (IK) & Dynamic Position Control
- **Concept:** Mapping task-space Cartesian goals to joint-space commands via the Jacobian pseudo-inverse.
- **Implementation:** The `IKController` computes joint velocities which are then integrated into target positions for MuJoCo's built-in PD Position Actuators. This ensures the arm moves using natural, finite motor torques rather than non-physical kinematic overrides.

### Mark 3: Operational Space Control (OSC)
- **Concept:** Direct force/torque control in task space using Task-Space Impedance (PD) Control.
- **Implementation:** The `OSCController` skips joint-position targets and directly calculates joint torques (`tau`) using the Operational Space Mass Matrix ($\Lambda$). Includes Dynamically Consistent Nullspace Projection to control redundant degrees of freedom without disturbing the end-effector.

### Mark 4: Robust State Machines & Trajectory Planning
- **Concept:** Dynamic "Sense & Plan" pipelines that react to the environment rather than blindly executing static waypoints.
- **Implementation:** Real-time generation of `TaskSpaceTrajectory` (Cubic Cartesian Interpolation) based on the *actual* position of the cube. Features built-in recovery protocols for dropped payloads mid-grasp.

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

---

## 🏗️ Architectural Principles

### Separation of Concerns
Never mix robot definitions with task logic. The project follows a strict directory structure:

| Directory | Purpose |
|---|---|
| `robots/` | XMLs, URDFs, and a `robot.py` wrapper — **Hardware** |
| `controllers/` | Math and solvers (IK, OSC, Grasping) — **Low-Level Control** |
| `perception/` | Cameras, Segmentation, OpenCV — **Sensing** |
| `planners/` | RRT, Task/Joint Space Trajectories — **Foresight** |
| `tasks/` | State machines and viewer loops — **Execution** |

### Universal API
Every robot (Franka, ViperX, UR5e) must expose the **exact same API** via `config.py` (e.g., `ACTIVE_JOINTS`, `ARM_DOF`, `Q_HOME`). This allows task scripts (like `pick_and_place.py`) to be completely **robot-agnostic**.

## Deep Dive Documentation

For granular details on exactly how this framework overcomes complex robotic challenges, refer to our specialized domain readmes:

- 🏗️ **[Core Architecture & Physics Tuning](src/README.md)**: Deep dive into how we solved MuJoCo simulation bugs, achieved an industrial-strength grasp ("The Iron Grip"), and the critical mathematical transition from purely Kinematic Inverse-Jacobians to Dynamic Operational Space Control (OSC).
- 👁️ **[Machine Perception](src/perception/README.md)**: Details on generating ground-truth Segmentation Masks vs constructing Computer Vision RGB/HSV thresholding pipelines.
