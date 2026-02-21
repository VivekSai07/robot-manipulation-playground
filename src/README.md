# MuJoCo Development & Robotics Notes

> **Project:** Universal Pick & Place Framework (Mark Series)
> **Core Tech:** MuJoCo 3.x, Pinocchio, Python

---

## Table of Contents

- [Architectural Principles](#-1-architectural-principles)
- [Physics & Simulation Bugs](#-2-physics--simulation-bugs-and-how-we-fixed-them)
- [The Iron Grip — Mastering MuJoCo Contacts](#-3-the-iron-grip--mastering-mujoco-contacts)
- [The Grand Lesson: Kinematics vs. Dynamics](#-4-the-grand-lesson-kinematics-vs-dynamics)

---

## 🏗️ 1. Architectural Principles

> The transition from *"writing scripts"* to *"building a robotics framework."*

### Separation of Concerns

Never mix robot definitions with task logic. The project follows a strict directory structure:

| Directory | Purpose |
|---|---|
| `robots/` | XMLs, URDFs, and a `robot.py` wrapper — **Hardware** |
| `controllers/` | Math and solvers like IK — **Brain** |
| `tasks/` | State machines and viewer loops — **Execution** |

### Universal API

Every robot (Franka, ViperX, WidowX) must expose the **exact same API** via `config.py` (e.g., `ACTIVE_JOINTS`, `ARM_DOF`, `Q_HOME`). This allows `pick_and_place.py` to be completely **robot-agnostic**.

### Native MuJoCo vs. URDF

While Pinocchio natively loves URDFs, we learned we can completely bypass them by extracting Jacobians and Kinematics directly from MuJoCo's engine (`mj_jacSite`, `mj_kinematics`). This is a massive time-saver for robots without clean URDFs (like the ViperX).

---

## 🔴 2. Physics & Simulation Bugs (And How We Fixed Them)

### Bug 1: The "Blind Grasp" — `TypeError: unhashable type: 'numpy.ndarray'`

- **Mistake:** Trying to detect grasps by checking geom string names, and incorrectly hashing MuJoCo's returned NumPy arrays.
- **Learning:** Collision geoms often don't have names in pre-built XMLs.
- **Fix:** Use Body IDs. Specifically, use `int(m.geom_bodyid[geom_index])` to cleanly extract the integer ID straight from MuJoCo's internal C-arrays, bypassing Python TypeErrors.

---

### Bug 2: The "Floor Crusher" — Bulldozing the Ground

- **Mistake:** Setting the IK target Z-height exactly to the center of the cube (`z = 0.02`).
- **Learning:** The End-Effector (EE) frame is usually at the wrist, but the fingertips extend several centimeters below it.
- **Fix:** Always add a `GRIPPER_Z_OFFSET` (e.g., `0.105` for Franka, `0.025` for ViperX) to the target position so the wrist stops high enough for the fingers to align with the payload.

---

### Bug 3: The "Zero-Gravity Cube" — Cube Floats into Space

- **Mistake:** Applying gravity compensation to the entire simulation state:
  ```python
  # WRONG
  d.qfrc_applied[:m.nv] = d.qfrc_bias[:m.nv]
  ```
- **Learning:** When adding a free-floating object (like a cube with `<freejoint/>`), MuJoCo adds 6 degrees of freedom to `m.nv`. Applying `qfrc_bias` to the cube literally turns off gravity for it.
- **Fix:** Restrict gravity compensation strictly to the robot's arm DOFs:
  ```python
  # CORRECT
  d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
  ```

---

### Bug 4: The "Jammed Gripper"

- **Mistake:** Initializing the ViperX `qpos` array with `0.0` for all joints.
- **Learning:** If a physical joint has a hard lower limit (e.g., ViperX fingers limit is `0.021`), initializing it at `0.0` creates a massive collision/violation at `t=0`. The physics solver violently clamps it, permanently jamming the actuator.
- **Fix:** Always initialize gripper joints to a valid `GRIPPER_OPEN` position explicitly.

---

## 🟢 3. The "Iron Grip" — Mastering MuJoCo Contacts

> Why the cube kept slipping out of the fingers, and how to create a perfect grasp.

### Premature Liftoff

The robot was lifting the exact millisecond `is_grasped` registered `True`. **Fix:** Force the state machine to wait for the full grasp duration so actuator force can fully build up before transitioning.

### Slippery Contacts

MuJoCo defaults are soft and slippery. Inside `panda.xml`, the collision geoms were upgraded with:

```xml
friction="5.0 0.5 0.0001"
condim="4"
solref="0.01 1"
solimp="0.9 0.95 0.001"
```

| Parameter | Effect |
|---|---|
| `friction="5.0 0.5 0.0001"` | Cranks up sliding and torsional friction |
| `condim="4"` | Enables torsional friction (stops the cube from twisting) |
| `solref` / `solimp` | Hardens the contact solver to stop microscopic bouncing/chatter |

### Weak Actuators

The default `deepmind actuator8` PD controller was only generating ~2 Newtons of force. **Fix:** Multiply `gainprm` and `biasprm` by **5** to deliver a ~10N industrial-strength grip.

---

## 🚀 4. The Grand Lesson: Kinematics vs. Dynamics

> The single most important realization of the **Mark-2** update.

### ❌ The Mistake — Kinematic Override

```python
# WRONG: Forcing velocities
d.qvel[:ARM_DOF] = dq
```

Overwriting the physics engine's state every frame turns the arm into an **"infinite-mass"** object. As the arm swings, it completely ignores Newton's 3rd Law. The physics solver can't calculate natural contact forces, causing the fingers to "teleport" and rip the cube out of the grip.

### ✅ The Solution — Dynamic Position Control

```python
# RIGHT: Integration + Actuator Control
q_target[idx] += dq[idx] * m.opt.timestep
d.ctrl[idx] = q_target[idx]
```

Instead of fighting the physics engine, we numerically integrate the IK velocity into a **target position** and send it to the robot's built-in PD Position Actuators via `d.ctrl`.

### The Result

The arm now moves using **finite, simulated motor torques**. Because it has real compliance, mass, and inertia, the contact solver can maintain a steady normal force on the cube — locking it securely in the gripper no matter how fast the arm swings.

---

*End of Log. Ready for Mark-3.*