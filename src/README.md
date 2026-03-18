# MuJoCo Development & Robotics Notes

> **Project:** Universal Pick & Place Framework (Mark Series)
> **Core Tech:** MuJoCo 3.x, Pinocchio, Python

---

## Table of Contents

- [Physics & Simulation Bugs](#-1-physics--simulation-bugs-and-how-we-fixed-them)
- [The Iron Grip — Mastering MuJoCo Contacts](#-2-the-iron-grip--mastering-mujoco-contacts)
- [The Grand Lesson: Kinematics vs. Dynamics](#-3-the-grand-lesson-kinematics-vs-dynamics)
- [Mark-11: VLA & Open-Vocabulary Bugs](#-4-mark-11-vla--open-vocabulary-bugs)
- [Mark-12: Florence-2 & Deep Learning Traps](#️-5-mark-12-florence-2--deep-learning-traps)
- [Mark-13: The APF End-Effector Blindspot](#-6-mark-13-the-apf-end-effector-blindspot)

## 🔴 1. Physics & Simulation Bugs (And How We Fixed Them)


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

## 🟢 2. The "Iron Grip" — Mastering MuJoCo Contacts

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

## 🚀 3. The Grand Lesson: Kinematics vs. Dynamics

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

## 🧠 4. Mark-11: VLA & Open-Vocabulary Bugs

> Integrating OWL-ViT and LLM reasoning into physical simulation control loops.

### Bug 5: The "Screw Motion" (Curved Trajectories)
- **Symptom:** When descending to pick up the object, the robot arm moved in a curved, sweeping arc, knocking the target out of the way.
- **Cause:** Using standard `pin.SE3.Interpolate()` couples translation and rotation. If the RRT planner leaves a 1-degree rotational error, the arm performs a 3D "screw motion" to fix the rotation while translating.
- **Fix:** Decouple the interpolation in the `TaskSpaceTrajectory` planner. Force the translation to follow a strict Euclidean straight line, and let Pinocchio handle the rotational SLERP independently.

### Bug 6: The "Ghost Payload" & Start-State Clamping
- **Symptom:** The RRT planner consistently failed with `❌ RRT Error: Start position is currently in collision!` right after grasping an object.
- **Cause:** Two paradoxes:
    1. To grip an object, the virtual fingers must microscopically penetrate the object's collision mesh, which RRT flags as an illegal collision.
    2. During RRT's virtual "what-if" planning phase, the robot arm swings around, but the physical payload remains frozen in mid-air, causing the swinging arm to crash into it.
- **Fix (The Magician's Trick):** Instantly teleport all semantic objects high into the sky (Z=5.0) just before calculating the RRT path, then teleport them back to the gripper before unpausing the physics engine.

### Bug 7: Micro-Slips and Sensor Noise
- **Symptom:** Grazing the wall caused the task to abort, even though the robot didn't drop the object.
- **Cause:** Rigid body physics are brittle. A slight bump caused the contact force to drop to 0 for exactly 1 millisecond. The script instantly saw `is_holding == False` and aborted.
- **Fix:** Implemented a Sensor Debounce Filter. If contact is lost, start a `slip_debounce_time` tracker. Only abort the mission if the object remains lost for `>0.25` seconds.

### Bug 8: OpenMP DLL Crash (The Python Library Collision)
- **Symptom:** Program crashes instantly with `OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.`
- **Cause:** PyTorch (for the VLM) and OpenCV (for the camera window) both attempt to load their own internal C++ OpenMP multiprocessing libraries, triggering a memory safety panic on Windows.
- **Fix:** Injected `os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'` at the very top of the perception pipeline.

---

## 👁️ 5. Mark-12: Florence-2 & Deep Learning Traps

> Tackling Hugging Face architectural constraints and conversational NLP.

### Bug 9: The `flash_attn` Trap (Static AST Panic)
- **Symptom:** Script crashes with `ImportError: ... requires flash_attn` even though it's an optional dependency. After faking the module, it crashes with `ValueError: flash_attn.__spec__ is not set.`
- **Cause:** Hugging Face's older dynamic module loader (`check_imports`) scans the text of Microsoft's custom model code and panics if it sees the word `flash_attn`. Furthermore, `importlib` requires mock modules to have a valid Python Module Spec.
- **Fix:** Robustly monkey-patched the system by injecting a perfectly mocked module type with a valid `__spec__` into `sys.modules`, bypassing the AST checker and safely tricking `is_flash_attn_2_available()` into returning `False`.
  ```python
  import types, importlib.machinery
  mock_flash_attn = types.ModuleType('flash_attn')
  mock_flash_attn.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
  sys.modules['flash_attn'] = mock_flash_attn
  ```

### Bug 10: The "Clenched Fist" Self-Collision
- **Symptom:** RRT fails immediately upon boot with a start-state collision, even when the table is empty.
- **Cause:** During the IDLE state while waiting for a text prompt, the script sent joint hold commands but didn't command the gripper. It defaulted to 0 (closed). The empty fingers clamped together, their collision meshes perfectly intersecting, causing the RRT safety check to fail.
- **Fix:** Explicitly commanded the gripper to stay open (255) while waiting in the IDLE state.

### Bug 11: Conversational NLP Failure (Phrase Grounding)
- **Symptom:** Florence-2 successfully found "red ball", but completely ignored "could you please pick up the red ball".
- **Cause:** Florence-2 is not an LLM chat agent; it uses `<CAPTION_TO_PHRASE_GROUNDING>`. It looks for physical pixels representing the concept of "could you please," fails, and returns nothing.
- **Fix:** Built a lightweight NLP text scrubber to dynamically strip polite filler words ("could you please", "can you grab", etc.) before passing the core noun phrase to the Vision-Language Model.

---

## 🛡️ 6. Mark-13: The APF End-Effector Blindspot

> Translating Cartesian Repulsions into Differential Kinematics for true Whole-Body Avoidance.

### Bug 12: The APF "End-Effector Blindspot"
- **Symptom:** During reactive obstacle avoidance, an incoming pendulum successfully repelled the gripper, but completely smashed into the robot's elbow/shoulder links. 
- **Cause:** APF math natively calculates distance from the threat to a single Cartesian point (the End-Effector). If a long robotic arm extends forward, dodging the EE in the XY plane does little to protect the massive joints sticking out mid-air.
- **Fix:** Upgraded from a simplistic Cartesian EE-repulsion shield to a **Whole-Body Differential Kinematics Shield**. By querying Pinocchio for the Jacobians of intermediate frames (Elbow, Wrist) at 500Hz, we project Cartesian threat repulsions dynamically into Joint-Space vector velocities ($\Delta q_{rep} = J_{elbow}^T F_{repulsive}$). We then inject this vector into the IK Controller's Null-Space Optimizer (`q_posture`), forcing the robot to physically tuck its elbow backwards to survive while maintaining flawless End-Effector task stability.

---

*End of Log.*
