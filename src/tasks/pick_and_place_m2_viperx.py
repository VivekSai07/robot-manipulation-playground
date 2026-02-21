import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.robots.ViperX.robot import ViperX
from src.robots.ViperX.config import (
    SCENE_PATH, Q_HOME, ARM_DOF, ACTIVE_JOINTS,
    GRIPPER_OPEN, GRIPPER_CLOSED
)

def main():
    print("🚀 Initializing ViperX Pick and Place...")

    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    # ViperX wrapper (No URDF needed, uses MuJoCo directly)
    robot = ViperX(m, d)
    
    # Mark-2 IK Controller
    # ViperX is a 6-DOF arm, so it uses 6 active joints
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    # Grasp Controller mapped to ViperX finger bodies
    grasp_sys = GraspController(m, d, left_finger="left_finger_link", right_finger="right_finger_link")

    # -------------------------------
    # Set Initial Pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    # CRITICAL FIX 1: Initialize gripper joints to a valid OPEN position!
    # If left at 0.0, they violate the 0.021 physical limit and permanently jam.
    d.qpos[6] = GRIPPER_OPEN
    d.qpos[7] = -GRIPPER_OPEN
    
    # Position tracking array for Position Actuators
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    # Capture initial downward orientation using the "pinch" site
    home_pose = robot.forward_kinematics(Q_HOME)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # -------------------------------
    # Dynamic Waypoints
    # -------------------------------
    # CRITICAL FIX 2: Raise Z-height to 0.045
    # The pinch site is centered, but the fingers extend 2.5cm further down.
    # Aiming for 0.045 allows the fingers to wrap the cube without hitting the floor.
    PICK_POS = np.array([0.35, 0.0, 0.045])
    PLACE_POS = np.array([0.0, 0.35, 0.1])
    
    states = [
        {"name": "Approach Pick", "pos": PICK_POS + np.array([0, 0, 0.15]), "gripper": GRIPPER_OPEN, "duration": 2.0},
        {"name": "Descend to Pick", "pos": PICK_POS, "gripper": GRIPPER_OPEN, "duration": 1.5},
        {"name": "Grasping", "pos": PICK_POS, "gripper": GRIPPER_CLOSED, "duration": 1.0},
        {"name": "Lift", "pos": PICK_POS + np.array([0, 0, 0.2]), "gripper": GRIPPER_CLOSED, "duration": 2.0},
        {"name": "Move to Place", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": GRIPPER_CLOSED, "duration": 3.0},
        {"name": "Lower to Place", "pos": PLACE_POS, "gripper": GRIPPER_CLOSED, "duration": 2.0},
        {"name": "Release", "pos": PLACE_POS, "gripper": GRIPPER_OPEN, "duration": 1.0},
        {"name": "Retract", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": GRIPPER_OPEN, "duration": 2.0},
        {"name": "Return to Home", "pos": HOME_POS, "gripper": GRIPPER_OPEN, "duration": 3.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0

    print("🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            q_current = d.qpos[:m.nq].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # 1. Compute IK target
                target_se3 = pin.SE3(fixed_rotation, state["pos"])
                
                # Compute differential velocity. Provide Q_HOME for nullspace resting.
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=np.array(Q_HOME))

                # 2. Velocity to Position Integration (CRITICAL FOR VIPERX)
                # ViperX uses <position> actuators, not velocity overrides.
                q_target[:6] += dq[:6] * m.opt.timestep
                
                # 3. Command Arm Position Actuators (Indices 0 to 5)
                d.ctrl[:6] = q_target[:6]

                # 4. Command Gripper Position Actuator (Index 6)
                d.ctrl[6] = state["gripper"]

                # 5. State Transitions (Contact-Aware)
                is_holding = grasp_sys.is_grasped()

                if state["name"] == "Grasping":
                    if is_holding and (d.time - state_start_time > state["duration"]):
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting payload.")
                        current_state_idx += 1
                        state_start_time = d.time
                else:
                    if done or (d.time - state_start_time > state["duration"]):
                        current_state_idx += 1
                        state_start_time = d.time
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            # CRITICAL FIX 3: Gravity Compensation restriction
            # Apply ONLY to the 8 joints of the ViperX arm/gripper.
            # If applied to all of m.nv, it turns off gravity for the target cube!
            d.qfrc_applied[:8] = d.qfrc_bias[:8]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()