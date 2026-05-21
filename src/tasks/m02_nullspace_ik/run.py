import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

def main():
    print("🚀 Initializing Franka Mark-2 Systems...")
    
    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    robot = FrankaPanda()
    
    # CRITICAL FIX 1: M2 IK uses active_joint_indices, not arm_dof!
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    # Initialize the new Grasp System
    grasp_sys = GraspController(m, d)

    # -------------------------------
    # Set Initial Pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    # Position tracking array for Position Actuators (replaces velocity override)
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    # Capture downward orientation and padded home pose
    q_home_pin = np.zeros(robot.model.nq)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # -------------------------------
    # Dynamic Waypoints (FIXED OFFSET)
    # -------------------------------
    # The panda_hand frame is at the wrist. 
    # The fingertips are ~10.5cm further down.
    GRIPPER_Z_OFFSET = 0.105 
    
    # Target the cube's center (z=0.02), plus the length of the gripper
    PICK_POS = np.array([0.45, 0.0, 0.02 + GRIPPER_Z_OFFSET])
    PLACE_POS = np.array([0.0, 0.5, 0.02 + GRIPPER_Z_OFFSET])
    
    # 255 = Open, 0 = Closed
    states = [
        {"name": "Approach Pick", "pos": PICK_POS + np.array([0, 0, 0.15]), "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "pos": PICK_POS, "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "pos": PICK_POS, "gripper": 0, "duration": 1.0}, # Contact-aware!
        {"name": "Lift", "pos": PICK_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "pos": PLACE_POS, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": PLACE_POS, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 255, "duration": 2.0},
        {"name": "Return to Home", "pos": HOME_POS, "gripper": 255, "duration": 3.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0

    print("🟢 Simulation Online. Engaging physics...")

    # -------------------------------
    # Active Physics Loop
    # -------------------------------
    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # Read State
            q_current = d.qpos[:robot.model.nq].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # 1. Compute IK target
                target_se3 = pin.SE3(fixed_rotation, state["pos"])
                
                # CRITICAL FIX 2: Pass 'q_home_pin' as q_posture.
                dq, err, done = ik.compute_velocity(
                    q_current, 
                    target_se3, 
                    q_posture=q_home_pin
                )

                # 2. Velocity to Position Integration
                max_speed = 1.0
                dq = np.clip(dq, -max_speed, max_speed)
                
                # CRITICAL FIX 3: Use Franka's PD Position Actuators!
                # Bypassing qvel allows the physics engine to calculate natural forces,
                # preventing the arm from "teleporting" and ripping the cube out.
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                # 3. Command Gripper
                grasp_sys.command(state["gripper"])

                # 4. State Transitions (Contact-Aware)
                is_holding = grasp_sys.is_grasped()

                if state["name"] == "Grasping":
                    # Wait for the full grasp duration to allow force to build up
                    if is_holding and (d.time - state_start_time > state["duration"]):
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting payload.")
                        current_state_idx += 1
                        state_start_time = d.time
                else:
                    # Normal time-based transition
                    if done or (d.time - state_start_time > state["duration"]):
                        current_state_idx += 1
                        state_start_time = d.time
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                # Sequence finished - hold final position steadily
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            # Gravity compensation (Franka arm joints only)
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()