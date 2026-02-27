import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.robots.universal_robots_ur5e.robot import UniversalRobotUR5e
from src.robots.universal_robots_ur5e.config import (
    SCENE_PATH, Q_HOME, ARM_DOF, ACTIVE_JOINTS,
    GRIPPER_OPEN, GRIPPER_CLOSED
)

def main():
    print("🚀 Initializing UR5e Mark-2 Systems...")

    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    robot = UniversalRobotUR5e()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    # 2F-85's primary sliding pads are nested under the follower links
    grasp_sys = GraspController(m, d, left_finger="left_follower", right_finger="right_follower")

    # -------------------------------
    # Set Initial Pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    # Initialize the 2F-85 Gripper joints to Open
    # (The Robotiq uses a complex underactuated linkage, but index 6 controls the main driver)
    d.qpos[6] = GRIPPER_OPEN
    
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    # Convert Q_HOME to a safely sized NumPy array for Pinocchio
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:6] = Q_HOME[:6]
    
    # Capture initial downward orientation
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # -------------------------------
    # Dynamic Waypoints
    # -------------------------------
    # The EE_FRAME is wrist_3_link. The 2F-85 gripper is ~14.5cm long.
    GRIPPER_Z_OFFSET = 0.145 
    
    PICK_POS = np.array([0.45, 0.0, 0.02 + GRIPPER_Z_OFFSET])
    PLACE_POS = np.array([0.0, 0.45, 0.1 + GRIPPER_Z_OFFSET])
    
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

            q_pin = pin.neutral(robot.model)
            q_pin[:6] = d.qpos[:6].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # 1. Compute IK target
                target_se3 = pin.SE3(fixed_rotation, state["pos"])
                dq, err, done = ik.compute_velocity(q_pin, target_se3, q_posture=q_home_pin)

                # 2. Velocity to Position Integration
                q_target[:6] += dq[:6] * m.opt.timestep
                
                # 3. Command Arm Actuators
                d.ctrl[:6] = q_target[:6]

                # 4. Command Gripper Actuator
                d.ctrl[6] = state["gripper"]

                # 5. State Transitions
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

            # Gravity Compensation (Only compensate the 6 arm joints. The gripper's mass is held by the arm!)
            d.qfrc_applied[:6] = d.qfrc_bias[:6]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()