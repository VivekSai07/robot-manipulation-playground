import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from controllers.ik_controller_m0 import IKController
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
)


def main():
    # -------------------------------
    # Robot + IK
    # -------------------------------
    robot = FrankaPanda()
    ik = IKController(robot, arm_dof=ARM_DOF)

    # -------------------------------
    # MuJoCo
    # -------------------------------
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    # -------------------------------
    # Initial pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]

    # -------------------------------
    # Capture home EE rotation
    # -------------------------------
    q_home_pin = np.zeros(robot.model.nq)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]

    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()

    # -------------------------------
    # Pick & Place targets
    # -------------------------------
    PICK_POS = np.array([0.5, 0.0, 0.1])
    PLACE_POS = np.array([0.0, 0.5, 0.2])

    states = [
        {"name": "Approach Pick", "pos": PICK_POS + np.array([0, 0, 0.2]), "gripper": 255, "duration": 3.0},
        {"name": "Descend to Pick", "pos": PICK_POS, "gripper": 255, "duration": 2.0},
        {"name": "Grasping", "pos": PICK_POS, "gripper": 0, "duration": 1.0},
        {"name": "Lift", "pos": PICK_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "pos": PLACE_POS, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": PLACE_POS, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 255, "duration": 2.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0

    print("🚀 Starting Pick and Place simulation...")

    # -------------------------------
    # Viewer loop
    # -------------------------------
    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            # -------------------------------
            # Current joint state from MuJoCo
            # -------------------------------
            q_current = d.qpos[:robot.model.nq].copy()

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # Target SE3
                target_se3 = pin.SE3(fixed_rotation, state["pos"])

                # ✅ NEW IK call
                dq = ik.compute_velocity(q_current, target_se3)

                # Apply arm velocity
                d.qvel[:robot.model.nv] = dq

                # Gripper control
                if m.nu >= 2:
                    d.ctrl[-1] = state["gripper"]
                    d.ctrl[-2] = state["gripper"]

                # State transition
                if d.time - state_start_time > state["duration"]:
                    current_state_idx += 1
                    state_start_time = d.time

                    if current_state_idx < len(states):
                        print(f"State → {states[current_state_idx]['name']}")

            else:
                d.qvel[:] = 0

            # Gravity compensation
            d.qfrc_applied[:robot.model.nv] = d.qfrc_bias[:robot.model.nv]

            mujoco.mj_step(m, d)
            viewer.sync()

            # real-time pacing
            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
