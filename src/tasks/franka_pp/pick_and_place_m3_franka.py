import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

def main():
    print("🚀 Initializing Franka Mark-3 (Trajectory Planned) Systems...")
    
    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    robot = FrankaPanda()
    
    # M2 IK is perfect as our lower-level joint solver
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    grasp_sys = GraspController(m, d)

    # -------------------------------
    # Set Initial Pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # -------------------------------
    # Dynamic Waypoints
    # -------------------------------
    GRIPPER_Z_OFFSET = 0.105 
    PICK_POS = np.array([0.45, 0.0, 0.02 + GRIPPER_Z_OFFSET])
    PLACE_POS = np.array([0.0, 0.5, 0.02 + GRIPPER_Z_OFFSET])
    
    states = [
        {"name": "Approach Pick", "pos": PICK_POS + np.array([0, 0, 0.15]), "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "pos": PICK_POS, "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "pos": PICK_POS, "gripper": 0, "duration": 1.0},
        {"name": "Lift", "pos": PICK_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "pos": PLACE_POS, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": PLACE_POS, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": PLACE_POS + np.array([0, 0, 0.2]), "gripper": 255, "duration": 2.0},
        {"name": "Return to Home", "pos": HOME_POS, "gripper": 255, "duration": 3.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0
    
    # NEW: Trajectory State
    current_trajectory = None

    print("🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            q_current = d.qpos[:robot.model.nq].copy()

            # CLEAR VIRTUAL SCENE: Reset visual debug markers every frame
            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # 1. Initialize Trajectory on State Transition
                if current_trajectory is None:
                    start_se3 = robot.forward_kinematics(q_current)
                    end_se3 = pin.SE3(fixed_rotation, state["pos"])
                    
                    current_trajectory = TaskSpaceTrajectory(
                        start_pose=start_se3,
                        end_pose=end_se3,
                        duration=state["duration"],
                        method="cubic"  # Try changing this to "linear" later to see the jerk!
                    )

                # 2. Query Trajectory for the current time step
                t_state = d.time - state_start_time
                target_se3 = current_trajectory.get_pose(t_state)

                # --- NEW: Trajectory Visualization ---
                if hasattr(viewer, 'user_scn') and current_trajectory is not None:
                    # Draw the path trail (Sample 20 points along the current trajectory)
                    for t_sample in np.linspace(0, state["duration"], 20):
                        pt = current_trajectory.get_pose(t_sample).translation
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE,
                            size=np.array([0.005, 0, 0]),
                            pos=pt,
                            mat=np.eye(3).flatten(),
                            rgba=np.array([0, 1, 0, 0.4]) # Semi-transparent green
                        )
                        viewer.user_scn.ngeom += 1

                    # Draw the active moving target (Red sphere)
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.012, 0, 0]),
                        pos=target_se3.translation,
                        mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 0.8]) # Solid red
                    )
                    viewer.user_scn.ngeom += 1
                # -------------------------------------
                
                # 3. Compute IK tracking the moving target
                dq, err, done = ik.compute_velocity(
                    q_current, 
                    target_se3, 
                    q_posture=q_home_pin
                )

                # Velocity limits & Integration
                max_speed = 1.0
                dq = np.clip(dq, -max_speed, max_speed)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                # Command Gripper
                grasp_sys.command(state["gripper"])

                # 4. State Transitions
                is_holding = grasp_sys.is_grasped()

                if state["name"] == "Grasping":
                    if is_holding and (t_state > state["duration"]):
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting payload.")
                        current_state_idx += 1
                        state_start_time = d.time
                        current_trajectory = None # Trigger new trajectory generation
                else:
                    if t_state > state["duration"]:
                        current_state_idx += 1
                        state_start_time = d.time
                        current_trajectory = None # Trigger new trajectory generation
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            # Gravity compensation
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()