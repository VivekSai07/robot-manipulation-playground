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

def generate_task_states(m, d, home_pos, z_offset):
    """
    SENSE & PLAN: Dynamically generates the waypoints based on the cube's ACTUAL current position.
    """
    cube_id = m.body("target_cube").id
    cube_pos = d.xpos[cube_id].copy()
    
    pick_pos = np.array([cube_pos[0], cube_pos[1], 0.02 + z_offset])
    place_pos = np.array([0.0, 0.5, 0.02 + z_offset])
    approach_offset = np.array([0, 0, 0.15])
    
    return [
        {"name": "Approach Pick", "pos": pick_pos + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "pos": pick_pos, "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "pos": pick_pos, "gripper": 0, "duration": 1.0},
        {"name": "Verify Lift", "pos": pick_pos + approach_offset, "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": place_pos + approach_offset, "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "pos": place_pos, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": place_pos, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": place_pos + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Return to Home", "pos": home_pos, "gripper": 255, "duration": 3.0},
    ]


def main():
    print("🚀 Initializing Franka Mark-4 (Robust Pipeline) Systems...")
    
    # ==========================================
    # 😈 CHAOS MONKEY TOGGLE
    # Set to True to intentionally sabotage the first grasp!
    SABOTAGE_TEST = True
    sabotage_done = False
    # ==========================================

    robot = FrankaPanda()
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

    mujoco.mj_forward(m, d)

    # -------------------------------
    # Initialize Dynamic State Machine
    # -------------------------------
    GRIPPER_Z_OFFSET = 0.105 
    states = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)
    
    current_state_idx = 0
    state_start_time = 0.0
    current_trajectory = None

    print("🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if current_state_idx < len(states):
                state = states[current_state_idx]

                # --- 😈 CHAOS MONKEY INJECTION ---
                # Teleport the cube 15cm away exactly in the middle of the "Grasping" phase
                if SABOTAGE_TEST and not sabotage_done and state["name"] == "Grasping":
                    cube_body_id = m.body("target_cube").id
                    cube_jnt_id = m.body_jntadr[cube_body_id]
                    cube_qpos_adr = m.jnt_qposadr[cube_jnt_id]
                    
                    # Shift it along the Y-axis by 0.15 meters
                    d.qpos[cube_qpos_adr + 1] += 0.15 
                    sabotage_done = True
                    print(f"\n[{d.time:.2f}s] 😈 CHAOS MONKEY: The cube slipped away!\n")
                # ---------------------------------

                # 1. Trajectory Generation
                if current_trajectory is None:
                    start_se3 = robot.forward_kinematics(q_current)
                    end_se3 = pin.SE3(fixed_rotation, state["pos"])
                    current_trajectory = TaskSpaceTrajectory(
                        start_pose=start_se3, end_pose=end_se3, 
                        duration=state["duration"], method="cubic"
                    )

                # 2. Query Trajectory
                t_state = d.time - state_start_time
                target_se3 = current_trajectory.get_pose(t_state)

                if hasattr(viewer, 'user_scn') and current_trajectory is not None:
                    for t_sample in np.linspace(0, state["duration"], 20):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.005, 0, 0]),
                            pos=current_trajectory.get_pose(t_sample).translation,
                            mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.4])
                        )
                        viewer.user_scn.ngeom += 1

                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                        pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.8])
                    )
                    viewer.user_scn.ngeom += 1
                
                # 3. Compute IK
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                dq = np.clip(dq, -1.0, 1.0)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                # 4. Command Gripper
                grasp_sys.command(state["gripper"])

                # 5. ROBUST STATE TRANSITIONS
                is_holding = grasp_sys.is_grasped()

                if t_state > state["duration"]:
                    
                    # VERIFY PHASE
                    if state["name"] == "Verify Lift":
                        if not is_holding:
                            print(f"[{d.time:.2f}s] ⚠️ Grasp Failed or Dropped! Initiating Recovery...")
                            # RECOVERY: Re-evaluate cube position, reset state machine!
                            states = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)
                            current_state_idx = 0
                            state_start_time = d.time
                            current_trajectory = None
                            continue 
                        else:
                            print(f"[{d.time:.2f}s] ✅ Grasp verified! Proceeding to place.")
                    
                    # Normal State Advancement
                    current_state_idx += 1
                    state_start_time = d.time
                    current_trajectory = None
                    if current_state_idx < len(states):
                        print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()