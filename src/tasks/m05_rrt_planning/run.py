import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.planners.trajectory_planner import TaskSpaceTrajectory, JointSpaceTrajectory
from src.planners.rrt import RRT
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

def solve_virtual_ik(robot, ik, q_start, target_se3, q_posture=None):
    q_virtual = q_start.copy()
    for _ in range(1000):
        dq, err, done = ik.compute_velocity(q_virtual, target_se3, q_posture=q_posture, tol=1e-3)
        q_virtual += dq * 0.05
        if done:
            break
    return q_virtual

def generate_task_states(m, d, home_pos, z_offset):
    cube_id = m.body("target_cube").id
    cube_pos = d.xpos[cube_id].copy()
    
    pick_pos = np.array([cube_pos[0], cube_pos[1], 0.02 + z_offset])
    place_pos = np.array([0.0, 0.5, 0.02 + z_offset])
    approach_offset = np.array([0, 0, 0.15])
    
    return [
        {"name": "Approach Pick", "pos": pick_pos + approach_offset, "gripper": 255, "duration": 3.0, "rrt": True},
        {"name": "Descend to Pick", "pos": pick_pos, "gripper": 255, "duration": 1.5, "rrt": False},
        {"name": "Grasping", "pos": pick_pos, "gripper": 0, "duration": 1.0, "rrt": False},
        {"name": "Verify Lift", "pos": pick_pos + approach_offset, "gripper": 0, "duration": 2.0, "rrt": False},
        {"name": "Move to Place", "pos": place_pos + approach_offset, "gripper": 0, "duration": 4.0, "rrt": True},
        {"name": "Lower to Place", "pos": place_pos, "gripper": 0, "duration": 2.0, "rrt": False},
        {"name": "Release", "pos": place_pos, "gripper": 255, "duration": 1.0, "rrt": False},
        {"name": "Retract", "pos": place_pos + approach_offset, "gripper": 255, "duration": 2.0, "rrt": False},
        {"name": "Return to Home", "pos": home_pos, "gripper": 255, "duration": 4.0, "rrt": True},
    ]


def main():
    print("🚀 Initializing Franka Mark-5 (RRT Motion Planning) Systems...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    grasp_sys = GraspController(m, d)

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

    GRIPPER_Z_OFFSET = 0.105 
    states = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)
    
    current_state_idx = 0
    state_start_time = 0.0
    
    current_trajectory = None
    current_rrt_path = None

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

                # ==========================================
                # RRT GENERATION PHASE
                # ==========================================
                if state.get("rrt", False) and current_rrt_path is None:
                    target_se3 = pin.SE3(fixed_rotation, state["pos"])
                    
                    # CRITICAL FIX 4: Nullspace Conflict
                    # Only apply home posture bias if we are actually returning home!
                    # Otherwise, the nullspace will violently pull the arm away from the place/pick positions,
                    # causing the IK solver to get stuck hovering in mid-air.
                    posture_bias = q_home_pin if state["name"] == "Return to Home" else None
                    q_goal = solve_virtual_ik(robot, ik, q_current, target_se3, q_posture=posture_bias)
                    
                    planner = RRT(m, d, ACTIVE_JOINTS, step_size=0.15)
                    raw_path = planner.plan(q_current[ACTIVE_JOINTS], q_goal[ACTIVE_JOINTS])
                    
                    if raw_path is None:
                        print("⚠️ RRT Failed! Falling back to linear trajectory...")
                        current_rrt_path = [] 
                    else:
                        current_rrt_path = []
                        for q_act in raw_path[1:]: 
                            q_full = q_current.copy()
                            q_full[ACTIVE_JOINTS] = q_act
                            current_rrt_path.append(q_full)
                        
                        state["wp_dur"] = max(0.2, state["duration"] / len(current_rrt_path))

                is_rrt_active = state.get("rrt", False) and current_rrt_path is not None and len(current_rrt_path) > 0

                # ==========================================
                # EXECUTE RRT (JOINT SPACE TRACKING)
                # ==========================================
                if is_rrt_active:
                    if current_trajectory is None:
                        q_next = current_rrt_path[0]
                        current_trajectory = JointSpaceTrajectory(q_current, q_next, state["wp_dur"], "cubic")
                        state_start_time = d.time

                    t_state = d.time - state_start_time
                    target_q = current_trajectory.get_position(t_state)
                    target_se3 = robot.forward_kinematics(target_q)
                    
                    if hasattr(viewer, 'user_scn'):
                        for wp in current_rrt_path:
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                                pos=robot.forward_kinematics(wp).translation, mat=np.eye(3).flatten(), rgba=np.array([0, 0.5, 1.0, 0.5])
                            )
                            viewer.user_scn.ngeom += 1

                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                            pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.8])
                        )
                        viewer.user_scn.ngeom += 1

                    for idx in ACTIVE_JOINTS:
                        q_target[idx] = target_q[idx]
                        d.ctrl[idx] = q_target[idx]

                    grasp_sys.command(state["gripper"])

                    if t_state > current_trajectory.duration:
                        current_rrt_path.pop(0)
                        current_trajectory = None
                        
                        if len(current_rrt_path) == 0:
                            current_state_idx += 1
                            state_start_time = d.time
                            current_rrt_path = None
                            if current_state_idx < len(states):
                                print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")
                        
                        continue 

                # ==========================================
                # EXECUTE CARTESIAN (TASK SPACE TRACKING)
                # ==========================================
                else:
                    if current_trajectory is None:
                        start_se3 = robot.forward_kinematics(q_current)
                        end_se3 = pin.SE3(fixed_rotation, state["pos"])
                        current_trajectory = TaskSpaceTrajectory(start_se3, end_se3, state["duration"], "cubic")
                        state_start_time = d.time

                    t_state = d.time - state_start_time
                    target_se3 = current_trajectory.get_pose(t_state)

                    if hasattr(viewer, 'user_scn') and current_trajectory is not None:
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                            pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.8])
                        )
                        viewer.user_scn.ngeom += 1

                    # FIX: Pass the dynamic posture_bias here as well!
                    posture_bias = q_home_pin if state["name"] == "Return to Home" else None
                    dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=posture_bias)
                    dq = np.clip(dq, -1.0, 1.0)
                    
                    for idx in ACTIVE_JOINTS:
                        q_target[idx] += dq[idx] * m.opt.timestep
                        d.ctrl[idx] = q_target[idx]

                    grasp_sys.command(state["gripper"])

                    is_holding = grasp_sys.is_grasped()
                    
                    ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                    can_transition = (t_state > state["duration"]) and (done or ignore_ik_error)

                    if can_transition:
                        if state["name"] == "Verify Lift":
                            if not is_holding:
                                print(f"[{d.time:.2f}s] ⚠️ Grasp Failed or Dropped! Initiating Recovery...")
                                states = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)
                                current_state_idx = 0
                                state_start_time = d.time
                                current_trajectory = None
                                current_rrt_path = None
                                continue 
                            else:
                                print(f"[{d.time:.2f}s] ✅ Grasp verified! Proceeding to place.")
                        
                        current_state_idx += 1
                        state_start_time = d.time
                        current_trajectory = None
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")
                        
                        continue

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