import os
import argparse
import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import ROBOT_DIR, Q_HOME, ARM_DOF, ACTIVE_JOINTS

SCENE_PATH = os.path.join(ROBOT_DIR, "model", "m1_scene.xml")

MAX_RETRIES = 3


def generate_task_states(m, d, home_pos, z_offset):
    """
    SENSE & PLAN: Dynamically generates waypoints from the cube's live position.
    """
    cube_id  = m.body("target_cube").id
    cube_pos = d.xpos[cube_id].copy()

    pick_pos      = np.array([cube_pos[0], cube_pos[1], 0.02 + z_offset])
    place_pos     = np.array([0.0, 0.5, 0.02 + z_offset])
    approach_offset = np.array([0, 0, 0.15])

    return [
        {"name": "Approach Pick",   "pos": pick_pos  + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "pos": pick_pos,                    "gripper": 255, "duration": 1.5},
        {"name": "Grasping",        "pos": pick_pos,                    "gripper": 0,   "duration": 1.0},
        {"name": "Verify Lift",     "pos": pick_pos  + approach_offset, "gripper": 0,   "duration": 2.0},
        {"name": "Move to Place",   "pos": place_pos + approach_offset, "gripper": 0,   "duration": 3.0},
        {"name": "Lower to Place",  "pos": place_pos,                   "gripper": 0,   "duration": 2.0},
        {"name": "Release",         "pos": place_pos,                   "gripper": 255, "duration": 1.0},
        {"name": "Retract",         "pos": place_pos + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Return to Home",  "pos": home_pos,                    "gripper": 255, "duration": 3.0},
    ]


def main():
    parser = argparse.ArgumentParser(description="Franka Mark-4: Robust Pick & Place with Recovery")
    parser.add_argument("--sabotage", action="store_true",
                        help="Enable chaos monkey — teleports cube mid-grasp to test recovery")
    args = parser.parse_args()

    sabotage_done = False

    print("🚀 Initializing Franka Mark-4 (Robust Pipeline) Systems...")
    if args.sabotage:
        print("😈 Chaos Monkey ENABLED — cube will be teleported mid-grasp.")

    robot    = FrankaPanda()
    ik       = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    grasp_sys = GraspController(m, d)

    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    home_pose      = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS       = home_pose.translation.copy()

    mujoco.mj_forward(m, d)

    GRIPPER_Z_OFFSET = 0.105
    states = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)

    current_state_idx = 0
    state_start_time  = 0.0
    current_trajectory = None
    retry_count = 0

    print("🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current  = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if current_state_idx < len(states):
                state  = states[current_state_idx]
                t_state = d.time - state_start_time

                # --- Chaos Monkey ---
                if args.sabotage and not sabotage_done and state["name"] == "Grasping":
                    cube_body_id  = m.body("target_cube").id
                    cube_qpos_adr = m.jnt_qposadr[m.body_jntadr[cube_body_id]]
                    d.qpos[cube_qpos_adr + 1] += 0.15
                    sabotage_done = True
                    print(f"\n[{d.time:.2f}s] 😈 CHAOS MONKEY: The cube slipped away!\n")

                # 1. Trajectory Generation
                if current_trajectory is None:
                    current_trajectory = TaskSpaceTrajectory(
                        start_pose=robot.forward_kinematics(q_current),
                        end_pose=pin.SE3(fixed_rotation, state["pos"]),
                        duration=state["duration"],
                        method="cubic"
                    )

                # 2. Query Trajectory
                target_se3 = current_trajectory.get_pose(t_state)

                # Trail + moving-target visualisation
                if hasattr(viewer, 'user_scn'):
                    for t_s in np.linspace(0, state["duration"], 20):
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.005, 0, 0]),
                            pos=current_trajectory.get_pose(t_s).translation,
                            mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.4])
                        )
                        viewer.user_scn.ngeom += 1
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                        pos=target_se3.translation, mat=np.eye(3).flatten(),
                        rgba=np.array([1, 0, 0, 0.8])
                    )
                    viewer.user_scn.ngeom += 1

                # 3. Compute IK
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                dq = np.clip(dq, -1.0, 1.0)
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx]    = q_target[idx]

                # 4. Gripper
                grasp_sys.command(state["gripper"])
                is_holding = grasp_sys.is_grasped()

                # 5. State Transitions
                if state["name"] == "Grasping":
                    # Contact-aware early exit
                    if is_holding and t_state > 0.3:
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting.")
                        current_state_idx += 1
                        state_start_time   = d.time
                        current_trajectory = None
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")
                    elif t_state > state["duration"]:
                        # Timed out without contact — Verify Lift will handle it
                        current_state_idx += 1
                        state_start_time   = d.time
                        current_trajectory = None
                        if current_state_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

                elif t_state > state["duration"]:
                    if state["name"] == "Verify Lift":
                        if not is_holding:
                            retry_count += 1
                            if retry_count >= MAX_RETRIES:
                                print(f"[{d.time:.2f}s] ❌ Max retries ({MAX_RETRIES}) reached. Aborting.")
                                break
                            print(f"[{d.time:.2f}s] ⚠️  Grasp failed! Recovery {retry_count}/{MAX_RETRIES} — rescanning cube...")
                            states             = generate_task_states(m, d, HOME_POS, GRIPPER_Z_OFFSET)
                            current_state_idx  = 0
                            state_start_time   = d.time
                            current_trajectory = None
                            continue
                        else:
                            print(f"[{d.time:.2f}s] ✅ Grasp verified! Proceeding to place.")
                            retry_count = 0

                    current_state_idx += 1
                    state_start_time   = d.time
                    current_trajectory = None
                    if current_state_idx < len(states):
                        print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                print(f"[{d.time:.2f}s] Sequence complete. Shutting down.")
                break

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
