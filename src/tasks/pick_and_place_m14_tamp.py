import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.tamp_brain import TAMPBrain
from src.planners.trajectory_planner import TaskSpaceTrajectory
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    Q_HOME, ARM_DOF, ACTIVE_JOINTS, ROBOT_DIR
)

# Load the TAMP Sorting Environment
TAMP_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "tamp_scene.xml")

def generate_tamp_states(pick_pos, place_pos):
    """
    State machine for a SINGLE sort operation.
    """
    approach_offset = np.array([0, 0, 0.15])
    return [
        {"name": "Approach Pick", "pos": pick_pos + approach_offset, "gripper": 255, "duration": 3.0},
        {"name": "Descend to Pick", "pos": pick_pos, "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "pos": pick_pos, "gripper": 0, "duration": 1.0},
        {"name": "Verify Lift", "pos": pick_pos + approach_offset, "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": place_pos + approach_offset, "gripper": 0, "duration": 4.0},
        {"name": "Lower to Place", "pos": place_pos, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": place_pos, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": place_pos + approach_offset, "gripper": 255, "duration": 2.0},
    ]

def main():
    print("🚀 Initializing Mark-14 Task and Motion Planning (TAMP) Agent...")
    
    robot = FrankaPanda()
    # CRITICAL FIX 1: Restored M12 IK Gains for stable, un-jittery movement
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    
    m = mujoco.MjModel.from_xml_path(TAMP_SCENE_PATH)
    d = mujoco.MjData(m)

    # 1. Move to Stow/Scan Pose
    q_stow = Q_HOME.copy()
    q_stow[0] = -1.57 # Rotate base -90 degrees
    d.qpos[:len(Q_HOME)] = q_stow
    
    q_stow_pin = pin.neutral(robot.model)
    q_stow_pin[:len(Q_HOME)] = q_stow
    
    mujoco.mj_forward(m, d)
    
    # 2. Scatter the Mess
    print("🎲 Scattering semantic mess on the table...")
    spawned_positions = []
    mess_objects = ["target_apple", "target_bottle", "target_metal_box", "target_wood_block"]
    
    for body_name in mess_objects:
        try:
            c_id = m.body(body_name).id
            adr = m.jnt_qposadr[m.body_jntadr[c_id]]
            for _ in range(50):
                px = np.random.uniform(0.35, 0.55)
                py = np.random.uniform(-0.15, 0.15) 
                valid = True
                for existing_p in spawned_positions:
                    if np.linalg.norm(np.array([px, py]) - existing_p) < 0.10: 
                        valid = False
                        break
                if valid:
                    d.qpos[adr] = px
                    d.qpos[adr+1] = py
                    spawned_positions.append(np.array([px, py]))
                    break 
        except ValueError:
            pass 
    mujoco.mj_forward(m, d)

    food_bin_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "food_bin")
    recycle_bin_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "recycle_bin")
    
    orig_food_pos = m.body_pos[food_bin_id].copy()
    orig_recycle_pos = m.body_pos[recycle_bin_id].copy()
    
    m.body_pos[food_bin_id][2] = -2.0
    m.body_pos[recycle_bin_id][2] = -2.0
    mujoco.mj_forward(m, d)
    
    # 3. Fire up the TAMP Brain
    brain = TAMPBrain(m, d)
    task_queue = brain.plan_cleanup()
    
    # Restore bin physical locations!
    m.body_pos[food_bin_id] = orig_food_pos
    m.body_pos[recycle_bin_id] = orig_recycle_pos
    mujoco.mj_forward(m, d)

    total_tasks = len(task_queue)
    completed_tasks = 0

    # Base kinematics
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # Reset Rotational Twist
    d.qpos[:len(Q_HOME)] = Q_HOME
    q_target = d.qpos.copy()
    mujoco.mj_forward(m, d)

    GRIPPER_Z_OFFSET = 0.105
    grasp_sys = None 
    
    states = []
    current_idx = 0
    state_start_time = d.time
    current_trajectory = None
    slip_debounce_time = 0.0
    
    mission_complete = False

    print("\n🟢 Simulation Online. Engaging Continuous Execution Loop...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            # ==========================================
            # THE SCHEDULER: POPPING NEW TASKS
            # ==========================================
            if len(states) == 0 and not mission_complete:
                if len(task_queue) > 0:
                    current_task = task_queue.pop(0)
                    completed_tasks += 1
                    print(f"\n==============================================")
                    print(f"📦 Executing Task {completed_tasks}/{total_tasks}: Sort '{current_task['prompt'].upper()}'")
                    print(f"==============================================")
                    
                    grasp_sys = GraspController(m, d, target=current_task["target_body"])
                    
                    target_z = current_task["pick_pos"][2]
                    
                    # =========================================================
                    # CRITICAL FIX 2: Floor Safety Clamp
                    # If the VLM caught a table shadow, the depth drops to 0.
                    # We absolutely forbid target_z from dropping below 0.025m 
                    # (the radius of the apple) to prevent floor crashing!
                    # =========================================================
                    target_z = max(0.025, target_z)
                    
                    pick_pos = np.array([
                        current_task["pick_pos"][0], 
                        current_task["pick_pos"][1], 
                        target_z + GRIPPER_Z_OFFSET
                    ])
                    place_pos = np.array([
                        current_task["place_pos"][0], 
                        current_task["place_pos"][1], 
                        target_z + 0.01 + GRIPPER_Z_OFFSET 
                    ])
                    
                    states = generate_tamp_states(pick_pos, place_pos)
                    current_idx = 0
                    state_start_time = d.time
                    current_trajectory = None
                    slip_debounce_time = 0.0
                else:
                    print("\n✅ All items sorted. Returning to Home.")
                    states = [{"name": "Return to Home", "pos": HOME_POS, "gripper": 255, "duration": 4.0}]
                    current_idx = 0
                    state_start_time = d.time
                    current_trajectory = None
                    mission_complete = True
                    grasp_sys = GraspController(m, d, target="target_apple") # Dummy fallback
                    grasp_sys.command(255)

            # ==========================================
            # CONTINUOUS TASK EXECUTION
            # ==========================================
            elif current_idx < len(states):
                state = states[current_idx]
                t_state = d.time - state_start_time

                if current_trajectory is None:
                    start_se3 = robot.forward_kinematics(q_current)
                    end_se3 = pin.SE3(fixed_rotation, state["pos"])
                    current_trajectory = TaskSpaceTrajectory(start_se3, end_se3, state["duration"], "cubic")
                    state_start_time = d.time

                t_traj = d.time - state_start_time
                target_se3 = current_trajectory.get_pose(t_traj)

                if hasattr(viewer, 'user_scn') and current_trajectory is not None:
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[viewer.user_scn.ngeom],
                        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.012, 0, 0]),
                        pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.8])
                    )
                    viewer.user_scn.ngeom += 1

                # =========================================================
                # CRITICAL FIX 3: Removed Nullspace Lock during Picking!
                # Forcing the elbow to stay home while reaching for the cube caused "Nullspace Sag",
                # which tilted the wrist diagonally and crushed the apple out of the gripper.
                # =========================================================
                posture_bias = q_stow_pin if state["name"] == "Return to Home" else None
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=posture_bias)
                dq = np.clip(dq, -1.0, 1.0)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])
                is_holding = grasp_sys.is_grasped()

                if state["gripper"] == 0 and state["name"] not in ["Grasping", "Lower to Place", "Release"]:
                    if not is_holding and t_state > 0.2:
                        if slip_debounce_time == 0.0:
                            slip_debounce_time = d.time
                        elif (d.time - slip_debounce_time) > 0.25:
                            print(f"\n🚨 SLIP DETECTED! Dropped item. Aborting current task...")
                            states = [] 
                            current_idx = 0
                            current_trajectory = None
                            slip_debounce_time = 0.0
                            continue 
                    else:
                        slip_debounce_time = 0.0 

                # =========================================================
                # CRITICAL FIX 4: Restored Descend to Pick to Ignore List
                # If the target is fractionally inside a collision box, 'done' is never triggered, 
                # causing an infinite freeze. We must allow the state to naturally timeout!
                # =========================================================
                ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                can_transition = (t_traj > state["duration"]) and (done or ignore_ik_error)

                if can_transition:
                    current_idx += 1
                    state_start_time = d.time
                    current_trajectory = None
                    slip_debounce_time = 0.0
                    
                    if current_idx < len(states):
                        print(f"   [{d.time:.2f}s] State → {states[current_idx]['name']}")
                    else:
                        states = []

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()
            
            brain.show_feed()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()