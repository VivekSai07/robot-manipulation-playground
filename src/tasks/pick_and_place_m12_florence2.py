import os
import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
import time
import threading
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import cv2

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.florence2_pipeline import Florence2Pipeline
from src.planners.trajectory_planner import TaskSpaceTrajectory, JointSpaceTrajectory
from src.planners.rrt import RRT
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    Q_HOME, ARM_DOF, ACTIVE_JOINTS, ROBOT_DIR
)

# Load the Semantic VLM Environment (With the obstacle wall!)
VLM_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "vlm_scene.xml")

command_queue = []
app_state = {"is_idle": True}

def input_thread():
    time.sleep(2.0) 
    while True:
        if app_state["is_idle"]:
            cmd = input("\n🗣️ ENTER COMMAND (e.g., 'red ball', 'metal box', 'drink'):\n> ")
            if cmd.strip():
                command_queue.append(cmd.strip())
                app_state["is_idle"] = False
        else:
            time.sleep(0.5)

def solve_virtual_ik(robot, ik, q_start, target_se3, q_posture=None):
    q_virtual = q_start.copy()
    for _ in range(1000):
        dq, err, done = ik.compute_velocity(q_virtual, target_se3, q_posture=q_posture, tol=1e-3)
        q_virtual += dq * 0.05
        if done:
            break
    return q_virtual

def generate_vlm_states(pick_pos, place_pos, stow_pos, z_offset):
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
        {"name": "Return to Stow", "pos": stow_pos, "gripper": 255, "duration": 4.0, "rrt": True},
    ]

def generate_recovery_states(current_pos, stow_pos):
    return [
        {"name": "Recovery Lift", "pos": current_pos + np.array([0, 0, 0.15]), "gripper": 255, "duration": 1.5, "rrt": False},
        {"name": "Return to Stow", "pos": stow_pos, "gripper": 255, "duration": 4.0, "rrt": True},
    ]

def get_closest_body(m, d, target_pos, candidates=["target_apple", "target_bottle", "target_metal_box"]):
    closest_body = None
    min_dist = float('inf')
    for name in candidates:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        dist = np.linalg.norm(d.xpos[body_id] - target_pos)
        if dist < min_dist:
            min_dist = dist
            closest_body = name
    return closest_body

def main():
    print("🚀 Initializing Mark-12 Vision-Language-Action (Florence-2) Agent...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    
    m = mujoco.MjModel.from_xml_path(VLM_SCENE_PATH)
    d = mujoco.MjData(m)

    analyzer = Florence2Pipeline(m, d)

    q_stow = Q_HOME.copy()
    q_stow[0] = -1.57 
    d.qpos[:len(Q_HOME)] = q_stow
    
    q_stow_pin = pin.neutral(robot.model)
    q_stow_pin[:len(Q_HOME)] = q_stow
    
    mujoco.mj_forward(m, d)
    
    print("🎲 Scattering semantic objects on the table...")
    spawned_positions = []
    for body_name in ["target_apple", "target_bottle", "target_metal_box"]:
        try:
            c_id = m.body(body_name).id
            adr = m.jnt_qposadr[m.body_jntadr[c_id]]
            for _ in range(50):
                px = np.random.uniform(0.35, 0.55)
                # Ensure objects spawn safely away from the wall's RRT clearance bubble!
                py = np.random.uniform(-0.25, 0.05) 
                valid = True
                for existing_p in spawned_positions:
                    if np.linalg.norm(np.array([px, py]) - existing_p) < 0.12: 
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

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()

    stow_pose = robot.forward_kinematics(q_stow_pin)
    STOW_POS = stow_pose.translation.copy()

    grasp_sys = GraspController(m, d, target="target_apple")
    
    GRIPPER_Z_OFFSET = 0.105
    states = []
    current_idx = 0
    state_start_time = 0.0
    q_target = d.qpos.copy()
    
    current_trajectory = None
    current_rrt_path = None
    slip_debounce_time = 0.0

    print("\n🟢 Simulation Online. Booting Cognitive Loop...")

    threading.Thread(target=input_thread, daemon=True).start()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if len(states) == 0:
                app_state["is_idle"] = True
                
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]
                    
                grasp_sys.command(255)
                
                if command_queue:
                    target_prompt = command_queue.pop(0)
                    print(f"\n🧠 Florence-2 reasoning about: '{target_prompt}'...")
                    
                    estimated_pos = analyzer.find_object(target_prompt)
                    
                    if estimated_pos is not None:
                        physical_body = get_closest_body(m, d, estimated_pos)
                        print(f"✅ Reasoning Complete! Grounded to physical body: {physical_body}")
                        
                        grasp_sys = GraspController(m, d, target=physical_body)
                        
                        body_id = m.body(physical_body).id
                        true_pos = d.xpos[body_id].copy()
                        
                        dynamic_z = 0.04 if "bottle" in physical_body else 0.02
                        PICK = np.array([true_pos[0], true_pos[1], dynamic_z + GRIPPER_Z_OFFSET])
                        PLACE = np.array([0.5, 0.5, dynamic_z + GRIPPER_Z_OFFSET]) 
                        
                        states = generate_vlm_states(PICK, PLACE, STOW_POS, GRIPPER_Z_OFFSET)
                        current_idx = 0
                        state_start_time = d.time
                        current_rrt_path = None
                        current_trajectory = None
                    else:
                        print("❌ VLM Error: Could not find an object matching that description in the scene.")

            elif current_idx < len(states):
                state = states[current_idx]
                t_state = d.time - state_start_time

                if state.get("rrt", False) and current_rrt_path is None:
                    target_se3 = pin.SE3(fixed_rotation, state["pos"])
                    posture_bias = q_stow_pin if state["name"] == "Return to Stow" else None
                    q_goal = solve_virtual_ik(robot, ik, q_current, target_se3, q_posture=posture_bias)
                    
                    hidden_cache = {}
                    for i, b_name in enumerate(["target_apple", "target_bottle", "target_metal_box"]):
                        b_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b_name)
                        if b_id != -1:
                            j_adr = m.jnt_qposadr[m.body_jntadr[b_id]]
                            hidden_cache[j_adr] = d.qpos[j_adr:j_adr+3].copy()
                            
                            d.qpos[j_adr:j_adr+3] = [10.0 + (i * 0.5), 10.0, 5.0] 
                            
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_collision(m, d)
            
                    # Wall is back! Add obstacle constraints.
                    planner = RRT(m, d, ACTIVE_JOINTS, step_size=0.15, max_iter=5000, 
                                  obstacle_names=["obstacle_wall"], clearance=0.03)
                    raw_path = planner.plan(q_current[ACTIVE_JOINTS], q_goal[ACTIVE_JOINTS])
            
                    for j_adr, orig_pos in hidden_cache.items():
                        d.qpos[j_adr:j_adr+3] = orig_pos
                    mujoco.mj_kinematics(m, d)
                    mujoco.mj_collision(m, d)
                    
                    if raw_path is None:
                        print("⚠️ RRT Failed! Falling back...")
                        current_rrt_path = [] 
                    else:
                        current_rrt_path = []
                        for q_act in raw_path[1:]: 
                            q_full = q_current.copy()
                            q_full[ACTIVE_JOINTS] = q_act
                            current_rrt_path.append(q_full)
                        state["wp_dur"] = max(0.2, state["duration"] / len(current_rrt_path))

                is_rrt_active = state.get("rrt", False) and current_rrt_path is not None and len(current_rrt_path) > 0

                if is_rrt_active:
                    if current_trajectory is None:
                        q_next = current_rrt_path[0]
                        current_trajectory = JointSpaceTrajectory(q_current, q_next, state["wp_dur"], "cubic")
                        state_start_time = d.time

                    t_traj = d.time - state_start_time
                    target_q = current_trajectory.get_position(t_traj)
                    
                    if hasattr(viewer, 'user_scn'):
                        for wp in current_rrt_path:
                            mujoco.mjv_initGeom(
                                viewer.user_scn.geoms[viewer.user_scn.ngeom],
                                type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                                pos=robot.forward_kinematics(wp).translation, mat=np.eye(3).flatten(), rgba=np.array([0, 0.5, 1.0, 0.5])
                            )
                            viewer.user_scn.ngeom += 1

                    for idx in ACTIVE_JOINTS:
                        q_target[idx] = target_q[idx]
                        d.ctrl[idx] = q_target[idx]

                else:
                    if current_trajectory is None:
                        start_se3 = robot.forward_kinematics(q_current)
                        end_se3 = pin.SE3(fixed_rotation, state["pos"])
                        current_trajectory = TaskSpaceTrajectory(start_se3, end_se3, state["duration"], "cubic")
                        state_start_time = d.time

                    t_traj = d.time - state_start_time
                    target_se3 = current_trajectory.get_pose(t_traj)

                    posture_bias = q_stow_pin if state["name"] == "Return to Stow" else None
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
                            print(f"\n🚨 SLIP DETECTED! Mission aborted. Initiating recovery...")
                            
                            current_physical_pose = robot.forward_kinematics(q_current)
                            states = generate_recovery_states(current_physical_pose.translation, STOW_POS)
                            current_idx = 0
                            state_start_time = d.time
                            current_trajectory = None
                            current_rrt_path = None
                            slip_debounce_time = 0.0
                            continue 
                    else:
                        slip_debounce_time = 0.0 

                if is_rrt_active:
                    if t_traj > current_trajectory.duration:
                        current_rrt_path.pop(0)
                        current_trajectory = None
                        if len(current_rrt_path) == 0:
                            current_idx += 1
                            state_start_time = d.time
                            current_rrt_path = None
                            slip_debounce_time = 0.0
                            if current_idx < len(states):
                                print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                            else:
                                print(f"🎉 Task Complete! Returning to standby.")
                                states = [] 
                else:
                    ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place"]
                    can_transition = (t_traj > state["duration"]) and (done or ignore_ik_error)

                    if can_transition:
                        current_idx += 1
                        state_start_time = d.time
                        current_trajectory = None
                        slip_debounce_time = 0.0
                        if current_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                        else:
                            print(f"🎉 Task Complete! Returning to standby.")
                            states = [] 

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()
            
            analyzer.show_live_feed()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()