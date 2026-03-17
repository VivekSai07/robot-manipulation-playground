import os
import time
import threading
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import cv2

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.vlm_pipeline import VLMPipeline
from src.planners.trajectory_planner import TaskSpaceTrajectory, JointSpaceTrajectory
from src.planners.rrt import RRT
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    Q_HOME, ARM_DOF, ACTIVE_JOINTS, ROBOT_DIR
)

# Load the Semantic VLM Environment
VLM_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "vlm_scene.xml")

# Global queue to safely pass text prompts from the input thread to the physics loop
command_queue = []

def input_thread():
    """Runs in the background to capture user text without freezing the physics engine."""
    time.sleep(2.0) # Wait for simulation to boot
    while True:
        cmd = input("\n🗣️ ENTER COMMAND (e.g., 'healthy snack', 'metal box', 'drink'):\n> ")
        if cmd.strip():
            command_queue.append(cmd.strip())

def solve_virtual_ik(robot, ik, q_start, target_se3, q_posture=None):
    """Solves for the joint-space goal position needed for RRT."""
    q_virtual = q_start.copy()
    for _ in range(1000):
        dq, err, done = ik.compute_velocity(q_virtual, target_se3, q_posture=q_posture, tol=1e-3)
        q_virtual += dq * 0.05
        if done:
            break
    return q_virtual

def generate_vlm_states(pick_pos, place_pos, stow_pos, z_offset):
    """Generates the trajectory state machine."""
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

def get_closest_body(m, d, target_pos, candidates=["target_apple", "target_bottle", "target_metal_box"]):
    """Finds the physics body that best matches the VLM's visual coordinate."""
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
    print("🚀 Initializing Mark-11 Vision-Language-Action (VLA) Agent...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    
    m = mujoco.MjModel.from_xml_path(VLM_SCENE_PATH)
    d = mujoco.MjData(m)

    # 1. Initialize the VLM Engine (Downloads models if necessary)
    analyzer = VLMPipeline(m, d)

    # 2. Define the "Perception Stow" Pose (Base turned -90 degrees to clear the camera view)
    q_stow = Q_HOME.copy()
    q_stow[0] = -1.57 
    d.qpos[:len(Q_HOME)] = q_stow
    
    q_stow_pin = pin.neutral(robot.model)
    q_stow_pin[:len(Q_HOME)] = q_stow
    
    mujoco.mj_forward(m, d)
    
    # 3. Scatter the Semantic Objects
    print("🎲 Scattering semantic objects on the table...")
    spawned_positions = []
    for body_name in ["target_apple", "target_bottle", "target_metal_box"]:
        try:
            c_id = m.body(body_name).id
            adr = m.jnt_qposadr[m.body_jntadr[c_id]]
            for _ in range(50):
                px = np.random.uniform(0.35, 0.55)
                py = np.random.uniform(-0.25, 0.15) 
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

    # CRITICAL FIX 1: Capture the natural forward-facing rotation from Q_HOME.
    # If we use the rotation from the -90 degree Stow Pose, the robot will try to do 
    # the entire task with a severely twisted wrist, forcing the elbow to flare out 
    # and smash into the wall during RRT planning!
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()

    stow_pose = robot.forward_kinematics(q_stow_pin)
    STOW_POS = stow_pose.translation.copy()

    # FIX 1: Initialize dummy GraspController to manage gripper state during IDLE
    grasp_sys = GraspController(m, d, target="target_apple")
    GRIPPER_Z_OFFSET = 0.105
    states = []
    current_idx = 0
    state_start_time = 0.0
    q_target = d.qpos.copy()
    
    current_trajectory = None
    current_rrt_path = None

    print("\n🟢 Simulation Online. Booting Cognitive Loop...")

    # Start the interactive terminal thread
    threading.Thread(target=input_thread, daemon=True).start()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            # ==========================================
            # IDLE STATE: Awaiting User Command
            # ==========================================
            if len(states) == 0:
                # Hold the stow position
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]
                
                # CRITICAL FIX 1: Hold the gripper OPEN while waiting!
                # If we don't command the gripper, it defaults to 0 (Closed).
                # When the fingers close empty, their collision meshes intersect,
                # causing RRT to instantly fail with "Start position in collision!"
                grasp_sys.command(255)
                
                # Check if user typed a command
                if command_queue:
                    target_prompt = command_queue.pop(0)
                    print(f"\n🧠 VLM reasoning about: '{target_prompt}'...")
                    
                    # 1. RUN VLM INFERENCE
                    estimated_pos = analyzer.find_object(target_prompt)
                    
                    if estimated_pos is not None:
                        # 2. MATCH TO PHYSICS BODY
                        physical_body = get_closest_body(m, d, estimated_pos)
                        print(f"✅ Reasoning Complete! Grounded to physical body: {physical_body}")
                        
                        # Re-initialize the Grasp Controller dynamically so it feels for the correct body!
                        grasp_sys = GraspController(m, d, target=physical_body)
                        
                        # Get perfect precision position from physics
                        body_id = m.body(physical_body).id
                        true_pos = d.xpos[body_id].copy()
                        
                        dynamic_z = 0.04 if "bottle" in physical_body else 0.02
                        PICK = np.array([true_pos[0], true_pos[1], dynamic_z + GRIPPER_Z_OFFSET])
                        PLACE = np.array([0.5, 0.5, dynamic_z + GRIPPER_Z_OFFSET]) 
                        
                        # 3. GENERATE RRT TRAJECTORY
                        states = generate_vlm_states(PICK, PLACE, STOW_POS, GRIPPER_Z_OFFSET)
                        current_idx = 0
                        state_start_time = d.time
                        current_rrt_path = None
                        current_trajectory = None
                    else:
                        print("❌ VLM Error: Could not find an object matching that description in the scene.")

            # ==========================================
            # ACTIVE TASK EXECUTION
            # ==========================================
            elif current_idx < len(states):
                state = states[current_idx]
                t_state = d.time - state_start_time

                # --- RRT Path Generation ---
                if state.get("rrt", False) and current_rrt_path is None:
                    target_se3 = pin.SE3(fixed_rotation, state["pos"])
                    posture_bias = q_stow_pin if state["name"] == "Return to Stow" else None
                    q_goal = solve_virtual_ik(robot, ik, q_current, target_se3, q_posture=posture_bias)
                    
                    planner = RRT(m, d, ACTIVE_JOINTS, step_size=0.15, max_iter=5000, 
                                  obstacle_names=["obstacle_wall"], clearance=0.04)
                    raw_path = planner.plan(q_current[ACTIVE_JOINTS], q_goal[ACTIVE_JOINTS])
                    
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

                # CRITICAL FIX: Check len(current_rrt_path) > 0 to prevent IndexError
                is_rrt_active = state.get("rrt", False) and current_rrt_path is not None and len(current_rrt_path) > 0

                # --- Execute Trajectory (RRT or Linear) ---
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

                # --- Reactive Drop Detection ---
                if state["gripper"] == 0 and state["name"] not in ["Grasping", "Lower to Place", "Release"]:
                    if not is_holding and t_state > 0.2:
                        print(f"\n🚨 SLIP DETECTED! Mission aborted.")
                        states = [] # Force IDLE state
                        current_idx = 0
                        current_trajectory = None
                        current_rrt_path = None
                        continue 

                # --- Normal State Transitions ---
                if is_rrt_active:
                    if t_traj > current_trajectory.duration:
                        current_rrt_path.pop(0)
                        current_trajectory = None
                        if len(current_rrt_path) == 0:
                            current_idx += 1
                            state_start_time = d.time
                            current_rrt_path = None
                            if current_idx < len(states):
                                print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                            else:
                                print(f"🎉 Task Complete! Returning to standby.")
                                states = [] # Trigger IDLE
                else:
                    # CRITICAL FIX 2: Removed "Descend to Pick" from ignore list so it waits for IK to physically arrive!
                    ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place"]
                    can_transition = (t_traj > state["duration"]) and (done or ignore_ik_error)

                    if can_transition:
                        current_idx += 1
                        state_start_time = d.time
                        current_trajectory = None
                        if current_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                        else:
                            print(f"🎉 Task Complete! Returning to standby.")
                            states = [] # Trigger IDLE

            # Update Physics and Viewer
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Keep OpenCV Window Alive
            analyzer.show_live_feed()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()