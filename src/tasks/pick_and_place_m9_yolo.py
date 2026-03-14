import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.yolo_pipeline import YOLOPipeline
from src.planners.trajectory_planner import TaskSpaceTrajectory, JointSpaceTrajectory
from src.planners.rrt import RRT
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    Q_HOME, ARM_DOF, ACTIVE_JOINTS, ROBOT_DIR
)

# Load the AI-specific environment with the angled camera
YOLO_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "yolo_scene.xml")

def solve_virtual_ik(robot, ik, q_start, target_se3, q_posture=None):
    q_virtual = q_start.copy()
    for _ in range(1000):
        dq, err, done = ik.compute_velocity(q_virtual, target_se3, q_posture=q_posture, tol=1e-3)
        q_virtual += dq * 0.05
        if done:
            break
    return q_virtual

def generate_task_states(pick_pos, place_pos, home_pos, z_offset):
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

def generate_recovery_states(home_pos):
    return [
        {"name": "Recovery Stow", "pos": home_pos, "gripper": 255, "duration": 3.0, "rrt": True},
        {"name": "Perception Scan", "pos": home_pos, "gripper": 255, "duration": 0.5, "rrt": False}
    ]

def main():
    # ========================================================
    # 🧠 AI SEMANTIC TARGET
    # We replaced the primitives with High-Fidelity Google Scanned Objects!
    # Try: "orange", "bowl", or "plate"
    TARGET_CLASS = "orange" 
    # ========================================================

    # Map the YOLO prediction string to our physical MuJoCo body 
    # so the tactile GraspController knows what to feel for!
    CLASS_TO_BODY = {
        "orange": "target_orange",
        "bowl": "target_plate", 
        "plate": "target_plate"
    }
    TARGET_BODY = CLASS_TO_BODY.get(TARGET_CLASS, "target_box")

    print(f"🚀 Initializing Mark-9 YOLOv8 Agent (Target: {TARGET_CLASS.upper()})...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    
    # Let MuJoCo handle the path resolution naturally so panda.xml works
    m = mujoco.MjModel.from_xml_path(YOLO_SCENE_PATH)
    d = mujoco.MjData(m)

    d.qpos[:len(Q_HOME)] = Q_HOME
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    
    # 1. Immediate Initial Stow (Clear the camera view)
    d.qpos[0] = -1.57 
    mujoco.mj_forward(m, d)
    
    print("🎲 Placing semantic objects on the table in fixed positions...")
    
    # Pre-defined known good coordinates within the camera view
    FIXED_POSITIONS = {
        "target_orange":   [0.45,  0.0],
        "target_plate": [0.45,  0.22],
        "target_box":   [0.45, -0.22]
    }
    
    for body_name, pos in FIXED_POSITIONS.items():
        try:
            c_id = m.body(body_name).id
            adr = m.jnt_qposadr[m.body_jntadr[c_id]]
            
            d.qpos[adr] = pos[0]
            d.qpos[adr+1] = pos[1]
        except ValueError:
            pass 
            
    mujoco.mj_forward(m, d)

    # Initialize the YOLO Engine
    analyzer = YOLOPipeline(m, d)
    
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    grasp_sys = GraspController(m, d, target=TARGET_BODY)
    GRIPPER_Z_OFFSET = 0.105

    # Start the agent directly in the scanning phase!
    states = generate_recovery_states(HOME_POS)
    current_idx = 1 
    state_start_time = 0.0
    q_target = d.qpos.copy()
    
    current_trajectory = None
    current_rrt_path = None

    print("\n🟢 Simulation Online. Engaging Neural Physics Loop...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if current_idx < len(states):
                state = states[current_idx]
                t_state = d.time - state_start_time

                # ==========================================
                # THE BRAIN: AI PERCEPTION SCANNING STATE
                # ==========================================
                if state["name"] == "Perception Scan":
                    for idx in ACTIVE_JOINTS:
                        d.ctrl[idx] = q_target[idx]
                        
                    if t_state > state["duration"]:
                        print(f"🧠 YOLO is searching for: {list(CLASS_TO_BODY.keys())}...")
                        
                        # Call the Neural Network
                        estimated_pos = analyzer.find_object(list(CLASS_TO_BODY.keys()))
                        
                        if estimated_pos is not None:
                            print(f"✅ AI Detection Successful! Generating strategy...")
                            
                            # The orange is spherical, so we don't need a massive height offset 
                            # like we did for the tall cup.
                            dynamic_z = 0.01
                                
                            PICK = np.array([estimated_pos[0], estimated_pos[1], dynamic_z + GRIPPER_Z_OFFSET])
                            PLACE = np.array([0.5, 0.5, dynamic_z + GRIPPER_Z_OFFSET]) 
                            
                            states = generate_task_states(PICK, PLACE, HOME_POS, GRIPPER_Z_OFFSET)
                            current_idx = 0
                            state_start_time = d.time
                            current_rrt_path = None
                            current_trajectory = None
                        else:
                            # AI failed to find it (maybe occluded?). Keep scanning.
                            state_start_time = d.time 
                    
                    d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
                    mujoco.mj_step(m, d)
                    viewer.sync()
                    
                    # Update the live OpenCV Window
                    analyzer.show_live_feed()
                    continue

                # ==========================================
                # RRT PATH GENERATION (Obstacle Avoidance)
                # ==========================================
                if state.get("rrt", False) and current_rrt_path is None:
                    target_se3 = pin.SE3(fixed_rotation, state["pos"])
                    posture_bias = q_home_pin if state["name"] in ["Return to Home", "Recovery Stow"] else None
                    q_goal = solve_virtual_ik(robot, ik, q_current, target_se3, q_posture=posture_bias)
                    
                    planner = RRT(m, d, ACTIVE_JOINTS, step_size=0.15, max_iter=5000, 
                                  clearance=0.04)
                    raw_path = planner.plan(q_current[ACTIVE_JOINTS], q_goal[ACTIVE_JOINTS])
                    
                    if raw_path is None:
                        print("⚠️ RRT Failed to find path. Re-attempting calculation...")
                        current_rrt_path = None 
                        continue 
                    else:
                        current_rrt_path = []
                        for q_act in raw_path[1:]: 
                            q_full = q_current.copy()
                            q_full[ACTIVE_JOINTS] = q_act
                            current_rrt_path.append(q_full)
                        
                        state["wp_dur"] = max(0.2, state["duration"] / len(current_rrt_path))

                is_rrt_active = state.get("rrt", False) and current_rrt_path is not None

                # ==========================================
                # EXECUTE TRAJECTORY
                # ==========================================
                if is_rrt_active:
                    if current_trajectory is None:
                        q_next = current_rrt_path[0]
                        current_trajectory = JointSpaceTrajectory(q_current, q_next, state["wp_dur"], "cubic")
                        state_start_time = d.time

                    t_traj = d.time - state_start_time
                    target_q = current_trajectory.get_position(t_traj)
                    
                    # Draw RRT Trail
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

                    posture_bias = q_home_pin if state["name"] in ["Return to Home", "Recovery Stow"] else None
                    dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=posture_bias)
                    dq = np.clip(dq, -1.0, 1.0)
                    
                    for idx in ACTIVE_JOINTS:
                        q_target[idx] += dq[idx] * m.opt.timestep
                        d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])
                is_holding = grasp_sys.is_grasped()

                # ==========================================
                # REACTIVE DROP DETECTION
                # ==========================================
                if state["gripper"] == 0 and state["name"] not in ["Grasping", "Lower to Place", "Release"]:
                    if not is_holding and t_state > 0.2:
                        print(f"\n[{d.time:.2f}s] 🚨 SLIP DETECTED! '{TARGET_CLASS}' lost during '{state['name']}'!")
                        print("🤖 Initiating Recovery Protocol...")
                        states = generate_recovery_states(HOME_POS)
                        current_idx = 0
                        state_start_time = d.time
                        current_trajectory = None
                        current_rrt_path = None
                        continue 

                # ==========================================
                # NORMAL STATE TRANSITIONS
                # ==========================================
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
                    ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                    can_transition = (t_traj > state["duration"]) and (done or ignore_ik_error)

                    if can_transition:
                        current_idx += 1
                        state_start_time = d.time
                        current_trajectory = None
                        if current_idx < len(states):
                            print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                        else:
                            print(f"[{d.time:.2f}s] 🎉 Mission Accomplished! Entering continuous run mode...")
                            states = generate_recovery_states(HOME_POS)
                            current_idx = 0
                            state_start_time = d.time

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Keep the AI Vision window updating smoothly!
            analyzer.show_live_feed()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()