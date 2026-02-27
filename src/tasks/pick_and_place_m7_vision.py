import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer
import cv2

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.vision_pipeline import VisionAnalyzer
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH, Q_HOME, ARM_DOF, ACTIVE_JOINTS
)

def main():
    print("🚀 Initializing Franka Mark-7 (Multi-Object NLP Vision) Systems...")
    
    # ========================================================
    # 🗣️ ENTER YOUR TEXT COMMAND HERE
    # Try: "pick up the green block", "grab the blue one", etc.
    USER_PROMPT = "pick up the blue block"
    # ========================================================

    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

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
    # Scatter All Cubes Randomly!
    # -------------------------------
    print("🎲 Scattering objects on the table with 10cm safety clearance...")
    spawned_positions = []
    
    for cube_name in ["target_cube_red", "target_cube_green", "target_cube_blue", "target_cube_yellow"]:
        try:
            cube_id = m.body(cube_name).id
            cube_jnt_id = m.body_jntadr[cube_id]
            cube_qpos_adr = m.jnt_qposadr[cube_jnt_id]
            
            for _ in range(50):
                px = np.random.uniform(0.35, 0.55)
                py = np.random.uniform(-0.25, 0.25)
                
                valid = True
                for existing_p in spawned_positions:
                    if np.linalg.norm(np.array([px, py]) - existing_p) < 0.10:
                        valid = False
                        break
                        
                if valid:
                    d.qpos[cube_qpos_adr] = px
                    d.qpos[cube_qpos_adr+1] = py
                    spawned_positions.append(np.array([px, py]))
                    break
        except ValueError:
            pass
            
    mujoco.mj_forward(m, d) 

    # -------------------------------
    # Vision Perception Phase
    # -------------------------------
    print(f"\n🗣️  User Command: '{USER_PROMPT}'")
    vision = VisionAnalyzer(m, d, camera_name="workspace_cam")
    
    estimated_cube_pos = vision.find_object_by_color(USER_PROMPT)

    if estimated_cube_pos is None:
        print("❌ Vision Error: Exiting pipeline.")
        return

    # Look up the true physics ID based on the parsed color to prove our math
    target_color = vision.parse_color_from_text(USER_PROMPT)
    target_body_name = f"target_cube_{target_color}"
    cube_id = m.body(target_body_name).id
    
    true_cube_pos = d.xpos[cube_id]
    error = np.linalg.norm(estimated_cube_pos - true_cube_pos)
    
    print(f"   Real Position:      {true_cube_pos}")
    print(f"   Estimated Position: {estimated_cube_pos}")
    print(f"   Calibration Error:  {error * 1000:.2f} mm")

    # -------------------------------
    # Dynamically Init Grasp Controller
    # -------------------------------
    # We MUST initialize the grasp system pointing to the correct body, 
    # otherwise the tactile sensors will be waiting to feel the wrong cube!
    grasp_sys = GraspController(m, d, target=target_body_name)

    # -------------------------------
    # Dynamic Waypoint Generation
    # -------------------------------
    GRIPPER_Z_OFFSET = 0.105 
    PICK_POS = np.array([estimated_cube_pos[0], estimated_cube_pos[1], 0.02 + GRIPPER_Z_OFFSET])
    PLACE_POS = np.array([0.0, 0.5, 0.02 + GRIPPER_Z_OFFSET])
    approach_offset = np.array([0, 0, 0.15])
    
    states = [
        {"name": "Approach Pick", "pos": PICK_POS + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Descend to Pick", "pos": PICK_POS, "gripper": 255, "duration": 1.5},
        {"name": "Grasping", "pos": PICK_POS, "gripper": 0, "duration": 1.0},
        {"name": "Verify Lift", "pos": PICK_POS + approach_offset, "gripper": 0, "duration": 2.0},
        {"name": "Move to Place", "pos": PLACE_POS + approach_offset, "gripper": 0, "duration": 3.0},
        {"name": "Lower to Place", "pos": PLACE_POS, "gripper": 0, "duration": 2.0},
        {"name": "Release", "pos": PLACE_POS, "gripper": 255, "duration": 1.0},
        {"name": "Retract", "pos": PLACE_POS + approach_offset, "gripper": 255, "duration": 2.0},
        {"name": "Return to Home", "pos": HOME_POS, "gripper": 255, "duration": 3.0},
    ]

    current_state_idx = 0
    state_start_time = 0.0

    print("\n🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                    pos=estimated_cube_pos, mat=np.eye(3).flatten(), rgba=np.array([1, 1, 1, 0.5])
                )
                viewer.user_scn.ngeom += 1

            if current_state_idx < len(states):
                state = states[current_state_idx]

                target_se3 = pin.SE3(fixed_rotation, state["pos"])
                posture_bias = q_home_pin if state["name"] == "Return to Home" else None
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=posture_bias)
                dq = np.clip(dq, -1.0, 1.0)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])
                
                is_holding = grasp_sys.is_grasped()
                t_state = d.time - state_start_time
                
                ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                can_transition = (t_state > state["duration"]) and (done or ignore_ik_error)

                if can_transition:
                    if state["name"] == "Verify Lift":
                        if not is_holding:
                            print(f"[{d.time:.2f}s] ⚠️ Grasp Failed!")
                        else:
                            print(f"[{d.time:.2f}s] ✅ Grasp verified! Proceeding to place.")
                    
                    current_state_idx += 1
                    state_start_time = d.time
                    if current_state_idx < len(states):
                        print(f"[{d.time:.2f}s] State → {states[current_state_idx]['name']}")

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Uncomment this to see the live view as the robot executes!
            # vision.show_live_feed()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()