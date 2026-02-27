import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.perception.segmentation_pipeline import SegmentationAnalyzer
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH, Q_HOME, ARM_DOF, ACTIVE_JOINTS
)

def main():
    # ========================================================
    # 🎯 CONFIGURATION
    # Try changing this to "red", "green", or "blue"
    TARGET_COLOR = "yellow" 
    TARGET_BODY = f"target_cube_{TARGET_COLOR}"
    # ========================================================

    print(f"🚀 Initializing Mark-7 Segmentation Pipeline (Target: {TARGET_COLOR})...")
    
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    
    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    # Standard Pose Setup
    d.qpos[:len(Q_HOME)] = Q_HOME
    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    
    # ---------------------------------------------------------
    # PERCEPTION STOW PHASE
    # Move the arm out of the camera's view before analyzing the scene.
    # We rotate the base 90 degrees to clear the table.
    # ---------------------------------------------------------
    d.qpos[0] = -1.57 # Rotate base to the side
    mujoco.mj_forward(m, d)
    
    # 1. Randomize all cubes while arm is out of the way
    print("🎲 Scattering objects on the table with 10cm safety clearance...")
    spawned_positions = []
    
    for color in ["red", "green", "blue", "yellow"]:
        try:
            c_id = m.body(f"target_cube_{color}").id
            adr = m.jnt_qposadr[m.body_jntadr[c_id]]
            
            for _ in range(50):
                px = np.random.uniform(0.35, 0.55)
                py = np.random.uniform(-0.25, 0.25)
                
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
            pass # Ignore if the cube isn't in the XML
    
    mujoco.mj_forward(m, d)

    # 2. Perfect Perception
    analyzer = SegmentationAnalyzer(m, d)
    
    print(f"🧠 Querying Segmentation Mask for '{TARGET_BODY}'...")
    estimated_pos = analyzer.find_object_by_name(TARGET_BODY)

    if estimated_pos is None:
        print(f"❌ Could not see '{TARGET_BODY}'! It might be off-table or occluded.")
        # Reset arm to home so user can see what happened in viewer
        d.qpos[:len(Q_HOME)] = Q_HOME
        mujoco.mj_forward(m, d)
    else:
        true_pos = d.xpos[m.body(TARGET_BODY).id]
        print(f"✅ Truth Found! Calibration Error: {np.linalg.norm(estimated_pos - true_pos)*1000:.2f}mm")

    # Move arm back to Home for the start of the task
    d.qpos[:len(Q_HOME)] = Q_HOME
    mujoco.mj_forward(m, d)
    
    # Capture orientation for the task
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # 3. Dynamic States
    if estimated_pos is not None:
        grasp_sys = GraspController(m, d, target=TARGET_BODY)
        GRIPPER_Z_OFFSET = 0.105
        PICK = np.array([estimated_pos[0], estimated_pos[1], 0.02 + GRIPPER_Z_OFFSET])
        PLACE = np.array([0.0, 0.5, 0.02 + GRIPPER_Z_OFFSET])
        approach_offset = np.array([0, 0, 0.15])
        
        states = [
            {"name": "Approach Pick", "pos": PICK + approach_offset, "gripper": 255, "dur": 2.0},
            {"name": "Descend to Pick", "pos": PICK, "gripper": 255, "dur": 1.5},
            {"name": "Grasping", "pos": PICK, "gripper": 0, "dur": 1.0},
            {"name": "Verify Lift", "pos": PICK + approach_offset, "gripper": 0, "dur": 2.0},
            {"name": "Move to Place", "pos": PLACE + approach_offset, "gripper": 0, "dur": 3.0},
            {"name": "Lower to Place", "pos": PLACE, "gripper": 0, "dur": 2.0},
            {"name": "Release", "pos": PLACE, "gripper": 255, "dur": 1.0},
            {"name": "Retract", "pos": PLACE + approach_offset, "gripper": 255, "dur": 2.0},
            {"name": "Return to Home", "pos": HOME_POS, "gripper": 255, "dur": 3.0},
        ]
    else:
        states = []

    current_idx = 0
    state_start = d.time
    q_target = d.qpos.copy()

    print("\n🟢 Simulation Online. Engaging physics...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()

            if hasattr(viewer, 'user_scn') and estimated_pos is not None:
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                    pos=estimated_pos, mat=np.eye(3).flatten(), rgba=np.array([1, 1, 1, 0.5])
                )
                viewer.user_scn.ngeom += 1

            if current_idx < len(states):
                s = states[current_idx]
                target_se3 = pin.SE3(fixed_rotation, s["pos"])
                
                posture_bias = q_home_pin if s["name"] == "Return to Home" else None
                dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=posture_bias)
                dq = np.clip(dq, -1.0, 1.0)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]
                
                grasp_sys.command(s["gripper"])
                
                is_holding = grasp_sys.is_grasped()
                t_state = d.time - state_start

                ignore_ik_error = s["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                can_transition = (t_state > s["dur"]) and (done or ignore_ik_error)

                if can_transition:
                    if s["name"] == "Verify Lift":
                        if not is_holding:
                            print(f"[{d.time:.2f}s] ⚠️ Grasp Failed! (Missing {TARGET_BODY})")
                        else:
                            print(f"[{d.time:.2f}s] ✅ Grasp verified! Proceeding to place.")

                    current_idx += 1
                    state_start = d.time
                    if current_idx < len(states):
                        print(f"[{d.time:.2f}s] State -> {states[current_idx]['name']}")

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()
            
            # Uncomment below to see the segmentation ID feed:
            # analyzer.show_debug_view()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()