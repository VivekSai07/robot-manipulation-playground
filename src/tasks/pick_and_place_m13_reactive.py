import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.controllers.apf_controller import APFController
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    ROBOT_DIR, Q_HOME, ARM_DOF, ACTIVE_JOINTS
)
import os

# Load the chaotic dynamic environment
DYNAMIC_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "dynamic_scene.xml")

def generate_reactive_states(pick_pos, place_pos, home_pos, z_offset):
    """
    State machine for Reactive Control.
    Notice we use 'apf': True for large spatial movements where dodging is required,
    but 'apf': False for delicate vertical insertions (so the arm doesn't abort a placement).
    """
    approach_offset = np.array([0, 0, 0.15])
    return [
        {"name": "Approach Pick", "pos": pick_pos + approach_offset, "gripper": 255, "apf": True},
        {"name": "Descend to Pick", "pos": pick_pos, "gripper": 255, "apf": False, "duration": 1.5},
        {"name": "Grasping", "pos": pick_pos, "gripper": 0, "apf": False, "duration": 1.0},
        {"name": "Verify Lift", "pos": pick_pos + approach_offset, "gripper": 0, "apf": False, "duration": 2.0},
        {"name": "Move to Place", "pos": place_pos + approach_offset, "gripper": 0, "apf": True},
        {"name": "Lower to Place", "pos": place_pos, "gripper": 0, "apf": False, "duration": 2.0},
        {"name": "Release", "pos": place_pos, "gripper": 255, "apf": False, "duration": 1.0},
        {"name": "Retract", "pos": place_pos + approach_offset, "gripper": 255, "apf": True},
        {"name": "Return to Home", "pos": home_pos, "gripper": 255, "apf": True},
    ]

def main():
    print("🚀 Initializing Mark-13 Reactive Agent (APF Dynamic Avoidance)...")
    
    robot = FrankaPanda()
    
    # Initialize both brains! 
    # IK handles Joint math, APF handles Cartesian reflexes.
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=5.0, kp_rot=3.0)
    # Increased repulsion strength to 0.5 and radius to 0.35 for more aggressive, earlier dodging.
    apf = APFController(k_att_pos=2.0, k_att_rot=2.0, k_rep_pos=0.5, influence_radius=0.20, max_v=1.5)
    
    m = mujoco.MjModel.from_xml_path(DYNAMIC_SCENE_PATH)
    d = mujoco.MjData(m)

    d.qpos[:len(Q_HOME)] = Q_HOME
    q_target = np.zeros(m.nq)
    q_target[:len(Q_HOME)] = Q_HOME

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:len(Q_HOME)] = Q_HOME
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # Look up dynamic obstacles
    threat_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "threat_ball")
    pendulum_jnt_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "pendulum_hinge")
    pendulum_dof_adr = m.jnt_dofadr[pendulum_jnt_id]
    
    # Pre-fetch key arm bodies for Whole-Arm Avoidance!
    elbow_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_link4")
    wrist_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "panda_link6")
    
    # Pre-fetch Pinocchio Frame IDs for Jacobian computation
    elbow_frame_id = robot.model.getFrameId("panda_link4")
    wrist_frame_id = robot.model.getFrameId("panda_link6")

    # Give the pendulum a massive initial shove to start the chaos!
    d.qvel[pendulum_dof_adr] = 5.0
    mujoco.mj_forward(m, d)

    grasp_sys = GraspController(m, d, target="target_cube")
    GRIPPER_Z_OFFSET = 0.105

    # Get target cube position for planning
    cube_id = m.body("target_cube").id
    cube_pos = d.xpos[cube_id].copy()
    
    PICK = np.array([cube_pos[0], cube_pos[1], 0.02 + GRIPPER_Z_OFFSET])
    PLACE = np.array([0.0, 0.5, 0.02 + GRIPPER_Z_OFFSET]) 
    
    states = generate_reactive_states(PICK, PLACE, HOME_POS, GRIPPER_Z_OFFSET)
    current_idx = 0
    state_start_time = 0.0

    print("\n🟢 Simulation Online. Engaging Reactive Physics Loop...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            q_current = d.qpos[:robot.model.nq].copy()
            current_physical_pose = robot.forward_kinematics(q_current)

            # --- Actively drive the pendulum so it never stops swinging! ---
            # Applying a slower, gentler sinusoidal torque to act like a motor pushing a swing
            d.qfrc_applied[pendulum_dof_adr] = np.sin(d.time * 1.5) * 1.0

            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0

            if current_idx < len(states):
                state = states[current_idx]
                t_state = d.time - state_start_time
                target_se3 = pin.SE3(fixed_rotation, state["pos"])

                # ==========================================
                # APF REACTIVE TRACKING (Dodging Mode)
                # ==========================================
                if state.get("apf", False):
                    # 1. Perceive Threats & Arm Posture
                    threat_pos = d.geom_xpos[threat_geom_id].copy()
                    elbow_pos = d.xpos[elbow_id].copy()
                    wrist_pos = d.xpos[wrist_id].copy()
                    
                    # Update Pinocchio Kinematics to calculate exact Jacobians for the elbow and wrist
                    pin.forwardKinematics(robot.model, robot.data, q_current)
                    pin.updateFramePlacements(robot.model, robot.data)
                    
                    J_elbow = pin.computeFrameJacobian(robot.model, robot.data, q_current, elbow_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    J_wrist = pin.computeFrameJacobian(robot.model, robot.data, q_current, wrist_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    
                    # 2. Compute Safe Cartesian Velocity AND Joint-Space Avoidance Torques
                    robot_links = [(wrist_pos, J_wrist), (elbow_pos, J_elbow)]
                    v_des_apf, dq_rep_null = apf.compute_velocity(
                        current_physical_pose, target_se3, [threat_pos], 
                        robot_links=robot_links, nv=robot.model.nv
                    )
                    
                    # 3. Decoupling Trick: Trick the IK controller into producing this exact Cartesian velocity
                    v_des_scaled = np.zeros(6)
                    v_des_scaled[:3] = v_des_apf[:3] / ik.kp_pos
                    v_des_scaled[3:] = v_des_apf[3:] / ik.kp_rot
                    
                    safe_target_se3 = current_physical_pose * pin.exp6(v_des_scaled)
                    
                    # 4. Inject Joint-Space Repulsion directly into the IK Null-Space!
                    q_posture_base = q_home_pin if state["name"] == "Return to Home" else q_current
                    posture_bias = q_posture_base + dq_rep_null
                    
                    dq, err, _ = ik.compute_velocity(q_current, safe_target_se3, q_posture=posture_bias)
                    
                    # 4. Check if we arrived at the true global target
                    dist_to_target = np.linalg.norm(current_physical_pose.translation - target_se3.translation)
                    can_transition = dist_to_target < 0.015 # Arrived within 1.5 cm!

                    # --- VISUALIZE APF SHIELD ---
                    if hasattr(viewer, 'user_scn'):
                        # Draw the true goal (Green Dot)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                            pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.8])
                        )
                        viewer.user_scn.ngeom += 1
                        
                        # Draw the APF Threat Bubble (Translucent Red Sphere)
                        mujoco.mjv_initGeom(
                            viewer.user_scn.geoms[viewer.user_scn.ngeom],
                            type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([apf.influence_radius, 0, 0]),
                            pos=threat_pos, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.15])
                        )
                        viewer.user_scn.ngeom += 1

                # ==========================================
                # LINEAR TRACKING (Precision Mode)
                # ==========================================
                else:
                    dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                    ignore_ik_error = state["name"] in ["Grasping", "Release", "Lower to Place", "Descend to Pick"]
                    can_transition = (t_state > state["duration"]) and (done or ignore_ik_error)

                # ==========================================
                # EXECUTE AND ADVANCE STATE
                # ==========================================
                dq = np.clip(dq, -1.5, 1.5)
                
                for idx in ACTIVE_JOINTS:
                    q_target[idx] += dq[idx] * m.opt.timestep
                    d.ctrl[idx] = q_target[idx]

                grasp_sys.command(state["gripper"])
                is_holding = grasp_sys.is_grasped()

                if can_transition:
                    if state["name"] == "Verify Lift" and not is_holding:
                        print(f"[{d.time:.2f}s] ⚠️ Grasp Failed! Resetting...")
                        current_idx = 0
                        state_start_time = d.time
                        continue 
                        
                    current_idx += 1
                    state_start_time = d.time
                    if current_idx < len(states):
                        print(f"[{d.time:.2f}s] State → {states[current_idx]['name']}")
                    else:
                        print(f"[{d.time:.2f}s] 🎉 Mission Accomplished! Closing simulation.")
                        break

            else:
                for idx in ACTIVE_JOINTS:
                    d.ctrl[idx] = q_target[idx]

            # Gravity compensation for the arm (Pendulum handles its own physics!)
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)

if __name__ == "__main__":
    main()