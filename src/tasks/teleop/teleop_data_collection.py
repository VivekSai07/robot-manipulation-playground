import time
import sys
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.controllers.keyboard_teleop import KeyboardTeleop
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)
from src.utils.data_logger import DataLogger

def randomize_environment(m, d, robot, q_home_pin, q_target, target_se3):
    """
    DOMAIN RANDOMIZATION: Teleports the cube to a new random location 
    and snaps the robot arm back to its starting home position.
    """
    # 1. Reset Robot
    d.qpos[:robot.model.nq] = q_home_pin
    d.qvel[:] = 0
    q_target[:len(q_home_pin)] = q_home_pin.copy()
    
    # 2. Randomize Cube Position
    cube_id = m.body("target_cube").id
    cube_jnt_id = m.body_jntadr[cube_id]
    
    # Ensure we are modifying the freejoint of the cube
    if m.jnt_type[cube_jnt_id] == mujoco.mjtJoint.mjJNT_FREE:
        cube_qpos_adr = m.jnt_qposadr[cube_jnt_id]
        
        # Spawn randomly in front of the robot
        random_x = np.random.uniform(0.35, 0.55)
        random_y = np.random.uniform(-0.25, 0.25)
        
        d.qpos[cube_qpos_adr] = random_x
        d.qpos[cube_qpos_adr+1] = random_y
        d.qpos[cube_qpos_adr+2] = 0.02 # Z height
        
        # Zero out the cube's momentum
        cube_dof_adr = m.jnt_dofadr[cube_jnt_id]
        d.qvel[cube_dof_adr:cube_dof_adr+6] = 0
        
    mujoco.mj_forward(m, d)
    
    # 3. Reset the Teleop Ghost Target to the physical arm
    current_pose = robot.forward_kinematics(q_home_pin)
    target_se3.translation = current_pose.translation.copy()
    target_se3.rotation = current_pose.rotation.copy()


def main():
    print("🚀 Initializing Franka Mark-6 (Data Collection Pipeline)...")

    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    
    grasp_sys = GraspController(m, d)
    teleop = KeyboardTeleop()
    logger = DataLogger()

    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    
    mujoco.mj_forward(m, d)

    current_pose = robot.forward_kinematics(q_home_pin)
    target_se3 = current_pose.copy()
    
    DROP_ZONE = np.array([0.0, 0.5, 0.02])

    print("\n🟢 Data Pipeline Online. Focus the Viewer Window to begin.")
    
    episode_count = 1
    step_count = 0

    with mujoco.viewer.launch_passive(m, d, key_callback=teleop.key_callback) as viewer:
        
        # Snapshot pure graphics settings to prevent WASD collisions
        default_flags = tuple(viewer.opt.flags)
        
        while viewer.is_running():
            step_start = time.time()
            
            for i in range(len(default_flags)):
                viewer.opt.flags[i] = default_flags[i]
            
            q_current = d.qpos[:robot.model.nq].copy()

            v_des, gripper_closed = teleop.get_command()
            target_se3.translation += v_des[:3] * m.opt.timestep

            current_physical_pose = robot.forward_kinematics(q_current)
            error_vector = target_se3.translation - current_physical_pose.translation
            leash_length = 0.1 
            
            if np.linalg.norm(error_vector) > leash_length:
                target_se3.translation = current_physical_pose.translation + (error_vector / np.linalg.norm(error_vector)) * leash_length

            # --- VISUAL RENDERING ---
            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0
                
                # Ghost Teleop Sphere
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                    pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.6])
                )
                viewer.user_scn.ngeom += 1

                # Drop Zone Target (Semi-transparent Green Box)
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_BOX, size=np.array([0.06, 0.06, 0.002]),
                    pos=DROP_ZONE, mat=np.eye(3).flatten(), rgba=np.array([0, 1, 0, 0.3])
                )
                viewer.user_scn.ngeom += 1
            # ------------------------

            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * m.opt.timestep
                d.ctrl[idx] = q_target[idx]

            gripper_cmd = 0 if gripper_closed else 255
            grasp_sys.command(gripper_cmd)

            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()

            # --- DATA LOGGING & TERMINAL HUD ---
            cube_id = m.body("target_cube").id
            cube_pos = d.xpos[cube_id].copy()
            
            logger.log_step(
                qpos=d.qpos[:robot.model.nq], qvel=d.qvel[:robot.model.nv],
                ee_pose=current_physical_pose, cube_pos=cube_pos, action=d.ctrl.copy()
            )
            step_count += 1
            
            # Print HUD on a single refreshing line
            sys.stdout.write(f"\r🎥 Episode: {episode_count} | Steps: {step_count} | [R] Reset | [Space] Grip | [WASD] Drive")
            sys.stdout.flush()
            # -----------------------------------

            # --- EPISODE MANAGER LOGIC ---
            dist_to_drop = np.linalg.norm(cube_pos[:2] - DROP_ZONE[:2])
            
            # Success: The cube is inside the green Drop Zone, resting on the table, and the gripper released it!
            is_cube_dropped = dist_to_drop < 0.06 and cube_pos[2] < 0.05 and not gripper_closed
            
            if teleop.reset_requested or is_cube_dropped:
                sys.stdout.write("\n") # Break the HUD line
                print(f"✅ Episode {episode_count} Complete! Saving data...")
                logger.save_episode()
                
                randomize_environment(m, d, robot, q_home_pin, q_target, target_se3)
                
                teleop.reset_requested = False
                episode_count += 1
                step_count = 0
                time.sleep(0.5) # Give human a moment to re-orient
                continue
            # -----------------------------

            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)
                
    print("\n\n🛑 Simulation Closed. Saving final data...")
    logger.save_episode()

if __name__ == "__main__":
    main()