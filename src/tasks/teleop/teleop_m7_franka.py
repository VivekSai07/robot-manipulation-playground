import time
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
# NEW: Import the Data Logger
from src.utils.data_logger import DataLogger

def main():
    print("🚀 Initializing Franka Mark-6 (Teleoperation) Systems...")

    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)
    
    grasp_sys = GraspController(m, d)
    teleop = KeyboardTeleop()
    
    # NEW: Initialize Logger
    logger = DataLogger()

    # -------------------------------
    # Set Initial Pose
    # -------------------------------
    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]
    
    q_target = np.zeros(m.nq)
    q_target[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    
    mujoco.mj_forward(m, d)

    # Initialize the teleoperation target pose to exactly where the arm starts!
    current_pose = robot.forward_kinematics(q_home_pin)
    target_se3 = current_pose.copy()

    print("\n🟢 Simulation Online.")
    print("====================================")
    print("🎮 TELEOP CONTROLS:")
    print("   [W / S] : Move Forward / Back (X-axis)")
    print("   [A / D] : Move Left / Right (Y-axis)")
    print("   [Q / E] : Move Up / Down (Z-axis)")
    print("   [SPACE] : Toggle Gripper")
    print("====================================\n")

    # Pass the teleop key_callback directly to the MuJoCo viewer!
    with mujoco.viewer.launch_passive(m, d, key_callback=teleop.key_callback) as viewer:
        
        # --- CRITICAL FIX: RENDERING OVERRIDE ---
        # Snapshot the clean, default graphics settings before the loop starts.
        default_flags = tuple(viewer.opt.flags)
        
        while viewer.is_running():
            step_start = time.time()
            
            # Forcibly lock the rendering flags every frame.
            # This completely nullifies MuJoCo's built-in WASD rendering hotkeys!
            for i in range(len(default_flags)):
                viewer.opt.flags[i] = default_flags[i]
            # ----------------------------------------
            
            q_current = d.qpos[:robot.model.nq].copy()

            # 1. Get Keyboard Commands
            v_des, gripper_closed = teleop.get_command()

            # 2. Update Virtual Target (Integrate Velocity to Position)
            target_se3.translation += v_des[:3] * m.opt.timestep

            # --- ADVANCED TELEOP: TARGET LEASHING ---
            current_physical_pose = robot.forward_kinematics(q_current)
            error_vector = target_se3.translation - current_physical_pose.translation
            leash_length = 0.1 # 10 centimeters max stretch
            
            if np.linalg.norm(error_vector) > leash_length:
                target_se3.translation = current_physical_pose.translation + (error_vector / np.linalg.norm(error_vector)) * leash_length
            # ----------------------------------------

            # Visualize the ghost target (Red Sphere)
            if hasattr(viewer, 'user_scn'):
                viewer.user_scn.ngeom = 0
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[viewer.user_scn.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([0.015, 0, 0]),
                    pos=target_se3.translation, mat=np.eye(3).flatten(), rgba=np.array([1, 0, 0, 0.6])
                )
                viewer.user_scn.ngeom += 1

            # 3. Compute IK
            dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
            dq = np.clip(dq, -2.0, 2.0)
            
            # 4. Integrate Joint Positions
            for idx in ACTIVE_JOINTS:
                q_target[idx] += dq[idx] * m.opt.timestep
                d.ctrl[idx] = q_target[idx]

            # 5. Command Gripper
            gripper_cmd = 0 if gripper_closed else 255
            grasp_sys.command(gripper_cmd)

            # 6. Physics Step
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
            mujoco.mj_step(m, d)
            viewer.sync()

            # --- NEW: DATA LOGGING ---
            # Read the target object position
            cube_id = m.body("target_cube").id
            cube_pos = d.xpos[cube_id].copy()
            
            # Record the state and action for this timestep!
            logger.log_step(
                qpos=d.qpos[:robot.model.nq],
                qvel=d.qvel[:robot.model.nv],
                ee_pose=current_physical_pose,
                cube_pos=cube_pos,
                action=d.ctrl.copy() # The exact motor commands sent this frame
            )
            # -------------------------

            # Real-time pacing
            elapsed = time.time() - step_start
            if elapsed < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed)
                
    # NEW: Save the dataset when the user closes the viewer!
    print("\n🛑 Simulation Closed. Saving data...")
    logger.save_episode()

if __name__ == "__main__":
    main()