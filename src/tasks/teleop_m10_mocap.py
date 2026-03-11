import time
import os
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.ik_controller_m2 import IKController
from src.controllers.grasp_controller import GraspController
from src.controllers.mediapipe_teleop import MediaPipeTeleop
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS,
    ROBOT_DIR
)

# Load the dedicated MoCap scene
MOCAP_SCENE_PATH = os.path.join(ROBOT_DIR, "model", "mocap_scene.xml")

def main():
    print("🚀 Initializing Franka Mark-10 (MediaPipe MoCap) Systems...")

    # -------------------------------
    # Initialize Core Systems
    # -------------------------------
    robot = FrankaPanda()
    ik = IKController(robot, active_joint_indices=ACTIVE_JOINTS, kp_pos=10.0, kp_rot=5.0)

    m = mujoco.MjModel.from_xml_path(MOCAP_SCENE_PATH)
    d = mujoco.MjData(m)
    
    grasp_sys = GraspController(m, d)
    
    # Initialize the Webcam AI (This will turn on your camera light!)
    print("📷 Firing up the webcam neural network...")
    teleop = MediaPipeTeleop(speed_scale=75.0, smoothing=0.6)

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

    # Get initial pose of the arm to sync the ghost target
    current_pose = robot.forward_kinematics(q_home_pin)
    target_se3 = current_pose.copy()

    # Look up the ID of the mocap body in the physics engine
    mocap_id = m.body("mocap_target").mocapid[0]
    
    # Snap the ghost sphere to the robot's starting hand position
    d.mocap_pos[mocap_id] = target_se3.translation.copy()

    print("\n🟢 Simulation Online.")
    print("====================================")
    print("✋ MOCAP CONTROLS:")
    print("   1. Pinch Index & Middle Finger to Engage Movement (Clutch)!")
    print("   2. Move your palm to drive the ghost sphere.")
    print("   3. Pinch Thumb & Index Finger to close the Gripper!")
    print("====================================\n")

    try:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                q_current = d.qpos[:robot.model.nq].copy()

                # 1. Get Neural Network Hand Tracking Command
                # NOTE: This naturally throttles the loop to ~30 FPS (the webcam speed)
                v_des, gripper_closed = teleop.get_command()

                # 2. Update Virtual Target 
                # Because the loop is throttled to ~30Hz, we integrate using 1/30 seconds!
                webcam_dt = 1.0 / 30.0
                target_se3.translation += v_des[:3] * webcam_dt

                # 3. Target Leashing (Safety Limiter)
                current_physical_pose = robot.forward_kinematics(q_current)
                error_vector = target_se3.translation - current_physical_pose.translation
                leash_length = 0.15 # Max 15cm stretch
                
                if np.linalg.norm(error_vector) > leash_length:
                    target_se3.translation = current_physical_pose.translation + (error_vector / np.linalg.norm(error_vector)) * leash_length

                # 4. Update the MoCap Ghost Sphere in MuJoCo
                d.mocap_pos[mocap_id] = target_se3.translation.copy()

                # 5. INNER PHYSICS ACCELERATOR (The Fix!)
                # Since the webcam delayed us by 33ms, we must rapidly step the 500Hz 
                # physics engine ~16 times to catch it back up to real-time!
                for _ in range(16):
                    # Compute IK
                    dq, err, done = ik.compute_velocity(q_current, target_se3, q_posture=q_home_pin)
                    dq = np.clip(dq, -2.0, 2.0)
                    
                    for idx in ACTIVE_JOINTS:
                        q_target[idx] += dq[idx] * m.opt.timestep
                        d.ctrl[idx] = q_target[idx]

                    # Command Gripper
                    gripper_cmd = 0 if gripper_closed else 255
                    grasp_sys.command(gripper_cmd)

                    # Step Physics
                    d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]
                    mujoco.mj_step(m, d)
                    
                    # Update q_current for the next micro-step IK calculation!
                    q_current = d.qpos[:robot.model.nq].copy()

                viewer.sync()
                    
    finally:
        print("\n🛑 Simulation Closed. Releasing Webcam...")
        teleop.cleanup()

if __name__ == "__main__":
    main()