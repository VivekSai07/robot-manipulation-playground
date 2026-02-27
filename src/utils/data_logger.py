import os
import time
import numpy as np
import pinocchio as pin

class DataLogger:
    """
    Mark-6 Data Logger for Imitation Learning / Behavior Cloning.
    Captures State and Action pairs at every physics step and packages them into .npz files.
    """
    def __init__(self, log_dir="data/teleop_episodes"):
        self.log_dir = log_dir
        # Create the data directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        self.reset()

    def reset(self):
        """Clears the internal buffers for a new recording episode."""
        self.observations = {
            "qpos": [],
            "qvel": [],
            "ee_pose_pos": [],
            "ee_pose_quat": [],
            "cube_pos": []
        }
        self.actions = []

    def log_step(self, qpos, qvel, ee_pose, cube_pos, action):
        """Records a single frame of State-Action data."""
        self.observations["qpos"].append(qpos.copy())
        self.observations["qvel"].append(qvel.copy())
        
        # Cartesian EE State
        self.observations["ee_pose_pos"].append(ee_pose.translation.copy())
        # Convert Pinocchio rotation matrix to Quaternion [x, y, z, w] for ML training
        quat = pin.Quaternion(ee_pose.rotation).coeffs()
        self.observations["ee_pose_quat"].append(quat.copy())
        
        # Object State
        self.observations["cube_pos"].append(cube_pos.copy())
        
        # Action (What the human commanded the motors to do)
        self.actions.append(action.copy())

    def save_episode(self):
        """Saves the recorded episode to disk in .npz format."""
        if len(self.actions) == 0:
            return None
            
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(self.log_dir, f"episode_{timestamp}.npz")
        
        # Convert all lists to fast NumPy arrays before saving
        np.savez(
            filename,
            qpos=np.array(self.observations["qpos"]),
            qvel=np.array(self.observations["qvel"]),
            ee_pose_pos=np.array(self.observations["ee_pose_pos"]),
            ee_pose_quat=np.array(self.observations["ee_pose_quat"]),
            cube_pos=np.array(self.observations["cube_pos"]),
            actions=np.array(self.actions)
        )
        
        print(f"\n💾 Episode successfully saved: {filename} ({len(self.actions)} steps recorded)")
        self.reset()
        return filename