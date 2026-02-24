import numpy as np
import pinocchio as pin

class TaskSpaceTrajectory:
    """
    Mark-3 Task Space Trajectory Generator
    Generates time-scaled paths between two SE3 poses (Cartesian Interpolation).
    """
    def __init__(self, start_pose, end_pose, duration, method="cubic"):
        self.start_pose = start_pose
        self.end_pose = end_pose
        self.duration = max(duration, 1e-5)
        self.method = method

    def get_pose(self, t):
        t_clamped = np.clip(t, 0.0, self.duration)
        alpha = t_clamped / self.duration
        
        if self.method == "cubic":
            s = 3 * (alpha ** 2) - 2 * (alpha ** 3)
        elif self.method == "linear":
            s = alpha
        else:
            s = alpha
            
        return pin.SE3.Interpolate(self.start_pose, self.end_pose, s)

class JointSpaceTrajectory:
    """
    Mark-5 Joint Space Trajectory Generator
    Generates time-scaled paths between two joint arrays (Joint Interpolation).
    Crucial for smoothly following RRT paths without wild wrist rotations!
    """
    def __init__(self, q_start, q_end, duration, method="cubic"):
        self.q_start = np.array(q_start)
        self.q_end = np.array(q_end)
        self.duration = max(duration, 1e-5)
        self.method = method

    def get_position(self, t):
        t_clamped = np.clip(t, 0.0, self.duration)
        alpha = t_clamped / self.duration
        
        if self.method == "cubic":
            s = 3 * (alpha ** 2) - 2 * (alpha ** 3)
        elif self.method == "linear":
            s = alpha
        else:
            s = alpha
            
        return self.q_start + s * (self.q_end - self.q_start)