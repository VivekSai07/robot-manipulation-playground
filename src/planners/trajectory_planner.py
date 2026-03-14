import numpy as np
import pinocchio as pin

class TaskSpaceTrajectory:
    """
    Mark-3 Task Space Trajectory Generator
    Generates time-scaled paths between two SE3 poses.
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
            
        # CRITICAL FIX: Decoupled Interpolation!
        # Instead of standard SE3 interpolation (which generates a screw/arc motion),
        # we force the translation to follow a strict Euclidean straight line,
        # while letting the Pinocchio handle the rotational SLERP safely.
        se3_interp = pin.SE3.Interpolate(self.start_pose, self.end_pose, s)
        straight_line_trans = self.start_pose.translation + s * (self.end_pose.translation - self.start_pose.translation)
        
        return pin.SE3(se3_interp.rotation, straight_line_trans)

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