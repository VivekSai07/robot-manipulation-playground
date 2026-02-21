import numpy as np
import pinocchio as pin

try:
    from pinocchio.robot_wrapper import RobotWrapper
except ImportError:
    RobotWrapper = pin.RobotWrapper

from .config import URDF_PATH, PACKAGE_DIR, EE_FRAME

class ViperX:
    """
    Standardized Pinocchio-compatible Robot Wrapper for ViperX.
    Mirrors the exact API of the FrankaPanda class.
    """
    def __init__(self):
        # Build Pinocchio model from URDF, resolving mesh paths dynamically
        self.robot = RobotWrapper.BuildFromURDF(
            URDF_PATH,
            package_dirs=[PACKAGE_DIR]
        )

        self.model = self.robot.model
        self.data = self.robot.data
        
        # Look up the ID for the end-effector frame from the URDF
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)
        
        # SAFEGUARD: Prevent silent C++ segfaults!
        # If the frame isn't found, getFrameId returns model.nframes. 
        # This will raise a clear Python error instead of crashing silently.
        if self.ee_frame_id >= self.model.nframes:
            raise ValueError(f"CRITICAL ERROR: EE_FRAME '{EE_FRAME}' not found in your URDF!")

    def forward_kinematics(self, q):
        """Computes the SE3 pose of the end-effector."""
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id]

    def jacobian(self, q):
        """Computes the 6xN geometric Jacobian in the LOCAL frame."""
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL
        )