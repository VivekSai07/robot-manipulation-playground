import numpy as np
import pinocchio as pin

try:
    from pinocchio.robot_wrapper import RobotWrapper
except ImportError:
    RobotWrapper = pin.RobotWrapper

from .config import URDF_PATH, PACKAGE_DIR, EE_FRAME


class FrankaPanda:
    def __init__(self):
        self.robot = RobotWrapper.BuildFromURDF(
            URDF_PATH,
            package_dirs=[PACKAGE_DIR]
        )

        self.model = self.robot.model
        self.data = self.robot.data
        self.ee_frame_id = self.model.getFrameId(EE_FRAME)

    def forward_kinematics(self, q):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id]

    def jacobian(self, q):
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL
        )
