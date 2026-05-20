import numpy as np
import pinocchio as pin


class IKController:
    def __init__(self, robot, arm_dof=7, kp=5.0, damp=1e-3):
        self.robot = robot
        self.arm_dof = arm_dof
        self.kp = kp
        self.damp = damp

    def compute_velocity(self, q_current, target_se3):
        # FK
        current_pose = self.robot.forward_kinematics(q_current)

        # error
        dMf = current_pose.inverse() * target_se3
        error = pin.log6(dMf).vector

        v_des = error * self.kp

        # Jacobian
        J = self.robot.jacobian(q_current)
        J_arm = J[:, :self.arm_dof]

        # damped least squares
        dq_arm = np.linalg.lstsq(
            J_arm.T @ J_arm + self.damp * np.eye(self.arm_dof),
            J_arm.T @ v_des,
            rcond=None,
        )[0]

        dq_full = np.zeros(self.robot.model.nv)
        dq_full[:self.arm_dof] = dq_arm
        return dq_full
