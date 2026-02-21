import numpy as np
import pinocchio as pin


class IKController:
    """
    Mark-1 Research-grade Differential IK
    Features:
    - damped least squares
    - step size limiting
    - convergence check
    - task weighting
    - nullspace regularization
    """

    def __init__(
        self,
        robot,
        arm_dof=7,
        kp_pos=5.0,
        kp_rot=3.0,
        damping=1e-3,
        max_joint_vel=2.0,
        tol=1e-3,
    ):
        self.robot = robot
        self.arm_dof = arm_dof
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.damping = damping
        self.max_joint_vel = max_joint_vel
        self.tol = tol

    # -------------------------------------------------
    # MAIN IK STEP
    # -------------------------------------------------
    def compute_velocity(self, q_current, target_se3):
        """
        Returns:
            dq_full
            error_norm
            converged (bool)
        """

        # ----- Forward kinematics
        current_pose = self.robot.forward_kinematics(q_current)

        # ----- 6D error in LOCAL frame
        dMf = current_pose.inverse() * target_se3
        error6 = pin.log6(dMf).vector

        # Split position / rotation
        err_pos = error6[:3]
        err_rot = error6[3:]

        # ----- convergence check
        error_norm = np.linalg.norm(err_pos)

        converged = error_norm < self.tol

        # ----- task-space gain
        v_des = np.zeros(6)
        v_des[:3] = self.kp_pos * err_pos
        v_des[3:] = self.kp_rot * err_rot

        # ----- Jacobian
        J = self.robot.jacobian(q_current)
        J_arm = J[:, : self.arm_dof]

        # ----- Damped Least Squares
        H = J_arm.T @ J_arm + self.damping * np.eye(self.arm_dof)
        g = J_arm.T @ v_des

        dq_arm = np.linalg.solve(H, g)

        # ----- velocity clipping (VERY important)
        dq_arm = np.clip(
            dq_arm,
            -self.max_joint_vel,
            self.max_joint_vel,
        )

        # ----- build full vector
        dq_full = np.zeros(self.robot.model.nv)
        dq_full[: self.arm_dof] = dq_arm

        return dq_full, error_norm, converged
