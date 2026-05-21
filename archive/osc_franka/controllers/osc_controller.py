import numpy as np
import pinocchio as pin

class OSCController:
    """
    Mark-3 Operational Space Controller (OSC)
    
    Features:
    - Task-Space Impedance (PD) Control
    - Dynamically Consistent Nullspace Projection (for redundant arms like Franka)
    - Active Joint Masking
    - Singularity Robustness via Damping
    """

    def __init__(
        self,
        robot,
        active_joint_indices,
        kp_pos=400.0,
        kp_rot=100.0,
        kd_pos=None,       # Defaults to critically damped: 2*sqrt(kp)
        kd_rot=None,
        nullspace_kp=20.0,
        nullspace_kd=None,
        damping=1e-3,
        task_weights=None
    ):
        self.robot       = robot
        self.active_idx  = np.array(active_joint_indices)
        self.nv_active   = len(self.active_idx)
        
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot

        # Critical damping: 2*sqrt(Kp) gives zero overshoot
        self.kd_pos = kd_pos if kd_pos is not None else 2.0 * np.sqrt(kp_pos)
        self.kd_rot = kd_rot if kd_rot is not None else 2.0 * np.sqrt(kp_rot)
        
        self.nullspace_kp  = nullspace_kp
        self.nullspace_kd  = nullspace_kd if nullspace_kd is not None else 2.0 * np.sqrt(nullspace_kp)
        
        self.damping = damping
        
        self.task_weights = np.ones(6) if task_weights is None else np.array(task_weights)

    def compute_torque(self, q_current, dq_current, target_se3, q_posture=None):
        """
        Computes the target joint torques to reach the target pose.

        Returns
        -------
        tau_full   : np.ndarray shape (nv,)  — full joint torque vector
        error_norm : float                   — translational error [metres]
        """
        # 1. Update Kinematics & Dynamics
        pin.computeAllTerms(self.robot.model, self.robot.data, q_current, dq_current)
        current_pose = self.robot.data.oMf[self.robot.ee_frame_id]

        # 2. 6D Error in LOCAL frame via log map
        dMf    = current_pose.inverse() * target_se3
        error6 = pin.log6(dMf).vector
        err_pos = error6[:3]
        err_rot = error6[3:]
        error_norm = np.linalg.norm(err_pos)

        # 3. Jacobian (Active Joints Only)
        J_full   = self.robot.jacobian(q_current)
        J_active = J_full[:, self.active_idx]
        J_task   = J_active * self.task_weights[:, np.newaxis]

        # 4. Current End-Effector Velocity
        v_current = J_active @ dq_current[self.active_idx]

        # 5. Desired Task-Space Acceleration (PD Control Law)
        a_des       = np.zeros(6)
        a_des[:3]   = self.kp_pos * err_pos - self.kd_pos * v_current[:3]
        a_des[3:]   = self.kp_rot * err_rot - self.kd_rot * v_current[3:]
        a_des      *= self.task_weights

        # 6. Joint-Space Mass Matrix (M)
        # ── Pinocchio only fills the UPPER triangle of M. ──
        # We must:
        #   (a) copy the buffer (np.array) so we don't mutate Pinocchio's internals
        #   (b) mirror the upper triangle to the lower to get a true symmetric matrix
        #   (c) regularize with a small diagonal term before inversion
        M_full   = np.array(self.robot.data.M)               # (a) copy
        M_full   = np.triu(M_full) + np.triu(M_full, 1).T   # (b) symmetrize
        M_active = M_full[np.ix_(self.active_idx, self.active_idx)]
        M_reg    = M_active + self.damping * np.eye(self.nv_active)  # (c) regularize

        # 7. Operational Space Mass Matrix (Lambda)
        #    Lambda = (J M^{-1} J^T)^{-1}
        # Use solve() rather than inv() for numerical stability
        M_inv      = np.linalg.solve(M_reg, np.eye(self.nv_active))
        Lambda_inv = J_task @ M_inv @ J_task.T
        Lambda     = np.linalg.pinv(Lambda_inv + self.damping * np.eye(6))

        # 8. Compute Task-Space Force (F) and map to joint torques
        F_task    = Lambda @ a_des
        tau_active = J_task.T @ F_task

        # 9. Dynamically Consistent Nullspace (Redundancy Resolution)
        if q_posture is not None:
            # Dynamically consistent pseudo-inverse: J_bar = M^{-1} J^T Lambda
            J_bar = M_inv @ J_task.T @ Lambda

            # Nullspace projector: N^T = I - J^T J_bar^T
            N_T = np.eye(self.nv_active) - J_task.T @ J_bar.T

            # PD control towards resting posture in nullspace
            q_err  =  q_posture[self.active_idx] - q_current[self.active_idx]
            dq_err = -dq_current[self.active_idx]
            tau_null_pd = self.nullspace_kp * q_err + self.nullspace_kd * dq_err

            # Project so nullspace torques don't disturb the EE task
            tau_active += N_T @ tau_null_pd

        # 10. Map back to full joint torque vector
        tau_full = np.zeros(self.robot.model.nv)
        tau_full[self.active_idx] = tau_active

        return tau_full, error_norm