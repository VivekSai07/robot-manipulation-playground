import numpy as np
import pinocchio as pin

class IKController:
    """
    Mark-2 Universal Robust Differential IK
    
    Features:
    - Robot agnostic (works for 4-DOF to 7-DOF+ arms)
    - Active Joint Masking (ignores grippers dynamically)
    - Nullspace Projection (posture resting / redundancy resolution for 7+ DOF)
    - Task-Space Weighting (allows ignoring impossible axes for 4/5 DOF arms)
    - Damped Least Squares (singularity robust)
    """

    def __init__(
        self,
        robot,
        active_joint_indices,
        kp_pos=5.0,
        kp_rot=3.0,
        damping=1e-3,
        max_joint_vel=2.0,
        nullspace_kp=1.0,
        task_weights=None
    ):
        """
        :param active_joint_indices: List of joint indices the IK is allowed to move.
                                     E.g., for Franka: [0, 1, 2, 3, 4, 5, 6].
        :param task_weights: 6D array [x, y, z, roll, pitch, yaw] mapping to 1 or 0.
                             E.g., for a 5-DOF ViperX: [1, 1, 1, 1, 1, 0] (ignore Yaw).
        """
        self.robot = robot
        self.active_idx = np.array(active_joint_indices)
        self.nv_active = len(self.active_idx)
        
        self.kp_pos = kp_pos
        self.kp_rot = kp_rot
        self.damping = damping
        self.max_joint_vel = max_joint_vel
        self.nullspace_kp = nullspace_kp
        
        # Default to full 6D control if not specified
        if task_weights is None:
            self.task_weights = np.ones(6)
        else:
            self.task_weights = np.array(task_weights)

    def compute_velocity(self, q_current, target_se3, q_posture=None, tol=1e-3):
        """
        Computes the target joint velocities.
        
        :param q_posture: Optional preferred joint configuration to rest in (Nullspace).
        """
        # 1. Forward Kinematics
        current_pose = self.robot.forward_kinematics(q_current)

        # 2. 6D Error in LOCAL frame
        dMf = current_pose.inverse() * target_se3
        error6 = pin.log6(dMf).vector

        # Split and scale by Kp
        err_pos = error6[:3]
        err_rot = error6[3:]
        
        error_norm = np.linalg.norm(err_pos)
        converged = error_norm < tol

        v_des = np.zeros(6)
        v_des[:3] = self.kp_pos * err_pos
        v_des[3:] = self.kp_rot * err_rot
        
        # Apply task weighting (Zeroes out error axes the robot shouldn't care about)
        v_des = v_des * self.task_weights

        # 3. Jacobian (Extract only the columns for the active arm joints)
        J_full = self.robot.jacobian(q_current)
        J_active = J_full[:, self.active_idx]
        
        # Apply task weighting to Jacobian rows
        J_task = J_active * self.task_weights[:, np.newaxis]

        # 4. Damped Least Squares (DLS)
        H = J_task.T @ J_task + self.damping * np.eye(self.nv_active)
        g = J_task.T @ v_des
        dq_active = np.linalg.solve(H, g)

        # 5. Nullspace Posture Control (Redundancy Resolution)
        # If the robot has more DOF than constrained task axes, use the extra freedom to 
        # move towards the preferred 'q_posture' without affecting the end-effector.
        if q_posture is not None:
            # Pseudo-inverse of the weighted Jacobian
            J_pinv = np.linalg.solve(H, J_task.T)
            Null_projector = np.eye(self.nv_active) - J_pinv @ J_task
            
            # Posture error (only for active joints)
            q_err = q_posture[self.active_idx] - q_current[self.active_idx]
            dq_null = self.nullspace_kp * q_err
            
            # Project posture velocity into the nullspace of the main task
            dq_active += Null_projector @ dq_null

        # 6. Velocity Clipping (Safety)
        dq_active = np.clip(dq_active, -self.max_joint_vel, self.max_joint_vel)

        # 7. Map back to full velocity vector (leaving grippers/unactuated joints at 0)
        dq_full = np.zeros(self.robot.model.nv)
        dq_full[self.active_idx] = dq_active

        return dq_full, error_norm, converged