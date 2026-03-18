import numpy as np
import pinocchio as pin

class APFController:
    """
    Mark-13 Artificial Potential Field (APF) Controller.
    A real-time local planner that generates collision-avoidance velocities.
    
    - Attractive Force: Pulls the End-Effector towards the target pose.
    - Repulsive Force: Pushes the End-Effector away from dynamic obstacles.
    """
    def __init__(self, k_att_pos=2.0, k_att_rot=2.0, k_rep_pos=0.08, influence_radius=0.25, max_v=1.5):
        """
        :param k_att_pos: Attractive gain for position.
        :param k_att_rot: Attractive gain for rotation.
        :param k_rep_pos: Repulsive gain (how violently it dodges).
        :param influence_radius: How close an obstacle must be (in meters) before the robot reacts.
        :param max_v: Maximum allowed Cartesian velocity to prevent explosive math.
        """
        self.k_att_pos = k_att_pos
        self.k_att_rot = k_att_rot
        self.k_rep_pos = k_rep_pos
        self.influence_radius = influence_radius
        self.max_v = max_v

    def compute_velocity(self, current_se3, target_se3, obstacle_positions, robot_links=None, nv=None):
        """
        Calculates the safe, repelled 6D velocity vector for the end-effector AND joint-space null posture.
        
        :param current_se3: Pinocchio SE3 of the current end-effector pose.
        :param target_se3: Pinocchio SE3 of the goal pose.
        :param obstacle_positions: List of 3D numpy arrays representing threat coordinates.
        :param robot_links: Optional List of tuples: (3D position, 6xNV Jacobian matrix) for intermediate links.
        :param nv: Number of robot joint velocities.
        :return: (v_des, dq_rep_null) - 6D desired Cartesian velocity AND nv-dimensional null-space repulsion vector.
        """
        # ==========================================
        # 1. ATTRACTIVE VELOCITY (The Goal)
        # ==========================================
        dMf = current_se3.inverse() * target_se3
        error6 = pin.log6(dMf).vector
        
        v_att = np.zeros(6)
        v_att[:3] = self.k_att_pos * error6[:3]
        v_att[3:] = self.k_att_rot * error6[3:]
        
        # ==========================================
        # 2. CARTESIAN REPULSIVE VELOCITY (The EE Shield)
        # ==========================================
        v_rep_world = np.zeros(6)
        current_pos = current_se3.translation
        
        for obs_pos in obstacle_positions:
            diff_xy = np.array([current_pos[0] - obs_pos[0], current_pos[1] - obs_pos[1], 0.0])
            dist_xy = np.linalg.norm(diff_xy)
            
            if dist_xy < 0.01:
                diff_xy = np.array([current_pos[0], current_pos[1], 0.0])
                dist_xy = np.linalg.norm(diff_xy)
                if dist_xy < 0.01: 
                    diff_xy = np.array([1.0, 0.0, 0.0])
                    dist_xy = 1.0
            
            if dist_xy < self.influence_radius:
                rep_magnitude = self.k_rep_pos * (1.0 / dist_xy - 1.0 / self.influence_radius) * (1.0 / (dist_xy ** 2))
                unit_dir = diff_xy / dist_xy
                v_rep_world[:3] += rep_magnitude * unit_dir
                
        # Transform World-Frame Repulsion into Local-Frame Velocity
        v_rep_local = np.zeros(6)
        R_world_to_local = current_se3.rotation.T
        v_rep_local[:3] = R_world_to_local @ v_rep_world[:3]

        v_des = v_att + v_rep_local
        
        v_linear_mag = np.linalg.norm(v_des[:3])
        if v_linear_mag > self.max_v:
            v_des[:3] = v_des[:3] * (self.max_v / v_linear_mag)

        # ==========================================
        # 3. JOINT-SPACE REPULSIVE VELOCITY (Whole-Body Shield)
        # ==========================================
        dq_rep_null = None
        if nv is not None:
            dq_rep_null = np.zeros(nv)
            if robot_links is not None:
                for obs_pos in obstacle_positions:
                    for link_pos, J_link in robot_links:
                        diff_xy = np.array([link_pos[0] - obs_pos[0], link_pos[1] - obs_pos[1], 0.0])
                        dist_xy = np.linalg.norm(diff_xy)
                        
                        if dist_xy < 0.01:
                            diff_xy = np.array([link_pos[0], link_pos[1], 0.0])
                            dist_xy = np.linalg.norm(diff_xy)
                            if dist_xy < 0.01:
                                diff_xy = np.array([1.0, 0.0, 0.0])
                                dist_xy = 1.0
                                
                        if dist_xy < self.influence_radius:
                            rep_magnitude = self.k_rep_pos * (1.0 / dist_xy - 1.0 / self.influence_radius) * (1.0 / (dist_xy ** 2))
                            unit_dir = diff_xy / dist_xy
                            
                            F_rep = np.zeros(6)
                            F_rep[:3] = rep_magnitude * unit_dir
                            
                            # J^T * F projects the 3D collision force directly into joint torques/velocities!
                            # We multiply by 2.0 to ensure the elbow ducking is aggressive enough against the IK target.
                            dq_rep_null += 2.0 * J_link.T @ F_rep

        return v_des, dq_rep_null