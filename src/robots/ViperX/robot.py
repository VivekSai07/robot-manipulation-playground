import numpy as np
import pinocchio as pin
import mujoco

class ViperX:
    """
    MuJoCo-native Pinocchio-compatible Robot Wrapper.
    Extracts Kinematics and Jacobians directly from MuJoCo,
    bypassing the need for an external URDF completely!
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # The 'pinch' site is exactly between the two fingers.
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pinch")

    def forward_kinematics(self, q):
        # Sync MuJoCo state
        self.data.qpos[:len(q)] = q
        mujoco.mj_kinematics(self.model, self.data)
        
        # Fetch position and rotation from the 'pinch' site
        pos = self.data.site_xpos[self.ee_site_id]
        rot = self.data.site_xmat[self.ee_site_id].reshape(3, 3)
        return pin.SE3(rot, pos.copy())

    def jacobian(self, q):
        # Sync MuJoCo state and update Center of Mass (required for jacobian calculation)
        self.data.qpos[:len(q)] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # Compute MuJoCo Jacobian (World Frame)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        
        # Rotate Jacobian to LOCAL Frame to match Pinocchio / IKController_m2 expectations
        R_T = self.data.site_xmat[self.ee_site_id].reshape(3, 3).T
        J_local = np.zeros((6, self.model.nv))
        J_local[:3, :] = R_T @ jacp
        J_local[3:, :] = R_T @ jacr
        
        return J_local