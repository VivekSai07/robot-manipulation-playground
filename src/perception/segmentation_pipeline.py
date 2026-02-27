import time
import numpy as np
import cv2
import mujoco

class SegmentationAnalyzer:
    """
    Mark-7 "Cheat Code" Perception Engine
    Uses MuJoCo's native segmentation masks to identify objects with 100% accuracy.
    Bypasses HSV color math entirely.
    """
    def __init__(self, model, data, camera_name="workspace_cam", width=640, height=480):
        self.m = model
        self.d = data
        self.cam_name = camera_name
        self.width = width
        self.height = height
        
        self.last_u = None
        self.last_v = None
        
        # Initialize specialized Renderers
        self.renderer_rgb = mujoco.Renderer(self.m, self.height, self.width)
        
        self.renderer_seg = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_seg.enable_segmentation_rendering()
        
        self.renderer_depth = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth.enable_depth_rendering()

    def get_data(self):
        """Renders RGB, Depth, and Segmentation maps."""
        self.renderer_rgb.update_scene(self.d, camera=self.cam_name)
        rgb = self.renderer_rgb.render()
        
        self.renderer_depth.update_scene(self.d, camera=self.cam_name)
        depth = self.renderer_depth.render()
        
        self.renderer_seg.update_scene(self.d, camera=self.cam_name)
        seg = self.renderer_seg.render()
        
        return rgb, depth, seg

    def find_object_by_name(self, body_name, z_offset=-0.02):
        """
        Locates a body by name using segmentation masks.
        """
        rgb, depth, seg = self.get_data()
        
        # 1. Look up the ID for the body
        body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            print(f"❌ Seg Error: Body '{body_name}' not found in model.")
            return None
            
        # Segmentation renders Geoms, not Bodies. Find all geoms belonging to this body.
        geom_ids = [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] == body_id]
        
        # 2. Find all pixels where the Segmentation ID matches one of our geoms
        mask = np.zeros((self.height, self.width), dtype=bool)
        
        # In MuJoCo's segmentation buffer:
        # Channel 0: Object ID
        # Channel 1: Object Type (mjOBJ_GEOM is usually 5)
        geom_type = mujoco.mjtObj.mjOBJ_GEOM
        
        for g_id in geom_ids:
            # We strictly enforce that the pixel matches both our target ID and is a geometry
            mask |= (seg[:, :, 0] == g_id) & (seg[:, :, 1] == geom_type)
        
        if not np.any(mask):
            self.last_u, self.last_v = None, None
            return None
            
        # 3. Calculate Centroid of the mask (Mean of indices)
        coords = np.argwhere(mask)
        v, u = coords.mean(axis=0).astype(int)
        
        self.last_u, self.last_v = u, v
        
        # 4. Sample Depth and Deproject
        z_dist = depth[v, u]
        world_pos = self.deproject(u, v, z_dist)
        
        # Add offset to reach the center of the block instead of its top surface
        world_pos[2] += z_offset
        
        return world_pos

    def deproject(self, u, v, z_dist):
        """Projects 2D pixel coordinates back into 3D World space."""
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * self.height / np.tan(np.deg2rad(fovy) / 2)
        cx, cy = self.width / 2.0, self.height / 2.0

        x_c = (u - cx) * z_dist / f
        y_c = (cy - v) * z_dist / f 
        z_c = -z_dist 
        
        cam_pos = self.d.cam_xpos[cam_id]
        cam_rot = self.d.cam_xmat[cam_id].reshape(3, 3)
        return cam_pos + cam_rot @ np.array([x_c, y_c, z_c])

    def show_debug_view(self):
        """Displays the perfect segmentation mask for visual verification."""
        rgb, _, seg = self.get_data()
        
        # Channel 0 contains the unique ID. We scale it up to create distinct colors.
        seg_img = seg[:, :, 0].astype(np.uint8) * 45 
        seg_bgr = cv2.applyColorMap(seg_img, cv2.COLORMAP_JET)
        
        if self.last_u is not None:
            cv2.drawMarker(seg_bgr, (self.last_u, self.last_v), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
            
        cv2.imshow("Robot Segmentation View (Truth)", seg_bgr)
        cv2.waitKey(1)