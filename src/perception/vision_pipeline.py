import time
import numpy as np
import cv2
import mujoco

class VisionAnalyzer:
    """
    Mark-7 Perception Engine (NLP & Color Upgrade)
    Handles RGB-D rendering, computer vision, text parsing, and Sim2Real 3D projection.
    """
    
    # Mathematical definitions of human colors in HSV space.
    # RELAXED THRESHOLDS: Lowered S and V to 100 to ensure we catch cubes in shadows,
    # while still remaining above the floor's Value (~76-102).
    COLOR_RANGES = {
        "red": [
            (np.array([0, 100, 100]), np.array([10, 255, 255])),
            (np.array([160, 100, 100]), np.array([180, 255, 255]))
        ],
        "green": [
            (np.array([35, 100, 100]), np.array([85, 255, 255]))
        ],
        "blue": [
            (np.array([110, 100, 100]), np.array([140, 255, 255]))
        ]
    }

    def __init__(self, model, data, camera_name="workspace_cam", width=640, height=480):
        self.m = model
        self.d = data
        self.cam_name = camera_name
        self.width = width
        self.height = height
        
        self.last_u = None
        self.last_v = None
        self.last_render_time = 0.0
        self.render_fps = 30.0
        
        self.renderer_rgb = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth.enable_depth_rendering()

    def get_images(self):
        self.renderer_rgb.update_scene(self.d, camera=self.cam_name)
        rgb_image = self.renderer_rgb.render()
        
        self.renderer_depth.update_scene(self.d, camera=self.cam_name)
        depth_buffer = self.renderer_depth.render()
        
        return rgb_image, depth_buffer

    def parse_color_from_text(self, text):
        """Simple NLP parsing to extract color from a sentence."""
        text = text.lower()
        for color in self.COLOR_RANGES.keys():
            if color in text:
                return color
        return None

    def find_object_by_color(self, text_prompt, z_offset=-0.02):
        """
        Extracts color from the prompt, runs HSV masking, and de-projects it to 3D.
        """
        color_name = self.parse_color_from_text(text_prompt)
        
        if color_name is None:
            print(f"❌ Vision Error: Could not understand any known colors in prompt: '{text_prompt}'")
            return None
            
        print(f"🧠 NLP Parsed Target Color: [{color_name.upper()}]")
            
        rgb, depth = self.get_images()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # Build the mask dynamically based on the requested color
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        for (lower, upper) in self.COLOR_RANGES[color_name]:
            mask |= cv2.inRange(hsv, lower, upper)
        
        # Clean up noise (small shadows/reflections) using opening/closing
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self.last_u = None
            self.last_v = None
            return None
            
        largest_contour = max(contours, key=cv2.contourArea)
        
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            self.last_u = None
            self.last_v = None
            return None
            
        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        
        self.last_u = u
        self.last_v = v
        
        z_dist = depth[v, u]
        world_pos = self.deproject(u, v, z_dist)
        world_pos[2] += z_offset
        
        return world_pos

    def deproject(self, u, v, z_dist):
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        
        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * self.height / np.tan(np.deg2rad(fovy) / 2)
        cx = self.width / 2.0
        cy = self.height / 2.0

        x_c = (u - cx) * z_dist / f
        y_c = (cy - v) * z_dist / f 
        z_c = -z_dist 
        
        local_pos = np.array([x_c, y_c, z_c])
        
        cam_pos = self.d.cam_xpos[cam_id]
        cam_rot = self.d.cam_xmat[cam_id].reshape(3, 3)
        
        world_pos = cam_pos + cam_rot @ local_pos
        return world_pos

    def show_live_feed(self):
        current_time = time.time()
        if current_time - self.last_render_time < (1.0 / self.render_fps):
            return
            
        self.last_render_time = current_time

        rgb, _ = self.get_images()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        
        if self.last_u is not None and self.last_v is not None:
            cv2.drawMarker(bgr, (self.last_u, self.last_v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            
        cv2.imshow("Robot Live Camera Feed", bgr)
        cv2.waitKey(1)