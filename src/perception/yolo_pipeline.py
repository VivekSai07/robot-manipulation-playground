import time
import numpy as np
import cv2
import mujoco

# Import the YOLO Deep Learning model
from ultralytics import YOLO

class YOLOPipeline:
    """
    Mark-9 Deep Learning Perception Engine
    Uses YOLOv8 to semantically detect objects, draws bounding boxes, 
    and de-projects the center of the bounding box into 3D space.
    """
    def __init__(self, model, data, camera_name="workspace_cam", width=640, height=480):
        self.m = model
        self.d = data
        self.cam_name = camera_name
        self.width = width
        self.height = height
        
        print("🧠 Loading YOLOv8 Nano Neural Network... (This may download the model on first run)")
        # Load the pre-trained nano model. It is ultra-fast and perfect for CPU inference.
        self.yolo = YOLO('yolov8n.pt')
        
        # Initialize specialized Renderers
        self.renderer_rgb = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth.enable_depth_rendering()

        # State storage for the live feed
        self.last_rgb = None
        self.last_detection = None
        self.last_render_time = 0.0
        self.render_fps = 30.0

    def get_images(self):
        """Renders RGB and Depth maps."""
        self.renderer_rgb.update_scene(self.d, camera=self.cam_name)
        rgb = self.renderer_rgb.render()
        
        self.renderer_depth.update_scene(self.d, camera=self.cam_name)
        depth = self.renderer_depth.render()
        
        return rgb, depth

    def find_object(self, target_classes, z_offset=-0.02):
        """
        Passes the image through YOLO, finds the highest confidence detection for any of the 'target_classes',
        and calculates its 3D world coordinate.
        """
        rgb, depth = self.get_images()
        self.last_rgb = rgb.copy()
        self.last_detection = None
        
        # 1. AI Inference
        # verbose=False prevents YOLO from spamming the terminal every frame
        results = self.yolo(rgb, verbose=False)[0]
        
        best_conf = 0.0
        best_box = None
        detected_class_name = None
        
        # 2. Parse Neural Network Output
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detected_name = results.names[cls_id]
            
            # Find the target class with the highest confidence among the synonyms
            if detected_name in target_classes and conf > best_conf:
                best_conf = conf
                detected_class_name = detected_name
                # Extract the [x1, y1, x2, y2] bounding box coordinates
                best_box = box.xyxy[0].cpu().numpy()
                
        if best_box is None:
            return None
            
        # 3. Calculate Centroid of the Bounding Box
        x1, y1, x2, y2 = best_box
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Save state for the live visualizer
        self.last_detection = {
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "centroid": (u, v),
            "conf": best_conf,
            "name": detected_class_name
        }
        
        # 4. Sample Depth and Deproject to 3D Space
        z_dist = depth[v, u]
        world_pos = self.deproject(u, v, z_dist)
        
        # The depth camera hits the surface of the object facing the camera. 
        # Since our camera is at X=0.9 looking towards X=0.0, we push the 
        # coordinate slightly AWAY from the camera (-X) to reach the center of mass.
        world_pos[0] -= 0.06  # Push 6cm back into the object depth
        
        # We apply the Z offset to push the target coordinate up from the table surface
        world_pos[2] += z_offset
        
        return world_pos

    def deproject(self, u, v, z_dist):
        """Projects 2D pixel coordinates back into 3D World space."""
        cam_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, self.cam_name)
        fovy = self.m.cam_fovy[cam_id]
        f = 0.5 * self.height / np.tan(np.deg2rad(fovy) / 2)
        cx, cy = self.width / 2.0, self.height / 2.0

        # Construct local ray vector
        x_c = (u - cx) * z_dist / f
        y_c = (cy - v) * z_dist / f 
        z_c = -z_dist 
        
        # Convert local camera coordinate to global world coordinate
        cam_pos = self.d.cam_xpos[cam_id]
        cam_rot = self.d.cam_xmat[cam_id].reshape(3, 3)
        return cam_pos + cam_rot @ np.array([x_c, y_c, z_c])

    def show_live_feed(self):
        """Displays the RGB image with AI bounding boxes layered on top."""
        current_time = time.time()
        if current_time - self.last_render_time < (1.0 / self.render_fps):
            return
            
        self.last_render_time = current_time

        if self.last_rgb is not None:
            bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw the bounding box and confidence score if an object is tracked
            if self.last_detection:
                x1, y1, x2, y2 = self.last_detection["box"]
                u, v = self.last_detection["centroid"]
                conf = self.last_detection["conf"]
                name = self.last_detection["name"]
                
                # Green Bounding Box
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Red Crosshair at Centroid
                cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                # Text Label
                cv2.putText(bgr, f"{name} {conf*100:.1f}%", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            cv2.imshow("Robot AI Vision (YOLOv8)", bgr)
            cv2.waitKey(1)