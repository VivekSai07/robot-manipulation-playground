import os
# CRITICAL OMP FIX: Must be set BEFORE importing torch or cv2 to prevent DLL initialization crashes on Windows!
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import numpy as np
import cv2
import mujoco
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class VLMPipeline:
    """
    Mark-11 Vision-Language Model (VLM) Perception Engine.
    Uses Hugging Face's OWL-ViT to perform zero-shot, open-vocabulary object detection.
    Translates free-form text descriptions into 3D world coordinates.
    """
    def __init__(self, model, data, camera_name="workspace_cam", width=640, height=480):
        self.m = model
        self.d = data
        self.cam_name = camera_name
        self.width = width
        self.height = height
        
        # Hardware Acceleration (CUDA if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading OWL-ViT Foundation Model on [{self.device.upper()}]...")
        print("   (This will download ~600MB of weights on the first run)")
        
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.vlm = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.vlm.eval() # Set model to evaluation mode
        
        # Initialize Renderers
        self.renderer_rgb = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth = mujoco.Renderer(self.m, self.height, self.width)
        self.renderer_depth.enable_depth_rendering()

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

    # CRITICAL FIX: Z-offset tuned to -0.025 (2.5cm) to perfectly target the center of mass 
    # of the 5cm diameter spheres and 5cm tall boxes in our simulation!
    def find_object(self, text_prompt, z_offset=-0.025, conf_threshold=0.08):
        """
        Passes the image and text prompt through the Vision-Language Model.
        Calculates the 3D world coordinate of the best matching object.
        """
        rgb, depth = self.get_images()
        self.last_rgb = rgb.copy()
        self.last_detection = None
        
        # Format the prompt for OWL-ViT. It works best when describing a photo.
        formatted_prompt = f"a photo of a {text_prompt}"
        
        # Process inputs for the Transformer
        inputs = self.processor(text=[[formatted_prompt]], images=rgb, return_tensors="pt").to(self.device)
        
        # 1. Run Neural Network Inference
        with torch.no_grad(): # Disable gradients for faster inference
            outputs = self.vlm(**inputs)
            
        # Target image size for bounding box scaling [Height, Width]
        target_sizes = torch.tensor([rgb.shape[:2]]).to(self.device)
        
        # The method was renamed to support zero-shot text grounding.
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes
        )[0]
        
        scores = results["scores"]
        boxes = results["boxes"]
        
        if len(scores) == 0:
            return None
            
        # 2. Extract the highest confidence match
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        
        # Manually enforce our confidence threshold since the newer API handles arguments slightly differently
        if best_score < conf_threshold:
            return None
            
        best_box = boxes[best_idx].detach().cpu().numpy()
        
        x1, y1, x2, y2 = best_box
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Save state for the live visualizer
        self.last_detection = {
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "centroid": (u, v),
            "conf": best_score,
            "prompt": text_prompt
        }
        
        # 3. Sample Depth and Deproject to 3D Space
        z_dist = depth[v, u]
        world_pos = self.deproject(u, v, z_dist)
        
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

    def show_live_feed(self):
        """Displays the RGB image with VLM bounding boxes layered on top."""
        current_time = time.time()
        if current_time - self.last_render_time < (1.0 / self.render_fps):
            return
            
        self.last_render_time = current_time

        if self.last_rgb is not None:
            # Convert MuJoCo RGB to OpenCV BGR
            bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            
            if self.last_detection:
                x1, y1, x2, y2 = self.last_detection["box"]
                u, v = self.last_detection["centroid"]
                conf = self.last_detection["conf"]
                prompt = self.last_detection["prompt"]
                
                # Purple Bounding Box for VLM
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.drawMarker(bgr, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(bgr, f"{prompt}: {conf*100:.1f}%", (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
            cv2.imshow("Cognitive Vision (OWL-ViT)", bgr)
            cv2.waitKey(1)