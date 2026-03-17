import os
# CRITICAL OMP FIX: Prevent DLL initialization crashes on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import numpy as np
import cv2
import mujoco
import torch
from PIL import Image

# =========================================================================
# CRITICAL HACK: Robustly Mock `flash_attn` to bypass Hugging Face's AST 
# and importlib checks on Windows where compiling flash_attn is nearly impossible.
# We create a dummy module type with a valid __spec__ so is_flash_attn_2_available() 
# gracefully returns False instead of crashing.
# =========================================================================
import types
mock_flash_attn = types.ModuleType('flash_attn')
# Give it a dummy spec so importlib.util.find_spec doesn't crash
import importlib.machinery
mock_flash_attn.__spec__ = importlib.machinery.ModuleSpec('flash_attn', None)
sys.modules['flash_attn'] = mock_flash_attn
# =========================================================================

from transformers import AutoProcessor, AutoModelForCausalLM

class Florence2Pipeline:
    """
    Mark-12 Vision-Language Model (VLM) Perception Engine.
    Uses Microsoft's Florence-2 Foundation Model for Open-Vocabulary Grounding.
    """
    def __init__(self, model, data, camera_name="workspace_cam", width=640, height=480):
        self.m = model
        self.d = data
        self.cam_name = camera_name
        self.width = width
        self.height = height
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Loading Microsoft Florence-2 Foundation Model on [{self.device.upper()}]...")
        print("   (This will download ~1.5GB of weights on the first run)")
        
        # Florence-2 requires trust_remote_code=True because of its custom unified architecture
        model_id = "microsoft/Florence-2-base"
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.vlm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(self.device)
        self.vlm.eval()
        
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

    def find_object(self, text_prompt, z_offset=-0.025):
        """
        Passes the image and text prompt through Florence-2 using Phrase Grounding.
        """
        rgb, depth = self.get_images()
        self.last_rgb = rgb.copy()
        self.last_detection = None
        
        # --- CRITICAL FIX 1: NLP Filler Stripping ---
        # Florence-2 performs precise Phrase Grounding, not conversational chat.
        # We strip out polite filler words so the model only processes the noun phrase.
        clean_prompt = text_prompt.lower()
        fillers = [
            "could you please pick up the ", "can you please pick up the ",
            "could you pick up the ", "can you pick up the ",
            "please pick up the ", "pick up the ",
            "could you please grab the ", "can you please grab the ",
            "could you grab the ", "can you grab the ",
            "please grab the ", "grab the ",
            "where is the ", "find the "
        ]
        for filler in fillers:
            if clean_prompt.startswith(filler):
                clean_prompt = clean_prompt.replace(filler, "")
        clean_prompt = clean_prompt.strip()
        
        # Florence-2 uses specific Task Prompts. We want Phrase Grounding.
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = task_prompt + clean_prompt
        
        # Florence-2 Processor expects PIL Images
        pil_image = Image.fromarray(rgb)
        
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
        
        # 1. Run Neural Network Text Generation
        with torch.no_grad():
            generated_ids = self.vlm.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3, # Beam search for better reasoning
                do_sample=False
            )
            
        # 2. Decode the output text and parse the bounding box coordinates
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(self.width, self.height)
        )
        
        # Extract results
        results = parsed_answer.get(task_prompt, {})
        bboxes = results.get("bboxes", [])
        
        if len(bboxes) == 0:
            return None
            
        # Florence-2 doesn't output traditional "confidence scores", it just returns 
        # the boxes it thinks match the text, sorted by relevance. We take the best one.
        best_box = bboxes[0]
        
        x1, y1, x2, y2 = best_box
        u = int((x1 + x2) / 2)
        v = int((y1 + y2) / 2)
        
        # Save state for the live visualizer
        self.last_detection = {
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "centroid": (u, v),
            "prompt": clean_prompt
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
            bgr = cv2.cvtColor(self.last_rgb, cv2.COLOR_RGB2BGR)
            
            if self.last_detection:
                x1, y1, x2, y2 = self.last_detection["box"]
                u, v = self.last_detection["centroid"]
                prompt = self.last_detection["prompt"]
                
                # Cyan Bounding Box for Florence-2
                cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.drawMarker(bgr, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(bgr, f"{prompt} (Florence-2)", (x1, max(20, y1 - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
            cv2.imshow("Cognitive Vision (Florence-2)", bgr)
            cv2.setWindowProperty("Cognitive Vision (Florence-2)", cv2.WND_PROP_TOPMOST, 1)
            cv2.waitKey(1)