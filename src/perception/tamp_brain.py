import mujoco
import numpy as np
from src.perception.florence2_pipeline import Florence2Pipeline

class TAMPBrain:
    """
    Mark-14 Task and Motion Planning (TAMP) Sequencer.
    Acts as a high-level cognitive wrapper around the Florence-2 VLM.
    Scans a scene, matches objects to predefined logic, and builds a task queue.
    """
    def __init__(self, model, data):
        self.m = model
        self.d = data
        
        self.vision = Florence2Pipeline(self.m, self.d)
        
        # ---------------------------------------------------------
        # THE KNOWLEDGE BASE
        # - X-Offsets: Pushes the target backward (-X) into the true 
        #   center to compensate for the angled camera.
        # - Z-Offsets: Tuned to grab the bottle higher (-0.02) to avoid 
        #   palm collisions, and to cradle the apple from below (-0.035)
        #   to prevent it from sliding out during lateral movement.
        # ---------------------------------------------------------
        self.knowledge_base = {
            "red apple":                 {"body": "target_apple",      "bin": "food_bin",    "z_offset": -0.035, "x_offset": -0.015},
            "green cylinder bottle":     {"body": "target_bottle",     "bin": "recycle_bin", "z_offset": -0.02,  "x_offset": -0.015},
            # "dark gray metal box":       {"body": "target_metal_box",  "bin": "recycle_bin", "z_offset": -0.02,  "x_offset": -0.010},
            # "brown wood block":          {"body": "target_wood_block", "bin": "recycle_bin", "z_offset": -0.02,  "x_offset": -0.010}
        }

    def plan_cleanup(self):
        """
        Actively scans the scene using the VLM and builds an execution schedule.
        Returns a list of task dictionaries.
        """
        print("\n🧠 TAMP Brain: Formulating macroscopic cleanup plan...")
        task_queue = []
        
        for prompt, info in self.knowledge_base.items():
            print(f"   🔍 Inspecting scene for '{prompt}'...")
            
            # Query Florence-2
            estimated_pos = self.vision.find_object(prompt, z_offset=info["z_offset"])
            
            if estimated_pos is not None:
                print(f"   ✅ Found '{prompt}'! Added to sorting schedule.")
                
                # Apply the X-offset to compensate for the camera parallax!
                estimated_pos[0] += info.get("x_offset", 0.0)
                
                # Resolve the physical world coordinates of the destination bin
                bin_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, info["bin"])
                bin_pos = self.d.xpos[bin_id].copy()
                
                # Append a structured Action Task to the queue
                task_queue.append({
                    "target_body": info["body"],
                    "prompt": prompt,
                    "pick_pos": estimated_pos,
                    "place_pos": bin_pos,
                    "z_offset": info["z_offset"]
                })
            else:
                print(f"   ⏭️ '{prompt}' not found. Skipping.")
                
        print(f"\n📋 TAMP Brain: Schedule complete! {len(task_queue)} tasks queued for execution.\n")
        return task_queue
        
    def show_feed(self):
        """Pass-through to keep the OpenCV debug window alive during planning."""
        self.vision.show_live_feed()