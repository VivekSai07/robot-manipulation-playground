## Perception Pipelines

We explore two main approaches to machine perception: a realistic computer vision approach, and a simulated "ground truth" segmentation approach. Below are notes covering how each pipeline works and how they are integrated into our tasks.

### 1. Vision Perception (NLP + HSV Color Masking)
**Files:** `src/perception/vision_pipeline.py` & `src/tasks/pick_and_place_m7_vision.py`

**How it works:**
- **NLP Parsing:** The pipeline takes a natural language string containing the target color (e.g., `"pick up the green block"`), and parses it to a known target color.
- **Rendering:** Using MuJoCo's camera rendering capabilities, it pulls current RGB frame and Depth buffer images from `workspace_cam`.
- **HSV Masking:** Taking the RGB image, it converts it to HSV space and applies pre-defined `COLOR_RANGES` bounds. It effectively highlights pixels that fall within the threshold of the target color, removing noise with morphological transformations.
- **Centroid Calculation & Deprojection:** By calculating the largest contour in the mask, it tracks its pixel coordinate `(u, v)`. Using the Depth buffer at that exact pixel, the intrinsic formulas (focal length, fov) and extrinsic matrices (camera position/rotation) project the 2D coordinate backwards into an estimated 3D World space coordinate.
- **Task Integration:** `pick_and_place_m7_vision.py` scatters objects natively, queries the estimated 3D position from the `vision_pipeline`, and builds dynamic waypoints to execute an Inverse Kinematics (IK) trajectory to grab the cube. Since this is an estimation, it prints out the Calibration Error compared to the true physics location. 

### 2. Segmentation Perception ("Cheat Code" / True Masking)
**Files:** `src/perception/segmentation_pipeline.py` & `src/tasks/pick_and_place_m7_segmentation.py`

**How it works:**
- **Segmentation Rendering:** Instead of processing light, reflection, and shadow calculations like a camera, we utilize MuJoCo's "cheat code" level simulator feature: **Segmentation Rendering**. This renders the scene such that every object surface is a solid, distinct integer ID instead of a visual color.
- **Ground Truth Masking:** By asking MuJoCo for the exact integer ID of a body name (e.g., `"target_cube_yellow"`), our pipeline filters the 2D segmentation image and instantly, flawlessly builds a true mask around the object. 
- **Centroid & Deprojection:** Similar to the vision pipeline, we grab the 2D coordinates `(u, v)`, sample the depth render, and calculate the 3D position in the world.
- **Task Integration:** `pick_and_place_m7_segmentation.py` expects a flawless tracking response. This method bypasses visual errors like shadows or lighting, providing almost zero calibration error. These exact physics-based pipelines are an integral part of AI generation: models can look up millions of frames of data automatically without any human annotating bounding boxes or testing visual color models.
