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

### 3. Neural Object Detection (YOLOv8)
**Files:** `src/perception/yolo_pipeline.py` & `src/tasks/pick_and_place_m9_yolo.py`

**How it works:**
- **Inference Setup:** Instead of relying on color thresholds or simulator "cheat codes" (segmentation masks), we deploy a lightweight, real-time neural network: Google's `ultralytics` YOLOv8 framework. 
- **Inference & Bounding Boxes:** The pipeline extracts the RGB camera frame natively and passes it to the YOLO engine. YOLO returns bounding box coordinates (min/max X and Y) and confidence scores for known geometric shapes or scanned objects (e.g., bowl, orange).
- **Centroid Deprojection:** By calculating the center of the 2D bounding box and querying the MuJoCo depth buffer at that pixel, we deproject the coordinate into 3D task space to feed directly into the robot's IK controllers.

### 4. Vision Language Reasoning (Qwen-VL)
**Files:** `src/perception/vlm_pipeline.py` & `src/tasks/pick_and_place_m11_vlm.py`

**How it works:**
- **Semantic Understanding:** Moving beyond rigid, pre-trained YOLO classes, we integrate the HuggingFace `Qwen-VL-Chat` model. The user can type ambiguous prompts like "grab the healthy snack" or "pick up the metal box".
- **Zero-Shot Grounding:** The VLM analyzes the camera's RGB frame alongside the user's text prompt, reasons about the image semantics, and natively outputs a formatted bounding box around the target object without any prior training on those specific 3D assets.
- **Physics Matching:** Because the bounding box is semantic, the pipeline projects the 3D coordinate and finds the closest physical `mujoco.mjtObj.mjOBJ_BODY` in the scene, automatically mapping a linguistic concept to a graspable physics mesh.

### 5. Advanced Vision-Language-Action (Florence-2)
**Files:** `src/perception/florence2_pipeline.py` & `src/tasks/pick_and_place_m12_florence2.py`

**How it works:**
- **High-Fidelity Reasoning:** Replaces Qwen-VL with Microsoft's `Florence-2` Foundation Model. Florence provides significantly faster, more accurate dense captioning and region-proposal grounding.
- **Open-Vocabulary Detection:** The pipeline leverages Florence's `<CAPTION_TO_PHRASE_GROUNDING>` task type to generate extremely precise pixel polygons/bounding boxes based solely on natural language text prompts.
- **Task Integration:** Provides the sensory backbone for the Mark-12 framework, allowing the robot to execute unbroken cognitive loops: idle scanning, language parsing, visual grounding, RRT trajectory planning, and tactile grasping with slip-recovery protocols. 
