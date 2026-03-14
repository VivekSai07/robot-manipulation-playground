import mujoco
import cv2
import numpy as np
from src.robots.franka_panda.config import SCENE_PATH

# 1. Load the new YOLO scene instead of the default SCENE_PATH
import os
yolo_scene_path = os.path.join(os.path.dirname(SCENE_PATH), "yolo_scene.xml")

m = mujoco.MjModel.from_xml_path(yolo_scene_path)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

# 2. Render the camera view
camera_name = "workspace_cam"
cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

width, height = 640, 480
renderer = mujoco.Renderer(m, height=height, width=width)
renderer.update_scene(d, camera=camera_name)

# Extract RGB
rgb_image = renderer.render()
# Convert RGB (MuJoCo) -> BGR (OpenCV)
bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

# 3. Save the image to disk
output_path = "yolo_camera_view.png"
cv2.imwrite(output_path, bgr_image)
print(f"✅ Camera view saved to {output_path}! Please open it to verify the angle.")
