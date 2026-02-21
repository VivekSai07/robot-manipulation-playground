import os
import mujoco
import mujoco.viewer

# Path to panda.xml
MODEL_PATH = os.path.join(
    "src",
    "robots",
    "franka_panda",
    "model",
    "scene.xml"
)

# Load model
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
