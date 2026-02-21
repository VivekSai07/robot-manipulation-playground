import os

ROBOT_DIR = os.path.dirname(__file__)

# Add URDF and Package paths just like Franka
URDF_PATH = os.path.join(ROBOT_DIR, "urdf", "vx300s.urdf")
SCENE_PATH = os.path.join(ROBOT_DIR, "model", "scene.xml")
PACKAGE_DIR = ROBOT_DIR

ARM_DOF = 6
ACTIVE_JOINTS = [0, 1, 2, 3, 4, 5]

# The end-effector link name defined inside your vx300s.urdf
# Note: If your URDF uses a different name for the wrist/gripper link, update this string!
EE_FRAME = "vx300s/ee_gripper_link" 

# Home position keyframe from vx300s.xml
Q_HOME = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

# ViperX uses positional commands for the gripper
GRIPPER_OPEN = 0.057
GRIPPER_CLOSED = 0.021