import os

ROBOT_DIR = os.path.dirname(__file__)
SCENE_PATH = os.path.join(ROBOT_DIR, "model", "scene.xml")

ARM_DOF = 6
ACTIVE_JOINTS = [0, 1, 2, 3, 4, 5]

# Home position keyframe from vx300s.xml
Q_HOME = [0.0, -0.96, 1.16, 0.0, -0.3, 0.0]

# ViperX uses positional commands for the gripper
GRIPPER_OPEN = 0.057
GRIPPER_CLOSED = 0.021