import os

ROBOT_DIR = os.path.dirname(__file__)

URDF_PATH = os.path.join(ROBOT_DIR, "urdf", "panda.urdf")
SCENE_PATH = os.path.join(ROBOT_DIR, "model", "scene.xml")
PACKAGE_DIR = ROBOT_DIR

ARM_DOF = 7
ACTIVE_JOINTS = [0, 1, 2, 3, 4, 5, 6]
EE_FRAME = "panda_hand"

Q_HOME = [
    0.0, -0.785, 0.0, -2.356,
    0.0, 1.571, 0.785,
    0.04, 0.04
]
