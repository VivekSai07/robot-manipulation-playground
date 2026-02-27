import time
import numpy as np
import pinocchio as pin
import mujoco
import mujoco.viewer

from src.controllers.osc_controller_m3 import OSCController
from src.controllers.grasp_controller import GraspController
from src.robots.franka_panda.robot import FrankaPanda
from src.robots.franka_panda.config import (
    SCENE_PATH,
    Q_HOME,
    ARM_DOF,
    ACTIVE_JOINTS
)

def main():
    print("🚀 Initializing Franka Mark-3 (OSC) Systems...")
    
    robot = FrankaPanda()
    
    # Raised gains for faster, crisper motion.
    # kd is set to critical damping: 2*sqrt(kp).
    osc = OSCController(
        robot,
        active_joint_indices=ACTIVE_JOINTS,
        kp_pos=400.0,          # was 150 → much stiffer spring, reaches target faster
        kp_rot=100.0,          # was 50
        nullspace_kp=20.0,     # was 10 → snappier posture recovery
    )

    m = mujoco.MjModel.from_xml_path(SCENE_PATH)
    d = mujoco.MjData(m)

    grasp_sys = GraspController(m, d)

    n_joints = min(len(Q_HOME), m.nq)
    d.qpos[:n_joints] = Q_HOME[:n_joints]

    q_home_pin = pin.neutral(robot.model)
    q_home_pin[:n_joints] = Q_HOME[:n_joints]
    
    home_pose = robot.forward_kinematics(q_home_pin)
    fixed_rotation = home_pose.rotation.copy()
    HOME_POS = home_pose.translation.copy()

    # --- GRIPPER Z OFFSET CALIBRATION ---
    # The panda_hand frame origin sits ~105mm above the fingertip contact surface.
    # However, we want the fingertips to be at cube *center* height, not the floor.
    # Cube half-height is typically 0.025m, so cube center is at z=0.025.
    # We command the EE frame to z = cube_center + offset so fingertips land on cube.
    #
    # If fingers still don't reach: DECREASE GRIPPER_Z_OFFSET in small steps (e.g. 0.005).
    # If fingers crash into floor: INCREASE it.
    CUBE_CENTER_Z   = 0.025   # height of cube center above ground (adjust to match your model)
    GRIPPER_Z_OFFSET = 0.095  # panda_hand frame → fingertip contact distance (was 0.105)

    PICK_XY  = np.array([0.45, 0.0])
    PLACE_XY = np.array([0.0,  0.5])

    PICK_POS  = np.array([*PICK_XY,  CUBE_CENTER_Z + GRIPPER_Z_OFFSET])
    PLACE_POS = np.array([*PLACE_XY, CUBE_CENTER_Z + GRIPPER_Z_OFFSET])

    APPROACH_HEIGHT = 0.18   # how high above pick/place to hover before descending

    # --- STATE MACHINE ---
    # Key timing fix: "Descend to Pick" is now 2.5s (was 1.5s) so the arm
    # *fully arrives* at pick height before the grasp command fires.
    # "Grasping" still waits for is_holding confirmation, but 1.5s max.
    states = [
        {"name": "Approach Pick",   "pos": PICK_POS  + np.array([0, 0, APPROACH_HEIGHT]), "gripper": 255, "duration": 1.5},
        {"name": "Descend to Pick", "pos": PICK_POS,                                       "gripper": 255, "duration": 2.5},  # was 1.5
        {"name": "Grasping",        "pos": PICK_POS,                                       "gripper": 0,   "duration": 1.5},  # was 1.0
        {"name": "Lift",            "pos": PICK_POS  + np.array([0, 0, APPROACH_HEIGHT]), "gripper": 0,   "duration": 1.5},  # was 2.0
        {"name": "Move to Place",   "pos": PLACE_POS + np.array([0, 0, APPROACH_HEIGHT]), "gripper": 0,   "duration": 2.5},  # was 3.0
        {"name": "Lower to Place",  "pos": PLACE_POS,                                      "gripper": 0,   "duration": 2.0},
        {"name": "Release",         "pos": PLACE_POS,                                      "gripper": 255, "duration": 1.0},
        {"name": "Retract",         "pos": PLACE_POS + np.array([0, 0, APPROACH_HEIGHT]), "gripper": 255, "duration": 1.5},
        {"name": "Return to Home",  "pos": HOME_POS,                                       "gripper": 255, "duration": 2.5},
    ]

    current_state_idx = 0
    state_start_time  = 0.0

    # --- POSITION CONVERGENCE HELPER ---
    # Transition to next state only when the EE is actually close to the target,
    # *and* the minimum duration has elapsed. This prevents premature transitions
    # that caused the early-close bug.
    POS_THRESHOLD = 0.015   # metres — "close enough" to target position

    print("🟢 Simulation Online. Engaging physics...")
    print(f"   PICK_POS  = {PICK_POS}")
    print(f"   PLACE_POS = {PLACE_POS}")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        mujoco.mj_forward(m, d)
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            q_current  = d.qpos[:robot.model.nq].copy()
            dq_current = d.qvel[:robot.model.nv].copy()

            if current_state_idx < len(states):
                state      = states[current_state_idx]
                target_pos = state["pos"]

                target_se3 = pin.SE3(fixed_rotation, target_pos)

                tau, err = osc.compute_torque(
                    q_current,
                    dq_current,
                    target_se3,
                    q_posture=q_home_pin
                )

                for i, j_idx in enumerate(ACTIVE_JOINTS):
                    d.ctrl[i] = tau[j_idx]

                grasp_sys.command(state["gripper"])

                elapsed = d.time - state_start_time
                min_time_passed = elapsed > state["duration"]

                if state["name"] == "Grasping":
                    is_holding = grasp_sys.is_grasped()
                    if is_holding and elapsed > 0.3:   # small delay to let fingers settle
                        print(f"[{d.time:.2f}s] ✅ Grasp secured! Lifting payload.")
                        current_state_idx += 1
                        state_start_time   = d.time
                    elif elapsed > state["duration"]:
                        # Timeout — proceed anyway to avoid hanging
                        print(f"[{d.time:.2f}s] ⚠️  Grasp timeout — proceeding (check GRIPPER_Z_OFFSET).")
                        current_state_idx += 1
                        state_start_time   = d.time
                else:
                    # For all non-grasp states: wait for BOTH time AND position convergence.
                    # This is the core fix for early-close / premature transitions.
                    pos_converged = err < POS_THRESHOLD
                    if min_time_passed and pos_converged:
                        current_state_idx += 1
                        state_start_time   = d.time
                        if current_state_idx < len(states):
                            next_name = states[current_state_idx]['name']
                            print(f"[{d.time:.2f}s] State → {next_name}  (err={err*1000:.1f}mm)")
                    elif min_time_passed and not pos_converged:
                        # Still converging — give it up to 2× the nominal duration
                        if elapsed > state["duration"] * 2.0:
                            print(f"[{d.time:.2f}s] ⚠️  Convergence timeout for '{state['name']}' (err={err*1000:.1f}mm). Moving on.")
                            current_state_idx += 1
                            state_start_time   = d.time

            # Gravity compensation applied to arm joints
            d.qfrc_applied[:ARM_DOF] = d.qfrc_bias[:ARM_DOF]

            mujoco.mj_step(m, d)
            viewer.sync()

            elapsed_wall = time.time() - step_start
            if elapsed_wall < m.opt.timestep:
                time.sleep(m.opt.timestep - elapsed_wall)

if __name__ == "__main__":
    main()