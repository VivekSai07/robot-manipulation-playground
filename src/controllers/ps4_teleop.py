import time
import numpy as np
import os

# Hide the pygame support prompt on startup
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

class PS4Teleop:
    """
    Mark-6 Human-in-the-Loop Teleoperation Interface (PS4 Upgrade)
    Maps PS4 analog sticks and buttons to 6D Cartesian velocities.
    """
    def __init__(self, speed=0.3, deadzone=0.1, smoothing=0.2):
        # Initialize the joystick module without opening a display window
        pygame.init()
        pygame.joystick.init()
        
        if pygame.joystick.get_count() == 0:
            raise Exception("🎮 No PS4 controller detected! Please connect it via Bluetooth or USB.")
            
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"🎮 Telemetry Link Established: {self.joystick.get_name()}")
        
        # 6D Velocity Vector [vx, vy, vz, wx, wy, wz]
        self.v_des = np.zeros(6)
        self.gripper_closed = False
        
        # Controller Parameters
        self.speed = speed
        self.deadzone = deadzone
        self.smoothing = smoothing # Controls how fast the robot accelerates (0.0 to 1.0)
        
        self.last_toggle_time = 0.0
        self.reset_requested = False

    def get_command(self):
        """
        Reads the hardware USB state of the controller, applies deadzones,
        and returns the desired velocity and gripper state.
        """
        # CRITICAL: This pumps the OS event queue to grab the latest USB data
        pygame.event.pump()
        
        # --- 1. Analog Stick Translation Control ---
        # Standard Pygame 2 SDL Mapping for PS4 Controllers:
        # Axis 0: Left Stick X (Left/Right)
        # Axis 1: Left Stick Y (Up/Down)
        # Axis 3: Right Stick Y (Up/Down) - Note: L2/R2 triggers shift this on some OS, but 3 is standard.
        
        left_x = self.joystick.get_axis(0)
        left_y = self.joystick.get_axis(1)
        right_y = self.joystick.get_axis(3)
        
        # Apply Deadzone (Ignore micro-movements)
        if abs(left_x) < self.deadzone: left_x = 0.0
        if abs(left_y) < self.deadzone: left_y = 0.0
        if abs(right_y) < self.deadzone: right_y = 0.0
        
        target_v = np.zeros(6)
        
        # Map Joysticks to Cartesian X, Y, Z
        # Pushing UP is negative on the joystick, so we negate it for Forward (+X)
        target_v[0] = -left_y * self.speed
        
        # Pushing RIGHT is positive on the joystick, we want Right (-Y)
        target_v[1] = -left_x * self.speed
        
        # Pushing Right Stick UP is negative, we want Up (+Z)
        target_v[2] = -right_y * self.speed
        
        # Apply Exponential Moving Average (EMA) to smooth the velocity
        self.v_des += (target_v - self.v_des) * self.smoothing
        
        # --- 2. Gripper Control (Cross / X Button) ---
        # Button 0 is traditionally the Cross button on PlayStation
        if self.joystick.get_button(0):
            current_time = time.time()
            if current_time - self.last_toggle_time > 0.3:  # 300ms cooldown
                self.gripper_closed = not self.gripper_closed
                self.last_toggle_time = current_time
                
        # --- 3. Episode Reset (Circle / O Button) ---
        # Button 1 is traditionally the Circle button
        if self.joystick.get_button(1):
            self.reset_requested = True
            
        return self.v_des.copy(), self.gripper_closed