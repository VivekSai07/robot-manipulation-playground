import time
import numpy as np

class KeyboardTeleop:
    """
    Mark-6 Human-in-the-Loop Teleoperation Interface
    Maps keyboard events from MuJoCo's passive viewer into 6D Cartesian velocities.
    """
    def __init__(self):
        # 6D Velocity Vector [vx, vy, vz, wx, wy, wz]
        self.v_des = np.zeros(6)
        self.gripper_closed = False
        
        # Max speed of the virtual target (meters per second)
        self.speed = 0.25  
        
        # Friction applied to the virtual target. 
        # At 500Hz physics, a 0.5 decay zeroes out velocity in 4ms! Since OS keyboard 
        # repeat rates are ~30Hz (33ms), the robot would severely stutter. 
        # 0.98 provides buttery smooth gliding between OS key events.
        self.decay = 0.98   
        
        # Spacebar cooldown to prevent rapid flickering from OS key-repeat
        self.last_toggle_time = 0.0

    def key_callback(self, keycode):
        """
        Triggered automatically by MuJoCo's passive viewer.
        Takes the raw integer keycode and updates the desired velocity.
        """
        # Spacebar (Keycode 32) -> Toggle Gripper
        if keycode == 32:
            current_time = time.time()
            if current_time - self.last_toggle_time > 0.3:  # 300ms cooldown
                self.gripper_closed = not self.gripper_closed
                self.last_toggle_time = current_time
            return

        # Convert ASCII keycode to uppercase character safely
        try:
            key = chr(keycode).upper()
        except ValueError:
            return

        # Map WASD / QE to Cartesian X, Y, Z axes
        if key == 'W':
            self.v_des[0] = self.speed       # Forward (+X)
        elif key == 'S':
            self.v_des[0] = -self.speed      # Backward (-X)
        elif key == 'A':
            self.v_des[1] = self.speed       # Left (+Y)
        elif key == 'D':
            self.v_des[1] = -self.speed      # Right (-Y)
        elif key == 'Q':
            self.v_des[2] = self.speed       # Up (+Z)
        elif key == 'E':
            self.v_des[2] = -self.speed      # Down (-Z)

    def get_command(self):
        """
        Returns the current velocity command and gripper state, 
        then gracefully decays the velocity so the robot coasts to a stop 
        when the user lets go of the keys.
        """
        current_v = self.v_des.copy()
        
        # Apply exponential decay to simulate friction/momentum
        self.v_des *= self.decay
        
        # Hard stop if velocity gets microscopically small
        np.where(np.abs(self.v_des) < 1e-4, 0.0, self.v_des)
        
        return current_v, self.gripper_closed