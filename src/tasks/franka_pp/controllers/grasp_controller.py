class GraspController:
    """
    Mark-1 Contact-Aware Grasp System
    - Uses Body IDs instead of Geom names (Robust)
    - Manages gripper actuation safely
    """
    def __init__(self, model, data, left_finger="left_finger", right_finger="right_finger", target="target_cube"):
        self.m = model
        self.d = data
        
        # Fetch Body IDs for robust collision detection
        # .id returns an integer
        self.left_id = self.m.body(left_finger).id
        self.right_id = self.m.body(right_finger).id
        self.target_id = self.m.body(target).id

    def is_grasped(self):
        """
        Returns True if BOTH fingers are making physical contact with the target body.
        """
        left_touch = False
        right_touch = False

        for i in range(self.d.ncon):
            c = self.d.contact[i]
            
            # Extract the raw integer body ID for the geoms in contact.
            # self.m.geom_bodyid array is much faster and avoids the numpy array wrapper.
            b1 = int(self.m.geom_bodyid[c.geom1])
            b2 = int(self.m.geom_bodyid[c.geom2])
            
            pair = {b1, b2}

            # Check if our target is in contact with the fingers
            if {self.left_id, self.target_id} == pair:
                left_touch = True
            if {self.right_id, self.target_id} == pair:
                right_touch = True

        return left_touch and right_touch

    def command(self, value):
        """
        Safely commands the gripper.
        For Panda, the gripper is a single tendon actuator (the very last one).
        """
        self.d.ctrl[-1] = value