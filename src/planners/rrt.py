import numpy as np
import mujoco

class Node:
    """A node in the RRT tree."""
    def __init__(self, q):
        self.q = np.array(q)
        self.parent = None

class RRT:
    """
    Mark-5 Joint-Space Rapidly-Exploring Random Tree (RRT)
    
    Features:
    - Native MuJoCo Collision Checking
    - Dynamic Clearance Bubbles for specific obstacles
    """
    def __init__(self, model, data, active_joint_indices, step_size=0.1, goal_bias=0.1, max_iter=3000, obstacle_names=None, clearance=0.0):
        self.m = model
        self.d = data
        self.active_idx = np.array(active_joint_indices)
        
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.clearance = clearance
        
        # Look up and cache all geometry IDs belonging to the designated obstacles
        self.obstacle_geom_ids = set()
        if obstacle_names:
            for name in obstacle_names:
                body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id != -1:
                    for i in range(self.m.ngeom):
                        if self.m.geom_bodyid[i] == body_id:
                            self.obstacle_geom_ids.add(i)
        
        self.q_min = self.m.jnt_range[self.active_idx, 0]
        self.q_max = self.m.jnt_range[self.active_idx, 1]
        self.q_min[self.q_min == 0] = -np.pi
        self.q_max[self.q_max == 0] = np.pi

    def _check_collision(self, q):
        """
        Temporarily sets the robot state and queries the MuJoCo collision engine.
        Applies a strict clearance threshold ONLY to the specified obstacles.
        """
        qpos_backup = self.d.qpos.copy()
        self.d.qpos[self.active_idx] = q
        
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        
        in_collision = False
        for i in range(self.d.ncon):
            contact = self.d.contact[i]
            g1, g2 = contact.geom1, contact.geom2
            
            # If the contact involves our wall, use the safety clearance!
            # Otherwise, use standard strict collision (to let the robot touch the floor/cube).
            is_obstacle = (g1 in self.obstacle_geom_ids) or (g2 in self.obstacle_geom_ids)
            threshold = self.clearance if is_obstacle else -1e-4
            
            if contact.dist < threshold:
                in_collision = True
                break
                
        self.d.qpos[:] = qpos_backup
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_collision(self.m, self.d)
        
        return in_collision

    def _sample(self, q_goal):
        """Sample a random joint configuration with a bias towards the goal."""
        if np.random.rand() < self.goal_bias:
            return q_goal
        return np.random.uniform(self.q_min, self.q_max)

    def _steer(self, q_near, q_rand):
        """Take one step_size from q_near towards q_rand."""
        direction = q_rand - q_near
        dist = np.linalg.norm(direction)
        if dist <= self.step_size:
            return q_rand
        return q_near + (direction / dist) * self.step_size

    def plan(self, q_start, q_goal):
        """
        Plan a collision-free path in joint space from q_start to q_goal.
        Returns a list of joint configurations (waypoints).
        """
        print("🧠 RRT: Calculating collision-free path...")
        
        if self._check_collision(q_start):
            print("❌ RRT Error: Start position is currently in collision!")
            return None
        if self._check_collision(q_goal):
            print("❌ RRT Error: Goal position results in a collision!")
            return None

        nodes = [Node(q_start)]
        
        for _ in range(self.max_iter):
            q_rand = self._sample(q_goal)
            
            # Find the nearest node currently in our tree
            nearest_node = min(nodes, key=lambda n: np.linalg.norm(n.q - q_rand))
            
            # Grow a branch towards the random sample
            q_new = self._steer(nearest_node.q, q_rand)
            
            # If the branch didn't hit a wall, add it to the tree!
            if not self._check_collision(q_new):
                new_node = Node(q_new)
                new_node.parent = nearest_node
                nodes.append(new_node)
                
                # Check if this new branch is close enough to the goal
                if np.linalg.norm(q_new - q_goal) <= self.step_size:
                    # Final check: can we connect to the goal directly?
                    if not self._check_collision(q_goal):
                        goal_node = Node(q_goal)
                        goal_node.parent = new_node
                        nodes.append(goal_node)
                        
                        path = self._extract_path(goal_node)
                        print(f"✅ RRT: Path found! ({len(nodes)} nodes explored, raw path length: {len(path)})")
                        
                        return self.smooth_path(path)
                        
        print("❌ RRT Error: Max iterations reached without finding a path.")
        return None

    def _extract_path(self, node):
        """Trace parent pointers back to start."""
        path = []
        current = node
        while current is not None:
            path.append(current.q)
            current = current.parent
        return path[::-1]  # Reverse to get Start -> Goal

    def smooth_path(self, path, iterations=100):
        """
        Shortcut smoothing: Randomly pick two points on the path and try to connect them 
        with a straight line. If it doesn't hit a collision, delete all the nodes in between!
        """
        if len(path) <= 2:
            return path
            
        print("✨ RRT: Smoothing and optimizing path...")
        smoothed_path = path.copy()
        
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                break
                
            # Pick two random indices along the path
            idx1, idx2 = sorted(np.random.choice(len(smoothed_path), 2, replace=False))
            
            if idx2 - idx1 <= 1:
                continue  # Already adjacent, nothing to shortcut
                
            q1 = smoothed_path[idx1]
            q2 = smoothed_path[idx2]
            
            # Check collisions along the straight line between them
            dist = np.linalg.norm(q2 - q1)
            num_samples = max(2, int(dist / self.step_size))
            
            collision_free = True
            for i in range(1, num_samples):
                alpha = i / num_samples
                q_interp = q1 + alpha * (q2 - q1)
                
                if self._check_collision(q_interp):
                    collision_free = False
                    break
                    
            if collision_free:
                # Shortcut successful! Slice out the jagged middle nodes.
                smoothed_path = smoothed_path[:idx1+1] + smoothed_path[idx2:]
                
        print(f"📉 RRT: Path smoothed to {len(smoothed_path)} essential waypoints.")
        return smoothed_path