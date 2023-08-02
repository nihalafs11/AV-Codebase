# @author Nihal Afsal 
# A python script comparing the recorded (x,y) vehicle locations to the achieved (x,y) vehicle locations using a PPC (Pure-Pursuit Control) algorithm.

import numpy as np
import math

class State:
    def __init__(self, x, y, yaw, L=3.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.L = L

    def predict(self, v, theta, dt):
        theta = np.clip(theta, -np.deg2rad(30), np.deg2rad(30))
        xdot = v * np.cos(self.yaw) * dt
        ydot = v * np.sin(self.yaw) * dt
        ywdot = v * np.tan(theta) / self.L * dt
        self.x += xdot
        self.y += ydot
        self.yaw += ywdot

def find_nearest(x, y, cx, cy):
    dx = x - cx
    dy = y - cy
    d = np.hypot(dy, dx)
    return np.argmin(d)

def generate_poses(lookahead_distance=1.5, offset_distance=0.2, delta_gain=0.2):
    # Generate x and y arrays
    x_values = np.arange(0, 20, 0.1)
    y_values = offset_distance * np.ones(x_values.shape)  

    # Create list to hold all poses
    all_poses = []

    # Loop over different speeds
    for speed in range(1, 4, 1):
        # Initialize state at starting position
        state = State(x=0.0, y=1.0, yaw=0.0)

        # Create list to hold poses for this speed
        speed_poses = []

        # Add initial pose to list
        speed_poses.append([state.x, state.y, lookahead_distance])

        # Loop over x positions
        for i in range(len(x_values)):
            # Calculate lookahead point
            lx = state.x + lookahead_distance * np.cos(state.yaw)
            ly = state.y + lookahead_distance * np.sin(state.yaw)

            # Find nearest point on path
            nearest_idx = find_nearest(lx, ly, x_values, y_values)

            # Calculate alpha and delta
            alpha = math.atan2(y_values[nearest_idx] - ly, x_values[nearest_idx] - lx) - state.yaw
            delta = math.atan2(2.0 * state.L * math.sin(alpha) / lookahead_distance, delta_gain)

            # Update state using the chosen speed and delta
            state.predict(v=speed, theta=delta, dt=0.1)

            # Add new pose to list
            speed_poses.append([state.x, state.y, lookahead_distance])

        # Add this set of poses to the global list
        all_poses.append(speed_poses)

    # Convert to numpy array and return
    all_poses = np.array(all_poses)
    return all_poses

# Example usage
print(generate_poses())
