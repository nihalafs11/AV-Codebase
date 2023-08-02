
# @author Nihal Afsal 
# A python script comparing 10 unique lookahead distance values for the Python PPC (Pure-Pursuit Control) algorithm

import numpy as np
import matplotlib.pyplot as plt
import math

class State:
    def __init__(self, x, y, yaw, L=3.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.L = L

    def predict(self, v, theta, dt):
        # Clip the steering angle to a range of -30 to 30 degrees
        theta = np.clip(theta, -np.deg2rad(30), np.deg2rad(30))
        # Calculate the change in x and y position and yaw angle
        xdot = v * np.cos(self.yaw) * dt
        ydot = v * np.sin(self.yaw) * dt
        ywdot = v * np.tan(theta) / self.L * dt
        # Update the state with the calculated changes
        self.x += xdot
        self.y += ydot
        self.yaw += ywdot

def find_nearest(x, y, cx, cy):
    # Calculate the distance between the point (x, y) and all points in (cx, cy)
    dx = x - cx
    dy = y - cy
    d = np.hypot(dy, dx)
    # Return the index of the nearest point in (cx, cy)
    return np.argmin(d)

# Create a reference trajectory
px = np.arange(0, 15, 0.1)
offset = 2.0
lookahead = 1.0
py = offset * np.ones(px.shape)

global_poses = []
for k in range(10):
    # Initialize the vehicle state
    state = State(x=0.0, y=1.0, yaw=0.0)
    pose = []
    pose.append([state.x, state.y, lookahead])

    for i in range(len(px)):
        # Calculate lookahead point (lx, ly)
        lx = state.x + lookahead * np.cos(state.yaw)
        ly = state.y + lookahead * np.sin(state.yaw)
        # Find the index of the nearest point on the reference trajectory
        idx = find_nearest(lx, ly, px, py)
        # Calculate the angle between the lookahead point and the nearest point
        alpha = math.atan2(py[idx] - ly, px[idx] - lx) - state.yaw
        # Calculate the steering angle
        delta = math.atan2(2.0 * state.L * math.sin(alpha) / lookahead, 1.0)
        # Update the vehicle state using the steering angle
        state.predict(v=1.0, theta=delta, dt=0.1)
        pose.append([state.x, state.y, lookahead])

    global_poses.append(pose)
    lookahead += 0.15

global_poses = np.array(global_poses)

# Plot the reference trajectory and vehicle poses
plt.figure(figsize=(8, 6))

plt.scatter(px, py, s=1, label="reference trajectory")

for j in range(len(global_poses[:, 0, 0])):
    plt.scatter(global_poses[j, :, 0], global_poses[j, :, 1], s=2, label=f"lookahead distance {global_poses[j, 0, 2]:.2f} m")

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Vehicle pose while varying pure pursuit lookahead distance")
plt.grid()
plt.legend()

plt.show()  # Keep the plot window open
