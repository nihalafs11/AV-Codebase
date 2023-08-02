# @author Nihal Afsal 
# A python script comparing 5 unique P values, 5 unique I values, and 5 unique D values values for the Python PID (Proportional Integral Derivative) control algorithm

import numpy as np
import matplotlib.pyplot as plt
import math

class State:
    def __init__(self, x, y, yaw, L=3.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.L = L

    def predict(self, velocity, angle, dt):
        angle = np.clip(angle, -np.deg2rad(30), np.deg2rad(30))
        xdot = velocity * np.cos(self.yaw) * dt
        ydot = velocity * np.sin(self.yaw) * dt
        yawdot = velocity * np.tan(angle) / self.L * dt
        self.x += xdot
        self.y += ydot
        self.yaw += yawdot

def find_nearest(x, y, cx, cy):
    dx = x - cx
    dy = y - cy
    d = np.hypot(dy, dx)
    return np.argmin(d)

class PID:
    def __init__(self, p, i, d):
        self.P = p
        self.I = i
        self.D = d
        self.previous_error = 0.0
        self.integral_error = 0.0

    def get_control(self, error, dt=0.1):
        proportional_error = error
        derivative_error = (error - self.previous_error) / dt
        self.integral_error += error * dt
        self.previous_error = error
        control = np.clip(
            proportional_error * self.P + self.integral_error * self.I + derivative_error * self.D,
            -np.deg2rad(30),
            np.deg2rad(30)
        )
        return control

# Initialize variables for simulation
px = np.arange(0, 20, 0.1)
offset = 2.0
p_coeff = 1.0
py = offset * np.ones(px.shape)
global_poses = []

# Simulate varying P values in PID controller
for k in range(5):
    pid = PID(p_coeff, 0.01, 1.0)
    state = State(x=0.0, y=1.0, yaw=0.0)
    pose = []
    pose.append([state.x, state.y, p_coeff])

    for i in range(len(px)):
        idx = find_nearest(state.x, state.y, px, py)
        alpha = py[idx] - state.y
        delta = pid.get_control(alpha)
        state.predict(velocity=1.0, angle=delta, dt=0.1)
        pose.append([state.x, state.y, p_coeff])

    global_poses.append(pose)
    p_coeff -= 0.15

global_poses = np.array(global_poses)

px = np.arange(0, 20, 0.1)
offset = 2.0
i_coeff = 0.005
py = offset * np.ones(px.shape)
global_poses2 = []

# Simulate varying I values in PID controller
for k in range(5):
    pid = PID(1.0, i_coeff, 1.0)
    state = State(x=0.0, y=1.0, yaw=0.0)
    pose = []
    pose.append([state.x, state.y, i_coeff])

    for j in range(len(px)):
        idx = find_nearest(state.x, state.y, px, py)
        alpha = py[idx] - state.y
        delta = pid.get_control(alpha)
        state.predict(velocity=1.0, angle=delta, dt=0.1)
        pose.append([state.x, state.y, i_coeff])

    global_poses2.append(pose)
    i_coeff += 0.025

global_poses2 = np.array(global_poses2)

px = np.arange(0, 20, 0.1)
offset = 2.0
d_coeff = 1.0
py = offset * np.ones(px.shape)
global_poses3 = []

# Simulate varying D values in PID controller
for k in range(5):
    pid = PID(1.0, 0.01, d_coeff)
    state = State(x=0.0, y=1.0, yaw=0.0)
    pose = []
    pose.append([state.x, state.y, d_coeff])

    for j in range(len(px)):
        idx = find_nearest(state.x, state.y, px, py)
        alpha = py[idx] - state.y
        delta = pid.get_control(alpha)
        state.predict(velocity=1.0, angle=delta, dt=0.1)
        pose.append([state.x, state.y, d_coeff])

    global_poses3.append(pose)
    d_coeff += 0.6

global_poses3 = np.array(global_poses3)

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(px, py, s=1, label="reference trajectory")

colors = ["red", "blue", "green", "orange", "purple"]

for j, color in zip(range(len(global_poses[:, 0, 0])), colors):
    plt.scatter(global_poses[j, :, 0], global_poses[j, :, 1], s=2, color=color, label="P value " + str(round(global_poses[j, 0, 2], 2)))

plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Vehicle pose as the P value of the PID controller changes")
plt.grid()
plt.legend()
plt.ylim(1, 3)
plt.xlim(0, 20)
plt.show()
