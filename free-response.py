# @author Nihal Afsal 
# A python script using matplotlib to plot vehicle’s free response as a function of time

import numpy as np
import matplotlib.pyplot as plt

m = 2000
c = 1500
k = 500
x0 = 50
v0 = 10

def vehicle_model(x, v):
	a = (-c * v - k * x) / m
	return a

# Parameters for the Euler method
time_end = 10
dt = 0.01
steps = int(time_end / dt)

# Arrays to store the results
time = np.zeros(steps)
position = np.zeros(steps)
velocity = np.zeros(steps)

# Initial conditions
position[0] = x0
velocity[0] = v0

# Euler method
for step in range(1, steps):
	time[step] = time[step - 1] + dt
	acceleration = vehicle_model(position[step - 1], velocity[step - 1])
	position[step] = position[step - 1] + velocity[step - 1] * dt
	velocity[step] = velocity[step - 1] + acceleration * dt

# Plotting the results
plt.plot(time, position)
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('Second-Order Vehicle Model')
plt.grid(True)
plt.show()

