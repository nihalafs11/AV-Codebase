
# @author Nihal Afsal 
# A python using matplotlib script to plot the response of a vehicle proportional controller

import numpy as np
import matplotlib.pyplot as plt

m = 2000
c = 1500
K_p = 6000

# Vehicle model with proportional control
def vehicle_model(v, e):
	a = (K_p * e - c * v) / m
	return a

# Parameters for the Euler method
time_end = 10
dt = 0.01
steps = int(time_end / dt)

# Arrays to store the results
time = np.zeros(steps)
velocity = np.zeros(steps)
error = np.zeros(steps)

# Initial conditions
velocity[0] = 0
error[0] = 0

# Euler method
for step in range(1, steps):
	time[step] = time[step - 1] + dt
    
	# Update error (command input is a ramp function)
	error[step] = time[step] - velocity[step - 1]
    
	# Update acceleration and velocity
	acceleration = vehicle_model(velocity[step - 1], error[step])
	velocity[step] = velocity[step - 1] + acceleration * dt

# Plotting the results
plt.plot(time, velocity)
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Proportional Controller for Unity Ramp Functions')
plt.grid(True)
plt.show()
