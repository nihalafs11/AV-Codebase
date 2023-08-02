
# @author Nihal Afsal 
# A python using matplotlib script to plot vehicle’s speed as a function of time

import numpy as np
import matplotlib.pyplot as plt

mass = 2000
resistance = 1500
force = 5 * 10**5

def speed(t):
	return (force / resistance) * (1 - np.exp(-resistance / mass * t))

time = np.linspace(0, 50, 1000)
velocity = speed(time)

plt.plot(time, velocity)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.title('Car Speed vs Time')
plt.grid(True)
plt.show()
