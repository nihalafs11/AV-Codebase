# A python script that has takes radar rosbag object detection data from a topic and then plots the x and y coordinates on a Python Matplotlib well formatted plot

import yaml

# We also import numpy for dealing with arrays and matplot lib for plotting
import numpy
from matplotlib import pyplot as plt

# Open and parse the yaml file
with open('topics.yaml', 'r') as file:
    rosbag_message = yaml.safe_load(file)

# According to this page (https://python.land/data-processing/python-yaml)
# the yaml file will be automatically loaded to a format that fits the data
# We verify that rosbag_message saves as a nested dictionary using type()

# Now I initialize stuff for the while loop
i = 0
datapoints = len(rosbag_message['objects'])
x = [0]*datapoints
y = [0]*datapoints

# Next I extract the x and y numbers from the yaml file and save as arrays
while i < len(rosbag_message['objects']):
    y[i] = rosbag_message['objects'][i]['pose']['pose']['position']['y']
    x[i] = rosbag_message['objects'][i]['pose']['pose']['position']['x']
    i += 1
# Note that I figured out the dictionary headers by printing the rosbag_message
# with various arguments like ['header'], ['id'], etc. until I could see what was going on
# To plot the locations of specific detected objects over time, the above code needs to be modified

# Since we have the x and y positions we can just directly plot them
plt.rc('grid', linestyle=':', color='red')
plt.scatter(y,x)
plt.title("Radar Detections from Radar.bag") 
plt.xlabel("Latitude") 
plt.ylabel("Longitude") 
plt.grid(True)
plt.show()

# Alternatively we can save this data as a csv file for later use
numpy.savetxt("radar_detections.csv", [ x, y ], delimiter=",")
