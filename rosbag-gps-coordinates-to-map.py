#@author Nihal Afsal
# An python script to extract gps point data from a csv to a sattelite map uing folium 

import folium
import pandas as pd

# Load GPS data from CSV file
data = pd.read_csv('gps_points.csv')

# Create a folium map centered on the mean latitude and longitude
m = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=25, tiles='OpenStreetMap')

# Add a circle marker for each GPS coordinate to the folium map
for lat, lon in zip(data['latitude'], data['longitude']):
    folium.CircleMarker(location=[lat, lon], radius=10, color='black', fill=True, fill_color='red', fill_opacity=1, weight=0.5).add_to(m)

# Save the folium map to an HTML file
m.save('gps_map.html')
