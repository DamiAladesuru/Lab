# %%
import geopy
from geopy.geocoders import Nominatim
import folium
import pickle
import geopandas as gpd
from folium import Choropleth, Circle, Marker 
from folium.plugins import HeatMap, MarkerCluster

#  %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gp
import geoparquet as gpq

# %% Set the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())


# %% Load grid 
with open('data/interim/gridgdf.pkl', 'rb') as f:
    gridgdf = pickle.load(f)
gridgdf.info()    
gridgdf.head()

# %%
# Load Germany grid to obtain the grid geometry
grid = gpd.read_file('data/interim/eeagrid_25832')
grid.plot()
grid.info()
grid.crs


# %% Join grid to gridgdf using cellcode
gridgdf_ = gridgdf.merge(grid, on='CELLCODE')
gridgdf_.info()
gridgdf_.head()

# %% Convert the DataFrame to a GeoDataFrame
gridgdf_ = gpd.GeoDataFrame(gridgdf_, geometry='geometry')

# %%
gridgdf_.crs

# %%
gridgdf_= gridgdf_.to_crs('EPSG:4326')











# %%
gridgdf_['centroid'] = gridgdf_['geometry'].apply(lambda polygon: polygon.centroid)
# %%
# Calculate the bounds of the data
min_lat = gridgdf_['centroid'].y.min()
max_lat = gridgdf_['centroid'].y.max()
min_lon = gridgdf_['centroid'].x.min()
max_lon = gridgdf_['centroid'].x.max()

# %% Calculate the center of the map
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# %%
# create new geodataframe containing only id, centroid, centroid_x and centroid_y
base_gdf = gridgdf_[['CELLCODE', 'centroid']]
# %%
base_gdf.info()

# %%
#from gridgdf_ drop 'centroid', 'centroid_x', 'centroid_y'
gridgdf_ = gridgdf_.drop(columns=['EOFORIGIN', 'NOFORIGIN', 'centroid'])

# %%
gridgdf_ = gridgdf_.reset_index().rename(columns={'index': 'id'})

# %% # Identify instances where 'fields' is 1
field_1 = gridgdf_[gridgdf_['fields'] == 1]
# Print the result
print(field_1)
# %%
field_1.info()

###################################################################
# %% Create the BASE map with the center location
m = folium.Map(location=[center_lat, center_lon], tiles='OpenStreetMap', zoom_start=8)
# other tiles include 'cartodbpositron'
m
# %%
# Plot a choropleth map with layes for mfs, grid and fields=1
folium.Choropleth(
    geo_data=gridgdf_,
    name='Mean Field Size (ha)',
    data=gridgdf_,
    columns=['id', 'mfs_ha'],
    key_on='feature.id',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    line_color='white',
    line_weight=0,
    highlight=False,
    smooth_factor=1.0,
    legend_name= 'Mean Field Size (ha)').add_to(m)

# Add the grid to the map
m.add_child(folium.GeoJson(gridgdf_, name='grid'))
m.add_child(folium.GeoJson(field_1, name='fields'))
# Add the layer control
folium.LayerControl().add_to(m)
# Show the map
m

