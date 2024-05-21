# %%
import geopy
from geopy.geocoders import Nominatim
import folium
import pickle
import geopandas as gpd
from folium import Choropleth, Circle, Marker 
from folium.plugins import HeatMap, MarkerCluster

# %% Set the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %%
# %% Load grid 
with open('data/interim/griddf.pkl', 'rb') as f:
    griddf = pickle.load(f)
griddf.info()    
griddf.head()

# %%
# Load Germany grid to obtain the grid geometry
grid = gpd.read_file('data/interim/eeagrid_25832')
grid.plot()
grid.info()
grid.crs

# %%
griddf_ = griddf_.to_crs(epsg=4326)

# %% Join grid to griddf using cellcode
griddf_ = griddf.merge(grid, on='CELLCODE')
griddf_.info()
griddf_.head()

# %% Convert the DataFrame to a GeoDataFrame
griddf_ = gpd.GeoDataFrame(griddf_, geometry='geometry')

# %%
griddf_['centroid'] = griddf_['geometry'].apply(lambda polygon: polygon.centroid)
# %%
# Calculate the bounds of the data
min_lat = griddf_['centroid'].y.min()
max_lat = griddf_['centroid'].y.max()
min_lon = griddf_['centroid'].x.min()
max_lon = griddf_['centroid'].x.max()

# %% Calculate the center of the map
center_lat = (min_lat + max_lat) / 2
center_lon = (min_lon + max_lon) / 2

# %%
# create new geodataframe containing only id, centroid, centroid_x and centroid_y
base_gdf = griddf_[['id', 'centroid', 'centroid_x', 'centroid_y']]
# %%
base_gdf.info()

# %%
#from griddf_ drop 'centroid', 'centroid_x', 'centroid_y'
griddf_ = griddf_.drop(columns=['EOFORIGIN', 'NOFORIGIN', 'centroid', 'centroid_x', 'centroid_y'])

# %%
griddf_ = griddf_.reset_index().rename(columns={'index': 'id'})

# %% # Identify instances where 'fields' is 1
field_1 = griddf_[griddf_['fields'] == 1]
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
    geo_data=griddf_,
    name='Mean Field Size (ha)',
    data=griddf_,
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
m.add_child(folium.GeoJson(griddf_, name='grid'))
m.add_child(folium.GeoJson(field_1, name='fields'))
# Add the layer control
folium.LayerControl().add_to(m)
# Show the map
m

