# %%
import folium # html mapping package
from folium import Choropleth, Circle, Marker 
from folium.plugins import HeatMap, MarkerCluster

# %% Load grid 
with open('data/interim/griddf.pkl', 'rb') as f:
    griddf = pickle.load(f)
griddf.info()    
griddf.head()

# %% ########################################
# Load Germany grid to obtain the grid geometry
grid = gpd.read_file('data/interim/eeagrid_25832')
grid.plot()
grid.info()
grid.crs

# %% Join grid to griddf using cellcode
griddf_ = griddf.merge(grid, on='CELLCODE')
griddf_.info()
griddf_.head()

# %% Convert the DataFrame to a GeoDataFrame
griddf_ = gpd.GeoDataFrame(griddf_, geometry='geometry')
# %%
# %%
# Generate centroids of the polygons
griddf_['centroid'] = griddf_['geometry'].centroid

# %%
# Obtain mean coordinates of the centroids
meanlongitude = griddf_['centroid'].x.mean()
meanlatitude = griddf_['centroid'].y.mean()

# %%
map1 = folium.Map(location=[meanlatitude, meanlongitude], tiles='OpenStreetMap', zoom_start=8)
map1
# %%
# %% # Identify instances where 'fields' is 1
field_1 = griddf_[griddf_['fields'] == 1]
# Print the result
print(field_1)
# %%
field_1.info()

# %%
# iterate through each row
for idx, row in field_1.iterrows():
    
# Add marker for each grid cell
    Marker([row['centroid'].y, row['centroid'].x]). add_to(map1) 
map1

# %%
griddf_ = griddf_.reset_index().rename(columns={'index': 'id'})

###################################################################
# %%
#create folium map for griddf_ showing chloropleth map of mean field size
# Create a Map instance
m = folium.Map(location=[52.5, 9.5], tiles = 'cartodbpositron', zoom_start=8, control_scale=True)
# Plot a choropleth map
# Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
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
# Add the layer control
folium.LayerControl().add_to(m)
# Show the map
m
# %%
#drop centroid column
griddf_ = griddf_.drop(columns=['centroid', 'EOFORIGIN', 'NOFORIGIN'])
# %%
# using m, show gids with fields ==1 as field layer
# Create a Map instance
m = folium.Map(location=[52.5, 9.5], tiles = 'cartodbpositron', zoom_start=8, control_scale=True)
# Plot a choropleth map
# Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
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


# %%
# Create a Map instance
m = folium.Map(location=[52.5, 9.5], tiles = 'OpenStreetMap', zoom_start=8, control_scale=True)

# %%
griddf_ = griddf_.to_crs(epsg=4326)


# %%
import geopy
import folium
# %%
