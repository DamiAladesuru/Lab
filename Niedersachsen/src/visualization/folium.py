# %%
import os
import numpy as np
import folium
import folium.plugins
from IPython.display import display

path = "~/Documents/DataAnalysis/Lab/Niedersachsen"
expanded_path = os.path.expanduser(path)
os.chdir(expanded_path)

print("Current directory:", os.getcwd())

from src.analysis.desc import gridgdf_desc as gd
from src.visualization import plotting_module as pm

# %% load data
gld, gridgdf = gd.silence_prints(gd.create_gridgdf)
# I always want to load gridgdf and process clean gridgdf separately so I can have uncleeaned data for comparison or sensitivity analysis
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)

'''
Parameters:
- gridgdf: GeodataFrame containing the grid data
'''

# %% let index of the gridgdf_ be the id and convert ndarray column to list
gridgdf_ = gridgdf_cl.reset_index().rename(columns={'index': 'id'})

#Check for ndarray types in gridgdf_ columns
for col in gridgdf_.columns:
    if gridgdf_[col].apply(lambda x: isinstance(x, np.ndarray)).any():
        print(f"Column '{col}' contains ndarrays.")
#convert groups column in gridgdf_ from np.ndarray to list
gridgdf_['groups'] = gridgdf_['groups'].apply(lambda x: x.tolist())

# %% Reproject to EPSG:4326 for Folium visualization
gdf_4326 = gridgdf_.to_crs("EPSG:4326")

# Temporarily reproject to a projected CRS for accurate centroid calculation
gdf_projected = gdf_4326.to_crs("EPSG:32632")

# Calculate centroids in the projected CRS
gdf_projected['centroid'] = gdf_projected.geometry.centroid

# Reproject centroids back to EPSG:4326 and convert them to lists
gdf_4326['centroid'] = gdf_projected['centroid'].to_crs("EPSG:4326").apply(lambda geom: [geom.y, geom.x])

# Extract latitude and longitude from centroids
latitudes = gdf_4326['centroid'].apply(lambda x: x[0])
longitudes = gdf_4326['centroid'].apply(lambda x: x[1])

# Calculate mean latitude and longitude
mean_lat = latitudes.mean()
mean_lon = longitudes.mean()

# Convert the GeoDataFrame to a GeoJSON string
geojson_data = gdf_4326.to_json()

# %% Initialize a Folium map centered on the mean centroid
m = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

# %% add a basic Choropleth layer for field size to the map
folium.Choropleth(
    geo_data=geojson_data,
    name='Field Size',
    data=gdf_4326,
    columns=['id', 'mfs_ha'],
    key_on='feature.properties.id',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Field Size'
).add_to(m)

# Add the GeoDataFrame to the map as a GeoJSON layer
folium.GeoJson(data=geojson_data, name='geojson').add_to(m)

# Add layer control to toggle between layers
folium.LayerControl().add_to(m)
# Display the map
m


# %% Initialize another Folium map to view the field size by year
m1 = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

grouped = gdf_4326.groupby('year')

# Iterate through each group and add a layer for each year
for year, group in grouped:
    # Convert the group to a GeoJSON string
    geojson_data = group.to_json()

    # Add a Choropleth layer to the map
    folium.Choropleth(
        geo_data=geojson_data,
        name=f'Field Size {year}',
        data=group,
        columns=['id', 'mfs_ha'], 
        key_on='feature.properties.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Field Size {year}'
    ).add_to(m1)

    # Add the GeoDataFrame to the map as a GeoJSON layer
    #folium.GeoJson(data=geojson_data, name=f'geojson_{year}').add_to(m1)

# Add layer control to toggle between layers
folium.LayerControl().add_to(m1)

# Save the map to an HTML file
m1.save('field_size_map_by_year.html')


# %% # group data by year and add markers to popup CELLCODE and mfs_ha for each grid
# Initialize a Folium map centered on the mean centroid
m2 = folium.Map(location=[mean_lat, mean_lon], zoom_start=10)

grouped = gdf_4326.groupby('year')

# Iterate through each group and add a layer for each year
for year, group in grouped:
    # Convert the group to a GeoJSON string
    geojson_data = group.to_json()

    # Add a Choropleth layer to the map
    folium.Choropleth(
        geo_data=geojson_data,
        name=f'Field Size {year}',
        data=group,
        columns=['id', 'mfs_ha'], 
        key_on='feature.properties.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=f'Field Size {year}'
    ).add_to(m2)
    
    # Create a marker cluster for the year
    marker_cluster = folium.plugins.MarkerCluster(name=f'Markers {year}').add_to(m2)
    
    for idx, row in group.iterrows():
        # Create a popup with the CELLCODE and mfs_ha
        popup = f"CELLCODE: {row['CELLCODE']}<br>Field Size: {row['mfs_ha']} ha"

        # Create a marker at the centroid of the grid cell
        marker = folium.Marker(location=row['centroid'],
                               popup=popup,
                               icon=folium.Icon(color='green', icon='info-sign'))

        # Add the marker to the marker cluster
        marker.add_to(marker_cluster)

# Add a GeoJson layer for the search functionality
geojson_layer = folium.GeoJson(
    geojson_data,
    name=f'GeoJson',
    tooltip=folium.GeoJsonTooltip(fields=['CELLCODE'], aliases=['CELLCODE:'])
).add_to(m2)

# Add a search control to the map
folium.plugins.Search(
    layer=geojson_layer,
    geom_type='Point',
    placeholder='Search for CELLCODE',
    search_label='CELLCODE',
    collapsed=False,
    position='bottomleft'
).add_to(m2)
        
# Add layer control to toggle between layers
folium.LayerControl().add_to(m2)

# Save the map to an HTML file
m2.save('src/visualization/yearly_map_with_cellcode_mfs_cl.html')

