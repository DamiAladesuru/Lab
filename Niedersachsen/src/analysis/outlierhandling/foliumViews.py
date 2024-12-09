'''viewing on folium'''
# %%
import os
import numpy as np
import folium
import folium.plugins
from IPython.display import display

# %%
def plot_gdf_on_map(gdf, value_column='area_ha', title='Field Size', zoom_start=10):
    """
    Plot a GeoDataFrame on a Folium map with annotations for 'kulturart' and 'Gruppe'.
    
    Parameters:
    gdf (GeoDataFrame): The GeoDataFrame to plot
    value_column (str): The column to use for choropleth coloring (default: 'area_ha')
    title (str): The title for the choropleth legend (default: 'Field Size')
    zoom_start (int): Initial zoom level for the map (default: 10)
    
    Returns:
    folium.Map: A Folium map object
    """
    
    # Reset index and rename for consistency
    gdf = gdf.reset_index().rename(columns={'index': 'id'})

    # Reproject to EPSG:4326 for Folium visualization
    gdf = gdf.to_crs("EPSG:4326")

    # Calculate centroids (using EPSG:32632 for accuracy, then converting back)
    gdf_projected = gdf.to_crs("EPSG:32632")
    gdf_projected['centroid'] = gdf_projected.geometry.centroid
    gdf['centroid'] = gdf_projected['centroid'].to_crs("EPSG:4326").apply(lambda geom: [geom.y, geom.x])

    # Extract latitude and longitude from centroids
    latitudes = gdf['centroid'].apply(lambda x: x[0])
    longitudes = gdf['centroid'].apply(lambda x: x[1])

    # Calculate mean latitude and longitude
    mean_lat = latitudes.mean()
    mean_lon = longitudes.mean()

    # Initialize a Folium map centered on the mean centroid
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=zoom_start)

    # Add a basic Choropleth layer for the specified column to the map
    folium.Choropleth(
        geo_data=gdf.to_json(),
        name=title,
        data=gdf,
        columns=['id', value_column],
        key_on='feature.properties.id',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=title
    ).add_to(m)

    # Add GeoJSON layer with popup information
    folium.GeoJson(
        gdf,
        name='geojson',
        style_function=lambda feature: {
            'fillColor': 'transparent',
            'color': 'blue',
            'weight': 1,
            'fillOpacity': 0.7,
        },
        popup=folium.GeoJsonPopup(
            fields=['kulturart', 'Gruppe', value_column],
            aliases=['Kulturart:', 'Gruppe:', f'{title}:'],
            localize=True,
            labels=True,
        )
    ).add_to(m)

    # Add layer control to toggle between layers
    folium.LayerControl().add_to(m)

    return m



# %%
m100_23 = gldf_max100[gldf_max100['year'] == 2023]

# %%
gld23 = gld[gld['year'] == 2023]



# %% Example usage:
m = plot_gdf_on_map(gld23)
m  # Display the map

