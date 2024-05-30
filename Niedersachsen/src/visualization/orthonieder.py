# %% 
import geopandas as gpd
import pandas as pd
import requests
import rasterio
import numpy as np
import os
import matplotlib.pyplot as plt
from rasterio.plot import show

# %% download, plot raster files from a GeoJSON file and store summary statistics in dataframe
# GeoJSON file containing the links to the raster files and other attributes
geojson_file = 'data/raw/lgln-opengeodata-dop20.geojson'

# Directory to save downloaded raster files
download_dir = 'data/raw/orthosniederrast'
os.makedirs(download_dir, exist_ok=True)

# Read the GeoJSON file
gdf = gpd.read_file(geojson_file)

# Print the columns and first few rows for verification
print("Columns in the GeoJSON file:", gdf.columns)
print("First few rows of the GeoJSON file:\n", gdf.head())

# Column name to filter on and the value to match
filter_column = 'tile_id'
filter_value = '324385864'

# Filter the GeoDataFrame
filtered_gdf = gdf[gdf[filter_column] == filter_value]

# Print the filtered rows for verification
print("Filtered rows:\n", filtered_gdf)

# Create a new DataFrame to save the filtered, downloaded, and plotted rows
downloaded_rows = []

# List to store information about the downloaded raster files
data = []

# Iterate over column with the URLs of the raster files
for idx, row in filtered_gdf.iterrows():
    url = row['rgb']  # Adjust this if the column name is different
    timestamp = row['Aktualitaet']
    filename = os.path.join(download_dir, f'raster_{idx}.tif')

    # Download the raster file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Append the row to the downloaded_rows list
    downloaded_rows.append(row)

    # Read the raster file to extract summary statistics
    with rasterio.open(filename) as src:
        data_array = src.read(1)  # Read the first band
        mean_value = np.mean(data_array)
        min_value = np.min(data_array)
        max_value = np.max(data_array)
        
        # Store the information in the list
        data.append({
            'timestamp': timestamp,
            'filename': filename,
            'mean_value': mean_value,
            'min_value': min_value,
            'max_value': max_value
        })

# Convert the list to a DataFrame
df = pd.DataFrame(data)

# Plot all raster files as layers over one another
fig, ax = plt.subplots(figsize=(10, 10))
for idx, row in df.iterrows():
    filename = row['filename']
    with rasterio.open(filename) as src:
        show(src, ax=ax, title="All Rasters Overlayed")
        
#add layer control for the raster files
plt.show()

#######################################################################
#transform coordinates for leaftlet
# %% 
from pyproj import Transformer

# Create a transformer
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326")

# Original coordinates
x, y = 438000.1, 5865999.9

# Transform the coordinates
lon, lat = transformer.transform(x, y)
print(lon, lat)


#######################################################################
# trying to plot raster files as layers over one another using folium
# %%
import os
import folium
import geopandas as gpd
import rasterio
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from PIL import Image

# Function to convert a raster file to a PNG image
def raster_to_png(raster_path, layer_name, output_dir):
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        norm = Normalize(vmin=data.min(), vmax=data.max())
        cmap = plt.cm.viridis

        img = cmap(norm(data))
        img = (img[:, :, :3] * 255).astype(np.uint8)

        # Save the image as a PNG file
        img_pil = Image.fromarray(img)
        image_path = os.path.join(output_dir, f'{layer_name}.png')
        img_pil.save(image_path)

        # Get bounds
        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]
    return image_path, bounds

# Create a directory to store the image overlays
output_dir = 'image_overlays'
os.makedirs(output_dir, exist_ok=True)

# Initialize a folium map
m = folium.Map(location=[45.0, -93.0], zoom_start=5)

# Initialize bounds list
all_bounds = []

# Add raster layers
raster_files = ['data/raw/orthosniederrast/raster_12226.tif', 'data/raw/orthosniederrast/raster_12227.tif', \
    'data/raw/orthosniederrast/raster_12230.tif', 'data/raw/orthosniederrast/raster_12238.tif']
for i, raster_file in enumerate(raster_files):
    if os.path.exists(raster_file):
        image_path, bounds = raster_to_png(raster_file, f'Raster_Layer_{i}', output_dir)
        img_overlay = folium.raster_layers.ImageOverlay(
            name=f'Raster Layer {i}',
            image=image_path,
            bounds=bounds,
            opacity=0.6
        )
        img_overlay.add_to(m)
        all_bounds.extend(bounds)

# Add a shapefile layer
gdf = cell23

# Add GeoJSON to folium map
geo_json = folium.GeoJson(
    data=gdf.to_json(),
    name='Shapefile Layer',
    style_function=lambda x: {
        'fillColor': 'transparent',
        'color': 'blue',
        'weight': 2
    }
)
geo_json.add_to(m)

# Add layer control to the map
folium.LayerControl().add_to(m)

# Calculate the bounds of the shapefile layer
gdf_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
gdf_bounds = [[gdf_bounds[1], gdf_bounds[0]], [gdf_bounds[3], gdf_bounds[2]]]
all_bounds.extend(gdf_bounds)

# Function to calculate the map bounds
def calculate_bounds(all_bounds):
    min_lat = min([b[0] for b in all_bounds])
    min_lon = min([b[1] for b in all_bounds])
    max_lat = max([b[0] for b in all_bounds])
    max_lon = max([b[1] for b in all_bounds])
    return [[min_lat, min_lon], [max_lat, max_lon]]

# Calculate and set map bounds
map_bounds = calculate_bounds(all_bounds)
m.fit_bounds(map_bounds)

# Save the map to an HTML file
m.save('map.html')

# Display the map
m

# %%
