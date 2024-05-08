# %%
import os
import geopandas as gpd
import shapely as sh
import matplotlib
import matplotlib.pyplot as plt
import fiona
import pyogrio
# %%
Grid = gpd.read_file("C:\\Users\\aladesuru\\Documents\\coding\\NRW\\Data\\Grid")
# %%
# option 1
chunk_size = 100  # Reduce the chunk size to reduce memory usage
for i in range(0, len(Grid), chunk_size):
    chunk = Grid[i:i+chunk_size]
    # Process the chunk of data
    # ...

# Generate a single plot that shows all the data
plt.figure()
Grid.plot('PageName', legend=True)
plt.show()

# %%
# option 2 - Simplify the geometry of each feature
tolerance = 0.01  # Adjust this value to control the level of simplification
simplified_grid = Grid.copy()
simplified_grid.head()

# %%
# List the columns in the GeoDataFrame
print(simplified_grid.columns)

# %%
simplified_grid['geometry'] = simplified_grid['geometry'].simplify(tolerance)

# Write the simplified shapefile to a new file
simplified_grid.to_file("simplified_shapefile.shp")

# %%
data = gpd.read_file("C:\\Users\\aladesuru\\Documents\\coding\\NRW\\Data\\EP_Kreis\\Simplified\\EP_Kreis25012023.shp")
data.head()

# %%
figsize = (20, 11)
data.plot('GN', legend=True, figsize=figsize)
data.plot('FID', legend=True, figsize=figsize)

# %%
os.getcwd()

# %%
Grid.head()
# %%
simplified_grid.plot('PageName', legend=True)

# %%
plt.show()
# %%
