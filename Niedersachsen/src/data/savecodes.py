# %% 
import pandas as pd
###################################################################
# attribute .columns to see just the column names in a dataframe,
# attribute .shape to just see the number of rows and columns.
# %%
# Filter data for the year 2016
gld16 = gld[gld['year'] == 2016]

# Show the plot
gld16.plot()
# Visualizing
 # %% to obtain x and y limits for sample year data
ax = df[df['year'] == 2023].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2023')
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()
plt.show()
print(f"2023 X limits: {x_lim}, Y limits: {y_lim}")
# %% to set x and y limits for sample year data
ax = df[df['year'] == 2023].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2023')
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

plt.show()

# Add a shapefile layer to leaflet
shapefile_path = 'path/to/shapefile.shp'
gdf = gpd.read_file(shapefile_path)
geo_json = GeoJSON(data=gdf.__geo_interface__, name='Shapefile Layer')
m.add_layer(geo_json)

# incase I want to calculate percentage change
df['mfs_pct_change'] = df.groupby('CELLCODE')['mfs'].pct_change()

# %% to transform coordinates e.g., to EPSG:4326 for leaftlet
from pyproj import Transformer

# Create a transformer
transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326")

# Original coordinates
x, y = 438000.1, 5865999.9

# Transform the coordinates
lon, lat = transformer.transform(x, y)
print(lon, lat)

# to fin the max, min value in a column
df['column'].max()
df['column'].min()

# to save only the selected data columns to csv
df[['Column1', 'Column2']].to_csv('path.csv', index=False)


# %% ########################################################
# check if data contains ecological area codes
#############################################################
# check for all years in all_years the min and max value of 'kulturcode' column
# %%
print(['year'], all_years.groupby('year')['kulturcode'].max())
print(['year'], all_years.groupby('year')['kulturcode'].min())

# %%
# Extract unique values from 'kulturcode' column for future use
unique_kulturcodes = all_years['kulturcode'].unique()
# Convert to DataFrame for easy CSV export
unique_kulturcodes_df = pd.DataFrame(unique_kulturcodes, columns=['UniqueKulturcodes'])
unique_kulturcodes_df.to_csv('reports/unique_kulturcodes.csv', index=False)

# %%
def stille_count(data, year):
    a = all_years[(all_years['kulturcode'] >= 545) & (all_years['kulturcode'] <= 587)]
    b = all_years[(all_years['kulturcode'] >= 52) & (all_years['kulturcode'] <= 66)]
    acount = a.groupby('year')['kulturcode'].value_counts()
    bcount = b.groupby('year')['kulturcode'].value_counts()
    joined = pd.concat([acount, bcount], axis=1)
    sorted = joined.sort_index()
    return sorted
print(stille_count(all_years, years))
# to csv
stille_count(all_years, years).to_csv('reports/statistics/stille_count.csv') #save to csv