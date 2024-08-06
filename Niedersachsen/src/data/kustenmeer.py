# import libraries
import geopandas as gpd
import matplotlib.pyplot as plt
import os


# %% Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify
print(os.getcwd())

# %% Load Landkreis file
landkreise = gpd.read_file( "N:/ds/data/Niedersachsen/verwaltungseinheiten/NDS_Landkreise.shp")
landkreise.info()

# %% Load pickle data file
gld = gpd.read_file('data/interim/gld.pkl')
gld.info()


# %% Plot landkreise "Küstenmeer Region Lüneburg" and "Küstenmeer Region Weser-Ems"
landkreise_kunst = landkreise[(landkreise['LANDKREIS'] == "Küstenmeer Region Lüneburg") | (landkreise['LANDKREIS'] == "Küstenmeer Region Weser-Ems")]
fig, ax = plt.subplots(figsize=(10, 10))
landkreise_kunst.plot(ax=ax)
ax.set_title('Landkreise: Küstenmeer Region Lüneburg & Weser-Ems')
plt.show()

# %% See fields belonging to "Küstenmeer Region Lüneburg" or "Küstenmeer Region Weser-Ems" in the LANDKREIS column
rows = gld[gld['LANDKREIS'].str.contains('Küstenmeer Region Lüneburg|Küstenmeer Region Weser-Ems', na=False)]
# Both equal 841 rows but Küstenmeer Region Lüneburg only has 84 rows in the dataset
yearly_counts = rows.groupby(['year', 'LANDKREIS']).size().reset_index(name='count')
# Convert to DataFrame and save to csv
df = yearly_counts
df.to_csv('reports/statistics/kunstenmeer_yearly_counts.csv', index=False)
# Save the filtered rows as a new DataFrame
kunstenmeer_df = rows.copy()
# Save the GeoDataFrame to a Shapefile
kunstenmeer_gdf = gpd.GeoDataFrame(kunstenmeer_df, geometry='geometry')
kunstenmeer_gdf.to_file('reports/gdf_files/kunstenmeer_fields/kunstenmeer_fields.shp', driver='ESRI Shapefile')
kunstenmeer_gdf.plot()