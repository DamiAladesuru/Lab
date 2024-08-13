# %%
import geopandas as gpd
import shapely as sh
from shapely.geometry import Polygon
import matplotlib
import matplotlib.pyplot as plt
import fiona
import pyogrio
from datetime import datetime
import pandas as pd
import os
import seaborn as sns
import zipfile

# %% Load data
# Define the base path
base_path = "N:/ds/data/Niedersachsen/Niedersachsen/Needed/schlaege_"

# Define the years you want to load
years = range(2012, 2024)

# Create a dictionary to store the data
data = {}

# Define the specific file names you want to load
specific_file_names = ["Schlaege_mitNutzung_2012.shp", "Schlaege_mitNutzung_2013.shp", "Schlaege_mitNutzung_2014.shp", "Schlaege_mitNutzung_2015.shp", "schlaege_2016.shp", "schlaege_2017.shp", "schlaege_2018.shp", "schlaege_2019.shp", "schlaege_2020.shp", "ud_21_s.shp", "Schlaege_2022_ende_ant.shp", "UD_23_S_AKT_ANT.shp"] #list of file names to be loaded

# Load the data for each year
for year in years:
    zip_file_path = f"{base_path}{year}.zip"
    
    # Open the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref: #creates Zipfile object in read mode and assigns it to zip_ref
        # Get the list of file names in the zip file
        file_names = zip_ref.namelist()
        
        # Check if the specific files you want to open are in the list
        for specific_file_name in specific_file_names:
            if specific_file_name in file_names:
                # If the file exists, read it with geopandas
                data[year] = gpd.read_file(f"/vsizip/{zip_file_path}/{specific_file_name}")
            else:
                print(f"File {specific_file_name} does not exist in {zip_file_path}.")
# %% CRS for each year
for year in years:
    print(f"CRS for {year}: {data[year].crs}")

# %% info for each year
for year in years:
    print(f"{year}: {data[year].info()}")

# %% Describe data for each year
for year in years:
    print(f"{year}: {data[year].describe()}") #summary statistics of numeric variables 
#You can also use the attribute .columns to see just the column names in a dataframe,
#or the attribute .shape to just see the number of rows and columns.

# %% Define the old and new names for the year column
old_names = ['JAHR', 'ANTJAHR', 'ANTRAGSJAH']
new_name = 'year'
# Change the column name for each dataframe
for year in years:
    for old_name in old_names:
        if old_name in data[year].columns:
            data[year] = data[year].rename(columns={old_name: new_name})

# %% Define the old and new names for kultur code column
old_names = ['KC_GEM', 'KC_FESTG', 'KC', 'NC_FESTG']
new_name = 'KULTURCODE'
# Change the column name for each dataframe
for year in years:
    for old_name in old_names:
        if old_name in data[year].columns:
            data[year] = data[year].rename(columns={old_name: new_name})

# %% Delete unneeded columns from the dataframes
# Define the column name you want to delete
column_names = ['Shape_Area', 'Shape_Leng', 'SHAPE_Leng', 'SHAPE_Area', 'SCHLAGNR', 'SCHLAGBEZ', 'FLAECHE', 'AKT_FL', 'AKTUELLEFL', 'KULTURCODE']
# Delete the columns for each dataframe
for year in years:
    for column_name in column_names:
        if column_name in data[year].columns:
            data[year] = data[year].drop(columns=column_name)

# %% info for each year to verify deletion
for year in years:
    print(f"{year}: {data[year].info()}")

# %% Head (first few rows) for each year
for year in years:
    print(f"{year}: {data[year].head()}")

# %% create area and perimeter columns for each year
for year in years:
    data[year]['area_ha'] = data[year].area / 10000
    data[year]['peri_km'] = data[year].length / 1000

# %% DeScribe data for these columns for each year
# Define the column name
column_names = ['area_ha', 'peri_km']
# Describe the columns for each dataframe
for year in years:
    if all(column_name in data[year].columns for column_name in column_names):
        print(f"{year}: {data[year][column_names].describe()}")

# %% change year data type
for year in years:
    data[year]['year'] = pd.to_datetime(data[year]['year'], format='%Y').dt.year
# %% Plot distribution curve for 'area_ha'
#for year in years:
## sns.histplot(data[year]['area_ha'], kde = True,
             ## stat="density", kde_kws=dict(cut=3),
             ## alpha=.4, edgecolor=(1, 1, 1, .4)) OR
## sns.kdeplot(data[year]['area_ha']) OR
   # sns.displot(data[year]['area_ha'], kind="kde", alpha=.4, rug=True)

# %% Concatenate all dataframes
all_data = pd.concat(list(data.values()), ignore_index=True)
# %% Get unique years
years = all_data['year'].unique()
years
# %% Head of the concatenated dataframe
all_data.head()

#%% describe
desc_stats = all_data.groupby('year')['area_ha'].describe()
# Export to CSV
desc_stats.to_csv('desc_stats.csv')
# %% Create a FacetGrid with a distribution plot for each year
g = sns.FacetGrid(all_data, col="year", col_wrap=4, height=4)
# Map a distribution plot to the FacetGrid
g.map(sns.histplot, "area_ha")
plt.show()
# %% grid
grid = gpd.read_file("C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/eea_10_km_eea-ref-grid-de_p_2013_v02_r00")
grid.plot()
grid.info()
# %% print minx, miny, maxx, maxy
print(all_data.geometry.total_bounds)

#%%
grid.head()
# %% Define the boundaries of your area (replace with your actual boundaries)
# Define the boundaries of your area (replace with your actual boundaries)
#minx, miny, maxx, maxy = all_data.geometry.total_bounds

# Define the size of the grid cells
#dx = 10
#dy = 10

# Create the grid
#grid = []

#for x in range(minx, maxx, dx):
    #for y in range(miny, maxy, dy):
       # grid.append(Polygon([(x,y), (x+dx, y), (x+dx, y+dy), (x, y+dy)]))

# Create a GeoDataFrame from the grid
#grid_gdf = gpd.GeoDataFrame(grid, columns=['geometry'])

# Plot the grid
#grid_gdf.plot()


