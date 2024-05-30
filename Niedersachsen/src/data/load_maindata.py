# %%
import geopandas as gpd
import shapely as sh
from shapely.geometry import Polygon
import fiona
import pyogrio
import pandas as pd
import os
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
######################################
#Preprocessing


# %% Set the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %% Load data
# Define the base path to load data
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
    
    # one thing that could be nice is to be able to check if data for certain year is already loaded. If it is, we can skip loading it again. This is useful especially because vpn often cut off when I take breaks and this can mean loading data again.
    #if year in data:
        #print(f"Data for {year} is already loaded.")
        #continue (this code slows things down further)
          
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
column_names = ['Shape_Area', 'Shape_Leng', 'SHAPE_Leng', 'SHAPE_Area', 'SCHLAGNR', 'SCHLAGBEZ', 'FLAECHE', 'AKT_FL', 'AKTUELLEFL']
# Delete the columns for each dataframe
for year in years:
    for column_name in column_names:
        if column_name in data[year].columns:
            data[year] = data[year].drop(columns=column_name)

# %% change year data type
for year in years:
    data[year]['year'] = pd.to_datetime(data[year]['year'], format='%Y').dt.year.astype(int)

# %% info for each year to verify deletion
for year in years:
    print(f"{year}: {data[year].info()}")

# %% Head (first few rows) for each year
for year in years:
    print(f"{year}: {data[year].head()}")

# %% Check if FLIK is dupicated within each year
for year in years:
    print(f"{year}: {data[year][['year', 'FLIK']].duplicated().sum()}")
    
# %%
# Load adminsitrative boundary GeoJSON file of Germany
degdf = gpd.read_file('data/raw/de_states.json')

# %% Filter the GeoDataFrame to get the boundary of Niedersachsen
nieder = degdf[degdf['name'] == "Niedersachsen"]
nieder.plot()
nieder.crs
# %% reproject nieder to epsg 25832
nieder=nieder.to_crs(epsg=25832)

# %% Join the data for each year to the Niedersachsen boundary
for year in years:
    data[year] = gpd.sjoin(data[year], nieder, how='inner', predicate='intersects')
    print(f"{year}: {data[year].info()}")
    # remove columns index_right, source, id, name
    data[year] = data[year].drop(columns=['index_right', 'source', 'id', 'name'])
# %% ############################################
# Append data for all years
# %% Concatenate all dataframes
all_years = pd.concat(list(data.values()), ignore_index=True)
all_years.info()

# %% Get unique years
years = all_years['year'].unique()
years

# %% check for instances with missing values
missing_values = all_years[all_years.isnull().any(axis=1)]
all_years.isnull().any(axis=1).sum()
# %% if there are missing values less than 1% of data, drop rows with missing values	
#all_years = all_years.dropna()

# check for all years in all_years the min and max value of 'KULTURCODE' column
# %%
print(['year'], all_years.groupby('year')['KULTURCODE'].max())

# %%
print(['year'], all_years.groupby('year')['KULTURCODE'].min())

# %% cahange the data type of 'KULTURCODE' column to integer
all_years['KULTURCODE'] = all_years['KULTURCODE'].astype(int)

# %% check if data contains ecological area codes
def stille_count(data, year):
    a = data[(data['KULTURCODE'] >= 545) & (data['KULTURCODE'] <= 587)]
    b = data[(data['KULTURCODE'] >= 52) & (data['KULTURCODE'] <= 66)]
    acount = a.groupby('year')['KULTURCODE'].value_counts()
    bcount = b.groupby('year')['KULTURCODE'].value_counts()
    joined = pd.concat([acount, bcount], axis=1)
    sorted = joined.sort_index()
    return sorted
print(stille_count(all_years, years))
# to csv
stille_count(all_years, years).to_csv('reports/statistics/stille_count.csv') #save to csv


# %% #################################
# Additional required columns for data
#1. Field area  (m2 and ha) 
all_years['area_m2'] = all_years.area
all_years['area_ha'] = all_years['area_m2']*(1/10000)
#2. Perimeter (m)
all_years['peri_m'] = all_years.length
#3. Shape index
import math as m
def shapeindex(a, p):
    SI = p/(2*m.sqrt(m.pi*a))
    return SI
all_years['shp_index'] = all_years.apply(lambda row: shapeindex(row['area_m2'], row['peri_m']), axis=1)

#4. Fractal dimension
def fractaldimension(a, p):
    FD = (2*m.log(p))/m.log(a)
    return FD
all_years['fract'] = all_years.apply(lambda row: fractaldimension(row['area_m2'], row['peri_m']), axis=1)

all_years.head()
# %%
#all_years.to_file("N:/ds/priv/aladesuru/NiedersachsenData/allyears_nieder/allyears_nieder.shp") #save to shapefile for visual inspection in e.g., ArcGIS

# %% ########################################
# Field/Landscape level descriptive statistics
    # total number of fields per year
    # min, max and mean value of field size, peri and shape index per year across landscape. We could have a box plot of these values across years.
all_years.info()
desc_stats = all_years.groupby('year')[['area_m2', 'area_ha', 'peri_m', 'shp_index', 'fract']].describe()
# Calculate the sum of each column
column_sums = all_years.groupby('year')[['area_m2', 'area_ha', 'peri_m', 'shp_index', 'fract']].sum()
desc_stats.to_csv('reports/statistics/ldscp_desc.csv') #save to csv
column_sums.to_csv('reports/statistics/sums.csv') #save to csv

# %% ######################################
# Reset the index and add the old index as a new column 'id' which could be used to search for duplicated entries after joining grid
all_years = all_years.reset_index().rename(columns={'index': 'id'})

#############################################
# %% ########################################
# Load Germany grid, join to main data and remore duplicates using largest intersection
grid = gpd.read_file('data/raw/eea_10_km_eea-ref-grid-de_p_2013_v02_r00')
grid.plot()
grid.info()
grid.crs
# %%
grid=grid.to_crs(epsg=25832)
# %% Save reprojected grid to shapefile for visual inspection in e.g., ArcGIS
#grid.to_file('data/interim/eeagrid_25832/eeagrid_25832.shp')
#%%
grid.head()

# %% Perform a spatial join to add grid information to the all_years data based on the geometry of the data.
allyears_grid = gpd.sjoin(all_years, grid, how='left', predicate="intersects")
allyears_grid.info()
allyears_grid.head()

# %% check for instances with missing values
missing_ = allyears_grid[allyears_grid.isnull().any(axis=1)]
allyears_grid.isnull().any(axis=1).sum() #0

# %% Check for duplicates in the 'identifier' column
duplicates = allyears_grid.duplicated('id')
# Print the number of duplicates
print(duplicates.sum())

#%%
# --- Create a sample with all double assigned polygons from allyears_grid, which are 
#     crossing grid borders and, therefore, are assigned to more than one
#     grid.
double_assigned = allyears_grid[allyears_grid.index.isin(allyears_grid[allyears_grid.index.duplicated()].index)]

#%%
# - Delete all double assigned polygons, i.d. polygons that are assigned to more
#   than one grid
allyears_grid = allyears_grid[~allyears_grid.index.isin(allyears_grid[allyears_grid.index.duplicated()].index)]

#%%
# --- Estimate the largest intersection for each polygon with a grid in the
#     double assigned sample. Use the unit of ha.
double_assigned['intersection'] = [
    a.intersection(grid[grid.index == b].\
      geometry.values[0]).area/10000 for a, b in zip(
       double_assigned.geometry.values, double_assigned.index_right
    )]

#%%
# --- Sort by intersection area and keep only the  row with the largest intersection.
double_assigned = double_assigned.sort_values(by='intersection').\
         groupby('id').last().reset_index()

#%%
# --- Add the data double_assigned to the allyears_grid data and name it gridleveldata(gld)
gld = pd.concat([allyears_grid,double_assigned])

#%%
# --- Only keep needed columns
gld = gld[["id","FLIK","year","area_m2","peri_m","shp_index","fract","CELLCODE","geometry"]]

######################################################
    
# %% Save file to pickle
#gld.to_pickle('data/interim/gld.pkl')
# gld.to_file("N:/ds/priv/aladesuru/NiedersachsenData/all_years_grid/gld.shp") #save to shapefile
   

# %%
