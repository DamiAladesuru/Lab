# %%
import geopandas as gpd
import shapely as sh
from shapely.geometry import Polygon
import fiona
import pyogrio
import pandas as pd
import os
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
column_names = ['Shape_Area', 'Shape_Leng', 'SHAPE_Leng', 'SHAPE_Area', 'SCHLAGNR', 'SCHLAGBEZ', 'FLAECHE', 'AKT_FL', 'AKTUELLEFL', 'KULTURCODE']
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

# %% Check for duplicates instance of 'year' and 'FLIK' columns
for year in years:
    print(f"{year}: {data[year][['year', 'FLIK']].duplicated().sum()}")
    
# %% Concatenate all dataframes
all_years = pd.concat(list(data.values()), ignore_index=True)
all_years.info()
# %% Get unique years
years = all_years['year'].unique()
years

# %% #################################
# Additional required columns for data
#1. Field area  (m2)  
all_years['area_m2'] = all_years.area
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

# %% Reset the index and add the old index as a new column 'id'
all_years = all_years.reset_index().rename(columns={'index': 'id'})
######################################

# %% ########################################
# Field/Landscape level descriptive statistics
    # total number of fields per year
    # min, max and mean value of field size, peri and shape index per year across landscape. We could have a box plot of these values across years.
all_years.info()
desc_stats = all_years.groupby('year')[['area_m2', 'peri_m', 'shp_index', 'fract']].describe()

desc_stats.to_csv('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/reports/statistics/ldscp_desc_stats.csv') #save to csv
#############################################
# %% ########################################
# Load Germany grid, join to main data and remore duplicates using largest intersection
grid = gpd.read_file("C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/data/raw/eea_10_km_eea-ref-grid-de_p_2013_v02_r00")
grid.plot()
grid.info()
grid.crs
# %%
grid=grid.to_crs(epsg=25832)
# %% Save reprojected grid to shapefile for visual inspection in e.g., ArcGIS
#grid.to_file("C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/data/processed/eeagrid_25832/eeagrid_25832.shp")
#%%
grid.head()

# %% Perform a spatial join to add grid information to the all_years data based on the geometry of the data.
allyears_grid = gpd.sjoin(all_years, grid, how='left', predicate="intersects")
allyears_grid.info()
allyears_grid.head()

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
#gld.to_pickle("C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/data/interim/gld.pkl")
# all_years_grid.to_file("N:/ds/priv/aladesuru/NiedersachsenData/all_years_grid/all_years_grid.shp") #save to shapefile
# - Export as csv without geometry
#gld.drop(columns=['geometry']).\
    #to_csv("C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/Code/Niedersachsen/data/interim/niedersachsen_2012-2023_inclGridCellcode.csv")
    


