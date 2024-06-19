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

# %% Define the old and new names for the year column
old_names = ['JAHR', 'ANTJAHR', 'ANTRAGSJAH']
new_name = 'year'
# Change the column name for each dataframe
for year in years:
    for old_name in old_names:
        if old_name in data[year].columns:
            data[year] = data[year].rename(columns={old_name: new_name})

# %% change year data type
for year in years:
    data[year]['year'] = pd.to_datetime(data[year]['year'], format='%Y').dt.year.astype(int)
    
# %% Delete unneeded columns from the dataframes
# Define the column name you want to delete
column_names = ['Shape_Area', 'Shape_Leng', 'SHAPE_Leng', 'SHAPE_Area', 'SCHLAGNR', 'SCHLAGBEZ', 'FLAECHE', 'AKT_FL', 'AKTUELLEFL']
# Delete the columns for each dataframe
for year in years:
    for column_name in column_names:
        if column_name in data[year].columns:
            data[year] = data[year].drop(columns=column_name)

# %% Define the old and new names for kultur code column
old_names = ['KC_GEM', 'KC_FESTG', 'KC', 'NC_FESTG', 'KULTURCODE']
new_name = 'kulturcode'
# Change the column name for each dataframe
for year in years:
    for old_name in old_names:
        if old_name in data[year].columns:
            data[year] = data[year].rename(columns={old_name: new_name})
            
# %% Iterate over each year to check if all unique kulturcode values are numbers or if there are characters
for year in years:
    # Extract unique kulturcode values for the current year
    unique_kulturcodes = data[year]['kulturcode'].unique()
    # Check for non-numeric kulturcode values
    non_numeric_kulturcodes = [code for code in unique_kulturcodes if not str(code).replace('.', '', 1).isdigit()]
    
    if non_numeric_kulturcodes:
        print(f"{year}: Non-numeric kulturcode values found: {non_numeric_kulturcodes}")
    else:
        print(f"{year}: All kulturcode values are numeric.")
        
        
# %% Assuming all kulturcode values are numeric
for year in years:
    # Convert kulturcode to integer
    data[year]['kulturcode'] = data[year]['kulturcode'].astype(int)
    print(f"{year}: kulturcode data type is now {data[year]['kulturcode'].dtype}")

# %% info for each year to verify modification
for year in years:
    print(f"{year}: {data[year].info()}")

# %% Head (first few rows) for each year
for year in years:
    print(f"{year}: {data[year].head()}")

# %% Check if FLIK is dupicated within each year
for year in years:
    print(f"{year}: {data[year][['year', 'FLIK']].duplicated().sum()}")
#yes, there are duplicates so we can not use this as unique identifier  
# %% So, we reset the index and add the old index as a new column 'id' which could be used to search for duplicated entries after joining land
for year in years:
    data[year] = data[year].reset_index().rename(columns={'index': 'id'})
    print(f"{year}: {data[year][['year', 'id']].duplicated().sum()}")
    #no duplicates


#########################################################################
#remove fields outside of land boundary and append data for all years
#########################################################################
# %% Extract the ZIP file containing the administrative shapefiles
# Path to the ZIP file
zip_path = 'C:/Users/aladesuru/Downloads/verwaltungseinheiten.zip'
# Target directory where files will be extracted
target_directory = 'data/raw/verwaltungseinheiten'
# Create the target directory if it doesn't exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)
# Open the ZIP file and extract its contents
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_directory)

# %% Load Land boundary 
shapefile_path = os.path.join(target_directory, "NDS_Landesflaeche.shp")
land = gpd.read_file(shapefile_path)
print(land.head())
land.info()
land.crs

# %% join land to data for each year to remove fields outside of land boundary
for year in years:
    data[year] = gpd.sjoin(data[year], land, how='inner', predicate='intersects')
    # remove columns index_right, source, id, name
    data[year] = data[year].drop(columns=['index_right', 'LAND'])
    print(f"{year}: {data[year].info()}")
    
# %% just to be sure, we check for duplicates after joining land
for year in years:
    print(f"{year}: {data[year][['year', 'id']].duplicated().sum()}") # no duplicates created

# %% if all looks good and there are no duplicates, we drop id and proceed with appending
for year in years:
    data[year] = data[year].drop(columns=['id'])
    print(f"{year}: {data[year].info()}")

# Append data for all years
# %% Concatenate all dataframes
all_years = pd.concat(list(data.values()), ignore_index=True)
all_years.info()

# %% Save to pickle
all_years.to_pickle('data/interim/data.pkl')
# %% save o parquet
all_years.to_parquet('data/interim/data.parquet')

# %% check for instances with missing values
missing_values = all_years[all_years.isnull().any(axis=1)]
all_years.isnull().any(axis=1).sum()
# %% if there are missing values less than 1% of data, drop rows with missing values	
#all_years = all_years.dropna()

# %% 
# Create columns for geometric measures
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


# %% Save to pickle
all_years.to_pickle('data/interim/data_withmetrics.pkl')
# %% save to parquet
all_years.to_parquet('data/interim/data_withmetrics.parquet')

#########################################################################
#join regional information to data
#########################################################################
# %% 1. Reset the index and add the old index as a new column 'id' which could be used to search for duplicated entries after joining districts
all_years = all_years.reset_index().rename(columns={'index': 'id'})

# %% Load Landkreis boundaries 
landkreise = gpd.read_file(os.path.join(target_directory, "NDS_Landkreise.shp"))
landkreise.info()

# %%
# Check the current CRS
print(f"Original CRS: {landkreise.crs}")
# %% Reproject to WGS84
if landkreise.crs != "EPSG:4326":
    regions = landkreise.to_crs("EPSG:4326")

# Plot the regions map
fig, ax = plt.subplots(figsize=(10, 10))
regions.plot(ax=ax, color='lightgrey', edgecolor='black')
# Annotate the map with regional district names
for idx, row in regions.iterrows():
    # Calculate the centroid of each district
    centroid = row['geometry'].centroid
    plt.text(centroid.x, centroid.y, row['LANDKREIS'], fontsize=8, ha='center')
# Add title and adjust layout
plt.title('Lower Saxony Regional Districts')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
# Show the plot
plt.show()

# %% Perform a spatial join to add landkreis information to the all_years data based on the geometry of the data.
allyears_landkreise = gpd.sjoin(all_years, landkreise, how='left', predicate="intersects")
allyears_landkreise.info()
allyears_landkreise.head()

# %% check for instances with missing values
missing_ = allyears_landkreise[allyears_landkreise.isnull().any(axis=1)]
allyears_landkreise.isnull().any(axis=1).sum() #0

# %% Check for duplicates in the 'identifier' column
duplicates = allyears_landkreise.duplicated('id')
# Print the number of duplicates
print(duplicates.sum()) #78053

#%% if there are duplicates, drop them
# --- Create a sample with all double assigned polygons from allyears_landkreise, which are 
#     crossing landkreis borders and, therefore, are assigned to more than one
#     landkreise.
double_landkreis = allyears_landkreise[allyears_landkreise.index.isin(allyears_landkreise[allyears_landkreise.index.duplicated()].index)]

#%%
# - Delete all double assigned polygons from data
allyears_landkreise = allyears_landkreise[~allyears_landkreise.index.isin(allyears_landkreise[allyears_landkreise.index.duplicated()].index)]

#%%
# --- Estimate the largest intersection for each polygon with a duplicated landkreise in the
#     double landkreis sample. Use the unit of ha.
double_landkreis['intersection'] = [
    a.intersection(landkreise[landkreise.index == b].\
      geometry.values[0]).area/10000 for a, b in zip(
       double_landkreis.geometry.values, double_landkreis.index_right
    )]

#%%
# --- Sort by intersection area and keep only the  row with the largest intersection.
double_landkreis = double_landkreis.sort_values(by='intersection').\
         groupby('id').last().reset_index()

#%%
# --- Add the data double_landkreis to the allyears_landkreise data and name it allyears_districts
allyears_districts = pd.concat([allyears_landkreise,double_landkreis])
allyears_districts.info()

#%%
# --- Only keep needed columns
allyears_districts = allyears_districts.drop(columns=['id', 'index_right', 'LK', 'intersection'])


# %% Plot landkreise "Küstenmeer Region Lüneburg" and "Küstenmeer Region Weser-Ems"
landkreise_kunst = landkreise[(landkreise['LANDKREIS'] == "Küstenmeer Region Lüneburg") | (landkreise['LANDKREIS'] == "Küstenmeer Region Weser-Ems")]
fig, ax = plt.subplots(figsize=(10, 10))
landkreise_kunst.plot(ax=ax)
ax.set_title('Landkreise: Küstenmeer Region Lüneburg & Weser-Ems')
plt.show()

# %% See fields belonging to "Küstenmeer Region Lüneburg" or "Küstenmeer Region Weser-Ems" in the LANDKREIS column
rows = allyears_districts[allyears_districts['LANDKREIS'].str.contains('Küstenmeer Region Lüneburg|Küstenmeer Region Weser-Ems', na=False)]
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


#########################################################################
#join grid to data
#########################################################################
# %% ########################################
# Load Germany grid, join to main data and remore duplicates using largest intersection
grid = gpd.read_file('data/raw/eea_10_km_eea-ref-grid-de_p_2013_v02_r00')
grid.plot()
grid.info()
grid.crs
# %%
grid=grid.to_crs(allyears_districts.crs)
grid.crs
grid.head()

# %% maybe filter out nieder using Land. I do not think the grid generated covers all of Nieder
#gridd = grid.reset_index().rename(columns={'index': 'id'})
#grid_nieder = gpd.sjoin(gridd, land, how='inner', predicate='intersects')
# %% Check for duplicates in the 'identifier' column
#dupli = grid_nieder.duplicated('id')
# Print the number of duplicates
#print(dupli.sum())
#grid_nieder = grid_nieder.drop(columns=['id', 'index_right'])
#grid_nieder.info()
#grid_nieder.plot()

# %% Reset the index and add the old index as a new column 'id' which could be used to search for duplicated entries after joining grid
allyears_districts = allyears_districts.reset_index().rename(columns={'index': 'id'})
allyears_districts.info()

# %% Perform a spatial join to add grid information to the allyears_districts data based on the geometry of the data.
allyears_dgrid = gpd.sjoin(allyears_districts, grid, how='left', predicate="intersects")
allyears_dgrid.info()
allyears_dgrid.head()

# %% check for instances with missing values
missing_2 = allyears_dgrid[allyears_dgrid.isnull().any(axis=1)]
print(len(missing_2)) #0

# %% Check for duplicates in the 'identifier' column
duplicates2 = allyears_dgrid.duplicated('id')
# Print the number of duplicates
print(duplicates2.sum())

#%%
# --- Create a sample with all double assigned polygons from allyears_grid, which are 
#     crossing grid borders and, therefore, are assigned to more than one
#     grid.
double_assigned = allyears_dgrid[allyears_dgrid.index.isin(allyears_dgrid[allyears_dgrid.index.duplicated()].index)]

#%%
# - Delete all double assigned polygons, i.d. polygons that are assigned to more
#   than one grid
allyears_dgrid = allyears_dgrid[~allyears_dgrid.index.isin(allyears_dgrid[allyears_dgrid.index.duplicated()].index)]

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
# --- Add the data double_assigned to the allyears_dgrid data and name it gridleveldata(gld)
gld = pd.concat([allyears_dgrid, double_assigned])
gld.info()

#%%
# --- Only keep needed columns
gld = gld.drop(columns=['id', 'index_right', 'EOFORIGIN', 'NOFORIGIN', 'intersection'])
    
# %% Save file to pickle and parquet
gld.to_pickle('data/interim/gld.pkl')
gld.to_parquet('data/interim/gld.parquet')
   
######################################################

# %%
# Plotting gld
gld[gld['year'] == 2012].plot(figsize=(10, 6))  # Adjust alpha for transparency as needed
plt.title('Geospatial Distribution in 2012')
plt.show()

gld[gld['year'] == 2013].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2013')
plt.show()

gld[gld['year'] == 2014].plot(figsize=(10, 6)) 
plt.title('Geospatial Distribution in 2014')
plt.show()

gld[gld['year'] == 2015].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2015')
plt.show()

gld[gld['year'] == 2016].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2016')
plt.show()

gld[gld['year'] == 2017].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2017')
plt.show()

gld[gld['year'] == 2018].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2018')
plt.show()

# Plotting gld
gld[gld['year'] == 2019].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2019')
plt.show()

gld[gld['year'] == 2020].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2020')
plt.show()

gld[gld['year'] == 2021].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2021')
plt.show()

gld[gld['year'] == 2022].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2022')
plt.show()

gld[gld['year'] == 2023].plot(figsize=(10, 6))
plt.title('Geospatial Distribution in 2023')
plt.show()
# %%
