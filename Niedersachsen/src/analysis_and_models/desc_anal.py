# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m

# %% Change the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %% Load pickle file
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
gld.info()    
gld.head()    
#########################################
# %% Field/ landscape level descriptive statistics
#########################################
    # total number of fields per year
    # min, max and mean value of field size, peri and shape index
    # per year across landscape. We could have a box plot of these values
    # across years.
#ldscp1_desc_stats = gld.groupby('year')[['area_m2', 'peri_m', 'shp_index',\
    #'fract']].describe()
#ldscp1_desc_stats.to_csv('reports/statistics/ldscp1_desc_stats.csv') 
#save to csv


# %% #######################################
# Grid level descriptive statistics
#######################################
    # total number of grids within the geographic boundaries of the
    # study area
print("gridcount =", gld.groupby('year')['CELLCODE'].nunique())

# Create table of year, grid id, number of fields in grid, mean field size,
# sd_fs, mean peri, sd_peri, mean shape index, sd_shape index.
griddf = gld[['year', 'CELLCODE']].drop_duplicates().copy()

# %% Before we continue, first check if number of entries for area_m2, peri_m, shp and fract within each cellcode is thesame
counts = gld.groupby('CELLCODE')[['area_m2', 'peri_m', 'shp_index', 'fract']].count()
same_counts = (counts['area_m2'] == counts['peri_m']) & (counts['area_m2'] == counts['shp_index']) & (counts['area_m2'] == counts['fract'])
different_counts = counts[~same_counts]
different_counts

# %%
# 1. Number of fields per grid
#fields = gld.groupby(['year', 'CELLCODE'])['area_m2'].count().reset_index()
#fields.columns = ['year', 'CELLCODE', 'fields']
#fields.head()
#griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])

# 2. Sum of field size per grid
fs_sum = gld.groupby(['year', 'CELLCODE'])['area_m2'].sum().reset_index()
fs_sum.columns = ['year', 'CELLCODE', 'fs_sum']
fs_sum.head()
griddf = pd.merge(griddf, fs_sum, on=['year', 'CELLCODE'])

# 3. Mean field size in the grid
griddf['mfs_ha'] = (griddf['fs_sum'] / griddf['fields'])*(1/10000)

# 4. Standard deviation of field size in the grid (ha)
sdfs_ha = gld.groupby(['year', 'CELLCODE'])['area_m2'].std()*(1/10000)
sdfs_ha = sdfs_ha.reset_index()
sdfs_ha.columns = ['year', 'CELLCODE', 'sdfs_ha']
griddf = pd.merge(griddf, sdfs_ha, on=['year', 'CELLCODE'])

# Since thesame, then we can use fields column as number of fields in the grid
# 5. Sum of field peri per grid
peri_sum = gld.groupby(['year', 'CELLCODE'])['peri_m'].sum().reset_index()
peri_sum.columns = ['year', 'CELLCODE', 'peri_sum']
peri_sum.head()
griddf = pd.merge(griddf, peri_sum, on=['year', 'CELLCODE'])

# 6. Mean perimeter in the grids
griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])

# 7. Standard deviation of perimeter in the grids
sdperi = gld.groupby(['year', 'CELLCODE'])['peri_m'].std()
sdperi = sdperi.reset_index()
sdperi.columns = ['year', 'CELLCODE', 'sdperi']
griddf = pd.merge(griddf, sdperi, on=['year', 'CELLCODE'])

# 8. Mean shape index in the grids
mean_shp = gld.groupby(['year', 'CELLCODE'])['shp_index'].mean().reset_index()
mean_shp.columns = ['year', 'CELLCODE', 'mean_shp']
griddf = pd.merge(griddf, mean_shp, on=['year', 'CELLCODE'])

# 9. Standard deviation of shape index in the grids
sd_shp = gld.groupby(['year', 'CELLCODE'])['shp_index'].std().reset_index()
sd_shp.columns = ['year', 'CELLCODE', 'sd_shp']
griddf = pd.merge(griddf, sd_shp, on=['year', 'CELLCODE'])

# 10. Mean fractal dimension in the grids
mean_fract = gld.groupby(['year', 'CELLCODE'])['fract'].mean().reset_index()
mean_fract.columns = ['year', 'CELLCODE', 'mean_fract']
griddf = pd.merge(griddf, mean_fract, on=['year', 'CELLCODE'])

# 11. Standard deviation of fractal dimension in the grids
sd_fract = gld.groupby(['year', 'CELLCODE'])['fract'].std().reset_index()
sd_fract.columns = ['year', 'CELLCODE', 'sd_fract']
griddf = pd.merge(griddf, sd_fract, on=['year', 'CELLCODE'])

griddf.head()

#############################################################
# Calculating changes in grid level aspect values over years
###############################################################
# %%
# Create table of differences over years in each grid of number of fields in grid, mean field size and mean shape index
griddf = griddf.sort_values(['CELLCODE', 'year'])  # Ensure the data is sorted by 'CELLCODE' and 'year'
griddf['MFSChng'] = griddf.groupby('CELLCODE')['mfs_ha'].diff()
griddf['MFSChng'] = griddf['MFSChng'].fillna(0)

griddf['MSIChng'] = griddf.groupby('CELLCODE')['mean_shp'].diff()
griddf['MSIChng'] = griddf['MSIChng'].fillna(0)

griddf['MfractChng'] = griddf.groupby('CELLCODE')['mean_fract'].diff()
griddf['MfractChng'] = griddf['MfractChng'].fillna(0)

griddf.head(15)


# %% check for null values in griddf
# Count the number of null values in each column
null_counts = griddf.isnull().sum()
# Print the null counts
print(null_counts)
# Filter rows with at least one null value
null_rows = griddf[griddf.isnull().any(axis=1)]
# Print the rows with null values
print(null_rows)


# %% # Identify instances where 'fields' is 1
field_1 = griddf[griddf['fields'] == 1]
# Print the result
print(field_1)
#save to csv
field_1.to_csv('reports/instances_with_field_1.csv')


# %% Descriptive statistcs of the grid level metrics by year
grid_desc_stats = griddf.groupby('year')[['fields', 'mfs_ha', 'MFSChng', 'sdfs_ha', 'mperi', \
    'sdperi', 'mean_shp', 'MSIChng', 'sd_shp', 'mean_fract', 'MfractChng', 'sd_fract']].describe()
# drop 25%, 50% and 75% columns for each metric
grid_desc_stats.to_csv('reports/statistics/grid_desc_stats.csv') 
#save to csv
grid_sums = griddf.groupby('year')[['fields', 'mfs_ha', 'sdfs_ha', 'mperi', \
    'sdperi', 'mean_shp', 'sd_shp', 'mean_fract', 'sd_fract']].sum()
grid_sums.to_csv('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen/reports/statistics/gridsums.csv') #save to csv



# %% Save to csv and pkl
#griddf.to_csv('data/interim/griddf.csv')
# griddf.to_pickle('data/interim/griddf.pkl')

# Use line plot with shaded area to show the sd of each metric across grids
# over years.
   

############################################################
# Temporal analysis: change in number of grid as number of field
# increases or decreses over years within the grids
############################################################

# %%
#import dtale
#d = dtale.show(griddf)
#d.open_browser()
# %%
#import qgrid
#qgrid_widget = qgrid.show_grid(griddf, show_toolbar=True)
#qgrid_widget


