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
ldscp1_desc_stats = gld.groupby('year')[['area_m2', 'peri_m', 'shp_index',\
    'fract']].describe()
ldscp1_desc_stats.to_csv('reports/statistics/ldscp1_desc_stats.csv') 
#save to csv

#######################################
# %% Grid level descriptive statistics
#######################################
    # total number of grids within the geographic boundaries of the
    # study area
print("gridcount =", gld['CELLCODE'].nunique())

griddf = gld[['year', 'CELLCODE']].copy() # Create a dataframe with columns year, grid id
    # mean and sd distribution across grids over years.
# %%
# 1. Number of fields per grid
fields = gld.groupby(['year', 'CELLCODE'])['area_m2'].count().reset_index()
fields.columns = ['year', 'CELLCODE', 'fields']
fields.head()
griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])
griddf.head()

# %%2. Sum of field size per grid
sum_fs = gld.groupby(['year', 'CELLCODE'])['area_m2'].sum().reset_index()
sum_fs.columns = ['year', 'CELLCODE', 'sum_fs']
sum_fs.head()
griddf = pd.merge(griddf, sum_fs, on=['year', 'CELLCODE'])
griddf.head()

# %% 2. Mean field size in the grid
mfs1 = gld.groupby(['year', 'CELLCODE'])['area_m2'].mean().reset_index() # regular mean formular
mfs1.head()
# %%MPS formular in hectares
#mfs = (gld.groupby(['year', 'CELLCODE'])['area_m2'].sum() / gld.groupby(['year', 'CELLCODE'])['area_m2'].count())*(1/10000)
#mfs = mfs.reset_index()
mfs = (gld.groupby(['year', 'CELLCODE'])['area_m2'].sum() / griddf['fields'])*(1/10000)
mfs = mfs.reset_index()
mfs.columns = ['year', 'CELLCODE', 'mfs']
mfs.head()
# %% Standard deviation of field size in the grid
griddf['SD_FS1'] = gld.groupby(['year', 'CELLCODE'])['area_m2'].std().reset_index()
griddf['SD_FS'] = m.sqrt(/gld.groupby(['year', 'CELLCODE'])['area_m2'].count())
griddf.head()
# Create table of year, grid id, number of fields in grid, mean field size,
# sd_fs, mean peri, sd_peri, mean shape index, sd_shape index.
# Use line plot with shaded area to show the sd of each metric across grids
# over years.
   

############################################################
# Temporal analysis: change in number of grid as number of field
# increases or decreses over years within the grids
############################################################

# %%
# 1. Number of fields per grid
fields_df = gld.groupby(['year', 'CELLCODE'])['area_m2'].count().reset_index()
fields_df.columns = ['year', 'CELLCODE', 'fields']
griddf = pd.merge(griddf, fields_df, on=['year', 'CELLCODE'])

# 2. Mean field size in the grid
mean_df = gld.groupby(['year', 'CELLCODE'])['area_m2'].mean().reset_index()
mean_df.columns = ['year', 'CELLCODE', 'MFS1']
griddf = pd.merge(griddf, mean_df, on=['year', 'CELLCODE'])

# And so on for the other calculations...
# %%
missing_cellcode = gld[gld['CELLCODE'].isnull()]
# %%
missing_cellcode
# %% count of missing cellcode
missing_cellcode['year'].value_counts()

# %%
