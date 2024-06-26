# %%
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
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

# %% Load gridgdf 
with open('data/interim/gridgdf.pkl', 'rb') as f:
    gridgdf = pickle.load(f)
gridgdf.info()    
gridgdf.head()

# %% ##############################################################
#Maximum mean fractal dimension
###################################################################
# 1. obtain cellcode of grid with maximum mean fract
CELLCODEMX = gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'CELLCODE'].values[0]
# 2. obtain year of grid with maximum mean fract
gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'year'].values[0]
# %% 3. plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLCODEMX) & (gld['year'] == 2018)].plot()
# %% 4. get list of mean_fract for all years for the gridcell with maximum mean_fract
# this allows to see how this value changed over the years
gridgdf[gridgdf['CELLCODE'] == CELLCODEMX].groupby('year')['mean_fract'].apply(list)
# %%
# obtain more stats of this grid cell
# a. fract sum of grid and year with maximum mean fract
mxa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'fract_sum'].values[0]
# b. number of field in grid with maximum mean fract
mxb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'fields'].values[0]
# c. mean field size in grid with maximum mean fract
mxc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'mfs_ha'].values[0]
print(mxa, mxb, mxc)
# d. mean shape index in grid with maximum mean fract
mxd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'mean_shp'].values[0]
print(mxa, mxb, mxc, mxd)

# %% ##############################################################
#Minimum mean fractal dimension
###################################################################
CELLCODEMN = gridgdf.loc[gridgdf['mean_fract'] == gridgdf.mean_fract.min(), 'CELLCODE'].values[0]
#obtain year of grid with minimum mean fract
gridgdf.loc[gridgdf['mean_fract'] == gridgdf.mean_fract.min(), 'year'].values[0]
# %% plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLCODEMN) & (gld['year'] == 2016)].plot()
# %% get list of mean_fract for all years for the gridcell with minimum mean_fract
gridgdf[gridgdf['CELLCODE'] == CELLCODEMN].groupby('year')['mean_fract'].apply(list)

# if cell 10kmE441N331 has the minimum mean_fract across the dataset,
# but this value is only that small in 2016, could there be other grids
# with more consistently small mean_fract or a more varying grid?
# 
# Maybe check the minimum mean_fract for all years

# %% get list of yearly min mean_fract
gridgdf.groupby('year')['mean_fract'].min()
# loop over all years to get the cellcode with minimum mean_fract
for year in gridgdf['year'].unique():
    print(year)
    print(gridgdf.loc[gridgdf['mean_fract'] == gridgdf[gridgdf['year'] == year]['mean_fract'].min(), 'CELLCODE'].values[0])
# create a df of year, minimum mean_fract and cellcode
minfract_df = gridgdf.loc[gridgdf.groupby('year')['mean_fract'].idxmin()]
minfract_df = minfract_df[['year', 'mean_fract', 'CELLCODE']]
print(minfract_df)
# save to csv
minfract_df.to_csv('reports/statistics/minfract_df.csv')

# one thing I learnt from this is that grid 10kmE441N330 actually has the
# minimum mean_Fract in most years and this grid is one at the edge
# of the study area.
###################################################################
# %% obtain more stats of CELLCODEMN in 2016
# fract sum of grid and year with minimum mean fract
mna = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMN) & (gridgdf['year'] == 2016), 'fract_sum'].values[0]
# number of field in grid with minimum mean fract
mnb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMN) & (gridgdf['year'] == 2016), 'fields'].values[0]
# mean field size in grid with maximum mean fract
mnc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMN) & (gridgdf['year'] == 2016), 'mfs_ha'].values[0]
# mean shape index in grid with maximum mean fract
mnd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMN) & (gridgdf['year'] == 2016), 'mean_shp'].values[0]
print(mna, mnb, mnc, mnd)




# %% ##############################################################
#Maximum mean shape index
###################################################################
# 1. obtain cellcode of grid with maximum mean shape index
CELLSHPMX = gridgdf.loc[gridgdf['mean_shp'] == gridgdf.mean_shp.max(), 'CELLCODE'].values[0]
# 2. obtain year of grid with maximum mean shape index
gridgdf.loc[gridgdf['mean_shp'] == gridgdf.mean_shp.max(), 'year'].values[0]
# %% 3. plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLSHPMX) & (gld['year'] == )].plot()
# %% 4. get list of mean_shp for all years for the gridcell with maximum mean shape index
# this allows to see how this value changed over the years
gridgdf[gridgdf['CELLCODE'] == CELLSHPMX].groupby('year')['mean_shp'].apply(list)
# %%
# obtain more stats of this grid cell
# a. shp sum of grid and year with maximum mean shape index
mxshpa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMX) & (gridgdf['year'] == 2018), 'shp_sum'].values[0]
# b. number of field in grid with maximum mean shape index
mxshpb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMX) & (gridgdf['year'] == 2018), 'fields'].values[0]
# c. mean field size in grid with maximum mean shape index
mxshpc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMX) & (gridgdf['year'] == 2018), 'mfs_ha'].values[0]
# d. mean fract in grid with maximum mean shape index
mxshpd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMX) & (gridgdf['year'] == 2018), 'mean_fract'].values[0]
print(mxshpa, mxshpb, mxshpc, mxshpd)

# %% ##############################################################
#Minimum mean shape index
###################################################################
CELLSHPMN = gridgdf.loc[gridgdf['mean_shp'] == gridgdf.mean_shp.min(), 'CELLCODE'].values[0]
#obtain year of grid with minimum mean shape index
gridgdf.loc[gridgdf['mean_shp'] == gridgdf.mean_shp.min(), 'year'].values[0]
# %% plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLSHPMN) & (gld['year'] == 2016)].plot()
# %% get list of mean_shp for all years for the gridcell with minimum mean_shp
gridgdf[gridgdf['CELLCODE'] == CELLSHPMN].groupby('year')['mean_shp'].apply(list)

# %% get list of yearly min mean_shp
gridgdf.groupby('year')['mean_shp'].min()
# loop over all years to get the cellcode with minimum mean_shp
for year in gridgdf['year'].unique():
    print(year)
    print(gridgdf.loc[gridgdf['mean_shp'] == gridgdf[gridgdf['year'] == year]['mean_shp'].min(), 'CELLCODE'].values[0])
# create a df of year, minimum mean_shp and cellcode
minshp_df = gridgdf.loc[gridgdf.groupby('year')['mean_shp'].idxmin()]
minshp_df = minshp_df[['year', 'mean_shp', 'CELLCODE']]
print(minshp_df)
# save to csv
minshp_df.to_csv('reports/statistics/minshp_df.csv')

# %% obtain more stats of CELLSHPMN in 2016
# fract sum of grid and year with minimum mean shape index
mnshpa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMN) & (gridgdf['year'] == 2016), 'shp_sum'].values[0]
# number of field in grid with minimum mean shape index
mnshpb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMN) & (gridgdf['year'] == 2016), 'fields'].values[0]
# mean field size in grid with maximum mean shape index
mnshpc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMN) & (gridgdf['year'] == 2016), 'mfs_ha'].values[0]
# mean fract in grid with maximum mean shape index
mnshpd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLSHPMN) & (gridgdf['year'] == 2016), 'mean_fract'].values[0]
print(mnshpa, mnshpb, mnshpc, mnshpd)



# %% ##############################################################
#Maximum mean field size
###################################################################
# 1. obtain cellcode of grid with maximum mean field size
CELLMFSMX = gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'CELLCODE'].values[0]
# 2. obtain year of grid with minimum mean field size
gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'year'].values[0]
# %% 3. plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLMFSMX) & (gld['year'] == 2018)].plot()
# %% 4. get list of mfs_ha for all years for the gridcell with maximum mfs_ha
# this allows to see how this value changed over the years
gridgdf[gridgdf['CELLCODE'] == CELLMFSMX].groupby('year')['mfs_ha'].apply(list)
# %%
# obtain more stats of this grid cell
# a. field size sum of grid and year with maximum mean field size
mxfsa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMX) & (gridgdf['year'] == 2018), 'fs_sum'].values[0]
# b. number of field in grid with maximum mean field size
mxfsb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMX) & (gridgdf['year'] == 2018), 'fields'].values[0]
# c. mean fract in grid with maximum mean field size
mxfsc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMX) & (gridgdf['year'] == 2018), 'mean_fract'].values[0]
# d. mean shp in grid with maximum mean field size
mxfsd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMX) & (gridgdf['year'] == 2018), 'mean_shp'].values[0]
print(mxfsa, mxfsb, mxfsc, mxfsd)

# %% ##############################################################
#Minimum mean field size
###################################################################
CELLMFSMN = gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.min(), 'CELLCODE'].values[0]
#obtain year of grid with minimum mean field size
gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.min(), 'year'].values[0]
# %% plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == CELLMFSMN) & (gld['year'] == 2016)].plot()
# %% get list of mean field size for all years for the gridcell with minimum mean field size
gridgdf[gridgdf['CELLCODE'] == CELLMFSMN].groupby('year')['mfs_ha'].apply(list)
# %% get list of yearly min mfs_ha
gridgdf.groupby('year')['mfs_ha'].min()
# loop over all years to get the cellcode with minimum mean field size
for year in gridgdf['year'].unique():
    print(year)
    print(gridgdf.loc[gridgdf['mfs_ha'] == gridgdf[gridgdf['year'] == year]['mfs_ha'].min(), 'CELLCODE'].values[0])
# create a df of year, minimum mfs_ha and cellcode
minmfs_df = gridgdf.loc[gridgdf.groupby('year')['mfs_ha'].idxmin()]
minmfs_df = minmfs_df[['year', 'mfs_ha', 'CELLCODE']]
print(minmfs_df)
# save to csv
minmfs_df.to_csv('reports/statistics/minmfs_df.csv')
# %% obtain more stats of CELLMFSMN in min year
# field size sum of grid and year with minimum mean field size
mnfsa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMN) & (gridgdf['year'] == 2016), 'fs_sum'].values[0]
# number of field in grid with minimum mean field size
mnfsb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMN) & (gridgdf['year'] == 2016), 'fields'].values[0]
# mean fract in grid with maximum mean field size
mnfsc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMN) & (gridgdf['year'] == 2016), 'mean_fract'].values[0]
# mean shp in grid with maximum mean field size
mnfsd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLMFSMN) & (gridgdf['year'] == 2016), 'mean_shp'].values[0]
print(mnfsa, mnfsb, mnfsc, mnfsd)
# %%
