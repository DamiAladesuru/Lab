# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m


# %% Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify
print(os.getcwd())

# %% Load pickle file
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
gld.info()    
gld.head()    


# %% ########################################
# Field/Landscape level descriptive statistics
    # total number of fields per year
    # min, max and mean value of field size, peri and shape index per year across landscape. We could have a box plot of these values across years.
gld.info()
desc_stats = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'shp_index', 'fract']].describe()
# Calculate the sum of each column
column_sums = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'shp_index', 'fract']].sum()
# Calculate the median of each column
column_medians = gld.groupby('year')[['area_ha', 'peri_m', 'par', 'shp_index', 'fract']].sum()
 #save to csv
desc_stats.to_csv('reports/statistics/ldscp_desc.csv')
column_sums.to_csv('reports/statistics/sums.csv')
column_medians.to_csv('reports/statistics/medians.csv')

# %% #######################################
# Grid level descriptive statistics
#######################################
    # total number of grids and regions within the geographic boundaries of the
    # study area
print("gridcount =", gld.groupby('year')[['CELLCODE', 'LANDKREIS']].nunique())

# Create table of year, grid id, number of fields in grid, mean field size,
# sd_fs, mean peri, sd_peri, mean shape index, sd_shape index.
griddf = gld[['year', 'CELLCODE', 'LANDKREIS']].drop_duplicates().copy()
griddf.info()

# %% Before we continue, first check if number of entries for area_m2, peri_m, shp and fract within each cellcode is thesame
counts = gld.groupby('CELLCODE')[['area_ha', 'peri_m', 'par', 'shp_index', 'fract']].count()
same_counts = (counts['area_ha'] == counts['peri_m'])\
    & (counts['area_ha'] == counts['par'])\
        & (counts['area_ha'] == counts['shp_index']) & (counts['area_ha'] == counts['fract'])
different_counts = counts[~same_counts]
different_counts

# %%
# Number of fields per grid
fields = gld.groupby(['year', 'CELLCODE'])['geometry'].count().reset_index()
fields.columns = ['year', 'CELLCODE', 'fields']
griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])

# Sum of field size per grid
fsha_sum = gld.groupby(['year', 'CELLCODE'])['area_ha'].sum().reset_index()
fsha_sum.columns = ['year', 'CELLCODE', 'fsha_sum']
griddf = pd.merge(griddf, fsha_sum, on=['year', 'CELLCODE'])

# Mean field size per grid
griddf['mfs_ha'] = (griddf['fsha_sum'] / griddf['fields'])

# Median field size per grid
griddf['midfs_ha'] = gld.groupby(['year', 'CELLCODE'])['area_ha'].median().reset_index()['area_ha']

# Standard deviation of field size per grid (ha)
sdfs_ha = gld.groupby(['year', 'CELLCODE'])['area_ha'].std().reset_index()
sdfs_ha.columns = ['year', 'CELLCODE', 'sdfs_ha']
griddf = pd.merge(griddf, sdfs_ha, on=['year', 'CELLCODE'])

# Sum of field peri per grid
peri_sum = gld.groupby(['year', 'CELLCODE'])['peri_m'].sum().reset_index()
peri_sum.columns = ['year', 'CELLCODE', 'peri_sum']
griddf = pd.merge(griddf, peri_sum, on=['year', 'CELLCODE'])

# Mean perimeter per grids
griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])

# Median perimeter per grid
griddf['midperi'] = gld.groupby(['year', 'CELLCODE'])['peri_m'].median().reset_index()['peri_m']

# Standard deviation of perimeter per grids
sdperi = gld.groupby(['year', 'CELLCODE'])['peri_m'].std().reset_index()
sdperi.columns = ['year', 'CELLCODE', 'sdperi']
griddf = pd.merge(griddf, sdperi, on=['year', 'CELLCODE'])

######################################################################
#Shape
######################################################################
# simple perimeter to area ratio
# Sum of Par per grid
par_sum = gld.groupby(['year', 'CELLCODE'])['par'].sum().reset_index()
par_sum.columns = ['year', 'CELLCODE', 'par_sum']
griddf = pd.merge(griddf, par_sum, on=['year', 'CELLCODE'])

# Mean Par per grid
griddf['mean_par'] = (griddf['par_sum'] / griddf['fields'])

# p/a ratio of grid as sum of peri divided by sum of area per grid
#griddf['grid_par'] = (griddf['peri_sum'] / griddf['fsha_sum']) #compare to mean par

# Median par per grid
griddf['midpar'] = gld.groupby(['year', 'CELLCODE'])['par'].median().reset_index()['par']

# %% Standard deviation of par per grids
sdpar = gld.groupby(['year', 'CELLCODE'])['par'].std().reset_index()
sdpar.columns = ['year', 'CELLCODE', 'sdpar']
griddf = pd.merge(griddf, sdpar, on=['year', 'CELLCODE'])

# corrected perimeter to area ratio
# Sum of cpar per grid
cpar_sum = gld.groupby(['year', 'CELLCODE'])['cpar'].sum().reset_index()
cpar_sum.columns = ['year', 'CELLCODE', 'cpar_sum']
griddf = pd.merge(griddf, cpar_sum, on=['year', 'CELLCODE'])

# p/a ratio of grid as sum of peri divided by sum of area per grid
#def gridcpa(p, a):
    #GCPA = (0.282*p)/(m.sqrt(a))
    #return GCPA
#griddf['grid_cpar'] = griddf.apply(lambda row: gridcpa(row['peri_sum'], row['fsha_sum']), axis=1)
#compare to mean cpar and grid_par

# Mean cpar per grid
griddf['mean_cpar'] = (griddf['cpar_sum'] / griddf['fields'])

# Median cpar per grid
griddf['midcpar'] = gld.groupby(['year', 'CELLCODE'])['cpar'].median().reset_index()['cpar']

# Standard deviation of cpar per grids
sd_cpar = gld.groupby(['year', 'CELLCODE'])['cpar'].std().reset_index()
sd_cpar.columns = ['year', 'CELLCODE', 'sd_cpar']
griddf = pd.merge(griddf, sd_cpar, on=['year', 'CELLCODE'])

# shape index
# Sum of shape index per grid
shp_sum = gld.groupby(['year', 'CELLCODE'])['shp_index'].sum().reset_index()
shp_sum.columns = ['year', 'CELLCODE', 'shp_sum']
griddf = pd.merge(griddf, shp_sum, on=['year', 'CELLCODE'])

# Mean shape index per grid
mean_shp = gld.groupby(['year', 'CELLCODE'])['shp_index'].mean().reset_index()
mean_shp.columns = ['year', 'CELLCODE', 'mean_shp']
griddf = pd.merge(griddf, mean_shp, on=['year', 'CELLCODE'])

# Median shape index per grid
griddf['midshp'] = gld.groupby(['year', 'CELLCODE'])['shp_index'].median().reset_index()['shp_index']

# Standard deviation of shape index per grid
sd_shp = gld.groupby(['year', 'CELLCODE'])['shp_index'].std().reset_index()
sd_shp.columns = ['year', 'CELLCODE', 'sd_shp']
griddf = pd.merge(griddf, sd_shp, on=['year', 'CELLCODE'])

# Sum of mean fractal dimension per grid
fract_sum = gld.groupby(['year', 'CELLCODE'])['fract'].sum().reset_index()
fract_sum.columns = ['year', 'CELLCODE', 'fract_sum']
fract_sum.head()
griddf = pd.merge(griddf, fract_sum, on=['year', 'CELLCODE'])

# Mean fractal dimension per grid
mean_fract = gld.groupby(['year', 'CELLCODE'])['fract'].mean().reset_index()
mean_fract.columns = ['year', 'CELLCODE', 'mean_fract']
griddf = pd.merge(griddf, mean_fract, on=['year', 'CELLCODE'])

# Median fractal dimension per grid
griddf['midfract'] = gld.groupby(['year', 'CELLCODE'])['fract'].median().reset_index()['fract']

# Standard deviation of fractal dimension in the grids
sd_fract = gld.groupby(['year', 'CELLCODE'])['fract'].std().reset_index()
sd_fract.columns = ['year', 'CELLCODE', 'sd_fract']
griddf = pd.merge(griddf, sd_fract, on=['year', 'CELLCODE'])

griddf.head()

#############################################################
# Calculating changes in grid level aspect values over years
###############################################################
# %%
# Create columns of differences over years in each grid of number of fields in grid, mean field size and mean shape index
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


# %% Descriptive statistics of the grid level metrics by year
grid_desc_stats = griddf.groupby('year')[['fields', 'mfs_ha', 'fs_sum', 'MFSChng', 'mperi', \
    'mean_shp', 'shp_sum', 'MSIChng', 'mean_fract', 'fract_sum', 'MfractChng']].describe()
# drop 25%, 50% and 75% columns for each metric
grid_desc_stats.to_csv('reports/statistics/grid/grid_desc_stats.csv') 
# save to csv
grid_sums = griddf.groupby('year')[['fields', 'mfs_ha', 'fs_sum', 'MFSChng', 'mperi', \
    'mean_shp', 'shp_sum', 'MSIChng', 'mean_fract', 'fract_sum', 'MfractChng']].sum()
grid_sums.to_csv('reports/statistics/grid/gridsums.csv') #save to csv


# %% ########################################
# Load Germany grid_landkreise to obtain the geometry
# %% Load pickle file
with open('data/interim/grid_landkreise.pkl', 'rb') as f:
    geom = pickle.load(f)
geom.info()    
geom.crs

# %% Join grid to griddf using cellcode
gridgdf = griddf.merge(geom, on='CELLCODE')
gridgdf.info()
gridgdf.head()

# Convert the DataFrame to a GeoDataFrame
gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')

#%% Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

# %% Save to csv and pkl
#griddf.to_csv('data/interim/griddf.csv')
gridgdf.to_pickle('data/interim/gridgdf.pkl')

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
