# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m
from functools import reduce # For merging multiple DataFrames


# %% Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')


# %% Load pickle file and merge kulturcode_mastermap
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
gld.info()    
gld.head()    

kulturcode_mastermap = pd.read_csv('reports/Kulturcode/kulturcode_mastermap.csv', encoding='windows-1252')

# %% merge the kulturcode_mastermap with data dataframe on 'kulturcode' column
gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
gld = gld.drop(columns=['kulturart', 'kulturart_sourceyear'])
gld.info()

# subset gld for where category1 == environmental
gld_envi = gld[gld['category1'] == 'environmental']
gld_envi.info()

# %% Save to pkl
#gld_envi.to_pickle('data/interim/gld_envi.pkl')
#gld_envi.info()


# %% General data descriptive statistics
gen_stats = gld_envi[['year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
gen_stats.to_csv('reports/statistics/envi_gen_stats.csv') #save to csv

# stats per year
yearly_genstats = gld_envi.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
yearly_genstats.to_csv('reports/statistics/envi_yearly_genstats.csv')# Save to CSV
# Calculate the sum of each column
column_sums = gld_envi.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].sum()
column_sums['stat'] = 'sum'
# Calculate the median of each column
column_medians = gld_envi.groupby('year')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].median()
column_medians['stat'] = 'median'
column_sums.to_csv('reports/statistics/envi_sums.csv')
column_medians.to_csv('reports/statistics/envi_medians.csv')
# Combine all statistics into a single DataFrame
yearly_stats = pd.concat([yearly_genstats, column_sums, column_medians])
yearly_stats.to_csv('reports/statistics/envi_yearly_stats.csv')# Save to CSV
# edit the yearly_stats file to make it more readable


# %% #######################################
# Grid level descriptive statistics
#######################################
    # total number of grids and regions within the geographic boundaries of the
    # study area
print("gridcount =", gld_envi.groupby('year')[['CELLCODE', 'LANDKREIS']].nunique())

# %% Before we continue, first check if number of entries for area_m2, peri_m, shp and fract within each cellcode is thesame
counts = gld_envi.groupby('CELLCODE')[['area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].count()
same_counts = (counts['area_ha'] == counts['peri_m'])\
    & (counts['area_ha'] == counts['par']) & (counts['area_ha'] == counts['cpar'])\
        & (counts['area_ha'] == counts['shp_index']) & (counts['area_ha'] == counts['fract'])
different_counts = counts[~same_counts]
different_counts

# %%
# Create df of year, grid id, landkreis, number of fields in grid, diversity of 
# groups, mean field size, sd_fs, mean peri, sd_peri, mean shape index, sd_shape index.
griddf = gld_envi[['year', 'LANDKREIS', 'CELLCODE']].drop_duplicates().copy()
griddf.info()

# %%
# Number of fields per grid
fields = gld_envi.groupby(['year', 'CELLCODE'])['geometry'].count().reset_index()
fields.columns = ['year', 'CELLCODE', 'fields']
griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])

# Number of unique groups per grid
group_count = gld_envi.groupby(['year', 'CELLCODE'])['Gruppe'].nunique().reset_index()
group_count.columns = ['year', 'CELLCODE', 'group_count']
griddf = pd.merge(griddf, group_count, on=['year', 'CELLCODE'])

# List of unique groups per grid
groups = gld_envi.groupby(['year', 'CELLCODE'])['Gruppe'].unique().reset_index()
groups.columns = ['year', 'CELLCODE', 'groups']
griddf = pd.merge(griddf, groups, on=['year', 'CELLCODE'])
griddf.head()

# Sum of field size per grid
fsha_sum = gld_envi.groupby(['year', 'CELLCODE'])['area_ha'].sum().reset_index()
fsha_sum.columns = ['year', 'CELLCODE', 'fsha_sum']
griddf = pd.merge(griddf, fsha_sum, on=['year', 'CELLCODE'])

# Mean field size per grid
griddf['mfs_ha'] = (griddf['fsha_sum'] / griddf['fields'])

# Median field size per grid
griddf['midfs_ha'] = gld_envi.groupby(['year', 'CELLCODE'])['area_ha'].median().reset_index()['area_ha']

# Standard deviation of field size per grid (ha)
sdfs_ha = gld_envi.groupby(['year', 'CELLCODE'])['area_ha'].std().reset_index()
sdfs_ha.columns = ['year', 'CELLCODE', 'sdfs_ha']
griddf = pd.merge(griddf, sdfs_ha, on=['year', 'CELLCODE'])

# Sum of field peri per grid
peri_sum = gld_envi.groupby(['year', 'CELLCODE'])['peri_m'].sum().reset_index()
peri_sum.columns = ['year', 'CELLCODE', 'peri_sum']
griddf = pd.merge(griddf, peri_sum, on=['year', 'CELLCODE'])

# Mean perimeter per grids
griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])

# Median perimeter per grid
griddf['midperi'] = gld_envi.groupby(['year', 'CELLCODE'])['peri_m'].median().reset_index()['peri_m']

# Standard deviation of perimeter per grids
sdperi = gld_envi.groupby(['year', 'CELLCODE'])['peri_m'].std().reset_index()
sdperi.columns = ['year', 'CELLCODE', 'sdperi']
griddf = pd.merge(griddf, sdperi, on=['year', 'CELLCODE'])

######################################################################
#Shape
######################################################################
# simple perimeter to area ratio
# Sum of Par per grid
par_sum = gld_envi.groupby(['year', 'CELLCODE'])['par'].sum().reset_index()
par_sum.columns = ['year', 'CELLCODE', 'par_sum']
griddf = pd.merge(griddf, par_sum, on=['year', 'CELLCODE'])

# Mean Par per grid
griddf['mean_par'] = (griddf['par_sum'] / griddf['fields'])

# p/a ratio of grid as sum of peri divided by sum of area per grid
#griddf['grid_par'] = (griddf['peri_sum'] / griddf['fsha_sum']) #compare to mean par

# Median par per grid
griddf['midpar'] = gld_envi.groupby(['year', 'CELLCODE'])['par'].median().reset_index()['par']

# Standard deviation of par per grids
sdpar = gld_envi.groupby(['year', 'CELLCODE'])['par'].std().reset_index()
sdpar.columns = ['year', 'CELLCODE', 'sdpar']
griddf = pd.merge(griddf, sdpar, on=['year', 'CELLCODE'])

# corrected perimeter to area ratio
# Sum of cpar per grid
cpar_sum = gld_envi.groupby(['year', 'CELLCODE'])['cpar'].sum().reset_index()
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
griddf['midcpar'] = gld_envi.groupby(['year', 'CELLCODE'])['cpar'].median().reset_index()['cpar']

# Standard deviation of cpar per grids
sd_cpar = gld_envi.groupby(['year', 'CELLCODE'])['cpar'].std().reset_index()
sd_cpar.columns = ['year', 'CELLCODE', 'sd_cpar']
griddf = pd.merge(griddf, sd_cpar, on=['year', 'CELLCODE'])

# shape index
# Sum of shape index per grid
shp_sum = gld_envi.groupby(['year', 'CELLCODE'])['shp_index'].sum().reset_index()
shp_sum.columns = ['year', 'CELLCODE', 'shp_sum']
griddf = pd.merge(griddf, shp_sum, on=['year', 'CELLCODE'])

# Mean shape index per grid
mean_shp = gld_envi.groupby(['year', 'CELLCODE'])['shp_index'].mean().reset_index()
mean_shp.columns = ['year', 'CELLCODE', 'mean_shp']
griddf = pd.merge(griddf, mean_shp, on=['year', 'CELLCODE'])

# Median shape index per grid
griddf['midshp'] = gld_envi.groupby(['year', 'CELLCODE'])['shp_index'].median().reset_index()['shp_index']

# Standard deviation of shape index per grid
sd_shp = gld_envi.groupby(['year', 'CELLCODE'])['shp_index'].std().reset_index()
sd_shp.columns = ['year', 'CELLCODE', 'sd_shp']
griddf = pd.merge(griddf, sd_shp, on=['year', 'CELLCODE'])

# Sum of mean fractal dimension per grid
fract_sum = gld_envi.groupby(['year', 'CELLCODE'])['fract'].sum().reset_index()
fract_sum.columns = ['year', 'CELLCODE', 'fract_sum']
fract_sum.head()
griddf = pd.merge(griddf, fract_sum, on=['year', 'CELLCODE'])

# Mean fractal dimension per grid
mean_fract = gld_envi.groupby(['year', 'CELLCODE'])['fract'].mean().reset_index()
mean_fract.columns = ['year', 'CELLCODE', 'mean_fract']
griddf = pd.merge(griddf, mean_fract, on=['year', 'CELLCODE'])

# Median fractal dimension per grid
griddf['midfract'] = gld_envi.groupby(['year', 'CELLCODE'])['fract'].median().reset_index()['fract']

# Standard deviation of fractal dimension in the grids
sd_fract = gld_envi.groupby(['year', 'CELLCODE'])['fract'].std().reset_index()
sd_fract.columns = ['year', 'CELLCODE', 'sd_fract']
griddf = pd.merge(griddf, sd_fract, on=['year', 'CELLCODE'])

griddf.head()

# %% subset selected columns in different dfs for easier handling
griddf_sums = griddf.filter(['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count', \
    'fsha_sum', 'peri_sum', 'par_sum', 'cpar_sum', 'shp_sum', 'fract_sum'], axis=1)

griddf_means = griddf.filter(['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count', \
    'mfs_ha', 'mperi', 'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract'], axis=1)

griddf_medians = griddf.filter(['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count', \
    'midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp', 'midfract'], axis=1)

griddf_stds = griddf.filter(['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count', \
    'sdfs_ha', 'sdperi', 'sdpar', 'sd_cpar', 'sd_shp', 'sd_fract'], axis=1)
griddf_stds.to_csv('reports/statistics/grid/envi_griddf_stds.csv', encoding='windows-1252') #save to csv

#############################################################
# Calculating absolute and reative changes in grid level aspect values over years
###############################################################
# %% Change for griddf_counts
# # Filter the DataFrame to get the values for the year 2012
griddf_counts = griddf.filter(['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count'], axis=1)
griddf_counts_2012 = griddf_counts[griddf_counts['year'] == 2012].set_index('CELLCODE')
# Merge the 2012 values back to the original DataFrame
griddf_counts = griddf_counts.merge(griddf_counts_2012, on='CELLCODE', suffixes=('', '_2012'))
# Ensure the data is sorted by 'CELLCODE' and 'year'
griddf_counts = griddf_counts.sort_values(['CELLCODE', 'year'])
griddf_counts = griddf_counts.assign(
    FieldsChng=griddf_counts.groupby('CELLCODE')['fields'].diff().fillna(0),
    FieldsChng12=griddf_counts['fields'] - griddf_counts['fields_2012'],
    GroupsChng=griddf_counts.groupby('CELLCODE')['group_count'].diff().fillna(0),
    GroupsChng12=griddf_counts['group_count'] - griddf_counts['group_count_2012']
)   
# Drop the columns with '_2012'
griddf_counts = griddf_counts.drop(columns=['year_2012', 'LANDKREIS_2012', 'fields_2012', 'group_count_2012'])
griddf_counts.info()

# %% Change for griddf_sums
griddf_sums_2012 = griddf_sums[griddf_sums['year'] == 2012].set_index('CELLCODE')
griddf_sums = griddf_sums.merge(griddf_sums_2012, on='CELLCODE', suffixes=('', '_2012'))
griddf_sums = griddf_sums.sort_values(['CELLCODE', 'year'])
griddf_sums = griddf_sums.assign(
    FSsumChng=griddf_sums.groupby('CELLCODE')['fsha_sum'].diff().fillna(0),
    FSsumChng12=griddf_sums['fsha_sum'] - griddf_sums['fsha_sum_2012'],
    PsumChng=griddf_sums.groupby('CELLCODE')['peri_sum'].diff().fillna(0),
    PsumChng12=griddf_sums['peri_sum'] - griddf_sums['peri_sum_2012'],
    PAsumChng=griddf_sums.groupby('CELLCODE')['par_sum'].diff().fillna(0),
    PAsumChng12=griddf_sums['par_sum'] - griddf_sums['par_sum_2012'],
    CPAsumChng=griddf_sums.groupby('CELLCODE')['cpar_sum'].diff().fillna(0),
    CPAsumChng12=griddf_sums['cpar_sum'] - griddf_sums['cpar_sum_2012'],
    SIsumChng=griddf_sums.groupby('CELLCODE')['shp_sum'].diff().fillna(0),
    SIsumChng12=griddf_sums['shp_sum'] - griddf_sums['shp_sum_2012'],
    fractsumChng=griddf_sums.groupby('CELLCODE')['fract_sum'].diff().fillna(0),
    fractsumChng12=griddf_sums['fract_sum'] - griddf_sums['fract_sum_2012']
)

# drop the columns with '_2012'
griddf_sums = griddf_sums.drop(columns=['year_2012', 'LANDKREIS_2012', 'fields_2012', 'group_count_2012',\
    'fsha_sum_2012', 'peri_sum_2012', 'par_sum_2012', 'cpar_sum_2012', 'shp_sum_2012', 'fract_sum_2012'])

griddf_sums.info()
griddf_sums.to_csv('reports/statistics/grid/envi_griddf_sums.csv', encoding='windows-1252') #save to csv 

# %% Change for griddf_means   
griddf_2012_means = griddf_means[griddf_means['year'] == 2012].set_index('CELLCODE')
griddf_means = griddf_means.merge(griddf_2012_means, on='CELLCODE', suffixes=('', '_2012'))
griddf_means = griddf_means.sort_values(['CELLCODE', 'year'])
griddf_means = griddf_means.assign(
    MFSChng=griddf_means.groupby('CELLCODE')['mfs_ha'].diff().fillna(0),
    MFSChng12=griddf_means['mfs_ha'] - griddf_means['mfs_ha_2012'],
    MperiChng=griddf_means.groupby('CELLCODE')['mperi'].diff().fillna(0),
    MperiChng12=griddf_means['mperi'] - griddf_means['mperi_2012'],
    MPAChng=griddf_means.groupby('CELLCODE')['mean_par'].diff().fillna(0),
    MPAChng12=griddf_means['mean_par'] - griddf_means['mean_par_2012'],
    MCPAChng=griddf_means.groupby('CELLCODE')['mean_cpar'].diff().fillna(0),
    MCPAChng12=griddf_means['mean_cpar'] - griddf_means['mean_cpar_2012'],
    MSIChng=griddf_means.groupby('CELLCODE')['mean_shp'].diff().fillna(0),
    MSIChng12=griddf_means['mean_shp'] - griddf_means['mean_shp_2012'],
    MfractChng=griddf_means.groupby('CELLCODE')['mean_fract'].diff().fillna(0),
    MfractChng12=griddf_means['mean_fract'] - griddf_means['mean_fract_2012']
)

# drop the columns with '_2012'
griddf_means = griddf_means.drop(columns=['year_2012', 'LANDKREIS_2012', 'fields_2012', 'group_count_2012',\
    'mfs_ha_2012','mperi_2012', 'mean_par_2012', 'mean_cpar_2012', 'mean_shp_2012', 'mean_fract_2012'])
griddf_means.info()
griddf_means.to_csv('reports/statistics/grid/envi_griddf_means.csv', encoding='windows-1252') #save to csv

# %% Change for griddf_medians
griddf_2012_medians = griddf_medians[griddf_medians['year'] == 2012].set_index('CELLCODE')
griddf_medians = griddf_medians.merge(griddf_2012_medians, on='CELLCODE', suffixes=('', '_2012'))
griddf_medians = griddf_medians.sort_values(['CELLCODE', 'year'])
griddf_medians = griddf_medians.assign(
    MidFSChng=griddf_medians.groupby('CELLCODE')['midfs_ha'].diff().fillna(0),
    MidFSChng12=griddf_medians['midfs_ha'] - griddf_medians['midfs_ha_2012'],
    MdperiChng=griddf_medians.groupby('CELLCODE')['midperi'].diff().fillna(0),
    MdperiChng12=griddf_medians['midperi'] - griddf_medians['midperi_2012'],
    MidPAChng=griddf_medians.groupby('CELLCODE')['midpar'].diff().fillna(0),
    MidPAChng12=griddf_medians['midpar'] - griddf_medians['midpar_2012'],
    MidCPAChng=griddf_medians.groupby('CELLCODE')['midcpar'].diff().fillna(0),
    MidCPAChng12=griddf_medians['midcpar'] - griddf_medians['midcpar_2012'],
    MidSIChng=griddf_medians.groupby('CELLCODE')['midshp'].diff().fillna(0),
    MidSIChng12=griddf_medians['midshp'] - griddf_medians['midshp_2012'],
    MidfractChng=griddf_medians.groupby('CELLCODE')['midfract'].diff().fillna(0),
    MidfractChng12=griddf_medians['midfract'] - griddf_medians['midfract_2012']
)
# drop the columns with '_2012'
griddf_medians = griddf_medians.drop(columns=['year_2012', 'LANDKREIS_2012', 'fields_2012',\
    'group_count_2012', 'midfs_ha_2012', 'midperi_2012', 'midpar_2012', 'midcpar_2012', 'midshp_2012', 'midfract_2012'])

griddf_medians.info()
griddf_medians.to_csv('reports/statistics/grid/envi_griddf_medians.csv', encoding='windows-1252') #save to csv

# %%
# Merge the DataFrames to create an extended grid level DataFrame
dfs = [griddf_counts, griddf_sums, griddf_means, griddf_medians, griddf_stds]
griddf_ext = reduce(lambda left, right: pd.merge(left, right, on=['year', 'LANDKREIS', 'CELLCODE', 'fields', 'group_count']), dfs)

##########################################################################################################
# Calculating descriptive statistics for the grid level metrics
##########################################################################################################
# %% Mean statistics of the grid level metrics by year
gridext_means = griddf_ext.groupby('year')[['fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp', 'midfract'
]].mean()
# save to csv
gridext_means.to_csv('reports/statistics/grid/envi_gridext_means.csv')

gridext_means_chng = griddf_ext.groupby('year')[['FieldsChng', 'FieldsChng12', 'GroupsChng', 'GroupsChng12',\
    'FSsumChng', 'FSsumChng12', 'PsumChng', 'PsumChng12', 'MFSChng', 'MFSChng12', 'MperiChng', 'MperiChng12',\
        'MPAChng', 'MPAChng12', 'MCPAChng', 'MCPAChng12', 'MSIChng', 'MSIChng12', 'MfractChng', 'MfractChng12',\
            'MidFSChng', 'MidFSChng12', 'MdperiChng', 'MdperiChng12', 'MidPAChng', 'MidPAChng12',\
                'MidCPAChng', 'MidCPAChng12', 'MidSIChng', 'MidSIChng12', 'MidfractChng', 'MidfractChng12']].mean()
# save to csv
gridext_means_chng.to_csv('reports/statistics/grid/gridextchng_means.csv')

# %% Median statistics of the grid level metrics by year
gridext_medians = griddf_ext.groupby('year')[['fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp',\
        'midfract']].median()
# save to csv
gridext_medians.to_csv('reports/statistics/grid/envi_gridext_medians.csv')

gridext_medians_chng = griddf_ext.groupby('year')[['FieldsChng', 'FieldsChng12', 'GroupsChng', 'GroupsChng12',\
    'FSsumChng', 'FSsumChng12', 'PsumChng', 'PsumChng12', 'MFSChng', 'MFSChng12', 'MperiChng', 'MperiChng12',\
        'MPAChng', 'MPAChng12', 'MCPAChng', 'MCPAChng12', 'MSIChng', 'MSIChng12', 'MfractChng', 'MfractChng12',\
            'MidFSChng', 'MidFSChng12', 'MdperiChng', 'MdperiChng12', 'MidPAChng', 'MidPAChng12',\
                'MidCPAChng', 'MidCPAChng12', 'MidSIChng', 'MidSIChng12', 'MidfractChng', 'MidfractChng12']].median()
# save to csv
gridext_means_chng.to_csv('reports/statistics/grid/envi_gridextchng_medians.csv')

# %% Quantile statistics of the grid level metrics by year
gridext_20 = griddf_ext.groupby('year')[['fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp',\
        'midfract']].quantile(0.25)
# save to csv
gridext_20.to_csv('reports/statistics/grid/envi_gridext_20.csv')

gridext_50 = griddf_ext.groupby('year')[['fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp',\
        'midfract']].quantile(0.5)
# save to csv
gridext_50.to_csv('reports/statistics/grid/envi_gridext_50.csv')

gridext_75 = griddf_ext.groupby('year')[['fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar', 'midshp',\
        'midfract']].quantile(0.75)
# save to csv
gridext_75.to_csv('reports/statistics/grid/envi_gridext_75.csv')

# %% Mean statistics of the grid level metrics all years
means = griddf_ext[['year', 'fields', 'group_count', 'fsha_sum', 'peri_sum', 'mfs_ha', 'mperi',\
    'mean_par', 'mean_cpar', 'mean_shp', 'mean_fract','midfs_ha', 'midperi', 'midpar', 'midcpar',\
        'midshp', 'midfract', 'FieldsChng', 'FieldsChng12', 'GroupsChng', 'GroupsChng12', 'FSsumChng',\
            'FSsumChng12', 'PsumChng', 'PsumChng12', 'MFSChng', 'MFSChng12', 'MperiChng', 'MperiChng12',\
                'MPAChng', 'MPAChng12', 'MCPAChng', 'MCPAChng12', 'MSIChng', 'MSIChng12', 'MfractChng',\
                    'MfractChng12', 'MidFSChng', 'MidFSChng12', 'MdperiChng', 'MdperiChng12', 'MidPAChng',\
                        'MidPAChng12', 'MidCPAChng', 'MidCPAChng12', 'MidSIChng', 'MidSIChng12',\
                            'MidfractChng', 'MidfractChng12']].mean()

# save to csv
means.to_csv('reports/statistics/envi_means.csv')


# %% ########################################
# Load Germany grid_landkreise to obtain the geometry
# %% Load pickle file
with open('data/interim/grid_landkreise.pkl', 'rb') as f:
    geom = pickle.load(f)
geom.info()    
geom.crs

# %% Join grid to griddf using cellcode
gridgdf = griddf_ext.merge(geom, on='CELLCODE')
gridgdf.info()
gridgdf.head()

# Convert the DataFrame to a GeoDataFrame
gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')

#%% Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

# %% Save to csv and pkl
griddf.to_csv('data/interim/envi_griddf.csv', encoding='windows-1252')
gridgdf.to_pickle('data/interim/envi_gridgdf.pkl')

# Use line plot with shaded area to show the sd of each metric across grids
# over years.
   

