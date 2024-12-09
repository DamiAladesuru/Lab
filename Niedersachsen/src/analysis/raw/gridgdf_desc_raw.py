# %%
import pandas as pd
import geopandas as gpd
import os
import math as m
import logging
import numpy as np
from shapely.geometry import box
import pickle
# Silence the print statements in a function call
import contextlib
import io


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.data import dataload as dl
from src.data import eca_new as eca
from src.analysis.raw import gld_desc_raw as gdr

''' This script contains functions for:
    - modifying gld to include columns for basic additional metrics and kulturcode descriptions.
    - creating griddf and gridgdf (without removing any assumed outlier and for subsamples).
    - computing descriptive statistics for gridgdf.
The functions are called in the tor_raw script
'''

# %%
def square_cpar(gld): #shape index adjusted for square fields
    gld['cpar2'] = ((0.25 * gld['peri_m']) / (gld['area_m2']**0.5))
    return gld


# %% A.
def create_griddf(gld):
    columns = ['CELLCODE', 'year', 'LANDKREIS']
    
    # 1. Extract the specified columns and drop duplicates
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")
    
    # 2. Compute mean statistics at grid level
    # for statistics, group by year and cellcode because you want to look at each year
    # and each grid cell and compute the statistics for each grid cell
    # Number of fields per grid
    fields = gld.groupby(['CELLCODE', 'year'])['geometry'].count().reset_index()
    fields.columns = ['CELLCODE', 'year', 'fields']
    griddf = pd.merge(griddf, fields, on=['CELLCODE', 'year'])

    # Number of unique groups per grid
    group_count = gld.groupby(['CELLCODE', 'year'])['Gruppe'].nunique().reset_index()
    group_count.columns = ['CELLCODE', 'year', 'group_count']
    griddf = pd.merge(griddf, group_count, on=['CELLCODE', 'year'])

    # Sum of field size per grid (m2)
    fsm2_sum = gld.groupby(['CELLCODE', 'year'])['area_m2'].sum().reset_index()
    fsm2_sum.columns = ['CELLCODE', 'year', 'fsm2_sum']
    griddf = pd.merge(griddf, fsm2_sum, on=['CELLCODE', 'year'])
    
    # Sum of field size per grid (ha)
    fsha_sum = gld.groupby(['CELLCODE', 'year'])['area_ha'].sum().reset_index()
    fsha_sum.columns = ['CELLCODE', 'year', 'fsha_sum']
    griddf = pd.merge(griddf, fsha_sum, on=['CELLCODE', 'year'])

    # Mean field size per grid
    griddf['mfs_ha'] = (griddf['fsha_sum'] / griddf['fields'])
    
    # Median field size per grid
    griddf['medfs_ha'] = gld.groupby(['CELLCODE', 'year'])['area_ha'].median().reset_index()['area_ha']

    # Sum of field perimeter per grid
    peri_sum = gld.groupby(['CELLCODE', 'year'])['peri_m'].sum().reset_index()
    peri_sum.columns = ['CELLCODE', 'year', 'peri_sum']
    griddf = pd.merge(griddf, peri_sum, on=['CELLCODE', 'year'])

    # Mean perimeter per grids
    griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])
    
    # Median perimeter per grid
    griddf['medperi'] = gld.groupby(['CELLCODE', 'year'])['peri_m'].median().reset_index()['peri_m']

    # Rate of fields per hectare of land per grid
    griddf['fields_ha'] = (griddf['fields'] / griddf['fsha_sum'])
    
    ######################################################################
    #Shape
    ######################################################################
    # perimeter to area ratio
    # Sum of par per grid
    par_sum = gld.groupby(['CELLCODE', 'year'])['par'].sum().reset_index()
    par_sum.columns = ['CELLCODE', 'year', 'par_sum']
    griddf = pd.merge(griddf, par_sum, on=['CELLCODE', 'year'])

    # Mean par per grid
    griddf['mean_par'] = (griddf['par_sum'] / griddf['fields'])
    
    # Median par per grid
    griddf['medpar'] = gld.groupby(['CELLCODE', 'year'])['par'].median().reset_index()['par']
    
    # p/a ratio of grid as sum of peri divided by sum of area per grid
    #griddf['grid_par'] = ((griddf['peri_sum'] / griddf['fsm2_sum'])) #compare to mean par 
    
    griddf = griddf.drop(columns=['par_sum', 'fsm2_sum'])
    
    return griddf


# check for duplicates in the griddf
def check_duplicates(griddf):
    duplicates = griddf[griddf.duplicated(subset=['CELLCODE', 'year'], keep=False)]
    print(f"Number of duplicates in griddf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")
            
#yearly gridcell differences and differences from first year
def calculate_yearlydiff(griddf): #yearly gridcell differences
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()
    
    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = griddf_ext.groupby('CELLCODE')[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_yearly_percdiff'] = (griddf_ext.groupby('CELLCODE')[col].diff() / griddf_ext.groupby('CELLCODE')[col].shift(1)).fillna(0) * 100
    
    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_ext = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_ext    


# %%
def calculate_diff_fromy1(griddf): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()

    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Get the first occurrence of each unique CELLCODE in the griddf_ext DataFrame. 
    y1_df = griddf_ext.groupby('CELLCODE').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['CELLCODE'] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Merge the first year values back into the original DataFrame
    griddf_ext = pd.merge(griddf_ext, y1_df, on='CELLCODE', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = griddf_ext[col] - griddf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((griddf_ext[col] - griddf_ext[f'{col}_y1']) / griddf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    griddf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_exty1 = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_exty1


# %% 
def combine_griddfs(griddf_ext, griddf_exty1):
    # Ensure the merge is based on 'CELLCODE' and 'year'
    # Select columns from griddf_exty1 that are not in griddf_ext (excluding 'CELLCODE' and 'year')
    columns_to_add = [col for col in griddf_exty1.columns if col not in griddf_ext.columns or col in ['CELLCODE', 'year']]

    # Merge the DataFrames on 'CELLCODE' and 'year', keeping the existing columns in griddf_ext
    combined_griddf = pd.merge(griddf_ext, griddf_exty1[columns_to_add], on=['CELLCODE', 'year'], how='left')
    
    return combined_griddf


# %%
def to_gdf(griddf_ext):
    # Load Germany grid_landkreise to obtain the geometry
    with open('data/interim/grid_landkreise.pkl', 'rb') as f:
        geom = pickle.load(f)
    geom.info()
    
    gridgdf = griddf_ext.merge(geom, on='CELLCODE')
    # Convert the DataFrame to a GeoDataFrame
    gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')
    # Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
    gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
    gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

    
    return gridgdf


def create_gridgdf_raw(gridfile_suf='', t=100, apply_t=False, gld_file='data/interim/gld_wtkc.pkl'):
    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dynamically refer to filename based on parameters
    if gridfile_suf:
        gridgdf_filename = os.path.join(output_dir, f'gridgdf_raw_{gridfile_suf}.pkl')
    else:
        gridgdf_filename = os.path.join(output_dir, 'gridgdf_raw.pkl')

    # Load gld, applying threshold t filtering only if specified
    gld_ext = gdr.adjust_gld(t=t, filename=gld_file,
                             apply_t=apply_t)


    if os.path.exists(gridgdf_filename):
        gridgdf_raw = pd.read_pickle(gridgdf_filename)
        print(f"Loaded gridgdf from {gridgdf_filename}")
    else:
        griddf = create_griddf(gld_ext)
        dupli = check_duplicates(griddf)
        # calculate differences
        griddf_ydiff = calculate_yearlydiff(griddf)
        griddf_exty1 = calculate_diff_fromy1(griddf)
        griddf_ext = combine_griddfs(griddf_ydiff, griddf_exty1)
        
        # Check for infinite values in all columns
        for column in griddf_ext.columns:
            infinite_values = griddf_ext[column].isin([np.inf, -np.inf])
            print(f"Infinite values present in {column}:", infinite_values.any())

            # Optionally, print the rows with infinite values
            if infinite_values.any():
                print(f"Rows with infinite values in {column}:")
                print(griddf_ext[infinite_values])

            # Handle infinite values by replacing them with NaN
            griddf_ext[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        gridgdf_raw = to_gdf(griddf_ext)
        gridgdf_raw.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")

    return gld_ext, gridgdf_raw


# %% B.
#########################################################################
# compute mean and median for columns in gridgdf. save the results to a csv file
def desc_grid(gridgdf):
    def compute_grid_allyear_stats(gridgdf):
        # 1. Compute general all year data descriptive statistics
        grid_allyears_stats = gridgdf.select_dtypes(include='number').describe()
        # Add a column to indicate the type of statistic
        grid_allyears_stats['statistic'] = grid_allyears_stats.index
        # Reorder columns to place 'statistic' at the front
        grid_allyears_stats = grid_allyears_stats[['statistic'] + list(grid_allyears_stats.columns[:-1])]
        
        # Save the descriptive statistics to a CSV file
        output_dir = 'reports/statistics'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'grid_allyears_stats.csv')
        if not os.path.exists(filename):
            grid_allyears_stats.to_csv(filename, index=False)
            print(f"Saved gen_stats to {filename}")
        
        return grid_allyears_stats
    grid_allyears_stats = compute_grid_allyear_stats(gridgdf)
    
    def compute_grid_year_average(gridgdf):
        # 2. Group by 'year' and calculate useful stats across grids
        grid_yearly_stats = gridgdf.groupby('year').agg(
            fields_sum=('fields', 'sum'),
            fields_mean=('fields', 'mean'),
            fields_std = ('fields', 'std'),
            fields_av_yearly_diff=('fields_yearly_diff', 'mean'),
            fields_adiff_y1=('fields_diff_from_y1', 'mean'),
            fields_apercdiff_y1=('fields_percdiff_to_y1', 'mean'),
                    
            group_count_mean=('group_count', 'mean'),
            group_count_av_yearly_diff=('group_count_yearly_diff', 'mean'),
            group_count_adiff_y1=('group_count_diff_from_y1', 'mean'),
            group_count_apercdiff_y1=('group_count_percdiff_to_y1', 'mean'),

            fsha_sum_sum=('fsha_sum', 'sum'),
            fsha_sum_mean=('fsha_sum', 'mean'),
            fsha_sum_std = ('fsha_sum', 'std'),
            fsha_sum_av_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
            fsha_sum_adiff_y1=('fsha_sum_diff_from_y1', 'mean'),
            fsha_sum_apercdiff_y1=('fsha_sum_percdiff_to_y1', 'mean'),

            mfs_ha_mean=('mfs_ha', 'mean'),
            mfs_ha_std=('mfs_ha', 'std'),
            mfs_ha_av_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
            mfs_ha_adiff_y1=('mfs_ha_diff_from_y1', 'mean'),
            mfs_ha_apercdiff_y1=('mfs_ha_percdiff_to_y1', 'mean'),

            mperi_mean=('mperi', 'mean'), #averge region's mean perimeter
            mperi_std = ('mperi', 'std'),
            mperi_av_yearly_diff=('mperi_yearly_diff', 'mean'),
            mperi_adiff_y1=('mperi_diff_from_y1', 'mean'),
            mperi_apercdiff_y1=('mperi_percdiff_to_y1', 'mean'),

            mean_par_mean=('mean_par', 'mean'),
            mean_par_std=('mean_par', 'std'),
            mean_par_av_yearly_diff=('mean_par_yearly_diff', 'mean'),
            mean_par_adiff_y1=('mean_par_diff_from_y1', 'mean'),
            mean_par_apercdiff_y1=('mean_par_percdiff_to_y1', 'mean'),
            
            
            fields_ha_mean=('fields_ha', 'mean'),
            fields_ha_std=('fields_ha', 'std'),
            fields_ha_av_yearly_diff=('fields_ha_yearly_diff', 'mean'),
            fields_ha_adiff_y1=('fields_ha_diff_from_y1', 'mean'),
            fields_ha_apercdiff_y1=('fields_ha_percdiff_to_y1', 'mean')

        ).reset_index()
            
        return grid_yearly_stats
    grid_yearly_stats = compute_grid_year_average(gridgdf)

    return grid_allyears_stats, grid_yearly_stats

# Silence the print statements in a function call
def silence_prints(func, *args, **kwargs):
    # Create a string IO stream to catch any print outputs
    with io.StringIO() as f, contextlib.redirect_stdout(f):
        return func(*args, **kwargs)  # Call the function without print outputs
######################################################################################


