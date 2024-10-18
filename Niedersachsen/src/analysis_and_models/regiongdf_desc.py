# %%
import pandas as pd
import geopandas as gpd
import os
import math as m
import logging
import numpy as np
from shapely.geometry import box
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis_and_models import regiongdf_desc2 as gd


# %% A. create regional aggregated df
def create_regiodf(gld):
    columns = ['year', 'LANDKREIS', 'CELLCODE', 'total_perimeter', 'total_edges', 'mean_edges']

    # 1. Extract the specified columns and drop duplicates
    regiondf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created regiondf with shape {regiondf.shape}")
    logging.info(f"Columns in regiondf: {regiondf.columns}")
    
    # 2. Compute mean statistics at region level
    # for statistics, group by year and landkreis because you want to look at each year
    # and each region and compute the statistics for each region 
    # Number of fields per region
    fields = gld.groupby(['year', 'LANDKREIS'])['geometry'].count().reset_index()
    fields.columns = ['year', 'LANDKREIS', 'fields']
    regiondf = pd.merge(regiondf, fields, on=['year', 'LANDKREIS'])

    # Number of unique groups per region
    group_count = gld.groupby(['year', 'LANDKREIS'])['Gruppe'].nunique().reset_index()
    group_count.columns = ['year', 'LANDKREIS', 'group_count']
    regiondf = pd.merge(regiondf, group_count, on=['year', 'LANDKREIS'])

    # Sum of field size per region (m2)
    fsm2_sum = gld.groupby(['year', 'LANDKREIS'])['area_m2'].sum().reset_index()
    fsm2_sum.columns = ['year', 'LANDKREIS', 'fsm2_sum']
    regiondf = pd.merge(regiondf, fsm2_sum, on=['year', 'LANDKREIS'])
    
    # Sum of field size per region (ha)
    fsha_sum = gld.groupby(['year', 'LANDKREIS'])['area_ha'].sum().reset_index()
    fsha_sum.columns = ['year', 'LANDKREIS', 'fsha_sum']
    regiondf = pd.merge(regiondf, fsha_sum, on=['year', 'LANDKREIS'])

    # Mean field size per region
    regiondf['mfs_ha'] = (regiondf['fsha_sum'] / regiondf['fields'])

    # Sum of field perimeter per region
    peri_sum = gld.groupby(['year', 'LANDKREIS'])['peri_m'].sum().reset_index()
    peri_sum.columns = ['year', 'LANDKREIS', 'peri_sum']
    regiondf = pd.merge(regiondf, peri_sum, on=['year', 'LANDKREIS'])

    # Mean perimeter per regions
    regiondf['mperi'] = (regiondf['peri_sum'] / regiondf['fields'])

    # Rate of fields per hectare of land per region
    regiondf['fields_ha'] = (regiondf['fields'] / regiondf['fsha_sum'])
    
    ######################################################################
    #Shape
    ######################################################################
    # perimeter to area ratio
    # Sum of par per region
    par_sum = gld.groupby(['year', 'LANDKREIS'])['par'].sum().reset_index()
    par_sum.columns = ['year', 'LANDKREIS', 'par_sum']
    regiondf = pd.merge(regiondf, par_sum, on=['year', 'LANDKREIS'])

    # Mean par per region
    regiondf['mean_par'] = (regiondf['par_sum'] / regiondf['fields'])
   
    # Sum of cpar per region
    cpar_sum = gld.groupby(['year', 'LANDKREIS'])['cpar'].sum().reset_index()
    cpar_sum.columns = ['year', 'LANDKREIS', 'cpar_sum']
    regiondf = pd.merge(regiondf, cpar_sum, on=['year', 'LANDKREIS'])

    # Mean cpar per region
    regiondf['mean_cpar'] = (regiondf['cpar_sum'] / regiondf['fields'])

    # corrected perimeter to area ratio adjusted for square fields
    # Sum of cpar2 per region
    cpar2_sum = gld.groupby(['year', 'LANDKREIS'])['cpar2'].sum().reset_index()
    cpar2_sum.columns = ['year', 'LANDKREIS', 'cpar2_sum']
    regiondf = pd.merge(regiondf, cpar2_sum, on=['year', 'LANDKREIS'])
    
    # Mean cpar2 per region
    regiondf['mean_cpar2'] = (regiondf['cpar2_sum'] / regiondf['fields'])
    
    # p/a ratio of region as sum of peri divided by sum of area per region
    regiondf['region_par'] = ((regiondf['peri_sum'] / regiondf['fsm2_sum'])) #compare to mean par 
    
    #new region par
    regiondf['region_par2'] = ((regiondf['total_perimeter'] / regiondf['fsm2_sum'])) #compare to mean par
            
    regiondf = regiondf.drop(columns=['par_sum', 'cpar_sum', 'cpar2_sum', 'fsm2_sum'])
    
    return regiondf


# check for duplicates in the regiondf
def check_duplicates(regiondf):
    duplicates = regiondf[regiondf.duplicated(subset=['year', 'LANDKREIS'], keep=False)]
    print(f"Number of duplicates in regiondf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")
            
def calculate_differences(regiondf): #yearly regioncell differences and differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    regiondf_ext = regiondf.copy()
    
    # Ensure the data is sorted by 'LANDKREIS' and 'year'
    regiondf_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
    numeric_columns = regiondf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = regiondf_ext.groupby('LANDKREIS')[col].diff().fillna(0)

    # Filter numeric columns to exclude columns with '_yearly_diff'
    numeric_columns_no_diff = [col for col in numeric_columns if not col.endswith('_yearly_diff')]

    # Calculate difference relative to the first year
    y1_df = regiondf_ext.groupby('LANDKREIS').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['LANDKREIS'] + list(numeric_columns_no_diff)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns_no_diff})

    # Merge the first year values back into the original DataFrame
    regiondf_ext = pd.merge(regiondf_ext, y1_df, on='LANDKREIS', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns_no_diff:
        new_columns[f'{col}_diff_from_y1'] = regiondf_ext[col] - regiondf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((regiondf_ext[col] - regiondf_ext[f'{col}_y1']) / regiondf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    regiondf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns_no_diff], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    regiondf_ext = pd.concat([regiondf_ext, new_columns_df], axis=1)

    return regiondf_ext    


def to_gdf(regiondf_ext):
    # Load Germany region_landkreise to obtain the geometry
    with open('data/interim/region_landkreise.pkl', 'rb') as f:
        geom = pickle.load(f)
    geom.info()
    
    regiongdf = regiondf_ext.merge(geom, on='LANDKREIS')
    # Convert the DataFrame to a GeoDataFrame
    regiongdf = gpd.GeoDataFrame(regiongdf, geometry='geometry')
    # Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
    regiongdf.drop(columns=['LANDKREIS_y'], inplace=True)
    regiongdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)

    
    return regiongdf



def create_regiongdf_wtoutlier():
    output_dir = 'data/interim'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    regiongdf_filename = os.path.join(output_dir, 'regiongdf_wtoutlier.pkl')

    # adjust gld
    gld_ext = adjust_gld()

    # Load or create regiongdf_wtoutlier with gld_ext
    if os.path.exists(regiongdf_filename):
        regiongdf = pd.read_pickle(regiongdf_filename)
        print(f"Loaded regiongdf from {regiongdf_filename}")
    else:
        regiondf = create_regiondf(gld_ext)
        dupli = check_duplicates(regiondf)
        regiondf_ext = calculate_differences(regiondf)
        
        # Check for infinite values in all columns
        for column in regiondf_ext.columns:
            infinite_values = regiondf_ext[column].isin([np.inf, -np.inf])
            print(f"Infinite values present in {column}:", infinite_values.any())

            # Optionally, print the rows with infinite values
            if infinite_values.any():
                print(f"Rows with infinite values in {column}:")
                print(regiondf_ext[infinite_values])

            # Handle infinite values by replacing them with NaN
            regiondf_ext[column].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        regiongdf_wtoutlier = to_gdf(regiondf_ext)
        regiongdf_wtoutlier.to_pickle(regiongdf_filename)
        print(f"Saved regiongdf to {regiongdf_filename}")

    return gld_ext, regiongdf_wtoutlier


def create_regiongdf():
    output_dir = 'data/interim'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gld_trimmed_filename = os.path.join(output_dir, 'gld_trimmed.pkl')
    regiongdf_filename = os.path.join(output_dir, 'regiongdf.pkl')

    # Load or create gld_trimmed
    if os.path.exists(gld_trimmed_filename):
        gld_trimmed = pd.read_pickle(gld_trimmed_filename)
        print(f"Loaded gld_trimmed from {gld_trimmed_filename}")
    else:
        gld_trimmed = adjust_trim_gld()
        gld_trimmed.to_pickle(gld_trimmed_filename)
        print(f"Saved gld_trimmed to {gld_trimmed_filename}")

    # Load or create regiongdf
    if os.path.exists(regiongdf_filename):
        regiongdf = pd.read_pickle(regiongdf_filename)
        print(f"Loaded regiongdf from {regiongdf_filename}")
    else:
        regiondf = create_regiondf(gld_trimmed)
        dupli = check_duplicates(regiondf)
        regiondf_ext = calculate_differences(regiondf)
        print(f"Info for regiondf_ext:")
        print(regiondf_ext.info())        
        regiongdf = to_gdf(regiondf_ext)
        regiongdf.to_pickle(regiongdf_filename)
        print(f"Saved regiongdf to {regiongdf_filename}")
        
    regiongdf = trim_regiongdf(regiongdf, 'mfs_ha', 1)
    
    # outliers 3
    outliers_region2 = regiongdf[regiongdf['fields'] < 100]
    outliers_region2 = outliers_region2.drop(columns=['geometry'])
    #to csv
    outliers_region2.to_csv('data/interim/outliers_region2.csv', index=False)
    
    regiongdf = regiongdf[regiongdf['fields'] > 100]

    return gld_trimmed, regiongdf


# %% B.
#########################################################################
# compute mean and median for columns in regiongdf. save the results to a csv file
def desc_region(regiongdf):
    def compute_region_allyear_stats(regiongdf):
        # 1. Compute general all year data descriptive statistics
        region_allyears_stats = regiongdf.select_dtypes(include='number').describe()
        # Add a column to indicate the type of statistic
        region_allyears_stats['statistic'] = region_allyears_stats.index
        # Reorder columns to place 'statistic' at the front
        region_allyears_stats = region_allyears_stats[['statistic'] + list(region_allyears_stats.columns[:-1])]
        
        # Save the descriptive statistics to a CSV file
        output_dir = 'reports/statistics'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = os.path.join(output_dir, 'region_allyears_stats.csv')
        if not os.path.exists(filename):
            region_allyears_stats.to_csv(filename, index=False)
            print(f"Saved gen_stats to {filename}")
        
        return region_allyears_stats
    region_allyears_stats = compute_region_allyear_stats(regiongdf)
    
    def compute_region_year_average(regiongdf):
        # 2. Group by 'year' and calculate useful stats across regions
        region_yearly_stats = regiongdf.groupby('year').agg(
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

            peri_sum_mean=('peri_sum', 'mean'),
            peri_sum_std = ('peri_sum', 'std'),
            peri_sum_av_yearly_diff=('peri_sum_yearly_diff', 'mean'),
            peri_sum_adiff_y1=('peri_sum_diff_from_y1', 'mean'),
            peri_sum_apercdiff_y1=('peri_sum_percdiff_to_y1', 'mean'),

            mperi_mean=('mperi', 'mean'),
            mperi_std = ('mperi', 'std'),
            mperi_av_yearly_diff=('mperi_yearly_diff', 'mean'),
            mperi_adiff_y1=('mperi_diff_from_y1', 'mean'),
            mperi_apercdiff_y1=('mperi_percdiff_to_y1', 'mean'),

            totperi_sum=('total_perimeter', 'sum'),
            totperi_mean=('total_perimeter', 'mean'),
            totperi_std = ('total_perimeter', 'std'),
            totperi_av_yearly_diff=('total_perimeter_yearly_diff', 'mean'),
            totperi_adiff_y1=('total_perimeter_diff_from_y1', 'mean'),
            totperi_apercdiff_y1=('total_perimeter_percdiff_to_y1', 'mean'),

            fields_ha_mean=('fields_ha', 'mean'),
            fields_ha_std=('fields_ha', 'std'),
            fields_ha_av_yearly_diff=('fields_ha_yearly_diff', 'mean'),
            fields_ha_adiff_y1=('fields_ha_diff_from_y1', 'mean'),
            fields_ha_apercdiff_y1=('fields_ha_percdiff_to_y1', 'mean'),

            mean_par_mean=('mean_par', 'mean'),
            mean_par_std=('mean_par', 'std'),
            mean_par_av_yearly_diff=('mean_par_yearly_diff', 'mean'),
            mean_par_adiff_y1=('mean_par_diff_from_y1', 'mean'),
            mean_par_apercdiff_y1=('mean_par_percdiff_to_y1', 'mean'),

            mean_cpar2_mean=('mean_cpar2', 'mean'),
            mean_cpar2_std=('mean_cpar2', 'std'),
            mean_cpar2_av_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),
            mean_cpar2_adiff_y1=('mean_cpar2_diff_from_y1', 'mean'),
            mean_cpar2_apercdiff_y1=('mean_cpar2_percdiff_to_y1', 'mean'),

            region_par_mean=('region_par', 'mean'),
            region_par_std=('region_par', 'std'),
            region_par_av_yearly_diff=('region_par_yearly_diff', 'mean'),
            region_par_adiff_y1=('region_par_diff_from_y1', 'mean'),
            region_par_apercdiff_y1=('region_par_percdiff_to_y1', 'mean'),


            region_par2_mean=('region_par2', 'mean'),
            region_par2_std=('region_par2', 'std'),
            region_par2_av_yearly_diff=('region_par2_yearly_diff', 'mean'),
            region_par2_adiff_y1=('region_par2_diff_from_y1', 'mean'),
            region_par2_apercdiff_y1=('region_par2_percdiff_to_y1', 'mean'),
            
            totuedges_mean=('unique_edges', 'mean'),
            totuedges_std=('unique_edges', 'std'),
            totuedges_av_yearly_diff=('unique_edges_yearly_diff', 'mean'),
            totuedges_adiff_y1=('unique_edges_diff_from_y1', 'mean'),
            totuedges_apercdiff_y1=('unique_edges_percdiff_to_y1', 'mean'),
            
            medges_mean=('mean_edges', 'mean'),
            medges_std=('mean_edges', 'std'),
            medges_av_yearly_diff=('mean_edges_yearly_diff', 'mean'),
            medges_adiff_y1=('mean_edges_diff_from_y1', 'mean'),
            medges_apercdiff_y1=('mean_edges_percdiff_to_y1', 'mean')

        ).reset_index()
            
        return region_yearly_stats
    region_yearly_stats = compute_region_year_average(regiongdf)

    return region_allyears_stats, region_yearly_stats


######################################################################################
# %% 
