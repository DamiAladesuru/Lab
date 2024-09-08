# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m
from functools import reduce # For merging multiple DataFrames
import logging
import numpy as np
from shapely.geometry import box

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.data import dataload as dl
from src.data import eca_new as eca
from src.analysis_and_models import describe_single as d

'''
This script takes gld and subsets it based on deffierent category columns.
Therefore, when calling the function subset_data(), always state a subsets
name and use this to replace whatever subsets was called in the other 
functions in the script.

'''

# %%
def add_missing_year_data(df, cellcode, from_year, to_year):
    # Filter the rows for the specified CELLCODE and from_year
    filtered_rows = df[(df['CELLCODE'] == cellcode) & (df['year'] == from_year)]
    
    # Create a copy of the filtered rows and update the year to to_year
    new_rows = filtered_rows.copy()
    new_rows['year'] = to_year
    
    # Concatenate the new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    
    return df

def subset_data():
        # Load base data
    gld = dl.load_data(loadExistingData=True)
    # add additional columns to the data
    gld = d.square_cpar(gld)
    kulturcode_mastermap = eca.process_kulturcode()
    gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
    gld = gld.drop(columns=['sourceyear'])
    # call the function to add missing year data
    gld = add_missing_year_data(gld, '10kmE438N336', 2016, 2017) 
    # Subset gld for different category
    #category1_subsets = {category: gld.loc[gld['category1'] == category] for category in gld['category1'].unique()}
    category2_subsets = {category: gld.loc[gld['category2'] == category] for category in gld['category2'].unique()}
    
    return gld, category2_subsets

# general data descriptive statistics grouped by year
def yearly_gen_statistics(category2_subsets):
    percentiles = [0.25, 0.5, 0.75]
    yearlygen_stats = {}

    # Loop through each gld in the dictionary
    for key, gld in category2_subsets.items():
        # Group by year
        grouped = gld.groupby('year')
        
        # Aggregate the desired statistics
        yearly_stat = grouped.agg(
            count = ('area_ha', 'count'),
            
            area_ha_sum=('area_ha', 'sum'),
            area_ha_mean=('area_ha', 'mean'),
            area_ha_median=('area_ha', 'median'),
            area_ha_25=('area_ha', lambda x: np.percentile(x, 25)),
            area_ha_50=('area_ha', lambda x: np.percentile(x, 50)),
            area_ha_75=('area_ha', lambda x: np.percentile(x, 75)),
            
            peri_m_sum=('peri_m', 'sum'),
            peri_m_mean=('peri_m', 'mean'),
            peri_m_median=('peri_m', 'median'),
            peri_m_25=('peri_m', lambda x: np.percentile(x, 25)),
            peri_m_50=('peri_m', lambda x: np.percentile(x, 50)),
            peri_m_75=('peri_m', lambda x: np.percentile(x, 75)),
                        
            par_sum=('par', 'sum'),
            par_mean=('par', 'mean'),
            par_median=('par', 'median'),
            par_25=('par', lambda x: np.percentile(x, 25)),
            par_50=('par', lambda x: np.percentile(x, 50)),
            par_75=('par', lambda x: np.percentile(x, 75)),
                                    
            cpar2_sum=('cpar2', 'sum'),
            cpar2_mean=('cpar2', 'mean'),
            cpar_median=('cpar2', 'median'),
            cpar2_25=('cpar2', lambda x: np.percentile(x, 25)),
            cpar2_50=('cpar2', lambda x: np.percentile(x, 50)),
            cpar2_75=('cpar2', lambda x: np.percentile(x, 75))
                                                
        ).reset_index()
        
        # Store the result in the dictionary
        yearlygen_stats[key] = yearly_stat

    return yearlygen_stats


# from category2_subsets create a datframes with the following columns: 'year', 'CELLCODE' and 'LANDKREIS'.
def create_griddf(category2_subsets):
    columns = ['year', 'LANDKREIS', 'CELLCODE']
    griddfs = {}

    for key, gld in category2_subsets.items():
        # Extract the specified columns and remove duplicates
        griddf = gld[columns].drop_duplicates().copy()
        #logging.info(f"Created griddf_{key} with shape {griddf.shape}")
        #logging.info(f"Columns in griddf_{key}: {griddf.columns}")
        
        # Number of fields per grid
        fields = gld.groupby(['year', 'CELLCODE'])['geometry'].count().reset_index()
        fields.columns = ['year', 'CELLCODE', 'fields']
        griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])

        # Number of unique groups per grid
        group_count = gld.groupby(['year', 'CELLCODE'])['Gruppe'].nunique().reset_index()
        group_count.columns = ['year', 'CELLCODE', 'group_count']
        griddf = pd.merge(griddf, group_count, on=['year', 'CELLCODE'])

        # List of unique groups per grid
        groups = gld.groupby(['year', 'CELLCODE'])['Gruppe'].unique().reset_index()
        groups.columns = ['year', 'CELLCODE', 'groups']
        griddf = pd.merge(griddf, groups, on=['year', 'CELLCODE'])

        # Sum of field size per grid (m2)
        fsm2_sum = gld.groupby(['year', 'CELLCODE'])['area_m2'].sum().reset_index()
        fsm2_sum.columns = ['year', 'CELLCODE', 'fsm2_sum']
        griddf = pd.merge(griddf, fsm2_sum, on=['year', 'CELLCODE'])
        
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

        # Sum of field perimeter per grid
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
        
        # Rate of fields per hectare of land per grid
        griddf['fields_ha'] = (griddf['fields'] / griddf['fsha_sum'])
        
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

        # Median par per grid
        griddf['midpar'] = gld.groupby(['year', 'CELLCODE'])['par'].median().reset_index()['par']

        # Standard deviation of par per grids
        sdpar = gld.groupby(['year', 'CELLCODE'])['par'].std().reset_index()
        sdpar.columns = ['year', 'CELLCODE', 'sdpar']
        griddf = pd.merge(griddf, sdpar, on=['year', 'CELLCODE'])

        # corrected perimeter to area ratio
        # Sum of cpar per grid
        cpar_sum = gld.groupby(['year', 'CELLCODE'])['cpar'].sum().reset_index()
        cpar_sum.columns = ['year', 'CELLCODE', 'cpar_sum']
        griddf = pd.merge(griddf, cpar_sum, on=['year', 'CELLCODE'])

        # Mean cpar per grid
        griddf['mean_cpar'] = (griddf['cpar_sum'] / griddf['fields'])

        # Median cpar per grid
        griddf['midcpar'] = gld.groupby(['year', 'CELLCODE'])['cpar'].median().reset_index()['cpar']

        # Standard deviation of cpar per grids
        sd_cpar = gld.groupby(['year', 'CELLCODE'])['cpar'].std().reset_index()
        sd_cpar.columns = ['year', 'CELLCODE', 'sd_cpar']
        griddf = pd.merge(griddf, sd_cpar, on=['year', 'CELLCODE'])

        # corrected perimeter to area ratio adjusted for square fields
        # Sum of cpar2 per grid
        cpar2_sum = gld.groupby(['year', 'CELLCODE'])['cpar2'].sum().reset_index()
        cpar2_sum.columns = ['year', 'CELLCODE', 'cpar2_sum']
        griddf = pd.merge(griddf, cpar2_sum, on=['year', 'CELLCODE'])
        
        # Mean cpar2 per grid
        griddf['mean_cpar2'] = (griddf['cpar2_sum'] / griddf['fields'])
        
        # Median cpar2 per grid
        griddf['midcpar2'] = gld.groupby(['year', 'CELLCODE'])['cpar2'].median().reset_index()['cpar2']
        
        # Standard deviation of cpar2 per grids
        sd_cpar2 = gld.groupby(['year', 'CELLCODE'])['cpar2'].std().reset_index()
        sd_cpar2.columns = ['year', 'CELLCODE', 'sd_cpar2']
        griddf = pd.merge(griddf, sd_cpar2, on=['year', 'CELLCODE'])

        # p/a ratio of grid as sum of peri divided by sum of area per grid
        griddf['grid_par'] = ((griddf['peri_sum'] / griddf['fsm2_sum'])) #compare to mean par 
    
        #LSI
        griddf['lsi'] = (0.25 * griddf['peri_sum'] / (griddf['fsm2_sum']**0.5)) #ref. FRAGSTATS help
        
               
        # Store the new DataFrame in the dictionary
        griddfs[key] = griddf

    return griddfs

# check for duplicates in the griddfs
def check_duplicates(griddfs):
    for key, df in griddfs.items():
        duplicates = df[df.duplicated(subset=['year', 'CELLCODE'], keep=False)]
        print(f"Number of duplicates in griddf_{key}: {duplicates.shape[0]}")
        if duplicates.shape[0] > 0:
            print(duplicates)
        else:
            print("No duplicates found")
            

def calculate_differences(griddfs):
    # Create a copy of the original dictionary to avoid altering the original data
    griddfs_ext = {}

    for key, df in griddfs.items():
        # Ensure the data is sorted by 'CELLCODE' and 'year'
        df.sort_values(by=['CELLCODE', 'year'], inplace=True)
        numeric_columns = df.select_dtypes(include='number').columns

        # Create a dictionary to store the new columns
        new_columns = {}

        # Calculate yearly difference for numeric columns and store in the dictionary
        for col in numeric_columns:
            new_columns[f'{col}_yearly_diff'] = df.groupby('CELLCODE')[col].diff().fillna(0)

        # Filter numeric columns to exclude columns with '_yearly_diff'
        numeric_columns_no_diff = [col for col in numeric_columns if not col.endswith('_yearly_diff')]

        # Calculate difference relative to the first year
        y1_df = df.groupby('CELLCODE').first().reset_index()
        
        # Rename the numeric columns to indicate the first year
        y1_df = y1_df[['CELLCODE'] + list(numeric_columns_no_diff)]
        y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns_no_diff})

        # Merge the first year values back into the original DataFrame
        df = pd.merge(df, y1_df, on='CELLCODE', how='left')

        # Calculate the difference from the first year for each numeric column (excluding yearly differences)
        for col in numeric_columns_no_diff:
            new_columns[f'{col}_diff_from_y1'] = df[col] - df[f'{col}_y1']

        # Drop the temporary first year columns
        df.drop(columns=[f'{col}_y1' for col in numeric_columns_no_diff], inplace=True)

        # Concatenate the new columns to the original DataFrame all at once
        new_columns_df = pd.DataFrame(new_columns)
        df = pd.concat([df, new_columns_df], axis=1)

        # Update the dictionary with the modified DataFrame using the new key format
        griddfs_ext[key] = df

    return griddfs_ext

# compute mean and median for columns in griddfs_ext. save the results to a csv file
def compute_grid_year_average(griddfs_ext):
    grid_year_average = {}
    for key, df in griddfs_ext.items():
        # Group by 'year' and calculate the mean and median
        grid_year_average[key] = df.groupby('year').agg(
            sum_fields=('fields', 'sum'),
            mean_fields=('fields', 'mean'),
            std_fields = ('fields', 'std'),
            mean_fields_yearly_diff=('fields_yearly_diff', 'mean'),
            mean_fields_diff_y1=('fields_diff_from_y1', 'mean'),
            median_fields=('fields', 'median'), # could be useful to know if the gridcell with median value is the cell in
                                                #   the very centre of Niedersachsen  
            fields_10=('fields', lambda x: np.percentile(x, 10)),
            fields_25=('fields', lambda x: np.percentile(x, 25)),
            fields_50=('fields', lambda x: np.percentile(x, 50)),
            fields_75=('fields', lambda x: np.percentile(x, 75)),
            fields_90=('fields', lambda x: np.percentile(x, 90)),
                    
            mean_group_count=('group_count', 'mean'),
            mean_group_count_yearly_diff=('group_count_yearly_diff', 'mean'),
            mean_group_count_diff_y1=('group_count_diff_from_y1', 'mean'),
            median_group_count=('group_count', 'median'),
            
            mean_fsha_sum=('fsha_sum', 'mean'),
            mean_fsha_sum_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
            mean_fsha_sum_diff_y1=('fsha_sum_diff_from_y1', 'mean'),                
            median_fsha_sum=('fsha_sum', 'median'),
            fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),                        
            fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
            fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
            fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
            fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),
                    
            mean_mfs_ha=('mfs_ha', 'mean'),
            std_mfs = ('mfs_ha', 'std'),
            mean_mfs_ha_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
            mean_mfs_ha_diff_y1=('mfs_ha_diff_from_y1', 'mean'),                
            median_mfs_ha=('mfs_ha', 'median'),
            mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),                        
            mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
            mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
            mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
            mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),
            
            mean_peri_sum=('peri_sum', 'mean'),
            mean_peri_sum_yearly_diff=('peri_sum_yearly_diff', 'mean'),
            mean_peri_sum_diff_y1=('peri_sum_diff_from_y1', 'mean'),                
            median_peri_sum=('peri_sum', 'median'),
            peri_sum_10=('peri_sum', lambda x: np.percentile(x, 10)),                        
            peri_sum_25=('peri_sum', lambda x: np.percentile(x, 25)),
            peri_sum_50=('peri_sum', lambda x: np.percentile(x, 50)),
            peri_sum_75=('peri_sum', lambda x: np.percentile(x, 75)),
            peri_sum_90=('peri_sum', lambda x: np.percentile(x, 90)),

            mean_mperi=('mperi', 'mean'),
            mean_mperi_yearly_diff=('mperi_yearly_diff', 'mean'),
            mean_mperi_diff_y1=('mperi_diff_from_y1', 'mean'),
            median_mperi=('mperi', 'median'), 
            
            mean_fields_ha=('fields_ha', 'mean'),
            std_fields_ha = ('fields_ha', 'std'),
            mean_fields_ha_yearly_diff=('fields_ha_yearly_diff', 'mean'),
            mean_fields_ha_diff_y1=('fields_ha_diff_from_y1', 'mean'),
            median_fields_ha=('fields_ha', 'median'),
            
            mean_mean_par=('mean_par', 'mean'),
            std_mean_par = ('mean_par', 'std'),
            mean_mean_par_yearly_diff=('mean_par_yearly_diff', 'mean'),            
            mean_mean_par_diff_y1=('mean_par_diff_from_y1', 'mean'),            
            median_mean_par=('mean_par', 'median'),
                    
            mean_mean_cpar=('mean_cpar', 'mean'),
            std_mean_cpar = ('mean_cpar', 'std'),
            mean_mean_cpar_yearly_diff=('mean_cpar_yearly_diff', 'mean'),            
            mean_mean_cpar_diff_y1=('mean_cpar_diff_from_y1', 'mean'),            
            median_mean_cpar=('mean_cpar', 'median'),
            
            mean_mean_cpar2=('mean_cpar2', 'mean'),
            std_mean_cpar2 = ('mean_cpar2', 'std'),
            mean_mean_cpar2_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),            
            mean_mean_cpar2_diff_y1=('mean_cpar2_diff_from_y1', 'mean'),            
            median_mean_cpar2=('mean_cpar2', 'median'),

            mean_lsi=('lsi', 'mean'),
            std_lsi = ('lsi', 'std'),
            mean_lsi_yearly_diff=('lsi_yearly_diff', 'mean'),
            mean_lsi_diff_y1=('lsi_diff_from_y1', 'mean'),
            median_lsi=('lsi', 'median'),
                
            mean_grid_par=('grid_par', 'mean'),
            std_grid_par = ('grid_par', 'std'),
            mean_grid_par_yearly_diff=('grid_par_yearly_diff', 'mean'),
            mean_grid_par_diff_y1=('grid_par_diff_from_y1', 'mean'),
            median_grid_par=('grid_par', 'median')
        
        ).reset_index()
        
    return grid_year_average

def compute_landkreis_average(griddfs_ext):
    landkreis_average = {}
    for key, df in griddfs_ext.items():
        # Group by 'year' and calculate the mean and median
        landkreis_average[key] = df.groupby(['LANDKREIS', 'year']).agg(
            sum_fields=('fields', 'sum'),
            mean_fields=('fields', 'mean'),
            std_fields = ('fields', 'std'),
            mean_fields_yearly_diff=('fields_yearly_diff', 'mean'),
            mean_fields_diff_y1=('fields_diff_from_y1', 'mean'),
            median_fields=('fields', 'median'), # could be useful to know if the gridcell with median value is the cell in
                                                #   the very centre of Niedersachsen  
            fields_10=('fields', lambda x: np.percentile(x, 10)),
            fields_25=('fields', lambda x: np.percentile(x, 25)),
            fields_50=('fields', lambda x: np.percentile(x, 50)),
            fields_75=('fields', lambda x: np.percentile(x, 75)),
            fields_90=('fields', lambda x: np.percentile(x, 90)),
                    
            mean_group_count=('group_count', 'mean'),
            mean_group_count_yearly_diff=('group_count_yearly_diff', 'mean'),
            mean_group_count_diff_y1=('group_count_diff_from_y1', 'mean'),
            median_group_count=('group_count', 'median'),
            
            mean_fsha_sum=('fsha_sum', 'mean'),
            mean_fsha_sum_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
            mean_fsha_sum_diff_y1=('fsha_sum_diff_from_y1', 'mean'),                
            median_fsha_sum=('fsha_sum', 'median'),
            fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),                        
            fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
            fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
            fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
            fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),
                    
            mean_mfs_ha=('mfs_ha', 'mean'),
            std_mfs = ('mfs_ha', 'std'),
            mean_mfs_ha_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
            mean_mfs_ha_diff_y1=('mfs_ha_diff_from_y1', 'mean'),                
            median_mfs_ha=('mfs_ha', 'median'),
            mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),                        
            mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
            mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
            mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
            mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),
            
            mean_peri_sum=('peri_sum', 'mean'),
            mean_peri_sum_yearly_diff=('peri_sum_yearly_diff', 'mean'),
            mean_peri_sum_diff_y1=('peri_sum_diff_from_y1', 'mean'),                
            median_peri_sum=('peri_sum', 'median'),
            peri_sum_10=('peri_sum', lambda x: np.percentile(x, 10)),                        
            peri_sum_25=('peri_sum', lambda x: np.percentile(x, 25)),
            peri_sum_50=('peri_sum', lambda x: np.percentile(x, 50)),
            peri_sum_75=('peri_sum', lambda x: np.percentile(x, 75)),
            peri_sum_90=('peri_sum', lambda x: np.percentile(x, 90)),

            mean_mperi=('mperi', 'mean'),
            mean_mperi_yearly_diff=('mperi_yearly_diff', 'mean'),
            mean_mperi_diff_y1=('mperi_diff_from_y1', 'mean'),
            median_mperi=('mperi', 'median'), 
            
            mean_fields_ha=('fields_ha', 'mean'),
            std_fields_ha = ('fields_ha', 'std'),
            mean_fields_ha_yearly_diff=('fields_ha_yearly_diff', 'mean'),
            mean_fields_ha_diff_y1=('fields_ha_diff_from_y1', 'mean'),
            median_fields_ha=('fields_ha', 'median'),
            
            mean_mean_par=('mean_par', 'mean'),
            std_mean_par = ('mean_par', 'std'),
            mean_mean_par_yearly_diff=('mean_par_yearly_diff', 'mean'),            
            mean_mean_par_diff_y1=('mean_par_diff_from_y1', 'mean'),            
            median_mean_par=('mean_par', 'median'),
                    
            mean_mean_cpar=('mean_cpar', 'mean'),
            std_mean_cpar = ('mean_cpar', 'std'),
            mean_mean_cpar_yearly_diff=('mean_cpar_yearly_diff', 'mean'),            
            mean_mean_cpar_diff_y1=('mean_cpar_diff_from_y1', 'mean'),            
            median_mean_cpar=('mean_cpar', 'median'),
            
            mean_mean_cpar2=('mean_cpar2', 'mean'),
            std_mean_cpar2 = ('mean_cpar2', 'std'),
            mean_mean_cpar2_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),            
            mean_mean_cpar2_diff_y1=('mean_cpar2_diff_from_y1', 'mean'),            
            median_mean_cpar2=('mean_cpar2', 'median'),

            mean_lsi=('lsi', 'mean'),
            std_lsi = ('lsi', 'std'),
            mean_lsi_yearly_diff=('lsi_yearly_diff', 'mean'),
            mean_lsi_diff_y1=('lsi_diff_from_y1', 'mean'),
            median_lsi=('lsi', 'median'),
                
            mean_grid_par=('grid_par', 'mean'),
            std_grid_par = ('grid_par', 'std'),
            mean_grid_par_yearly_diff=('grid_par_yearly_diff', 'mean'),
            mean_grid_par_diff_y1=('grid_par_diff_from_y1', 'mean'),
            median_grid_par=('grid_par', 'median')
        
        ).reset_index()
        
    return landkreis_average

def create_gdf(griddfs_ext):
    # Load Germany grid_landkreise to obtain the geometry
    with open('data/interim/grid_landkreise.pkl', 'rb') as f:
        geom = pickle.load(f)
    geom.info()
    
    gridgdfs = {}
    
    # Join grid to griddfs_ext using cellcode   
    for key, df in griddfs_ext.items():
        gridgdf = df.merge(geom, on='CELLCODE')
        # Convert the DataFrame to a GeoDataFrame
        gridgdf = gpd.GeoDataFrame(gridgdf, geometry='geometry')
        # Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
        gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
        gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)
        gridgdfs[key] = gridgdf
    
    return gridgdfs


def process_descriptives():
    #current_date = dt.now().strftime("%Y%m%d")
    output_dir = 'reports/statistics/subsets/category2'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gld, category2_subsets = subset_data()

    for key, df in category2_subsets.items():
        print(f"Info for category2_subsets_{key}:")
        print(df.info())

    # General data descriptive statistics
    
    yearly_desc_dict = yearly_gen_statistics(category2_subsets)
    # Define the output Excel file path
    output_excel_filename = os.path.join(output_dir, 'yearlygen_stats.xlsx')

    # Use ExcelWriter to save each DataFrame as a sheet in the same Excel file
    with pd.ExcelWriter(output_excel_filename) as writer:
        for key, df in yearly_desc_dict.items():
            print(f"Info for yearlygen_stats_{key}:")
            print(df.info())
            # Write each DataFrame to a different sheet
            df.to_excel(writer, sheet_name=f'yearlygen_stats_{key}', index=False)
            print(f"Added yearlygen_stats_{key} to {output_excel_filename}")

    print(f"Saved all DataFrames to {output_excel_filename}")
            
    # Grid level data processing
    ####################################################   
    griddfs = create_griddf(category2_subsets)
    for key, df in griddfs.items():
        print(f"Info for griddf_{key}:")
        print(df.info())
           
    dupli = check_duplicates(griddfs)
    
    griddfs_ext = calculate_differences(griddfs)
    for key, df in griddfs_ext.items():
        print(f"Info for griddf_{key}:")
        print(df.info())

    grid_year_average = compute_grid_year_average(griddfs_ext)
    for key, df in grid_year_average.items():
        print(f"Info for grid_year_average_{key}:")
        print(df.info())
          

    landkreis_average = compute_landkreis_average(griddfs_ext)
    for key, df in landkreis_average.items():
        print(f"Info for landkreis_average_{key}:")
        print(df.info())
            
    gridgdf = create_gdf(griddfs_ext)
            
    return gld, category2_subsets, yearly_desc_dict, griddfs, griddfs_ext, grid_year_average, landkreis_average, gridgdf



if __name__ == '__main__':
    gld, category2_subsets, yearly_desc_dict, griddfs, griddfs_ext, grid_year_average, landkreis_average, gridgdf = process_descriptives()
    print("Done!")
