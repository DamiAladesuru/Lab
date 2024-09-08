# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m
from functools import reduce # For merging multiple DataFrames
import logging
import numpy as np
#from datetime import datetime as dt

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.data import dataload as dl

def subset_data():
    # Load base data
    gld = dl.load_data(loadExistingData=True)
    kulturcode_mastermap = pd.read_csv('reports/Kulturcode/kulturcode_mastermap.csv', encoding='windows-1252')
    gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
    gld = gld.drop(columns=['kulturart_sourceyear'])
    
    # Subset gld for different categories
    gld_envi = gld[gld['category1'] == 'Environmental']
    gld_others = gld[gld['category1'] == 'Others']

    return gld_envi, gld_others

# general data descriptive statistics grouped by year
def yearly_gen_statistics(gld_subsets):
    percentiles = [0.25, 0.5, 0.75]
    yearlygen_stats = {}

    # Loop through each gld in the dictionary
    for key, gld in gld_subsets.items():
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
                                    
            cpar_sum=('cpar', 'sum'),
            cpar_mean=('cpar', 'mean'),
            cpar_median=('cpar', 'median'),
            cpar_25=('cpar', lambda x: np.percentile(x, 25)),
            cpar_50=('cpar', lambda x: np.percentile(x, 50)),
            cpar_75=('cpar', lambda x: np.percentile(x, 75)),
                                                
            shp_index_sum=('shp_index', 'sum'),
            shp_index_mean=('shp_index', 'mean'),
            shp_index_median=('shp_index', 'median'),
            shp_index_25=('shp_index', lambda x: np.percentile(x, 25)),
            shp_index_50=('shp_index', lambda x: np.percentile(x, 50)),
            shp_index_75=('shp_index', lambda x: np.percentile(x, 75)),
                                                            
            fract_sum=('fract', 'sum'),
            fract_mean=('fract', 'mean'),
            fract_median=('fract', 'median'),
            fract_25=('fract', lambda x: np.percentile(x, 25)),
            fract_50=('fract', lambda x: np.percentile(x, 50)),
            fract_75=('fract', lambda x: np.percentile(x, 75))
        ).reset_index()
        
        # Store the result in the dictionary
        yearlygen_stats[key] = yearly_stat

    return yearlygen_stats


# from gld_subsets create a datframes with the following columns: 'year', 'CELLCODE' and 'LANDKREIS'.
def create_griddf(gld_subsets):
    columns = ['year', 'LANDKREIS', 'CELLCODE']
    griddfs = {}

    for key, gld in gld_subsets.items():
        # Extract the specified columns and remove duplicates
        griddf = gld[columns].drop_duplicates().copy()
        logging.info(f"Created griddf_{key} with shape {griddf.shape}")
        logging.info(f"Columns in griddf_{key}: {griddf.columns}")
        
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

        ######################################################################
        #Shape
        ######################################################################
        # simple perimeter to area ratio
        # Sum of Par per grid
        par_sum = gld.groupby(['year', 'CELLCODE'])['par'].sum().reset_index()
        par_sum.columns = ['year', 'CELLCODE', 'par_sum']
        griddf = pd.merge(griddf, par_sum, on=['year', 'CELLCODE'])

        # p/a ratio of grid as sum of peri divided by sum of area per grid
        griddf['grid_par'] = (griddf['peri_sum'] / griddf['fsha_sum']) #compare to mean par

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
    griddfs_ext = griddfs.copy()
    
    for key, df in griddfs_ext.items():
        # Ensure the data is sorted by 'CELLCODE' and 'year'
        df.sort_values(by=['CELLCODE', 'year'], inplace=True)
        numeric_columns = df.select_dtypes(include='number').columns

        # Calculate yearly difference for numeric columns
        for col in numeric_columns:
            df[f'{col}_yearly_diff'] = df.groupby('CELLCODE')[col].diff().fillna(0)

        # Calculate difference relative to the year 2012
        if 2012 in df['year'].values:
            # Create a DataFrame for 2012 values
            df_2012 = df[df['year'] == 2012][['CELLCODE'] + list(numeric_columns)]
            df_2012 = df_2012.rename(columns={col: f'{col}_2012' for col in numeric_columns})
            
            # Merge the 2012 values back to the original DataFrame
            df = pd.merge(df, df_2012, on='CELLCODE', how='left')

            # Calculate the difference from 2012 for each numeric column
            for col in numeric_columns:
                df[f'{col}_diff_from_2012'] = df[col] - df[f'{col}_2012']

            # Drop the temporary 2012 columns
            df.drop(columns=[f'{col}_2012' for col in numeric_columns], inplace=True)
        
        # Update the dictionary with the modified DataFrame
        griddfs_ext[key] = df

    return griddfs_ext

# compute mean and median for columns in griddfs_ext. save the results to a csv file
def compute_mean_median(griddfs_ext):
    mean_median = {}
    for key, df in griddfs_ext.items():
        # Group by 'year' and calculate the mean and median
        mean_median[key] = df.groupby('year').agg(
            mean_fields=('fields', 'mean'),
            mean_fields_yearly_diff=('fields_yearly_diff', 'mean'),
            mean_fields_diff12=('fields_diff_from_2012', 'mean'),
            median_fields=('fields', 'median'),
            median_fields_yearly_diff=('fields_yearly_diff', 'median'),
            median_fields_diff12=('fields_diff_from_2012', 'median'),
            fields_10=('fields', lambda x: np.percentile(x, 10)),
            fields_25=('fields', lambda x: np.percentile(x, 25)),
            fields_50=('fields', lambda x: np.percentile(x, 50)),
            fields_75=('fields', lambda x: np.percentile(x, 75)),
            fields_90=('fields', lambda x: np.percentile(x, 90)),
                        
            mean_group_count=('group_count', 'mean'),
            mean_group_count_yearly_diff=('group_count_yearly_diff', 'mean'),
            mean_group_count_diff12=('group_count_diff_from_2012', 'mean'),
            median_group_count_yearly_diff=('group_count_yearly_diff', 'median'),
            median_group_count=('group_count', 'median'),
            median_group_count_diff12=('group_count_diff_from_2012', 'median'),
            
            sum_fsha_sum=('fsha_sum', 'sum'),
            mean_fsha_sum=('fsha_sum', 'mean'),
            mean_fsha_sum_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
            mean_fsha_sum_diff12=('fsha_sum_diff_from_2012', 'mean'),                
            median_fsha_sum=('fsha_sum', 'median'),
            median_fsha_sum_yearly_diff=('fsha_sum_yearly_diff', 'median'),
            median_fsha_sum_diff12=('fsha_sum_diff_from_2012', 'median'),                        
            fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),                        
            fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
            fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
            fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
            fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),
            
            mean_mfs_ha=('mfs_ha', 'mean'),
            mean_mfs_ha_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
            mean_mfs_ha_diff12=('mfs_ha_diff_from_2012', 'mean'),                
            median_mfs_ha=('mfs_ha', 'median'),
            median_mfs_ha_yearly_diff=('mfs_ha_yearly_diff', 'median'),
            median_mfs_ha_diff12=('mfs_ha_diff_from_2012', 'median'),                        
            mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),                        
            mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
            mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
            mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
            mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),

            mean_midfs_ha=('midfs_ha', 'mean'),
            mean_midfs_ha_yearly_diff=('midfs_ha_yearly_diff', 'mean'),            
            mean_midfs_ha_diff12=('midfs_ha_diff_from_2012', 'mean'),
            median_midfs_ha=('midfs_ha', 'median'),
            median_midfs_ha_yearly_diff=('midfs_ha_yearly_diff', 'median'),
            median_midfs_ha_diff12=('midfs_ha_diff_from_2012', 'median'),

            mean_mperi=('mperi', 'mean'),
            mean_mperi_yearly_diff=('mperi_yearly_diff', 'mean'),
            mean_mperi_diff12=('mperi_diff_from_2012', 'mean'),
            median_mperi=('mperi', 'median'),
            median_mperi_yearly_diff=('mperi_yearly_diff', 'median'),            
            median_mperi_diff12=('mperi_diff_from_2012', 'median'),

            mean_midperi=('midperi', 'mean'),
            mean_midperi_yearly_diff=('midperi_yearly_diff', 'mean'),
            mean_midperi_diff12=('midperi_diff_from_2012', 'mean'),            
            median_midperi=('midperi', 'median'),            
            median_midperi_yearly_diff=('midperi_yearly_diff', 'median'),            
            median_midperi_diff12=('midperi_diff_from_2012', 'median'),            
            
            mean_grid_par=('grid_par', 'mean'),
            mean_grid_par_yearly_diff=('grid_par_yearly_diff', 'mean'),
            mean_grid_par_diff12=('grid_par_diff_from_2012', 'mean'),
            median_grid_par=('grid_par', 'median'),
            median_grid_par_yearly_diff=('grid_par_yearly_diff', 'median'),
            median_grid_par_diff12=('grid_par_diff_from_2012', 'median'),
                        
            mean_mean_par=('mean_par', 'mean'),
            mean_mean_par_yearly_diff=('mean_par_yearly_diff', 'mean'),
            mean_mean_par_diff12=('mean_par_diff_from_2012', 'mean'),
            median_mean_par=('mean_par', 'median'),
            median_mean_par_yearly_diff=('mean_par_yearly_diff', 'median'),
            median_mean_par_diff12=('mean_par_diff_from_2012', 'median'),
            
            mean_midpar=('midpar', 'mean'),
            mean_midpar_yearly_diff=('midpar_yearly_diff', 'mean'),            
            mean_midpar_diff12=('midpar_diff_from_2012', 'mean'),            
            median_midpar=('midpar', 'median'),
            median_midpa_yearly_diff=('midpar_yearly_diff', 'median'),
            median_midpa_diff12=('midpar_diff_from_2012', 'median'),

            mean_mean_cpar=('mean_cpar', 'mean'),
            mean_mean_cpar_yearly_diff=('mean_cpar_yearly_diff', 'mean'),            
            mean_mean_cpar_diff12=('mean_cpar_diff_from_2012', 'mean'),            
            median_mean_cpar=('mean_cpar', 'median'),
            median_mean_cpar_yearly_diff=('mean_cpar_yearly_diff', 'median'),
            median_mean_cpar_diff12=('mean_cpar_diff_from_2012', 'median'),
            
            mean_midcpar=('midcpar', 'mean'),
            mean_midcpar_yearly_diff=('midcpar_yearly_diff', 'mean'),            
            mean_midcpar_diff12=('midcpar_diff_from_2012', 'mean'),            
            median_midcpar=('midcpar', 'median'),            
            median_midcpar_yearly_diff=('midcpar_yearly_diff', 'median'),            
            median_midcpar_diff12=('midcpar_diff_from_2012', 'median'),

            mean_mean_shp=('mean_shp', 'mean'),
            mean_mean_shp_yearly_diff=('mean_shp_yearly_diff', 'mean'),
            mean_mean_shp_diff12=('mean_shp_diff_from_2012', 'mean'),
            median_mean_shp=('mean_shp', 'median'),
            median_mean_shp_yearly_diff=('mean_shp_yearly_diff', 'median'),            
            median_mean_shp_diff12=('mean_shp_diff_from_2012', 'median'),            

            mean_midshp=('midshp', 'mean'),
            mean_midshp_yearly_diff=('midshp_yearly_diff', 'mean'),            
            mean_midshp_diff12=('midshp_diff_from_2012', 'mean'),            
            median_midshp=('midshp', 'median'),
            median_midshp_yearly_diff=('midshp_yearly_diff', 'median'),            
            median_midshp_diff12=('midshp_diff_from_2012', 'median'),            

            mean_mean_fract=('mean_fract', 'mean'),
            mean_mean_fract_yearly_diff=('mean_fract_yearly_diff', 'mean'),
            mean_mean_fract_diff12=('mean_fract_diff_from_2012', 'mean'),
            median_mean_fract=('mean_fract', 'median'),
            median_mean_fract_yearly_diff=('mean_fract_yearly_diff', 'median'),
            median_mean_fract_diff12=('mean_fract_diff_from_2012', 'median'),

            mean_midfract=('midfract', 'mean'),
            mean_midfract_yearly_diff=('midfract_yearly_diff', 'mean'),
            mean_midfract_diff12=('midfract_diff_from_2012', 'mean'),                       
            median_midfract=('midfract', 'median'),
            median_midfract_yearly_diff=('midfract_yearly_diff', 'median'),
            median_midfract_diff12=('midfract_diff_from_2012', 'median')
        ).reset_index()
        
    return mean_median

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
    output_dir = 'reports/statistics/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gld_envi, gld_others = subset_data()

    gld_subsets = {
        'envi': gld_envi,
        'others': gld_others,
    }
    for key, df in gld_subsets.items():
        print(f"Info for gld_subsets_{key}:")
        print(df.info())

    # General data descriptive statistics
    ####################################################
    for key, dataset in gld_subsets.items():
        # Select the specified columns and calculate descriptive statistics
        gen_stats = dataset[['year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
        # Add a column to indicate the type of statistic
        gen_stats['statistic'] = gen_stats.index
        # Reorder columns to place 'statistic' at the front
        gen_stats = gen_stats[['statistic', 'year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']]
        # Save the descriptive statistics to a CSV file
        gen_stats_filename = os.path.join(output_dir, f'gen_stats_{key}.csv')#_{current_date}
        if not os.path.exists(gen_stats_filename):
            gen_stats.to_csv(gen_stats_filename, index=False)
            print(f"Saved gen_stats_{key} to {gen_stats_filename}")
        
    yearly_desc_dict = yearly_gen_statistics(gld_subsets)
    for key, df in yearly_desc_dict.items():
        print(f"Info for yearlygen_stats_{key}:")
        print(df.info())
        yearly_stats_filename = os.path.join(output_dir, f'yearlygen_stats_{key}.csv')
        if not os.path.exists(yearly_stats_filename):
            df.to_csv(yearly_stats_filename, index=False)
            print(f"Saved yearlygen_stats_{key} to {yearly_stats_filename}")
            
    # Grid level data processing
    ####################################################   
    griddfs = create_griddf(gld_subsets)
    for key, df in griddfs.items():
        print(f"Info for griddf_{key}:")
        print(df.info())
        griddf_filename = os.path.join(output_dir, f'griddf_{key}.csv')#_{current_date}
        if not os.path.exists(griddf_filename):
            df.to_csv(griddf_filename, encoding='windows-1252', index=False)
            print(f"Saved griddf_{key} to {griddf_filename}")
                
    dupli = check_duplicates(griddfs)
    
    griddfs_ext = calculate_differences(griddfs)
    for key, df in griddfs_ext.items():
        print(f"Info for griddf_{key}:")
        print(df.info())
        griddf_ext_filename = os.path.join(output_dir, f'griddf_{key}_extended.csv')#_{current_date}
        if not os.path.exists(griddf_ext_filename):
            df.to_csv(griddf_ext_filename, encoding='windows-1252', index=False)
            print(f"Saved griddf_{key}_extended to {griddf_ext_filename}")


    mean_median = compute_mean_median(griddfs_ext)
    for key, df in mean_median.items():
        print(f"Info for mean_median_{key}:")
        print(df.info())
        mean_median_filename = os.path.join(output_dir, f'mean_median_{key}.csv')#_{current_date}
        if not os.path.exists(mean_median_filename):
            df.to_csv(mean_median_filename, index=False)
            print(f"Saved mean_median_{key} to {mean_median_filename}")
            
    gridgdf = create_gdf(griddfs_ext)
    for key, gdf in gridgdf.items():
        gridgdf_filename = os.path.join('data', 'interim', f'gridgdf_{key}.pkl')#_{current_date}
        if not os.path.exists(gridgdf_filename):
            gdf.to_pickle(gridgdf_filename)
            print(f"Saved gridgdf_{key} to {gridgdf_filename}")
            
    return gld_subsets, yearly_desc_dict, griddfs, griddfs_ext, mean_median, gridgdf



if __name__ == '__main__':
    gld_subsets, yearly_desc_dict, griddfs, griddfs_ext, mean_median, gridgdf = process_descriptives()
    print("Done!")