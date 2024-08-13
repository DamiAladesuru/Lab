# %%
import pickle
import geopandas as gpd
import pandas as pd
import os
import math as m
from functools import reduce # For merging multiple DataFrames
import logging
import numpy as np

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.data import dataload as dl

output_dir = 'reports/statistics/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %%
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

# Call the subset_data function to get gld_envi and gld_others
gld_envi, gld_others = subset_data()

gld_subsets = {
    'envi': gld_envi,
    'others': gld_others,
}

# %%
for key, df in gld_subsets.items():
    print(f"Info for gld_subsets_{key}:")
    print(df.info())
# %% #####################################################################
# General data descriptive statistics
###########################################################################
for key, dataset in gld_subsets.items():
    # Select the specified columns and calculate descriptive statistics
    gen_stats = dataset[['year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']].describe()
    # Add a column to indicate the type of statistic
    gen_stats['statistic'] = gen_stats.index
    # Reorder columns to place 'statistic' at the front
    gen_stats = gen_stats[['statistic', 'year', 'area_ha', 'peri_m', 'par', 'cpar', 'shp_index', 'fract']]
    # Save the descriptive statistics to a CSV file
    gen_stats.to_csv(os.path.join(output_dir, f'gen_stats_{key}.csv'), index=False)

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

yearly_desc_dict = yearly_gen_statistics(gld_subsets)
for key, df in yearly_desc_dict.items():
    print(f"Info for yearlygen_stats_{key}:")
    print(df.info())
    df.to_csv(os.path.join(output_dir, f'yearlygen_stats_{key}.csv'), index=False)
    print(f"Saved yearlygen_stats_{key} to {os.path.join(output_dir, f'yearlygen_stats_{key}.csv')}")

# %% ##########################################################################
# Grid level data processing
###########################################################################
# %% from gld_subsets create a datframes with the following columns: 'year', 'CELLCODE' and 'LANDKREIS'.
def create_griddf(gld_subsets):
    columns = ['year', 'LANDKREIS', 'CELLCODE']
    griddfs = {}

    for key, gld in gld_subsets.items():
        # Extract the specified columns and remove duplicates
        griddf = gld[columns].drop_duplicates()
        
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

        # Mean Par per grid
        griddf['mean_par'] = (griddf['par_sum'] / griddf['fields'])

        # p/a ratio of grid as sum of peri divided by sum of area per grid
        #griddf['grid_par'] = (griddf['peri_sum'] / griddf['fsha_sum']) #compare to mean par

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
# Create and save griddfs
griddfs = create_griddf(gld_subsets)
for key, df in griddfs.items():
    print(f"Info for griddf_{key}:")
    print(df.info())
    df.to_csv(os.path.join(output_dir, f'griddf_{key}.csv'), index=False)
    print(f"griddf_{key} to {os.path.join(output_dir, f'griddf_{key}.csv')}")

# %%   
def extract_and_merge_2012_data(griddfs):
    for key, df in griddfs.items():
        # Extract 2012 data
        df_2012 = df[df['year'] == 2012].copy()
        # Set index to 'CELLCODE'
        df_2012.set_index('CELLCODE', inplace=True)
        # Drop the 'year' column from the 2012 data to avoid duplication in the merge
        df_2012.drop(columns=['year'], inplace=True)
        # Rename the 2012 columns to reflect they are from 2012
        df_2012.columns = [f"{col}_2012" for col in df_2012.columns]
        # Merge the 2012 data back into the original dataframe on 'CELLCODE'
        griddfs[key] = pd.merge(df, df_2012, on='CELLCODE', how='left')

    return griddfs

def create_diff_columns(griddfs_with2012):
    griddfs_ext = {}
    for key, df in griddfs_with2012.items():
        print(f"Info for griddf_with2012_{key}:")
        print(df.info())
    for key, df in griddfs_with2012.items():
        # sort by 'CELLCODE', 'year' and Create columns for the differences between one row and the next
        df.sort_values(by=['CELLCODE', 'year'], inplace=True)
        for col in df.columns:
            if col not in ['year', 'LANDKREIS', 'CELLCODE', 'groups','LANDKREIS_2012', 'groups_2012']:
                df[f"{col}_diff"] = df[col].diff()
 
        # Reset the index
        df.reset_index(drop=True, inplace=True)
        # Store the extended DataFrame in the dictionary
        griddfs_ext[key] = df
        
    return griddfs_ext

def create_diff12_columns(griddfs_ext):
    for key, df in griddfs_ext.items():
        # Convert all columns to numeric if possible
        df = df.apply(pd.to_numeric, errors='coerce')
        # Create columns for the differences between 2012 and other years
        for col in df.columns:
            if col.endswith('_2012'):
                # Extract the base name from the column name
                base_name = col[:-5]
                # Check if the corresponding 2012 column exists
                if f"{base_name}_2012" in df.columns:
                    # Create a new column with the difference between the year and 2012
                    df[f"{base_name}_diff12"] = df[col] - df[f"{base_name}_2012"]
        # Drop the 2012 columns
        df.drop(columns=[col for col in df.columns if col.endswith('_2012')], inplace=True)

    return griddfs_ext

# Call the functions with the defined variables
griddfs_with2012 = extract_and_merge_2012_data(griddfs)
griddfs_ext = create_diff_columns(griddfs_with2012)
griddfs_ext = create_diff12_columns(griddfs_ext)

# Output information
for key, df in griddfs_ext.items():
    print(f"Info for griddf_ext_{key}:")
    print(df.info())
    df.to_csv(os.path.join(output_dir, f'griddf_ext_{key}.csv'), index=False)
    print(f"Saved griddf_ext_{key} to {os.path.join(output_dir, f'griddf_ext_{key}.csv')}")

#####################################################################
# # Calculating descriptive statistics for the grid level metrics
#####################################################################
# %% compute mean and median for columns in griddfs_ext. save the results to a csv file
def compute_mean_median(griddfs_ext):
    mean_median = {}
    for key, df in griddfs_ext.items():
        # Group by 'year' and calculate the mean and median
        mean_median[key] = df.groupby('year').agg(
            mean_fields=('fields', 'mean'),
            median_fields=('fields', 'median'),
            mean_group_count=('group_count', 'mean'),
            median_group_count=('group_count', 'median'),
            mean_mfs_ha=('mfs_ha', 'mean'),
            median_mfs_ha=('mfs_ha', 'median'),
            mean_midfs_ha=('midfs_ha', 'mean'),
            median_midfs_ha=('midfs_ha', 'median'),
            mean_mperi=('mperi', 'mean'),
            median_mperi=('mperi', 'median'),
            mean_midperi=('midperi', 'mean'),
            median_midperi=('midperi', 'median'),
            mean_mean_par=('mean_par', 'mean'),
            median_mean_par=('mean_par', 'median'),
            mean_midpar=('midpar', 'mean'),
            median_midpar=('midpar', 'median'),
            mean_mean_cpar=('mean_cpar', 'mean'),
            median_mean_cpar=('mean_cpar', 'median'),
            mean_midcpar=('midcpar', 'mean'),
            median_midcpar=('midcpar', 'median'),
            mean_mean_shp=('mean_shp', 'mean'),
            median_mean_shp=('mean_shp', 'median'),
            mean_midshp=('midshp', 'mean'),
            median_midshp=('midshp', 'median'),
            mean_mean_fract=('mean_fract', 'mean'),
            median_mean_fract=('mean_fract', 'median'),
            mean_midfract=('midfract', 'mean'),
            median_midfract=('midfract', 'median'),
        ).reset_index()
        
    return mean_median

mean_median = compute_mean_median(griddfs_ext)
for key, df in mean_median.items():
    print(f"Info for mean_median_{key}:")
    print(df.info())
    df.to_csv(os.path.join(output_dir, f'mean_median_{key}.csv'), index=False)
    print(f"Saved mean_median_{key} to {os.path.join(output_dir, f'mean_median_{key}.csv')}")
# %%
