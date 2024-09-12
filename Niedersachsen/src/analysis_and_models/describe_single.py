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
# %%
from src.data import dataload as dl
from src.data import eca_new as eca

# additional gld columns
def bbox_width(geometry):
    minx, miny, maxx, maxy = geometry.bounds
    return maxx - minx

def bbox_length(geometry):
    minx, miny, maxx, maxy = geometry.bounds
    return maxy - miny

def square_cpar(gld): #shape index adjusted for square fields
    gld['cpar2'] = ((0.25 * gld['peri_m']) / (gld['area_m2']**0.5))
    return gld

def field_polspy(gld):
    gld['polspy'] = (gld['area_m2']) / ((gld['peri_m'] / 4)** 2)
    return gld

def bounding_box(gld):
    gld['bbox'] = gld['geometry'].apply(lambda geom: box(*geom.bounds))   
    return gld

def bbox_dimensions(gld):
    gld['bbox_width'] = gld['geometry'].apply(bbox_width)
    gld['bbox_length'] = gld['geometry'].apply(bbox_length)
    return gld

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

def load_gld():
        # Load base data
    gld = dl.load_data(loadExistingData=True)
    # add additional columns to the data
    gld = square_cpar(gld)
    kulturcode_mastermap = eca.process_kulturcode()
    gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
    gld = gld.drop(columns=['sourceyear'])
    # call the function to add missing year data
    gld = add_missing_year_data(gld, '10kmE438N336', 2016, 2017) 
    
    return gld

# %%
# general data descriptive statistics grouped by year
def yearly_gen_statistics(gld):
    percentiles = [0.25, 0.5, 0.75]
    # Group by year
    grouped = gld.groupby('year')
        
    # Aggregate the desired statistics
    yearlygen_stats = grouped.agg(
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
                    
        par_mean=('par', 'mean'),
        par_median=('par', 'median'),
        par_25=('par', lambda x: np.percentile(x, 25)),
        par_50=('par', lambda x: np.percentile(x, 50)),
        par_75=('par', lambda x: np.percentile(x, 75)),
                                                                
        cpar2_mean=('cpar2', 'mean'),
        cpar2_median=('cpar2', 'median'),
        cpar2_25=('cpar2', lambda x: np.percentile(x, 25)),
        cpar2_50=('cpar2', lambda x: np.percentile(x, 50)),
        cpar2_75=('cpar2', lambda x: np.percentile(x, 75)),
                                            
        polspy_mean=('polspy', 'mean'),
        polspy_median=('polspy', 'median'),
        polspy_25=('polspy', lambda x: np.percentile(x, 25)),
        polspy_50=('polspy', lambda x: np.percentile(x, 50)),
        polspy_75=('polspy', lambda x: np.percentile(x, 75)),
                                                        
        fract_sum=('fract', 'sum'),
        fract_mean=('fract', 'mean'),
        fract_median=('fract', 'median'),
        fract_25=('fract', lambda x: np.percentile(x, 25)),
        fract_50=('fract', lambda x: np.percentile(x, 50)),
        fract_75=('fract', lambda x: np.percentile(x, 75))
    ).reset_index()
    
    return yearlygen_stats


# create a datframes with the following columns: 'year', 'CELLCODE' and 'LANDKREIS'.
'''
for crop group analysis, also include the needed columns e.g., category2
'''

def create_griddf(gld):
    columns = ['year', 'LANDKREIS', 'CELLCODE', 'category2']

    # Extract the specified columns and remove duplicates
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")
    
    # for statistics group by year and cellcode because you want to look at each year
    # and each grid cell and compute the statistics for each grid cell
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
    # perimeter to area ratio
    # Sum of par per grid
    par_sum = gld.groupby(['year', 'CELLCODE'])['par'].sum().reset_index()
    par_sum.columns = ['year', 'CELLCODE', 'par_sum']
    griddf = pd.merge(griddf, par_sum, on=['year', 'CELLCODE'])

    # Mean par per grid
    griddf['mean_par'] = (griddf['par_sum'] / griddf['fields'])

    # Median par per grid
    griddf['midpar'] = gld.groupby(['year', 'CELLCODE'])['par'].median().reset_index()['par']

    # Standard deviation of par per grids
    sd_par = gld.groupby(['year', 'CELLCODE'])['par'].std().reset_index()
    sd_par.columns = ['year', 'CELLCODE', 'sd_par']
    griddf = pd.merge(griddf, sd_par, on=['year', 'CELLCODE'])
    
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
    
    # p/a ratio of grid as sum of peri divided by sum of area per grid
    griddf['grid_par'] = ((griddf['peri_sum'] / griddf['fsm2_sum'])) #compare to mean par 
    
    #LSI
    griddf['lsi'] = (0.25 * griddf['peri_sum'] / (griddf['fsm2_sum']**0.5)) #ref. FRAGSTATS help
            
    # Polsby-Popper shape index
    # Mean polspy per grid
    #mean_polspy = gld.groupby(['year', 'CELLCODE'])['polspy'].mean().reset_index()
    #mean_polspy.columns = ['year', 'CELLCODE', 'mean_polspy']
    #griddf = pd.merge(griddf, mean_polspy, on=['year', 'CELLCODE'])

    # Median polspy per grid
    #griddf['midpolspy'] = gld.groupby(['year', 'CELLCODE'])['polspy'].median().reset_index()['polspy']

    # Standard deviation of polspy per grid
    #sd_polspy = gld.groupby(['year', 'CELLCODE'])['polspy'].std().reset_index()
    #sd_polspy.columns = ['year', 'CELLCODE', 'sd_polspy']
    #griddf = pd.merge(griddf, sd_polspy, on=['year', 'CELLCODE'])
    
    # grid polspy
    #griddf['grid_polspy'] = ((griddf['fsm2_sum']) / ((griddf['peri_sum'] / 4) ** 2))
    
      
    return griddf


# check for duplicates in the griddf
def check_duplicates(griddf):
    duplicates = griddf[griddf.duplicated(subset=['year', 'CELLCODE'], keep=False)]
    print(f"Number of duplicates in griddf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")
            
def calculate_differences(griddf): #yearly gridcell differences and differences from first year
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

    # Filter numeric columns to exclude columns with '_yearly_diff'
    numeric_columns_no_diff = [col for col in numeric_columns if not col.endswith('_yearly_diff')]

    # Calculate difference relative to the first year
    y1_df = griddf_ext.groupby('CELLCODE').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['CELLCODE'] + list(numeric_columns_no_diff)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns_no_diff})

    # Merge the first year values back into the original DataFrame
    griddf_ext = pd.merge(griddf_ext, y1_df, on='CELLCODE', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns_no_diff:
        new_columns[f'{col}_diff_from_y1'] = griddf_ext[col] - griddf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((griddf_ext[col] - griddf_ext[f'{col}_y1']) / griddf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    griddf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns_no_diff], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    griddf_ext = pd.concat([griddf_ext, new_columns_df], axis=1)

    return griddf_ext    
    

# compute mean and median for columns in griddfs_ext. save the results to a csv file
def compute_grid_year_average(griddf_ext):
    # Group by 'year' and calculate the mean and median for grid year averages
    grid_year_average = griddf_ext.groupby('year').agg(
        fields_sum=('fields', 'sum'),
        fields_mean=('fields', 'mean'),
        fields_std = ('fields', 'std'),
        fields_av_yearly_diff=('fields_yearly_diff', 'mean'),
        fields_adiff_y1=('fields_diff_from_y1', 'mean'),
        fields_apercdiff_y1=('fields_percdiff_to_y1', 'mean'),
        fields_median=('fields', 'median'), # could be useful to know if the gridcell with median value is the cell in
                                             #   the very centre of Niedersachsen  
        fields_10=('fields', lambda x: np.percentile(x, 10)),
        fields_25=('fields', lambda x: np.percentile(x, 25)),
        fields_50=('fields', lambda x: np.percentile(x, 50)),
        fields_75=('fields', lambda x: np.percentile(x, 75)),
        fields_90=('fields', lambda x: np.percentile(x, 90)),
                   
        group_count_mean=('group_count', 'mean'),
        group_count_av_yearly_diff=('group_count_yearly_diff', 'mean'),
        group_count_adiff_y1=('group_count_diff_from_y1', 'mean'),
        group_count_apercdiff_y1=('group_count_percdiff_to_y1', 'mean'),
        group_count_median=('group_count', 'median'),

        fsha_sum_sum=('fsha_sum', 'sum'),
        fsha_sum_mean=('fsha_sum', 'mean'),
        fsha_sum_av_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
        fsha_sum_adiff_y1=('fsha_sum_diff_from_y1', 'mean'),
        fsha_sum_apercdiff_y1=('fsha_sum_percdiff_to_y1', 'mean'),
        fsha_sum_median=('fsha_sum', 'median'),
        fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),
        fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
        fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
        fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
        fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),

        mfs_ha_mean=('mfs_ha', 'mean'),
        mfs_ha_std=('mfs_ha', 'std'),
        mfs_ha_av_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
        mfs_ha_adiff_y1=('mfs_ha_diff_from_y1', 'mean'),
        mfs_ha_apercdiff_y1=('mfs_ha_percdiff_to_y1', 'mean'),
        mfs_ha_median=('mfs_ha', 'median'),
        mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),
        mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
        mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
        mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
        mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),

        peri_sum_mean=('peri_sum', 'mean'),
        peri_sum_av_yearly_diff=('peri_sum_yearly_diff', 'mean'),
        peri_sum_adiff_y1=('peri_sum_diff_from_y1', 'mean'),
        peri_sum_apercdiff_y1=('peri_sum_percdiff_to_y1', 'mean'),
        peri_sum_median=('peri_sum', 'median'),
        peri_sum_10=('peri_sum', lambda x: np.percentile(x, 10)),
        peri_sum_25=('peri_sum', lambda x: np.percentile(x, 25)),
        peri_sum_50=('peri_sum', lambda x: np.percentile(x, 50)),
        peri_sum_75=('peri_sum', lambda x: np.percentile(x, 75)),
        peri_sum_90=('peri_sum', lambda x: np.percentile(x, 90)),

        mperi_mean=('mperi', 'mean'),
        mperi_av_yearly_diff=('mperi_yearly_diff', 'mean'),
        mperi_adiff_y1=('mperi_diff_from_y1', 'mean'),
        mperi_apercdiff_y1=('mperi_percdiff_to_y1', 'mean'),
        mperi_median=('mperi', 'median'),

        fields_ha_mean=('fields_ha', 'mean'),
        fields_ha_std=('fields_ha', 'std'),
        fields_ha_av_yearly_diff=('fields_ha_yearly_diff', 'mean'),
        fields_ha_adiff_y1=('fields_ha_diff_from_y1', 'mean'),
        fields_ha_apercdiff_y1=('fields_ha_percdiff_to_y1', 'mean'),
        fields_ha_median=('fields_ha', 'median'),

        mean_par_mean=('mean_par', 'mean'),
        mean_par_std=('mean_par', 'std'),
        mean_par_av_yearly_diff=('mean_par_yearly_diff', 'mean'),
        mean_par_adiff_y1=('mean_par_diff_from_y1', 'mean'),
        mean_par_apercdiff_y1=('mean_par_percdiff_to_y1', 'mean'),
        mean_par_median=('mean_par', 'median'),

        mean_cpar_mean=('mean_cpar', 'mean'),
        mean_cpar_std=('mean_cpar', 'std'),
        mean_cpar_av_yearly_diff=('mean_cpar_yearly_diff', 'mean'),
        mean_cpar_adiff_y1=('mean_cpar_diff_from_y1', 'mean'),
        mean_cpar_apercdiff_y1=('mean_cpar_percdiff_to_y1', 'mean'),
        mean_cpar_median=('mean_cpar', 'median'),

        mean_cpar2_mean=('mean_cpar2', 'mean'),
        mean_cpar2_std=('mean_cpar2', 'std'),
        mean_cpar2_av_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),
        mean_cpar2_adiff_y1=('mean_cpar2_diff_from_y1', 'mean'),
        mean_cpar2_apercdiff_y1=('mean_cpar2_percdiff_to_y1', 'mean'),
        mean_cpar2_median=('mean_cpar2', 'median'),

        lsi_mean=('lsi', 'mean'),
        lsi_std=('lsi', 'std'),
        lsi_av_yearly_diff=('lsi_yearly_diff', 'mean'),
        lsi_adiff_y1=('lsi_diff_from_y1', 'mean'),
        lsi_apercdiff_y1=('lsi_percdiff_to_y1', 'mean'),
        lsi_median=('lsi', 'median'),

        grid_par_mean=('grid_par', 'mean'),
        grid_par_std=('grid_par', 'std'),
        grid_par_av_yearly_diff=('grid_par_yearly_diff', 'mean'),
        grid_par_adiff_y1=('grid_par_diff_from_y1', 'mean'),
        grid_par_apercdiff_y1=('grid_par_percdiff_to_y1', 'mean'),
        grid_par_median=('grid_par', 'median')


    ).reset_index()
        
    return grid_year_average


def compute_landkreis_average(griddf_ext):
    # Group by 'year' and 'LANDKREIS' and calculate the mean and median for grid year and landkreis averages
    landkreis_average = griddf_ext.groupby(['LANDKREIS', 'year']).agg(
        fields_sum=('fields', 'sum'),
        fields_mean=('fields', 'mean'),
        fields_std = ('fields', 'std'),
        fields_av_yearly_diff=('fields_yearly_diff', 'mean'),
        fields_adiff_y1=('fields_diff_from_y1', 'mean'),
        fields_apercdiff_y1=('fields_percdiff_to_y1', 'mean'),
        fields_median=('fields', 'median'), # could be useful to know if the gridcell with median value is the cell in
                                             #   the very centre of Niedersachsen  
        fields_10=('fields', lambda x: np.percentile(x, 10)),
        fields_25=('fields', lambda x: np.percentile(x, 25)),
        fields_50=('fields', lambda x: np.percentile(x, 50)),
        fields_75=('fields', lambda x: np.percentile(x, 75)),
        fields_90=('fields', lambda x: np.percentile(x, 90)),
                   
        group_count_mean=('group_count', 'mean'),
        group_count_av_yearly_diff=('group_count_yearly_diff', 'mean'),
        group_count_adiff_y1=('group_count_diff_from_y1', 'mean'),
        group_count_apercdiff_y1=('group_count_percdiff_to_y1', 'mean'),
        group_count_median=('group_count', 'median'),

        fsha_sum_sum=('fsha_sum', 'sum'),
        fsha_sum_mean=('fsha_sum', 'mean'),
        fsha_sum_av_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
        fsha_sum_adiff_y1=('fsha_sum_diff_from_y1', 'mean'),
        fsha_sum_apercdiff_y1=('fsha_sum_percdiff_to_y1', 'mean'),
        fsha_sum_median=('fsha_sum', 'median'),
        fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),
        fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
        fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
        fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
        fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),

        mfs_ha_mean=('mfs_ha', 'mean'),
        mfs_ha_std=('mfs_ha', 'std'),
        mfs_ha_av_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
        mfs_ha_adiff_y1=('mfs_ha_diff_from_y1', 'mean'),
        mfs_ha_apercdiff_y1=('mfs_ha_percdiff_to_y1', 'mean'),
        mfs_ha_median=('mfs_ha', 'median'),
        mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),
        mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
        mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
        mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
        mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),

        peri_sum_mean=('peri_sum', 'mean'),
        peri_sum_av_yearly_diff=('peri_sum_yearly_diff', 'mean'),
        peri_sum_adiff_y1=('peri_sum_diff_from_y1', 'mean'),
        peri_sum_apercdiff_y1=('peri_sum_percdiff_to_y1', 'mean'),
        peri_sum_median=('peri_sum', 'median'),
        peri_sum_10=('peri_sum', lambda x: np.percentile(x, 10)),
        peri_sum_25=('peri_sum', lambda x: np.percentile(x, 25)),
        peri_sum_50=('peri_sum', lambda x: np.percentile(x, 50)),
        peri_sum_75=('peri_sum', lambda x: np.percentile(x, 75)),
        peri_sum_90=('peri_sum', lambda x: np.percentile(x, 90)),

        mperi_mean=('mperi', 'mean'),
        mperi_av_yearly_diff=('mperi_yearly_diff', 'mean'),
        mperi_adiff_y1=('mperi_diff_from_y1', 'mean'),
        mperi_apercdiff_y1=('mperi_percdiff_to_y1', 'mean'),
        mperi_median=('mperi', 'median'),

        fields_ha_mean=('fields_ha', 'mean'),
        fields_ha_std=('fields_ha', 'std'),
        fields_ha_av_yearly_diff=('fields_ha_yearly_diff', 'mean'),
        fields_ha_adiff_y1=('fields_ha_diff_from_y1', 'mean'),
        fields_ha_apercdiff_y1=('fields_ha_percdiff_to_y1', 'mean'),
        fields_ha_median=('fields_ha', 'median'),

        mean_par_mean=('mean_par', 'mean'),
        mean_par_std=('mean_par', 'std'),
        mean_par_av_yearly_diff=('mean_par_yearly_diff', 'mean'),
        mean_par_adiff_y1=('mean_par_diff_from_y1', 'mean'),
        mean_par_apercdiff_y1=('mean_par_percdiff_to_y1', 'mean'),
        mean_par_median=('mean_par', 'median'),

        mean_cpar_mean=('mean_cpar', 'mean'),
        mean_cpar_std=('mean_cpar', 'std'),
        mean_cpar_av_yearly_diff=('mean_cpar_yearly_diff', 'mean'),
        mean_cpar_adiff_y1=('mean_cpar_diff_from_y1', 'mean'),
        mean_cpar_apercdiff_y1=('mean_cpar_percdiff_to_y1', 'mean'),
        mean_cpar_median=('mean_cpar', 'median'),

        mean_cpar2_mean=('mean_cpar2', 'mean'),
        mean_cpar2_std=('mean_cpar2', 'std'),
        mean_cpar2_av_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),
        mean_cpar2_adiff_y1=('mean_cpar2_diff_from_y1', 'mean'),
        mean_cpar2_apercdiff_y1=('mean_cpar2_percdiff_to_y1', 'mean'),
        mean_cpar2_median=('mean_cpar2', 'median'),

        lsi_mean=('lsi', 'mean'),
        lsi_std=('lsi', 'std'),
        lsi_av_yearly_diff=('lsi_yearly_diff', 'mean'),
        lsi_adiff_y1=('lsi_diff_from_y1', 'mean'),
        lsi_apercdiff_y1=('lsi_percdiff_to_y1', 'mean'),
        lsi_median=('lsi', 'median'),

        grid_par_mean=('grid_par', 'mean'),
        grid_par_std=('grid_par', 'std'),
        grid_par_av_yearly_diff=('grid_par_yearly_diff', 'mean'),
        grid_par_adiff_y1=('grid_par_diff_from_y1', 'mean'),
        grid_par_apercdiff_y1=('grid_par_percdiff_to_y1', 'mean'),
        grid_par_median=('grid_par', 'median')

        #mean_mean_polsby=('mean_polspy', 'mean'),
        #std_mean_polspy=('mean_polspy', 'std'),
        #mean_mean_polsby_yearly_diff=('mean_polspy_yearly_diff', 'mean'),
        #mean_mean_polsby_diff_y1=('mean_polspy_diff_from_y1', 'mean'),
        #median_mean_polsby=('mean_polspy', 'median'),
        
        #mean_grid_polspy=('grid_polspy', 'mean'),
        #std_grid_polspy=('grid_polspy', 'std'),
        #mean_grid_polspy_yearly_diff=('grid_polspy_yearly_diff', 'mean'),
        #mean_grid_polspy_diff_y1=('grid_polspy_diff_from_y1', 'mean'),
        #median_grid_polspy=('grid_polspy', 'median'),      

    ).reset_index()
        
    return landkreis_average

def compute_cropgroup_average(griddf_ext):
    # Group by 'year' and 'LANDKREIS' and calculate the mean and median for grid year and landkreis averages
    category2_average = griddf_ext.groupby(['category2', 'year']).agg(
        fields_sum=('fields', 'sum'),
        fields_mean=('fields', 'mean'),
        fields_std = ('fields', 'std'),
        fields_av_yearly_diff=('fields_yearly_diff', 'mean'),
        fields_adiff_y1=('fields_diff_from_y1', 'mean'),
        fields_apercdiff_y1=('fields_percdiff_to_y1', 'mean'),
        fields_median=('fields', 'median'), # could be useful to know if the gridcell with median value is the cell in
                                             #   the very centre of Niedersachsen  
        fields_10=('fields', lambda x: np.percentile(x, 10)),
        fields_25=('fields', lambda x: np.percentile(x, 25)),
        fields_50=('fields', lambda x: np.percentile(x, 50)),
        fields_75=('fields', lambda x: np.percentile(x, 75)),
        fields_90=('fields', lambda x: np.percentile(x, 90)),
                   
        group_count_mean=('group_count', 'mean'),
        group_count_av_yearly_diff=('group_count_yearly_diff', 'mean'),
        group_count_adiff_y1=('group_count_diff_from_y1', 'mean'),
        group_count_apercdiff_y1=('group_count_percdiff_to_y1', 'mean'),
        group_count_median=('group_count', 'median'),

        fsha_sum_sum=('fsha_sum', 'sum'),
        fsha_sum_mean=('fsha_sum', 'mean'),
        fsha_sum_av_yearly_diff=('fsha_sum_yearly_diff', 'mean'),
        fsha_sum_adiff_y1=('fsha_sum_diff_from_y1', 'mean'),
        fsha_sum_apercdiff_y1=('fsha_sum_percdiff_to_y1', 'mean'),
        fsha_sum_median=('fsha_sum', 'median'),
        fsha_sum_10=('fsha_sum', lambda x: np.percentile(x, 10)),
        fsha_sum_25=('fsha_sum', lambda x: np.percentile(x, 25)),
        fsha_sum_50=('fsha_sum', lambda x: np.percentile(x, 50)),
        fsha_sum_75=('fsha_sum', lambda x: np.percentile(x, 75)),
        fsha_sum_90=('fsha_sum', lambda x: np.percentile(x, 90)),

        mfs_ha_mean=('mfs_ha', 'mean'),
        mfs_ha_std=('mfs_ha', 'std'),
        mfs_ha_av_yearly_diff=('mfs_ha_yearly_diff', 'mean'),
        mfs_ha_adiff_y1=('mfs_ha_diff_from_y1', 'mean'),
        mfs_ha_apercdiff_y1=('mfs_ha_percdiff_to_y1', 'mean'),
        mfs_ha_median=('mfs_ha', 'median'),
        mfs_ha_10=('mfs_ha', lambda x: np.percentile(x, 10)),
        mfs_ha_25=('mfs_ha', lambda x: np.percentile(x, 25)),
        mfs_ha_50=('mfs_ha', lambda x: np.percentile(x, 50)),
        mfs_ha_75=('mfs_ha', lambda x: np.percentile(x, 75)),
        mfs_ha_90=('mfs_ha', lambda x: np.percentile(x, 90)),

        peri_sum_mean=('peri_sum', 'mean'),
        peri_sum_av_yearly_diff=('peri_sum_yearly_diff', 'mean'),
        peri_sum_adiff_y1=('peri_sum_diff_from_y1', 'mean'),
        peri_sum_apercdiff_y1=('peri_sum_percdiff_to_y1', 'mean'),
        peri_sum_median=('peri_sum', 'median'),
        peri_sum_10=('peri_sum', lambda x: np.percentile(x, 10)),
        peri_sum_25=('peri_sum', lambda x: np.percentile(x, 25)),
        peri_sum_50=('peri_sum', lambda x: np.percentile(x, 50)),
        peri_sum_75=('peri_sum', lambda x: np.percentile(x, 75)),
        peri_sum_90=('peri_sum', lambda x: np.percentile(x, 90)),

        mperi_mean=('mperi', 'mean'),
        mperi_av_yearly_diff=('mperi_yearly_diff', 'mean'),
        mperi_adiff_y1=('mperi_diff_from_y1', 'mean'),
        mperi_apercdiff_y1=('mperi_percdiff_to_y1', 'mean'),
        mperi_median=('mperi', 'median'),

        fields_ha_mean=('fields_ha', 'mean'),
        fields_ha_std=('fields_ha', 'std'),
        fields_ha_av_yearly_diff=('fields_ha_yearly_diff', 'mean'),
        fields_ha_adiff_y1=('fields_ha_diff_from_y1', 'mean'),
        fields_ha_apercdiff_y1=('fields_ha_percdiff_to_y1', 'mean'),
        fields_ha_median=('fields_ha', 'median'),

        mean_par_mean=('mean_par', 'mean'),
        mean_par_std=('mean_par', 'std'),
        mean_par_av_yearly_diff=('mean_par_yearly_diff', 'mean'),
        mean_par_adiff_y1=('mean_par_diff_from_y1', 'mean'),
        mean_par_apercdiff_y1=('mean_par_percdiff_to_y1', 'mean'),
        mean_par_median=('mean_par', 'median'),

        mean_cpar_mean=('mean_cpar', 'mean'),
        mean_cpar_std=('mean_cpar', 'std'),
        mean_cpar_av_yearly_diff=('mean_cpar_yearly_diff', 'mean'),
        mean_cpar_adiff_y1=('mean_cpar_diff_from_y1', 'mean'),
        mean_cpar_apercdiff_y1=('mean_cpar_percdiff_to_y1', 'mean'),
        mean_cpar_median=('mean_cpar', 'median'),

        mean_cpar2_mean=('mean_cpar2', 'mean'),
        mean_cpar2_std=('mean_cpar2', 'std'),
        mean_cpar2_av_yearly_diff=('mean_cpar2_yearly_diff', 'mean'),
        mean_cpar2_adiff_y1=('mean_cpar2_diff_from_y1', 'mean'),
        mean_cpar2_apercdiff_y1=('mean_cpar2_percdiff_to_y1', 'mean'),
        mean_cpar2_median=('mean_cpar2', 'median'),

        lsi_mean=('lsi', 'mean'),
        lsi_std=('lsi', 'std'),
        lsi_av_yearly_diff=('lsi_yearly_diff', 'mean'),
        lsi_adiff_y1=('lsi_diff_from_y1', 'mean'),
        lsi_apercdiff_y1=('lsi_percdiff_to_y1', 'mean'),
        lsi_median=('lsi', 'median'),

        grid_par_mean=('grid_par', 'mean'),
        grid_par_std=('grid_par', 'std'),
        grid_par_av_yearly_diff=('grid_par_yearly_diff', 'mean'),
        grid_par_adiff_y1=('grid_par_diff_from_y1', 'mean'),
        grid_par_apercdiff_y1=('grid_par_percdiff_to_y1', 'mean'),
        grid_par_median=('grid_par', 'median')
        
        #mean_mean_polsby=('mean_polspy', 'mean'),
        #std_mean_polspy=('mean_polspy', 'std'),
        #mean_mean_polsby_yearly_diff=('mean_polspy_yearly_diff', 'mean'),
        #mean_mean_polsby_diff_y1=('mean_polspy_diff_from_y1', 'mean'),
        #median_mean_polsby=('mean_polspy', 'median'),
        
        #mean_grid_polspy=('grid_polspy', 'mean'),
        #std_grid_polspy=('grid_polspy', 'std'),
        #mean_grid_polspy_yearly_diff=('grid_polspy_yearly_diff', 'mean'),
        #mean_grid_polspy_diff_y1=('grid_polspy_diff_from_y1', 'mean'),
        #median_grid_polspy=('grid_polspy', 'median'),      

    ).reset_index()
        
    return category2_average

def create_gdf(griddf_ext):
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


def process_descriptives():
    output_dir = 'reports/statistics/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gld = load_gld()
            
    # Grid level data processing
    ####################################################   
    griddf = create_griddf(gld)
    griddf_filename = os.path.join(output_dir, f'griddf.csv')
    if not os.path.exists(griddf_filename):
        griddf.to_csv(griddf_filename, encoding='windows-1252', index=False)
        print(f"Saved griddf to {griddf_filename}")
                    
    dupli = check_duplicates(griddf)
    
    griddf_ext = calculate_differences(griddf)
    print(f"Info for griddf_ext:")
    print(griddf_ext.info())
    griddf_ext_filename = os.path.join(output_dir, f'griddf_extended.csv')
    if not os.path.exists(griddf_ext_filename):
        griddf_ext.to_csv(griddf_ext_filename, encoding='windows-1252', index=False)
        print(f"Saved griddf_extended to {griddf_ext_filename}")


    grid_year_average = compute_grid_year_average(griddf_ext)
    print(f"Info for grid_year_average:")
    print(grid_year_average.info())
    grid_year_average_filename = os.path.join(output_dir, f'grid_year_average.csv')
    if not os.path.exists(grid_year_average_filename):
        grid_year_average.to_csv(grid_year_average_filename, index=False)
        print(f"Saved grid_year_average to {grid_year_average_filename}")
            

    landkreis_average = compute_landkreis_average(griddf_ext)
    print(f"Info for landkreis_average:")
    print(landkreis_average.info())
    landkreis_average_filename = os.path.join(output_dir, f'landkreis_average.csv')
    if not os.path.exists(landkreis_average_filename):
        landkreis_average.to_csv(landkreis_average_filename, index=False)
        print(f"Saved landkreis_average to {landkreis_average_filename}")
        
        
    category2_average = compute_cropgroup_average(griddf_ext)

            
    gridgdf = create_gdf(griddf_ext)
    gridgdf_filename = os.path.join('data', 'interim', f'gridgdf__.pkl')
    if not os.path.exists(gridgdf_filename):
        gridgdf.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")
        
    # General data descriptive statistics
    ####################################################
    # Select the specified columns and calculate descriptive statistics
    grid_gen_stats = gridgdf.select_dtypes(include='number').describe()
    # Add a column to indicate the type of statistic
    grid_gen_stats['statistic'] = grid_gen_stats.index
    # Reorder columns to place 'statistic' at the front
    grid_gen_stats = grid_gen_stats[['statistic'] + list(grid_gen_stats.columns[:-1])]
    # Save the descriptive statistics to a CSV file
    filename = os.path.join(output_dir, 'grid_gen_stats.csv')
    if not os.path.exists(filename):
        grid_gen_stats.to_csv(filename, index=False)
        print(f"Saved gen_stats to {filename}")

            
    return gld, griddf, griddf_ext, grid_year_average, landkreis_average, category2_average, gridgdf


# sample usage of the function process_descriptives


if __name__ == '__main__':
    gld, griddf, griddf_ext, grid_year_average, landkreis_average, category2_average, gridgdf = process_descriptives()
    print("Done!")
# %%
