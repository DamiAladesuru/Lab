# %%
import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np
import pickle
# Silence the print statements in a function call
import contextlib
import io


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.desc import gld_desc_raw as gdr

''' This script contains functions for:
    - modifying gld to include columns for basic additional metrics and kulturcode descriptions.
    - creating griddf and gridgdf (without removing any assumed outlier and for subsamples).
    - computing descriptive statistics for gridgdf.
The functions are called in the trend_of_fisc script
'''

# %%
def square_cpar(gld): #shape index adjusted for square fields
    gld['cpar'] = ((0.25 * gld['peri_m']) / (gld['area_m2']**0.5))
    return gld


# %% A.
def create_griddf(gld):
    """
    Create a griddf GeodataFrame with aggregated statistics that summarize field data.

    Parameters:
    gld (geodataFrame): Input geodataFrame with columns ['CELLCODE', 'year', 'LANDKREIS'] 
                        and additional numeric columns for aggregation.

    Returns:
    A  geoDataFrame with unique CELLCODE, year, and LANDKREIS rows,
                  enriched with aggregated fields.
    """
    
    required_columns = ['CELLCODE', 'year', 'LANDKREIS', 'geometry',\
        'Gruppe', 'area_m2', 'area_ha', 'peri_m', 'par', 'cpar']
    missing = [col for col in required_columns if col not in gld.columns]
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")
    
    # Step 1: Create base griddf
    columns = ['CELLCODE', 'year', 'LANDKREIS']
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")

    # Helper function for aggregation
    def add_aggregated_column(griddf, gld, column, aggfunc, new_col):
        logging.info(f"Adding column '{new_col}' using '{aggfunc}' on '{column}'.")
        temp = gld.groupby(['CELLCODE', 'year'])[column].agg(aggfunc).reset_index()
        temp.columns = ['CELLCODE', 'year', new_col]
        return pd.merge(griddf, temp, on=['CELLCODE', 'year'], how='left')

    # Define aggregations
    aggregations = [
        {'column': 'geometry', 'aggfunc': 'count', 'new_col': 'fields'},
        {'column': 'Gruppe', 'aggfunc': 'nunique', 'new_col': 'group_count'},
        {'column': 'area_m2', 'aggfunc': 'sum', 'new_col': 'fsm2_sum'},
        {'column': 'area_ha', 'aggfunc': 'sum', 'new_col': 'fsha_sum'},
        {'column': 'peri_m', 'aggfunc': 'sum', 'new_col': 'peri_sum'},
        {'column': 'par', 'aggfunc': 'sum', 'new_col': 'par_sum'},
        {'column': 'cpar', 'aggfunc': 'sum', 'new_col': 'cpar_sum'},
        {'column': 'area_ha', 'aggfunc': 'mean', 'new_col': 'mfs_ha'},
        {'column': 'peri_m', 'aggfunc': 'mean', 'new_col': 'mperi'},
        {'column': 'par', 'aggfunc': 'mean', 'new_col': 'mpar'},
        {'column': 'cpar', 'aggfunc': 'mean', 'new_col': 'mcpar'},
        {'column': 'area_ha', 'aggfunc': 'median', 'new_col': 'medfs_ha'},
        {'column': 'peri_m', 'aggfunc': 'median', 'new_col': 'medperi'},
        {'column': 'par', 'aggfunc': 'median', 'new_col': 'medpar'},
        {'column': 'cpar', 'aggfunc': 'median', 'new_col': 'medcpar'},
    ]

    # Apply each aggregation
    for agg in aggregations:
        griddf = add_aggregated_column(griddf, gld, agg['column'], agg['aggfunc'], agg['new_col'])

    # Rate of fields per hectare of land per grid
    griddf['fields_ha'] = (griddf['fields'] / griddf['fsha_sum'])

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

def process_griddf(gld_ext):
    '''a function that combines all the functions above to create a gridgdf
    in the next function, I have simply individually called each function
    but this combined functionality is easier to call in other scripts
    e.g., for creating landkreis or crop group subsamples
    '''
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
    gridgdf = to_gdf(griddf_ext)

    return gridgdf

def create_gridgdf():
    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dynamically refer to filename based on parameters
    gridgdf_filename = os.path.join(output_dir, 'gridgdf.pkl')

    # Load gld
    gld_ext = gdr.adjust_gld()
    gld_ext = square_cpar(gld_ext)

    if os.path.exists(gridgdf_filename):
        gridgdf = pd.read_pickle(gridgdf_filename)
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
        gridgdf = to_gdf(griddf_ext)
        gridgdf.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")

    return gld_ext, gridgdf

# %% drop gridgdf outliers i.e., grids which have  fields < 300
# but only if all fields < 300 for all years in which the grid is in the dataset
def clean_gridgdf(gridgdf):
    # Remove grids with fields < 300
    gridgdf_clean = gridgdf[~(gridgdf['fields'] < 300)]
    outliers = gridgdf[gridgdf['fields'] < 300]
    # Log the count of unique values of 'CELLCODE' in the outliers
    logging.info(f"Unique CELLCODES with fields < 300: {outliers['CELLCODE'].nunique()}")

    # Step 2: Create a DataFrame for unique CELLCODES
    # and total count of years for unique CELLOCODES in gridgdf and outliers
    gridgdf_unique = gridgdf.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()
    outliers_unique = outliers.groupby('CELLCODE').agg(total_occurrence=('year', 'count')).reset_index()

    # Step 3: Add a column to check if occurrences match
    merged_outliers = outliers_unique.merge(
        gridgdf_unique, 
        on='CELLCODE', 
        suffixes=('_outliers', '_gridgdf'),
        how='left'
    )
    merged_outliers['all_years_in_data'] = merged_outliers['total_occurrence_outliers'] == merged_outliers['total_occurrence_gridgdf']

    # Step 4: Filter out rows where 'all_years_in_data' is no
    unmatched_outliers = merged_outliers[merged_outliers['all_years_in_data'] == False]

    # Step 5: Filter original outliers DataFrame for unmatched CELLCODES
    unmatched_outlier_codes = unmatched_outliers['CELLCODE']
    unmatched_outliers_df = outliers[outliers['CELLCODE'].isin(unmatched_outlier_codes)]

    # Step 6: Join these rows to gridgdf_clean
    final_cleaned_gridgdf = pd.concat([gridgdf_clean, unmatched_outliers_df], ignore_index=True)
    
    # Step 7: Create a final outliers DataFrame without the unmatched CELLCODES
    final_outliers = outliers[~(outliers['CELLCODE'].isin(unmatched_outlier_codes))]
    
    # Step 8: Print the final count of unique values of 'CELLCODES' in the final outliers
    logging.info(f"Final unique CELLCODES in outliers: {final_outliers['CELLCODE'].nunique()}")

    return final_cleaned_gridgdf, final_outliers

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
            fields_av_yearlydiff=('fields_yearly_diff', 'mean'),
            fields_adiffy1=('fields_diff_from_y1', 'mean'),
            fields_apercdiffy1=('fields_percdiff_to_y1', 'mean'),
                    
            group_count_mean=('group_count', 'mean'),
            group_count_av_yearlydiff=('group_count_yearly_diff', 'mean'),
            group_count_adiffy1=('group_count_diff_from_y1', 'mean'),
            group_count_apercdiffy1=('group_count_percdiff_to_y1', 'mean'),

            fsha_sum_sum=('fsha_sum', 'sum'),
            fsha_sum_mean=('fsha_sum', 'mean'),
            fsha_sum_std = ('fsha_sum', 'std'),
            fsha_sum_av_yearlydiff=('fsha_sum_yearly_diff', 'mean'),
            fsha_sum_adiffy1=('fsha_sum_diff_from_y1', 'mean'),
            fsha_sum_apercdiffy1=('fsha_sum_percdiff_to_y1', 'mean'),

            mfs_ha_mean=('mfs_ha', 'mean'),
            mfs_ha_std=('mfs_ha', 'std'),
            mfs_ha_av_yearlydiff=('mfs_ha_yearly_diff', 'mean'),
            mfs_ha_adiffy1=('mfs_ha_diff_from_y1', 'mean'),
            mfs_ha_apercdiffy1=('mfs_ha_percdiff_to_y1', 'mean'),

            med_fsha_mean=('medfs_ha', 'mean'),
            med_fsha_med=('medfs_ha', 'median'),
            med_fsha_std=('medfs_ha', 'std'),
            med_fsha_av_yearlydiff=('medfs_ha_yearly_diff', 'mean'),
            med_fsha_adiffy1=('medfs_ha_diff_from_y1', 'mean'),
            med_fsha_apercdiffy1=('medfs_ha_percdiff_to_y1', 'mean'),

            med_fsha_yearlydiff_med=('medfs_ha_yearly_diff', 'median'),
            med_fsha_diffy1_med=('medfs_ha_diff_from_y1', 'median'),
            med_fsha_percdiffy1_med=('medfs_ha_percdiff_to_y1', 'median'),            

            mperi_mean=('mperi', 'mean'), #averge mean perimeter
            mperi_std = ('mperi', 'std'),
            mperi_av_yearlydiff=('mperi_yearly_diff', 'mean'),
            mperi_adiffy1=('mperi_diff_from_y1', 'mean'),
            mperi_apercdiffy1=('mperi_percdiff_to_y1', 'mean'),
            
            medperi_med=('medperi', 'median'),
            medperi_yearlydiff_med=('medperi_yearly_diff', 'median'),
            medperi_diffy1_med=('medperi_diff_from_y1', 'median'),
            medperi_percdiffy1_med=('medperi_percdiff_to_y1', 'median'),

            mpar_mean=('mpar', 'mean'),
            mpar_std=('mpar', 'std'),
            mpar_av_yearlydiff=('mpar_yearly_diff', 'mean'),
            mpar_adiffy1=('mpar_diff_from_y1', 'mean'),
            mpar_apercdiffy1=('mpar_percdiff_to_y1', 'mean'),
            
            medpar_mean=('medpar', 'mean'),
            medpar_std=('medpar', 'std'),
            medpar_av_yearlydiff=('medpar_yearly_diff', 'mean'),
            medpar_adiffy1=('medpar_diff_from_y1', 'mean'),
            medpar_apercdiffy1=('medpar_percdiff_to_y1', 'mean'),

            medpar_med=('medpar', 'median'),
            medpar_yearlydiff_med=('medpar_yearly_diff', 'median'),
            medpar_diffy1_med=('medpar_diff_from_y1', 'median'),
            medpar_percdiffy1_med=('medpar_percdiff_to_y1', 'median'),
            
            mcpar_mean=('mcpar', 'mean'),
            mcpar_std=('mcpar', 'std'),
            mcpar_av_yearlydiff=('mcpar_yearly_diff', 'mean'),
            mcpar_adiffy1=('mcpar_diff_from_y1', 'mean'),
            mcpar_apercdiffy1=('mcpar_percdiff_to_y1', 'mean'),
            
            medcpar_mean=('medcpar', 'mean'),
            medcpar_std=('medcpar', 'std'),
            medcpar_av_yearlydiff=('medcpar_yearly_diff', 'mean'),
            medcpar_adiffy1=('medcpar_diff_from_y1', 'mean'),
            medcpar_apercdiffy1=('medcpar_percdiff_to_y1', 'mean'),

            medcpar_med=('medcpar', 'median'),
            medcpar_yearlydiff_med=('medcpar_yearly_diff', 'median'),
            medcpar_diffy1_med=('medcpar_diff_from_y1', 'median'),
            medcpar_percdiffy1_med=('medcpar_percdiff_to_y1', 'median'),
            
            fields_ha_mean=('fields_ha', 'mean'),
            fields_ha_med=('fields_ha', 'median'),
            fields_ha_std=('fields_ha', 'std'),
            fields_ha_av_yearlydiff=('fields_ha_yearly_diff', 'mean'),
            fields_ha_adiffy1=('fields_ha_diff_from_y1', 'mean'),
            fields_ha_diffy1_med=('fields_ha_diff_from_y1', 'median'),
            fields_ha_apercdiffy1=('fields_ha_percdiff_to_y1', 'mean'),
            fields_ha_percdiffy1_med=('fields_ha_percdiff_to_y1', 'median')

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


