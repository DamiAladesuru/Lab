'''script for loading gld, modifying it as I want, creating gridgdf, creating subsamples, etc.'''

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

#
def square_cpar(gld): #shape index adjusted for square fields
    gld['cpar2'] = ((0.25 * gld['peri_m']) / (gld['area_m2']**0.5))
    return gld


# A.
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


# 
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


# 
def combine_griddfs(griddf_ext, griddf_exty1):
    # Ensure the merge is based on 'CELLCODE' and 'year'
    # Select columns from griddf_exty1 that are not in griddf_ext (excluding 'CELLCODE' and 'year')
    columns_to_add = [col for col in griddf_exty1.columns if col not in griddf_ext.columns or col in ['CELLCODE', 'year']]

    # Merge the DataFrames on 'CELLCODE' and 'year', keeping the existing columns in griddf_ext
    combined_griddf = pd.merge(griddf_ext, griddf_exty1[columns_to_add], on=['CELLCODE', 'year'], how='left')
    
    return combined_griddf


#
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


# B.
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

            medfs_ha_mean=('medfs_ha', 'mean'),
            medfs_ha_std=('medfs_ha', 'std'),
            medfs_ha_av_yearly_diff=('medfs_ha_yearly_diff', 'mean'),
            medfs_ha_adiff_y1=('medfs_ha_diff_from_y1', 'mean'),
            medfs_ha_apercdiff_y1=('medfs_ha_percdiff_to_y1', 'mean'),

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
            

            medpar_mean=('medpar', 'mean'),
            medpar_std=('medpar', 'std'),
            medpar_av_yearly_diff=('medpar_yearly_diff', 'mean'),
            medpar_adiff_y1=('medpar_diff_from_y1', 'mean'),
            medpar_apercdiff_y1=('medpar_percdiff_to_y1', 'mean'),
            
            
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
# load above and then run below individually to load, odify data and process to gridgdf

# %%
def adjust_gld():
    gld_path = 'data/interim/gld_wtkc.pkl'
    
    # Check if the file already exists
    if os.path.exists(gld_path):
        # Load data from the file if it exists
        gld = pd.read_pickle(gld_path)
        print("Loaded gld data from existing file.")
    else:
        # Load base data
        gld = dl.load_data(loadExistingData=True)
        # Add additional columns to the data
        kulturcode_mastermap = eca.process_kulturcode()
        gld = pd.merge(gld, kulturcode_mastermap, on='kulturcode', how='left')
        # Drop unnecessary columns
        gld = gld.drop(columns='sourceyear')
        # Save the processed data to a file
        gld.to_pickle(gld_path)
        print("Processed and saved new data.")
    
    gld = gld.drop(columns=['cpar', 'shp_index', 'fract'])
    
    return gld

gld = adjust_gld()
# %%
glda = gld[~gld['LANDKREIS'].isin(['Küstenmeer Region Weser-Ems', 'Lüneburg', 'Helmstedt'])]
glda =glda[~(glda['area_m2'] < 300)]
# %%
gld100 =gld[~(gld['area_m2'] < 100)]

# %%
gldf = gld[~gld['LANDKREIS'].isin(['Küstenmeer Region Weser-Ems', 'Lüneburg', 'Helmstedt'])]

# %%
def create_gridgdf_raw(loadedgld, gridfile_suf,
                       t=100, apply_t=False,
                       gld_file='data/interim/gldkc_min100.pkl'): #base file is gld_wtkc.pkl
    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Dynamically refer to filename based on parameters
    if gridfile_suf:
        gridgdf_filename = os.path.join(output_dir, f'gridgdf_raw_{gridfile_suf}.pkl')
    else:
        gridgdf_filename = os.path.join(output_dir, 'gridgdf_raw.pkl')

    if loadedgld is not None:
        gld_ext = loadedgld
    else:    
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

    return gld_ext, gridgdf_raw

# %%
_, gridgdf100 = create_gridgdf_raw(loadedgld = gld100, gridfile_suf='0912')

_, grid_yearly_100 = silence_prints(desc_grid, gridgdf100)

# %%

# %%
gldf_max100 =gld[(gld['area_m2'] < 100)]
gldf_max100df =gldf_max100.drop(columns='geometry')

# %% creeate subsamples gridgdf
def create_gridgdf_ss(gld, column_x, filename_suffix='ss'):

    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gridgdf_filename = os.path.join(output_dir, f'combined_gridgdf_{filename_suffix}.pkl')
        
    if os.path.exists(gridgdf_filename):
        combined_gridgdf_ss = pd.read_pickle(gridgdf_filename)
        print(f"Loaded gridgdf for {column_x} from {gridgdf_filename}")
        return None, combined_gridgdf_ss
    
    # Dictionary to store gridgdf_raw DataFrames for each unique value in column_x
    gridgdf_dict = {}

    # Loop through each unique value in column_x
    unique_values = gld[column_x].unique()
    for value in unique_values:
        # Filter gld for the current unique value
        gld_ext = gld[gld[column_x] == value]
        
        griddf = grdr.create_griddf(gld_ext)
        dupli = grdr.check_duplicates(griddf)
        
        # calculate differences
        griddf_ydiff = grdr.calculate_yearlydiff(griddf)
        griddf_exty1 = grdr.calculate_diff_fromy1(griddf)
        griddf_ext = grdr.combine_griddfs(griddf_ydiff, griddf_exty1)
                
        # Add a column indicating the subsample value
        griddf_ext['group'] = value
        
        gridgdf_raw = grdr.to_gdf(griddf_ext)

        # Store the gridgdf_raw in the dictionary
        gridgdf_dict[value] = gridgdf_raw

    # Combine all the DataFrames in the dictionary into one DataFrame
    combined_gridgdf_ss = pd.concat(gridgdf_dict.values(), ignore_index=True)

    # Save the combined DataFrame to a file
    combined_filename = os.path.join(output_dir, f'combined_gridgdf_{filename_suffix}.pkl')
    combined_gridgdf_ss.to_pickle(combined_filename)
    print(f"Saved combined gridgdf to {combined_filename}")

    return gridgdf_dict, combined_gridgdf_ss

# %%
gridgdf_dict, combined_gridgdf_ss = create_gridgdf_ss(gld_base, 'Gruppe', 'basess')

# %% subsampled dict at gld level
def create_gld_ss(gld, column_x):
    # Dictionary
    gld_dict = {}

    # Loop through each unique value in column_x
    unique_values = gld[column_x].unique()
    for value in unique_values:
        # Filter gld for the current unique value
        gld_ss = gld[gld[column_x] == value]
        
        gld_dict[value] = gld_ss

    return gld_dict

gld_dict = create_gld_ss(gld_base, 'Gruppe')


# %%
# Iterate over the dictionary returned by create_gridgdf_ss
for subsample_name, subsample in gridgdf_dict.items():
    unique_years = sorted(subsample['year'].unique())
    stack_plots_in_grid(
        subsample, 
        unique_years, 
        scatterplot_mpar_marea, 
        ncols=4, 
        figsize=(25, 15), 
        grid_title=f"Scatterplots for Subsample {subsample_name}"  # Use the subsample name for naming
    )


# %%
# Iterate over the dictionary returned by create_gld_ss
for subsample_name, subsample in gld_dict.items():
    unique_years = sorted(subsample['year'].unique())
    stack_plots_in_grid(
        subsample, 
        unique_years, 
        scatterplot_par_area, 
        ncols=4, 
        figsize=(25, 15), 
        grid_title=f"{subsample_name}"  # Use the subsample name for naming
    )
# %%
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=result['combined_grid_yearly'], x='year', y='mfs_ha_mean', hue='subsample',
             marker='o')

# Add titles and labels
plt.title('Trend of Average MFS (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average MFS (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Show the plot
plt.show()

# %%
# Initialize dictionaries to store results
grid_allyears_dict = {}
grid_yearly_dict = {}

# Iterate over the gridgdf_dict
for key, gdf_subsample in gridgdf_dict.items():
    # Silence prints and run desc_grid
    grid_allyears_raw, grid_yearly_raw = grdr.silence_prints(grdr.desc_grid, gdf_subsample)
    
    # Add the key as a new column to identify the subsample
    grid_allyears_raw['subsample'] = key
    grid_yearly_raw['subsample'] = key
    
    # Store in dictionaries
    grid_allyears_dict[key] = grid_allyears_raw
    grid_yearly_dict[key] = grid_yearly_raw

# Combine all DataFrames into one for each type
combined_grid_allyears = pd.concat(grid_allyears_dict.values(), ignore_index=True)
combined_grid_yearly = pd.concat(grid_yearly_dict.values(), ignore_index=True)

# Return or use the combined DataFrames and dictionaries
result = {
    'grid_allyears': grid_allyears_dict,
    'grid_yearly': grid_yearly_dict,
    'combined_grid_allyears': combined_grid_allyears,
    'combined_grid_yearly': combined_grid_yearly
}

# Example usage:
# result['combined_grid_allyears'] contains the combined DataFrame for all subsamples
# result['grid_allyears'] contains individual DataFrames in a dictionary

# %%
for key, gdf in gridgdf_dict.items():
    # Convert the GeoDataFrame to EPSG 4326
    geoData = gdf.to_crs(epsg=4326)
    
    # Plot the choropleth, including the key as part of the title
    plot_facet_choropleth_with_geoplot(
        geoData, 
        column='mean_par', 
        cmap='plasma', 
        year_col='year', 
        ncols=4, 
        title=f"Choropleth for Subsample: {key}"
    )

# %%
