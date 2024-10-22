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

from src.data import dataload as dl
from src.data import eca_new as eca

''' here we have the functions for modifying gld to include columns for additional metrics,
and kulturcode descriptions. We also have functions for creating griddf and gridgdf, and
for computing descriptive statistics for gridgdf. The functions are called in the 
trend_of_fisc script and other main analysis scripts.

Caution: there are two functions for adjusting gld, one for the full dataset and the other
for the trimmed dataset. The full dataset is used to create griddf and gridgdf, while the
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

# additional columns
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
def adjust_gld():
        # Load base data
    gld = dl.load_data(loadExistingData=True)
    # add additional columns to the data
    gld_ext = square_cpar(gld)
    kulturcode_mastermap = eca.process_kulturcode()
    gld_ext = pd.merge(gld_ext, kulturcode_mastermap, on='kulturcode', how='left')
    gld_ext = gld_ext.drop(columns=['sourceyear'])
    
    # call function to add missing year data
    gld_ext = add_missing_year_data(gld_ext, '10kmE438N336', 2016, 2017)
    
    
    return gld_ext

# %%
def adjust_trim_gld():
        # Load base data
    gld = dl.load_data(loadExistingData=True)
    # add additional columns to the data
    gld_ext = square_cpar(gld)
    kulturcode_mastermap = eca.process_kulturcode()
    gld_ext = pd.merge(gld_ext, kulturcode_mastermap, on='kulturcode', how='left')
    gld_ext = gld_ext.drop(columns=['sourceyear'])
    
    # call function to add missing year data
    gld_ext = add_missing_year_data(gld_ext, '10kmE438N336', 2016, 2017)
    
    # outlier
    outlier = gld_ext[gld_ext['area_ha'] > 20]    
    outlier.to_pickle('data/interim/outlier_above20ha.pkl')
     
    #trim data
    gld_trimmed = gld_ext[gld_ext['area_ha'] <= 20]
    
    return gld_trimmed


# %% A.
def create_griddf(gld):
    columns = ['year', 'LANDKREIS', 'CELLCODE']

    # 1. Extract the specified columns and drop duplicates
    griddf = gld[columns].drop_duplicates().copy()
    logging.info(f"Created griddf with shape {griddf.shape}")
    logging.info(f"Columns in griddf: {griddf.columns}")
    
    # 2. Compute mean statistics at grid level
    # for statistics, group by year and cellcode because you want to look at each year
    # and each grid cell and compute the statistics for each grid cell
    # Number of fields per grid
    fields = gld.groupby(['year', 'CELLCODE'])['geometry'].count().reset_index()
    fields.columns = ['year', 'CELLCODE', 'fields']
    griddf = pd.merge(griddf, fields, on=['year', 'CELLCODE'])

    # Number of unique groups per grid
    group_count = gld.groupby(['year', 'CELLCODE'])['Gruppe'].nunique().reset_index()
    group_count.columns = ['year', 'CELLCODE', 'group_count']
    griddf = pd.merge(griddf, group_count, on=['year', 'CELLCODE'])

    # Sum of field size per grid (m2)
    fsm2_sum = gld.groupby(['year', 'CELLCODE'])['area_m2'].sum().reset_index()
    fsm2_sum.columns = ['year', 'CELLCODE', 'fsm2_sum']
    griddf = pd.merge(griddf, fsm2_sum, on=['year', 'CELLCODE'])
    
    # Sum of field size per grid (ha)
    fsha_sum = gld.groupby(['year', 'CELLCODE'])['area_ha'].sum().reset_index()
    fsha_sum.columns = ['year', 'CELLCODE', 'fsha_sum']
    griddf = pd.merge(griddf, fsha_sum, on=['year', 'CELLCODE'])

    # Mean field size per grid
    griddf['mfs_ha'] = (griddf['fsha_sum'] / griddf['fields'])

    # Sum of field perimeter per grid
    peri_sum = gld.groupby(['year', 'CELLCODE'])['peri_m'].sum().reset_index()
    peri_sum.columns = ['year', 'CELLCODE', 'peri_sum']
    griddf = pd.merge(griddf, peri_sum, on=['year', 'CELLCODE'])

    # Mean perimeter per grids
    griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])

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
   
    # Sum of cpar per grid
    cpar_sum = gld.groupby(['year', 'CELLCODE'])['cpar'].sum().reset_index()
    cpar_sum.columns = ['year', 'CELLCODE', 'cpar_sum']
    griddf = pd.merge(griddf, cpar_sum, on=['year', 'CELLCODE'])

    # Mean cpar per grid
    griddf['mean_cpar'] = (griddf['cpar_sum'] / griddf['fields'])

    # corrected perimeter to area ratio adjusted for square fields
    # Sum of cpar2 per grid
    cpar2_sum = gld.groupby(['year', 'CELLCODE'])['cpar2'].sum().reset_index()
    cpar2_sum.columns = ['year', 'CELLCODE', 'cpar2_sum']
    griddf = pd.merge(griddf, cpar2_sum, on=['year', 'CELLCODE'])
    
    # Mean cpar2 per grid
    griddf['mean_cpar2'] = (griddf['cpar2_sum'] / griddf['fields'])
    
    # p/a ratio of grid as sum of peri divided by sum of area per grid
    griddf['grid_par'] = ((griddf['peri_sum'] / griddf['fsm2_sum'])) #compare to mean par 
    
    #LSI
    griddf['lsi'] = (0.25 * griddf['peri_sum'] / (griddf['fsm2_sum']**0.5)) #ref. FRAGSTATS help
            
    griddf = griddf.drop(columns=['par_sum', 'cpar_sum', 'cpar2_sum', 'fsm2_sum'])
    
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

def bar_plot(data, column, bins, labels):
    # Bin the data
    data[f'{column}_binned'] = pd.cut(data[column], bins=bins, labels=labels, right=False)

    # Calculate percentage frequency using the dynamically named binned column
    percentage_frequency = data[f'{column}_binned'].value_counts(normalize=True) * 100
    percentage_frequency = percentage_frequency.reindex(labels)  # Ensure the order matches the labels
    
    # Drop empty bins if any
    percentage_frequency = percentage_frequency[percentage_frequency > 0]

    # Convert labels to strings to avoid the warning
    percentage_frequency.index = percentage_frequency.index.astype(str)

    # Create the percentage frequency bar chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=percentage_frequency.index, y=percentage_frequency.values)
    plt.title(f'Percentage Distribution of {column} in Specified Ranges')
    plt.xlabel('Range')
    plt.ylabel('Percentage Frequency')
    plt.grid(False)
    plt.xticks(rotation=45)

    # Remove the top and right spines
    sns.despine(left=True, bottom=True)

    # Add labels to the bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    plt.show()

def trim_gridgdf(gridgdf, column, threshold):
    
    # 1. Original Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(gridgdf[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
    
    # 2. Original Bar Plot
    bins2 = [0, 1, 2, 3, 5, 9, float('inf')]
    labels2 = ['0-1', '1-2', '2-3', '3-5', '5-9', '>9']
    bar_plot(gridgdf, 'mfs_ha', bins2, labels2)
    
    # 3. Trimm data based on threshold
    gridgdf_trim = gridgdf[gridgdf[column] >= threshold]
    
    # 4 save outlier
    outlier_grid = gridgdf[gridgdf[column] < threshold]
    outlier_grid.to_pickle('data/interim/outlier_grid.pkl')
    
    # 5. Box Plot without Outliers (Trimmed Data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(gridgdf_trim[column])
    plt.title(f'Box Plot of {column} Without Values Below {threshold}')
    plt.show()
    
    # 6. Bar Plot for Trimmed Data
    bar_plot(gridgdf_trim, 'mfs_ha', bins2, labels2)
    
    return gridgdf_trim


def create_gridgdf_wtoutlier():
    output_dir = 'data/interim'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gridgdf_filename = os.path.join(output_dir, 'gridgdf_wtoutlier.pkl')

    # adjust gld
    gld_ext = adjust_gld()

    # Load or create gridgdf_wtoutlier with gld_ext
    if os.path.exists(gridgdf_filename):
        gridgdf = pd.read_pickle(gridgdf_filename)
        print(f"Loaded gridgdf from {gridgdf_filename}")
    else:
        griddf = create_griddf(gld_ext)
        dupli = check_duplicates(griddf)
        griddf_ext = calculate_differences(griddf)
        print(f"Info for griddf_ext:")
        print(griddf_ext.info())        
        gridgdf_wtoutlier = to_gdf(griddf_ext)
        gridgdf_wtoutlier.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")

    return gld_ext, gridgdf_wtoutlier


def create_gridgdf():
    output_dir = 'data/interim'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gld_trimmed_filename = os.path.join(output_dir, 'gld_trimmed.pkl')
    gridgdf_filename = os.path.join(output_dir, 'gridgdf.pkl')

    # Load or create gld_trimmed
    if os.path.exists(gld_trimmed_filename):
        gld_trimmed = pd.read_pickle(gld_trimmed_filename)
        print(f"Loaded gld_trimmed from {gld_trimmed_filename}")
    else:
        gld_trimmed = adjust_trim_gld()
        gld_trimmed.to_pickle(gld_trimmed_filename)
        print(f"Saved gld_trimmed to {gld_trimmed_filename}")

    # Load or create gridgdf
    if os.path.exists(gridgdf_filename):
        gridgdf = pd.read_pickle(gridgdf_filename)
        print(f"Loaded gridgdf from {gridgdf_filename}")
    else:
        griddf = create_griddf(gld_trimmed)
        dupli = check_duplicates(griddf)
        griddf_ext = calculate_differences(griddf)
        print(f"Info for griddf_ext:")
        print(griddf_ext.info())        
        gridgdf = to_gdf(griddf_ext)
        gridgdf.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")
        
    gridgdf = trim_gridgdf(gridgdf, 'mfs_ha', 1)

    return gld_trimmed, gridgdf


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

            peri_sum_mean=('peri_sum', 'mean'),
            peri_sum_std = ('peri_sum', 'std'),
            peri_sum_av_yearly_diff=('peri_sum_yearly_diff', 'mean'),
            peri_sum_adiff_y1=('peri_sum_diff_from_y1', 'mean'),
            peri_sum_apercdiff_y1=('peri_sum_percdiff_to_y1', 'mean'),

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

            lsi_mean=('lsi', 'mean'),
            lsi_std=('lsi', 'std'),
            lsi_av_yearly_diff=('lsi_yearly_diff', 'mean'),
            lsi_adiff_y1=('lsi_diff_from_y1', 'mean'),
            lsi_apercdiff_y1=('lsi_percdiff_to_y1', 'mean'),

            grid_par_mean=('grid_par', 'mean'),
            grid_par_std=('grid_par', 'std'),
            grid_par_av_yearly_diff=('grid_par_yearly_diff', 'mean'),
            grid_par_adiff_y1=('grid_par_diff_from_y1', 'mean'),
            grid_par_apercdiff_y1=('grid_par_percdiff_to_y1', 'mean'),


        ).reset_index()
            
        return grid_yearly_stats
    grid_yearly_stats = compute_grid_year_average(gridgdf)

    return grid_allyears_stats, grid_yearly_stats


######################################################################################
# %% 
