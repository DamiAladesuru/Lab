# %%
import pandas as pd
import geopandas as gpd
import os
import logging
import numpy as np
from shapely.geometry import box
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
# Silence the print statements in a function call
import contextlib
import io


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.data import dataload as dl
from src.data import eca_new as eca

''' This script contains functions for:
    - modifying gld to include columns for additional metrics, polygon edge count,
    unique count of polygon edges in a grid cell and kulturcode descriptions.
    - trimming and removing outliers from gld
    - creating griddf and gridgdf (without outlier).
    - computing descriptive statistics for gridgdf.
The functions are called in the trend_of_fisc script and other main analysis scripts.

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


def get_polygon_edges(polygon):
    """Get the exterior coordinates of a polygon or multipolygon."""
    if polygon.geom_type == 'MultiPolygon':
        return [list(p.exterior.coords) for p in polygon.geoms]
    elif polygon.geom_type == 'Polygon':
        return [list(polygon.exterior.coords)]
    else:
        return []

def extract_edges_and_lengths(polygon):
    """Extract edges and calculate their lengths for a single polygon."""
    edges = set()
    total_length = 0
    
    for coords in get_polygon_edges(polygon):
        for i in range(len(coords) - 1):
            edge = tuple(sorted([coords[i], coords[i + 1]]))
            length = np.sqrt((coords[i][0] - coords[i + 1][0])**2 + (coords[i][1] - coords[i + 1][1])**2)
            edges.add(edge)
            total_length += length
            
    return edges, total_length  # Return unique edges and total length

def calculate_unique_perimeter_and_edges(polygons, cache=None):
    """Calculate total perimeter and unique edges for a list of polygons."""
    all_edges = {}
    polygon_edge_counts = []
    
    for polygon in polygons:
        # Use cache to store results of edge extraction
        if cache is None:
            cache = {}
        
        if polygon not in cache:
            polygon_edges, _ = extract_edges_and_lengths(polygon)  # Call the helper function
            cache[polygon] = polygon_edges  # Store the result in cache
        else:
            polygon_edges = cache[polygon]  # Retrieve from cache

        for edge in polygon_edges:
            all_edges[edge] = all_edges.get(edge, 0) + 1
        polygon_edge_counts.append(len(polygon_edges))
    
    unique_edges = {k: v for k, v in all_edges.items() if v == 1}
    total_perimeter = sum(np.sqrt((e[0][0] - e[1][0])**2 + (e[0][1] - e[1][1])**2) for e in unique_edges)
    unique_edge_count = len(unique_edges)
    mean_unique_edges_per_polygon = np.mean(polygon_edge_counts)
    
    return total_perimeter, unique_edge_count, mean_unique_edges_per_polygon

def calculate_perimeter_and_edges(polygon, cache=None):
    """Calculate total edges and perimeter for a single polygon."""
    if cache is None:
        cache = {}
        
    if polygon not in cache:
        edges, total_length = extract_edges_and_lengths(polygon)  # Call the helper function
        cache[polygon] = (edges, total_length)  # Store the result in cache
    else:
        edges, total_length = cache[polygon]  # Retrieve from cache
    
    return len(edges), total_length  # Return count of edges and total perimeter


def process_geodataframe_grouped(gdf):
    """Process the GeoDataFrame to calculate metrics grouped by CELLCODE and year."""
    # Ensure geometry column is valid
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # Group by CELLCODE and year to calculate total perimeter and unique edges
    grouped_result = gdf.groupby(['CELLCODE', 'year']).apply(lambda group: pd.Series(calculate_unique_perimeter_and_edges(group['geometry'].tolist()))).reset_index()
    
    # Rename columns for clarity
    grouped_result.columns = ['CELLCODE', 'year', 'total_uperimeter', 'totunique_edges', 'mean_unique_edges']
    
    # Merge the grouped results back into the original GeoDataFrame
    gdf = gdf.merge(grouped_result, on=['CELLCODE', 'year'], how='left')

    return gdf

def process_geodataframe_individual(gdf):
    """Process the GeoDataFrame to calculate metrics for each individual polygon."""
    # Ensure geometry column is valid
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # Calculate total edges and perimeter for each individual polygon
    gdf['edges'], gdf['perimeter_i'] = zip(*gdf['geometry'].apply(calculate_perimeter_and_edges))
    
    return gdf


# %%
def adjust_trim_gld():
        # Load base data
    gld = dl.load_data(loadExistingData=True)
    # add additional columns to the data
    gld_ext = square_cpar(gld)
    kulturcode_mastermap = eca.process_kulturcode()
    gld_ext = pd.merge(gld_ext, kulturcode_mastermap, on='kulturcode', how='left')
    gld_ext = gld_ext.drop(columns=['sourceyear', 'cpar', 'shp_index', 'fract'])
    
    # call function to add missing year data
    gld_ext = add_missing_year_data(gld_ext, '10kmE438N336', 2016, 2017)
    
    # call the function to count edges and total perimeter
    # Process grouped metrics
    gld_ext = process_geodataframe_grouped(gld_ext)

    # Process individual metrics
    gld_ext = process_geodataframe_individual(gld_ext)  
    
    # outlier
    outlier = gld_ext[gld_ext['area_ha'] > 20]    
    outlier.to_pickle('data/interim/gridgdf/outlier_above20ha.pkl')
     
    #trim data
    gld_trimmed = gld_ext[gld_ext['area_ha'] <= 20]
    
    return gld_trimmed


# %% A.
def create_griddf(gld):
    columns = ['CELLCODE', 'year', 'LANDKREIS'] #, 'total_uperimeter', 'totunique_edges', 'mean_unique_edges'

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

    # Average total edges of polygons in grid
    #mean_edges = gld.groupby(['CELLCODE', 'year'])['edges'].mean().reset_index()
    #mean_edges.columns = ['CELLCODE', 'year', 'mean_edges']
    #griddf = pd.merge(griddf, mean_edges, on=['CELLCODE', 'year'])

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

    # Sum of field perimeter per grid
    peri_sum = gld.groupby(['CELLCODE', 'year'])['peri_m'].sum().reset_index()
    peri_sum.columns = ['CELLCODE', 'year', 'peri_sum']
    griddf = pd.merge(griddf, peri_sum, on=['CELLCODE', 'year'])

    # Mean perimeter per grids
    griddf['mperi'] = (griddf['peri_sum'] / griddf['fields'])

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
    
    # p/a ratio of grid as sum of peri divided by sum of area per grid
    #griddf['grid_par'] = ((griddf['peri_sum'] / griddf['fsm2_sum'])) #compare to mean par 
    
    #new grid par
    #griddf['grid_par'] = ((griddf['total_uperimeter'] / griddf['fsm2_sum'])) #compare to mean par
            
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


def trim_gridgdf(gridgdf, column, threshold):
    
    # 1. Original Box Plot
    plt.figure(figsize=(8, 6))
    sns.boxplot(gridgdf[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
    
    # 3. Trimm data based on threshold
    gridgdf_trim = gridgdf[gridgdf[column] >= threshold]
    
    # 4 save outlier
    if os.path.exists('data/interim/gridgdf/outlier_gridmfs_1.pkl'):
        print(f"outlier_grid for mfs_1 exists")
    else:
        outlier_grid = gridgdf[gridgdf[column] < threshold]
        outlier_grid.to_pickle('data/interim/gridgdf/outlier__gridmfs_1.pkl')
    
    # 5. Box Plot without Outliers (Trimmed Data)
    plt.figure(figsize=(8, 6))
    sns.boxplot(gridgdf_trim[column])
    plt.title(f'Box Plot of {column} Without Values Below {threshold}')
    plt.show()

    
    return gridgdf_trim

#%%
def create_gridgdf():
    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Define file paths
    gld_trimmed_filename = os.path.join(output_dir, 'gld_trimmed.pkl')
    gridgdf_filename = os.path.join(output_dir, 'gridgdf.pkl')
    outliers1_grid_filename = os.path.join(output_dir, 'outliers_mfs_1.pkl')
    outliers2_grid_filename = os.path.join(output_dir, 'outliers_field100.pkl')

    # Load or create gld_trimmed
    if os.path.exists(gld_trimmed_filename):
        gld_trimmed = pd.read_pickle(gld_trimmed_filename)
        print(f"Loaded gld_trimmed from {gld_trimmed_filename}")
    else:
        gld_trimmed = adjust_trim_gld()

    # Load or create gridgdf
    if os.path.exists(gridgdf_filename):
        gridgdf = pd.read_pickle(gridgdf_filename)
        print(f"Loaded gridgdf from {gridgdf_filename}")
    else:
        griddf = create_griddf(gld_trimmed)
        dupli = check_duplicates(griddf)
        # calculate differences
        griddf_ydiff = calculate_yearlydiff(griddf)
        griddf_exty1 = calculate_diff_fromy1(griddf)
        griddf_ext = combine_griddfs(griddf_ydiff, griddf_exty1)  
        print(f"Info for griddf_ext:")
        print(griddf_ext.info())        
        gridgdf = to_gdf(griddf_ext)

        # Extract outliers and remove them from gridgdf
        outliers1_grid = gridgdf[gridgdf['mfs_ha'] < 1]  # First condition
        gridgdf = gridgdf[gridgdf['mfs_ha'] >= 1]

        outliers2_grid = gridgdf[gridgdf['fields'] < 100]  # Second condition
        gridgdf = gridgdf[gridgdf['fields'] >= 100]
        
    # Save files only if they don't already exist
    if not os.path.exists(gld_trimmed_filename):
        gld_trimmed.to_pickle(gld_trimmed_filename)
        print(f"Saved gld_trimmed to {gld_trimmed_filename}")    

    if not os.path.exists(gridgdf_filename):
        gridgdf.to_pickle(gridgdf_filename)
        print(f"Saved gridgdf to {gridgdf_filename}")

    if not os.path.exists(outliers1_grid_filename):
        outliers1_grid.to_pickle(outliers1_grid_filename)
        print(f"Saved outliers1_grid to {outliers1_grid_filename}")

    if not os.path.exists(outliers2_grid_filename):
        outliers2_grid.to_pickle(outliers2_grid_filename)
        print(f"Saved outliers2_grid to {outliers2_grid_filename}")
        


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

# Other potential variables to include in the gridgdf desc
'''
                        
            mean_edges_mean=('mean_edges', 'mean'),
            mean_edges_std=('mean_edges', 'std'),
            mean_edges_av_yearly_diff=('mean_edges_yearly_diff', 'mean'),
            mean_edges_adiff_y1=('mean_edges_diff_from_y1', 'mean'),
            mean_edges_apercdiff_y1=('mean_edges_percdiff_to_y1', 'mean'),
            
            totuperi_sum=('total_uperimeter', 'sum'),
            totuperi_mean=('total_uperimeter', 'mean'),
            totuperi_std = ('total_uperimeter', 'std'),
            totuperi_av_yearly_diff=('total_uperimeter_yearly_diff', 'mean'),
            totuperi_adiff_y1=('total_uperimeter_diff_from_y1', 'mean'),
            totuperi_apercdiff_y1=('total_uperimeter_percdiff_to_y1', 'mean'),            

            grid_par_mean=('grid_par', 'mean'),
            grid_par_std=('grid_par', 'std'),
            grid_par_av_yearly_diff=('grid_par_yearly_diff', 'mean'),
            grid_par_adiff_y1=('grid_par_diff_from_y1', 'mean'),
            grid_par_apercdiff_y1=('grid_par_percdiff_to_y1', 'mean'),
            
            totuedges_mean=('totunique_edges', 'mean'),
            totuedges_std=('totunique_edges', 'std'),
            totuedges_av_yearly_diff=('totunique_edges_yearly_diff', 'mean'),
            totuedges_adiff_y1=('totunique_edges_diff_from_y1', 'mean'),
            totuedges_apercdiff_y1=('totunique_edges_percdiff_to_y1', 'mean'),
            
            muedges_mean=('mean_unique_edges', 'mean'),
            muedges_std=('mean_unique_edges', 'std'),
            muedges_av_yearly_diff=('mean_unique_edges_yearly_diff', 'mean'),
            muedges_adiff_y1=('mean_unique_edges_diff_from_y1', 'mean'),
            muedges_apercdiff_y1=('mean_unique_edges_percdiff_to_y1', 'mean')
'''


# Silence the print statements in a function call
def silence_prints(func, *args, **kwargs):
    # Create a string IO stream to catch any print outputs
    with io.StringIO() as f, contextlib.redirect_stdout(f):
        return func(*args, **kwargs)  # Call the function without print outputs
######################################################################################

