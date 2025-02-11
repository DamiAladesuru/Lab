# %% from grid
import os
import pandas as pd
import geopandas as gpd

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")


# %%
def aggreg8_grid2region(gridgdf):
    regiondf_4rg = gridgdf.groupby(['LANDKREIS', 'year'] ).agg(
        totfields=('fields', 'sum'),
        totugrids=('CELLCODE', 'nunique'),  # Count unique CELLCODE values
        totugroups=('group_count', 'nunique'),  # Count unique group_count values
        totarea=('fsha_sum', 'sum'),
        meanmfs=('mfs_ha', 'mean'),
        totperi=('peri_sum', 'sum'),
        meanmperi=('mperi', 'mean'),
        meanmpar=('mean_par', 'mean'),
        fields_ha=('fields_ha', 'mean')
    ).reset_index()
        
    return regiondf_4rg

# check for duplicates in the regiondf
def check_duplicates(regiondf):
    duplicates = regiondf[regiondf.duplicated(subset=['year', 'LANDKREIS'], keep=False)]
    print(f"Number of duplicates in regiondf: {duplicates.shape[0]}")
    if duplicates.shape[0] > 0:
        print(duplicates)
    else:
        print("No duplicates found")


#yearly region differences and differences from first year
def calculate_yearlydiff(regiondf): #yearly region differences
    # Create a copy of the original dictionary to avoid altering the original data
    regiondf_ext = regiondf.copy()
    
    # Ensure the data is sorted by 'LANDKREIS'' and 'year'
    regiondf_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
    numeric_columns = regiondf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = regiondf_ext.groupby('LANDKREIS')[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_yearly_percdiff'] = (regiondf_ext.groupby('LANDKREIS')[col].diff() / regiondf_ext.groupby('LANDKREIS')[col].shift(1)).fillna(0) * 100
    
    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    regiondf_ydiff = pd.concat([regiondf_ext, new_columns_df], axis=1)

    return regiondf_ydiff    


def calculate_diff_fromy1(regiondf): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    regiondf_ext = regiondf.copy()

    # Ensure the data is sorted by 'LANDKREIS' and 'year'
    regiondf_ext.sort_values(by=['LANDKREIS', 'year'], inplace=True)
    numeric_columns = regiondf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate difference relative to the first year
    y1_df = regiondf_ext.groupby('LANDKREIS').first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[['LANDKREIS'] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Merge the first year values back into the original DataFrame
    regiondf_ext = pd.merge(regiondf_ext, y1_df, on='LANDKREIS', how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = regiondf_ext[col] - regiondf_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((regiondf_ext[col] - regiondf_ext[f'{col}_y1']) / regiondf_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    regiondf_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    regiondf_exty1 = pd.concat([regiondf_ext, new_columns_df], axis=1)

    return regiondf_exty1


def combine_regiondfs(regiondf_ext, regiondf_exty1):
    # Ensure the merge is based on 'LANDKREIS' and 'year'
    # Select columns from regiondf_exty1 that are not in regiondf_ext (excluding 'LANDKREIS' and 'year')
    columns_to_add = [col for col in regiondf_exty1.columns if col not in regiondf_ext.columns or col in ['LANDKREIS', 'year']]

    # Merge the DataFrames on 'LANDKREIS' and 'year', keeping the existing columns in regiondf_ext
    combined_regiondf = pd.merge(regiondf_ext, regiondf_exty1[columns_to_add], on=['LANDKREIS', 'year'], how='left')
    
    return combined_regiondf


# %%
def to_gdf(regiondf_ext):
    # Load Landkreis file for regional boundaries
    base_dir = "N:/ds/data/Niedersachsen/verwaltungseinheiten"
    
    landkreise = gpd.read_file(os.path.join(base_dir, "NDS_Landkreise.shp"))
    #landkreise.info()
    
    regiongdf = pd.merge(regiondf_ext, landkreise, on='LANDKREIS')
    # Convert the DataFrame to a GeoDataFrame
    regiongdf = gpd.GeoDataFrame(regiongdf, geometry='geometry')
    # Drop the 'LK' column
    regiongdf.drop(columns='LK', inplace=True)
    regiongdf.info()
    
    return regiongdf


def create_regiongdf_fg():
    output_dir = 'data/interim/regiongdf'
    input_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gridgdf_filename = os.path.join(input_dir, 'gridgdf.pkl')
    regiongdf_filename = os.path.join(output_dir, 'region1gdf.pkl')

    # Load gridgdf
    gridgdf = pd.read_pickle(gridgdf_filename)
    #print(f"Loaded gridgdf from {gridgdf_filename}")
    
    # Load or create regiongdf
    if os.path.exists(regiongdf_filename):
        regiongdf = pd.read_pickle(regiongdf_filename)
        print(f"Loaded regiongdf from {regiongdf_filename}")
    else:

        regiondf = aggreg8_grid2region(gridgdf)
        check_duplicates(regiondf)
        #calculate yearly differences
        regiondf_ydiff = calculate_yearlydiff(regiondf)
        regiondf_exty1 = calculate_diff_fromy1(regiondf)
        regiondf_ext = combine_regiondfs(regiondf_ydiff, regiondf_exty1)
        regiongdf = to_gdf(regiondf_ext)
        regiongdf.to_pickle(regiongdf_filename)
        print(f"Saved regiongdf to {regiongdf_filename}")
        

    return regiongdf

def create_regiondf_fg():
    output_dir = 'data/interim/regiongdf'
    input_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gridgdf_filename = os.path.join(input_dir, 'gridgdf.pkl')

    # Load gridgdf
    gridgdf = pd.read_pickle(gridgdf_filename)
    print(f"Loaded gridgdf from {gridgdf_filename}")
    
    regiondf = aggreg8_grid2region(gridgdf)
    check_duplicates(regiondf)
    #calculate yearly differences
    regiondf_ydiff = calculate_yearlydiff(regiondf)
    regiondf_exty1 = calculate_diff_fromy1(regiondf)
    regiondf_ext = combine_regiondfs(regiondf_ydiff, regiondf_exty1)
    #r.to_csv('data/interim/regiongdf/regiondf.csv', index=False)
    #print(f"Saved regiondf")
    

    return regiondf_ext

# %%
