''' approach that works for calculating yearly differences and differences from the first year in a dataframe. '''

# %%
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

griddf_ext = calculate_yearlydiff(griddf)    


# %%
def calculate_diff_fromy1(griddf): #yearly differences from first year
    # Create a copy of the original dictionary to avoid altering the original data
    griddf_ext = griddf.copy()

    # Ensure the data is sorted by 'CELLCODE' and 'year'
    griddf_ext.sort_values(by=['CELLCODE', 'year'], inplace=True)
    numeric_columns = griddf_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate difference relative to the first year
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

griddf_exty1 = calculate_diff_fromy1(griddf)


# %% 
def combine_griddfs(griddf_ext, griddf_exty1):
    # Ensure the merge is based on 'CELLCODE' and 'year'
    # Select columns from griddf_exty1 that are not in griddf_ext (excluding 'CELLCODE' and 'year')
    columns_to_add = [col for col in griddf_exty1.columns if col not in griddf_ext.columns or col in ['CELLCODE', 'year']]

    # Merge the DataFrames on 'CELLCODE' and 'year', keeping the existing columns in griddf_ext
    combined_griddf = pd.merge(griddf_ext, griddf_exty1[columns_to_add], on=['CELLCODE', 'year'], how='left')
    
    return combined_griddf

# Combine griddf_ext and griddf_exty1
combined_griddf = combine_griddfs(griddf_ext, griddf_exty1)




# %%
