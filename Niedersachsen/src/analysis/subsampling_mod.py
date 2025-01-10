# %%
import os
import pandas as pd

from src.analysis.desc import gridgdf_desc as gd
from src.analysis.desc import gld_desc_raw as gdr

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

# call
# gld_dict = create_gld_ss(gld_base, 'Gruppe')

# %% creeate subsamples gridgdf dictionary
def create_gridgdf_ss(gld, column_x):

    output_dir = 'data/interim/gridgdf'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    gridgdf_filename = os.path.join(output_dir, f'combined_gridgdf_{column_x}.pkl')
        
    # Dictionary to store gridgdf DataFrames for each unique value in column_x
    gridgdf_dict = {}

    # Loop through each unique value in column_x
    unique_values = gld[column_x].unique()
    for value in unique_values:
        # Filter gld for the current unique value
        gld_ext = gld[gld[column_x] == value]
        # Create gridgdf for the subsample
        gridgdf = gd.process_griddf(gld_ext)    
        # Add a column indicating the subsample value
        gridgdf[column_x] = value

        # Store the gridgdf in the dictionary
        gridgdf_dict[value] = gridgdf

    # Combine all the DataFrames in the dictionary into one DataFrame
    combined_gridgdf_ss = pd.concat(gridgdf_dict.values(), ignore_index=True)

    # Save the combined DataFrame to a file
    combined_filename = os.path.join(output_dir, f'combined_gridgdf_{column_x}.pkl')
    if os.path.exists(gridgdf_filename):
        print(f"Combined gridgdf for {column_x} already saved to {gridgdf_filename}")
    else:
        combined_gridgdf_ss.to_pickle(combined_filename)
        print(f"Saved combined gridgdf to {combined_filename}")

    return gridgdf_dict, combined_gridgdf_ss

# call
# gridgdf_dict, combined_gridgdf_ss = create_gridgdf_ss(gld_base, 'Gruppe')
# gridgdf_dict contains individual DataFrames in a dictionary

# %%
def ss_desc(gridgdf_dict):
    # Initialize dictionaries to store descriptives results
    grid_allyears_dict = {}
    grid_yearly_dict = {}

    # Iterate over the gridgdf_dict
    for key, gdf_subsample in gridgdf_dict.items():
        # Silence prints and run desc_grid
        grid_allyears_raw, grid_yearly_raw = gd.silence_prints(gd.desc_grid, gdf_subsample)
        
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
    
    return result

# call
# result = ss_desc(gridgdf_dict)
# result['combined_grid_allyears'] contains the combined DataFrame for all subsamples
# result['grid_allyears'] contains individual DataFrames in a dictionary


#######################################
# for subsampling specific groups
#######################################

# %%
# Creating gridgdf without gld outliers i.e., relating to gld used for create_gridgdf func
# in gd module for specific crop group. We cannot trim grid here.
def griddf_speci_subsample(cropsubsample, col1='Gruppe', col2=None, gld_data=None):
    # Use gld data loaded in subsamping script
    if gld_data is not None:
        gld_trimmed = gld_data
        
    # Create subsample gridgdf
    if col2:
        # Subsample with flexibility to subsample from 'Gruppe' or a category column 
        gld_ss = gld_trimmed[(gld_trimmed[col1] == cropsubsample) | (gld_trimmed[col2] == cropsubsample)]
    else:
        # Subsample based on one column (default to 'Gruppe')
        gld_ss = gld_trimmed[gld_trimmed[col1] == cropsubsample]
    griddf = gd.create_griddf(gld_ss)
    dupli = gd.check_duplicates(griddf)
    # calculate differences
    griddf_ydiff = gd.calculate_yearlydiff(griddf)
    griddf_exty1 = gd.calculate_diff_fromy1(griddf)
    griddf_ext = gd.combine_griddfs(griddf_ydiff, griddf_exty1)       
    gridgdf = gd.to_gdf(griddf_ext)
    

    return gld_ss, gridgdf

# %% compute averages for all groups or categories in gld
#########################################################
# here I need to load gld with crop groups,
# sub sample for crop groups that are of interest
# compute yearly averages for the sub samples and differences from first year
# then plot the differences over the years 
def gldss_overyears(column):
    gld = gdr.adjust_gld()
    
    # Count and store the unique values in gld column 'Gruppe' or 'category3' etc.
    gruppe_values = gld[column].unique()
    gruppe_count = len(gruppe_values)
    
    # Create a dictionary to store results for each Gruppe
    ss_dict = {}
    
    for gruppe in gruppe_values:
        # Create subsample for each Gruppe
        gld_subsample = gld[gld[column] == gruppe]
        
        # Run other functions on the subsample
        gydesc = gdr.compute_year_average(gld_subsample)
        gydesc['fields_ha'] = gydesc['fields_total'] / gydesc['area_sum']
        cop = gdr.calculate_yearlydiff(gydesc)
        cop_y1 = gdr.calculate_diff_fromy1(gydesc)
        gydesc_new = gdr.combine_diffgriddfs(cop, cop_y1)
        
        # Store results for this Gruppe
        ss_dict[gruppe] = gydesc_new
    
    return gld, ss_dict, gruppe_count

# %% let subsamples be grouped as in category3
def group_dictdfs(ss_dict):
    environmental_groups = [
        'stilllegung/aufforstung', 
        'greening / landschaftselemente', 
        'aukm', 
        'aus der produktion genommen'
    ]
    
    ffc_groups = [
        'getreide',
        'gemüse',
        'leguminosen',
        'eiweißpflanzen',
        'hackfrüchte',
        'ölsaaten',
        'kräuter',
        'ackerfutter'
    ]
        
    others = [
        'sonstige flächen',
        'andere handelsgewächse',
        'zierpflanzen',
        'mischkultur',
        'energiepflanzen'
    ]
    
    dauerkulturen = [
        'dauerkulturen'
    ]
    
    dauergrünland = [
        'dauergrünland'
    ]

    # Create grouped dictionary with conditional check for existing keys in ss_dict
    ss_dict_gr = {
        'environmental': {key: ss_dict[key] for key in environmental_groups if key in ss_dict},
        'food_and_fodder': {key: ss_dict[key] for key in ffc_groups if key in ss_dict},
        'others': {key: ss_dict[key] for key in others if key in ss_dict},
        'dauerkulturen': {key: ss_dict[key] for key in dauerkulturen if key in ss_dict},
        'dauergrünland': {key: ss_dict[key] for key in dauergrünland if key in ss_dict}
    }
    
    return ss_dict_gr

