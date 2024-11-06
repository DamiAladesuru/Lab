# %%
from src.analysis import gridgdf_desc2 as gd
from src.analysis.raw import gld_desc_raw as gdr

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
def gldss_overyears():
    gld = gdr.adjust_gld()
    
    # Count and store the unique values in gld column 'Gruppe' or 'category3' etc.
    gruppe_values = gld['category3'].unique()
    gruppe_count = len(gruppe_values)
    
    # Create a dictionary to store results for each Gruppe
    ss_dict = {}
    
    for gruppe in gruppe_values:
        # Create subsample for each Gruppe
        gld_subsample = gld[gld['category3'] == gruppe]
        
        # Run other functions on the subsample
        gydesc = gdr.compute_year_average(gld_subsample)
        gydesc['fields_ha'] = gydesc['fields_total'] / gydesc['area_sum']
        cop = gdr.calculate_yearlydiff(gydesc)
        cop_y1 = gdr.calculate_diff_fromy1(gydesc)
        gydesc_new = gdr.combine_diffgriddfs(cop, cop_y1)
        
        # Store results for this Gruppe
        ss_dict[gruppe] = gydesc_new
    
    return gld, ss_dict, gruppe_count
# %%
