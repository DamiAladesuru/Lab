'''Script for loading data as subsamples of crops, and plotting
trends of metrics over time for each subsample.'''

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.desc import gridgdf_desc as gd
from src.analysis import subsampling_mod as ss
from src.visualization import plotting_module as pm

'''
gd module contains functions for loading entire data and calculating metrics
ss module contains functions for creating subsamples and calculating metrics
pm module contains functions for plotting
'''

# load data
##############################################
# %%
gld, _ = gd.silence_prints(gd.create_gridgdf) # gld contains all data
gridgdf_dict, _ = ss.create_gridgdf_ss(gld, 'category3') # gridgdf_dict contains subsamples
result = ss.ss_desc(gridgdf_dict) # result contains calculated metrics for subsamples
cat_yearlies = result['grid_yearly'] # cat_yearlies contains yearly metric values for each subsample
print(cat_yearlies.keys())

# plot metric trend for each subsample.
# We use multiline_metrics function from pm module.
##############################################
# %%
metrics = {
    'Median Field Size': 'med_fsha_percdiffy1_med',
    'Median Perimeter': 'medperi_percdiffy1_med',
    'Median PAR': 'medpar_percdiffy1_med',
    'Fields/TotalArea': 'fields_ha_percdiffy1_med'
}

color_mapping = {
    #https://personal.sron.nl/~pault/
    'Median Field Size': '#004488',
    'Median Perimeter': 'grey',
    'Median PAR': '#228833',
    'Fields/TotalArea': '#CC3311',
}

for category_name, df in cat_yearlies.items():
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"Skipping {category_name}: Invalid or empty DataFrame")
        continue  # Skip if not a valid DataFrame
    
    save_path = f"reports/figures/ToF/{category_name}_trends.svg"
    
    pm.multiline_metrics(
        df=df,
        title=f"{category_name} Metrics Over Time",
        ylabel="Aggregate Change (%) in Field Metric Value from 2012",
        metrics=metrics,
        format='svg',
        save_path=save_path,
        color_mapping=color_mapping
    )
    
    print(f"Plot saved: {save_path}")

#################################################################
# We can put all the plots in one figure with shared legend.
# We use multiline_metrics_shared_legend function from pm module.
#For this, we use the combined_grid_yearly from result.
#################################################################
# %%
cat_combined = result['combined_grid_yearly']

# %% 
metrics = {
    'Median Field Size': 'med_fsha_percdiffy1_med',
    'Median Perimeter': 'medperi_percdiffy1_med',
    'Median PAR': 'medpar_percdiffy1_med',
    'Fields/TotalArea': 'fields_ha_percdiffy1_med'
}

color_mapping = {
    'Median Field Size': '#004488',
    'Median Perimeter': 'grey',
    'Median PAR': '#228833',
    'Fields/TotalArea': '#CC3311',
}

pm.multiline_metrics_with_shared_legend(
    df=cat_combined,
    title="Trend of Change in Field Metric Values Over Time by Subsample",
    ylabel="Aggregate Change (%) in Field Metric Value from 2012",
    metrics=metrics,
    save_path="reports/figures/ToF/allcategory_trends_shared_legend.svg",
    color_mapping=color_mapping,
    format='svg'
)

########################################################
# Finally, here is the call code for multimetric_ss_plot
# This function plots multiple metrics for each grouped subgroup
#i.e., we see plot of eacg gruppe under defined category
########################################################
# Call the subsampling data function and load data
gld, results_gr, _ = ss.gldss_overyears(column = 'Gruppe')

# create dictionary subgrouping gruppes according to their category3 valaue
subgroups = ss.group_dictdfs(results_gr)
print(subgroups.keys())

# Define your metrics
metrics = {
    'MFS': 'area_median_percdiff_to_y1',
    'mperi': 'peri_median_percdiff_to_y1',
    'MeanPAR': 'medianPAR_percdiff_to_y1',
    'Fields/TotalArea': 'fields_ha_percdiff_to_y1'
}

# Iterate over each key in subgroup subsample dict to plot each subgroup
for subgroup_name, subgroup_dict in subgroups.items():
    title = f"{subgroup_name} Metrics Over Time"
    ylabel = "Metric Value"  # Customize as needed
    pm.multimetric_ss_plot(subgroup_dict, title, ylabel, metrics)


