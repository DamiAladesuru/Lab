'''Visualizations'''
# %% Importing modules
import os

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.raw import gridgdf_desc_raw as grdr
from src.analysis import gridgdf_desc2 as gd
from src.visualization import plotting_module as pm

''' in pm, we have intialize_plotting which we have to run first to set up the color dictionary and plot
 for the first time. After that we can use other functions directly and metric colors will be consistent across all plots.'''

# %% load data
gld_ext, gridgdf_raw = grdr.silence_prints(grdr.create_gridgdf_raw)
grid_allyears_raw, grid_yearly_raw = grdr.silence_prints(grdr.desc_grid,gridgdf_raw)


# %% without 'outlier' in gld and gridgdf
#gld_trimmed, gridgdf = gd.create_gridgdf()
#grid_allyears_stats, grid_yearly_stats = gd.desc_grid(gridgdf)

# %% define objects for plotting
multiline_df = grid_yearly_raw
#correlation_df = gridgdf_raw
#correlation_wtoutlier = gridgdf_raw

# %%
# Define the path for the color dictionary
color_dict_path = 'reports/figures/ToF/label_color_dict.pkl'

# Initial call to set up the color dictionary and plot absolute change in field metrics
pm.initialize_plotting(
    df=multiline_df,
    title='Trend of Absolute Change in Field Metric Value Over Time',
    ylabel='Average Absolute Change',
    metrics={
        'MFS': 'mfs_ha_adiff_y1',
        'mperi': 'mperi_adiff_y1',
        'MeanPAR': 'mean_par_adiff_y1',
        'Fields/Ha': 'fields_ha_adiff_y1'
    },
    color_dict_path=color_dict_path
)

# %% Multimetric plot of yearly average percentage change in field metrics
pm.multimetric_plot(
    df=multiline_df, # or grid_yearly_stats_wtoutlier
    title='Trend of Change in Field Metric Value Over Time',
    ylabel = 'Av. Percentage Change in Field Metric Value from 2012',
    metrics={
        'mperi': 'mperi_apercdiff_y1',
        'MFS': 'mfs_ha_apercdiff_y1',
        'Fields/Ha': 'fields_ha_apercdiff_y1',
        'MeanPAR': 'mean_par_apercdiff_y1'
        }
)

# %% correlation matrix
target_columns = ['mfs_ha_percdiff_to_y1', 'fields_ha_percdiff_to_y1', 'mean_par_percdiff_to_y1', 'mperi_percdiff_to_y1']
#target_columns = ['mfs_ha_adiff_y1', 'fields_ha_adiff_y1', 'grid_par_adiff_y1', 'grid_par2_adiff_y1', 'mean_par_apercdiff_y1', 'mperi_adiff_y1'] #for absolute change
new_column_names = ['Δ_mfs', 'Δ_f/ha', 'Δ_MeanPAR', 'Δ_mperi']

# %% single correlation matrix
pm.plot_correlation_matrix(correlation_df, 'Correlation Matrix of Field Metrics', target_columns, new_column_names)

# %% correlation matrix comparing with and without outliers
pm.plot_correlation_matrices(correlation_wtoutlier, correlation_df, 
                          'Correlation Matrix with Outliers', 
                          'Correlation Matrix without Outliers')


############################################
# analysis without grid
############################################
# %% load data without grid
from src.analysis.raw import gld_desc_raw as gdr
gld, gydesc = gdr.gld_overyears() # or _, gydesc_filt = gld_overyears_filt(x = 'sonstige flächen')

# %% Multimetric plot of yearly average percentage change in field metrics
pm.multimetric_plot(
    df=gydesc, 
    title='Trend of Change in Field Metric Value Over Time',
    ylabel = 'Percentage Change in Field Metric Value from 2012',
    metrics={
        'mperi': 'peri_mean_percdiff_to_y1',
        'MFS': 'area_mean_percdiff_to_y1',
        'Fields/Ha': 'fields_ha_percdiff_to_y1',
        'MeanPAR': 'meanPAR_percdiff_to_y1'
        }
)
# %% correlation matrix
target_columns = ['area_mean_percdiff_to_y1', 'fields_ha_percdiff_to_y1', 'meanPAR_percdiff_to_y1', 'peri_mean_percdiff_to_y1']
#target_columns = ['mfs_ha_adiff_y1', 'fields_ha_adiff_y1', 'grid_par_adiff_y1', 'grid_par2_adiff_y1', 'mean_par_apercdiff_y1', 'mperi_adiff_y1'] #for absolute change
new_column_names = ['Δ_mfs', 'Δ_f/ha', 'Δ_MeanPAR', 'Δ_mperi']

# single correlation matrix
pm.plot_correlation_matrix(gydesc, 'Correlation Matrix of Field Metrics', target_columns, new_column_names)

# %% 
##################################################
# multimetric facet plot for subsamples of data
##################################################	
# Call the subsampling data function
from src.analysis import subsampling_mod as ss

gld, results, gruppe_count = ss.gldss_overyears(column = 'Gruppe')
results_gr = ss.group_dictdfs(results)
print(results.keys())

# %%
# Define your metrics
metrics = {
    'MFS': 'area_mean_percdiff_to_y1',
    'mperi': 'peri_mean_percdiff_to_y1',
    'MeanPAR': 'meanPAR_percdiff_to_y1',
    'Fields/Ha': 'fields_ha_percdiff_to_y1'
}

# Call the modified multimetric_plot function either
# a. For one facet plot containing all subgroups
# pm.multimetric_ss_plot(results, '#', 'Percentage Change in Field Metric Value from 2012', metrics)

# or preferably, for subgrouped facet plots

# %%
# Iterate over each key in ss_dict_gr to plot each subgroup
for subgroup_name, subgroup_dict in results_gr.items():
    title = f"{subgroup_name} Metrics Over Time"
    ylabel = "Metric Value"  # Customize as needed
    pm.multimetric_ss_plot(subgroup_dict, title, ylabel, metrics)
