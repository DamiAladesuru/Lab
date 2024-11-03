'''Visualizations'''
# %% Importing modules
import os


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.raw import gridgdf_desc_raw as gdr
from src.visualization import plotting_module as pm

''' in pm, we have intialize_plotting which we have to run first to set up the color dictionary and plot
 for the first time. After that we can use other functions directly and metric colors will be consistent across all plots.'''

# %% load data
gld_ext, gridgdf_raw = gdr.silence_prints(gdr.create_gridgdf_raw)
grid_allyears_raw, grid_yearly_raw = gdr.silence_prints(gdr.desc_grid,gridgdf_raw)


# %% define objects for plotting
multiline_df = grid_yearly_raw # or gextyd for plotting trend without grid
correlation_df = gridgdf_raw
#correlation_wtoutlier = gridgdf_wtoutlier

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
pm.plot_correlation_matrix(correlation_df, 'Correlation Matrix of Field Metrics')

# %% correlation matrix comparing with and without outliers
pm.plot_correlation_matrices(correlation_wtoutlier, correlation_df, 
                          'Correlation Matrix with Outliers', 
                          'Correlation Matrix without Outliers')


# %% for subsample over years data
from src.analysis.raw import gld_desc_raw as gr

gld, dict, gruppe_count = gr.gld_overyears()

# Run multimetric_plot for subsample dictionary
pm.multimetric_ss_plot(dict, 'Metrics Over Years by Gruppe', 'Percentage Change in Field Metric Value from 2012',
                       metrics={'mperi': 'mperi_apercdiff_y1',
                                'MFS': 'mfs_ha_apercdiff_y1',
                                'Fields/Ha': 'fields_ha_apercdiff_y1',
                                'MeanPAR': 'mean_par_apercdiff_y1'
                                })