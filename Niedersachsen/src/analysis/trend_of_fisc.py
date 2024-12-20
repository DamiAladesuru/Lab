'''Visualizations'''
# %% Importing modules
import os

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.desc import gridgdf_desc as gd
from src.visualization import plotting_module as pm

''' We use this script to visualize FiSC. In pm, we have intialize_plotting which we have to run first to set up the color dictionary and plot
 for the first time. After that we can use other functions directly and metric colors will be consistent across all plots.'''

# %% load data
gld, gridgdf = gd.silence_prints(gd.create_gridgdf)
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)
_, grid_yearly = gd.silence_prints(gd.desc_grid,gridgdf_cl)

# %% define objects for plotting
multiline_df = grid_yearly
#correlation_df = gridgdf_cl
#correlation_wtoutlier = gridgdf

# %%
# Define the path for the color dictionary
color_dict_path = 'reports/figures/ToF/label_color_dict.pkl'

# Initial call to set up the color dictionary and plot absolute change in field metrics
pm.initialize_plotting(
    df=multiline_df,
    title='Trend of Absolute Change in Field Metric Value Over Time',
    ylabel='Average Absolute Change',
    metrics={
        'MFS': 'mfs_ha_adiffy1',
        'mperi': 'mperi_adiffy1',
        'MeanPAR': 'mpar_adiffy1',
        'Fields/Ha': 'fields_ha_adiffy1'
    },
    color_dict_path=color_dict_path
)

# %% Multimetric plot of yearly average percentage change in field metrics
pm.multimetric_plot(
    df=multiline_df, # or grid_yearly_stats_wtoutlier
    title='Trend of Change in Field Metric Value Over Time',
    ylabel = 'Av. Percentage Change in Field Metric Value from 2012',
    metrics={

        'MFS': 'mfs_ha_apercdiffy1',
        'mperi': 'mperi_apercdiffy1',
        'MeanPAR': 'mpar_apercdiffy1',
        'Fields/Ha': 'fields_ha_apercdiffy1',
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
from src.analysis.desc import gld_desc_raw as gdr
gy, gydesc = gdr.gld_overyears() 

# or apply group filter before creating gydesc
#xgroup =['unbef.mieten.auf al', 'pilze unter glas']
#_, gydesc_filt = gdr.gld_overyears_filt('par', xgroup)

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
#########################################
# working with subsamples of data
#########################################
# Call the subsampling data function and load data
from src.analysis import subsampling_mod as ss

gld, results_gr, _ = ss.gldss_overyears(column = 'Gruppe') # or _, results_cat,_  = ss.gldss_overyears(column = 'category3') #'category3'

# define objetcs for further use
results = ... # results_gr or results_cat

# create dictionary subgrouping gruppes according to their category3 valaue
subgroups = ss.group_dictdfs(results)
print(subgroups.keys())

# %% multimetric facet plot for each subgroup
##################################################
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
# Iterate over each key in subgroup subsample dict to plot each subgroup
for subgroup_name, subgroup_dict in subgroups.items():
    title = f"{subgroup_name} Metrics Over Time"
    ylabel = "Metric Value"  # Customize as needed
    pm.multimetric_ss_plot(subgroup_dict, title, ylabel, metrics)


# single line plots
##################################################
# %% process subsamples for single line plots
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

# first, create a column called 'Gruppe' in each df such that the column value is the key of the dictionary
# then, concatenate all the dfs into one df
for key in results:
    results[key]['Gruppe'] = key
    cols = list(results[key].columns)
    cols.insert(1, cols.pop(cols.index('Gruppe')))
    results[key] = results[key][cols]

# Concatenate all DataFrames into one
gr_combined = pd.concat(results.values(), ignore_index=True) # or cats_combined

# again, assign the concatenated df to a new dynamic variable
ss_combined = gr_combined # or cats_combined
ss_combined.info()

#%%  a. total land area over time
# for entire dataset
plt.figure(figsize=(8, 6))

plt.plot(gydesc['year'], gydesc['area_sum'], marker='o', color = 'purple')
''' gydesc is data for entire dataset i.e., without subsampling '''
plt.xlabel('Year')
plt.ylabel('Area Sum (ha)')
plt.title('Total Agricultural Area (ha) in Data Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

# disaggregated by Gruppe
fig = px.line(ss_combined, x='year', y='area_sum', color='Gruppe')
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Area Sum (ha)',
    title='Total Area for Each Gruppe Over Time',
    template='plotly_white'
)

#save plot as html
#fig.write_html('reports/figures/ToF/totare_groups.html')

fig.show()

#%% b. yearly change in land area 
#  for entire dataset
plt.figure(figsize=(8, 5))

plt.plot(gydesc['year'], gydesc['area_sum_percdiff_to_y1'], marker='o', color = 'purple') # or area_sum_diff_from_y1

plt.xlabel('Year')
plt.ylabel('Perc Change from 2012 in Area Sum (ha)') #retitle as needed
plt.title('Perc Change from 2012 in Total Area Over Time')
plt.grid(True)
plt.tight_layout()
# save plot as png
plt.savefig('reports/figures/ToF/totarech12_perc.png')

plt.show()

# %%
# disaggregated by Gruppe
fig = px.line(ss_combined, x='year', y='area_sum_diff_from_y1', color='Gruppe')
fig.update_layout(
    #showlegend=False,
    legend_title='Category3',
    xaxis_title='Year',
    yaxis_title='Change from 2012 in Area Sum (ha)',
    title='Change from 2012 in Total Area for Each Category3 Over Time',
    template='plotly_white'
)
# save plot as html
fig.write_html('reports/figures/ToF/totarech_cat12.html')

fig.show()

# disaggregated by Gruppe (percentage change)
fig = px.line(ss_combined, x='year', y='area_sum_percdiff_to_y1', color='Gruppe')
fig.update_layout(
    #showlegend=False,
    legend_title='Category3',
    xaxis_title='Year',
    yaxis_title='Perc Change from 2012 in Area Sum (ha)',
    title='Perc Change from 2012 in Total Area for Each Category3 Over Time',
    template='plotly_white'
)
# save plot as html
fig.write_html('reports/figures/ToF/totarepch_cat12.html')

fig.show()

