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
gld_ext, gridgdf_raw = grdr.silence_prints(grdr.create_gridgdf_raw, include_sonstige=True, filename_suffix='nole100') 
#gld_ext, gridgdf_raw = grdr.silence_prints(grdr.create_gridgdf_raw, include_sonstige=True)
#if include_sonstige = True, the data will include 'sonstige flächen' in the data and for that, you should not specify filename_suffix
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

# Define the exclude condition
exclude_condition = lambda gld: gld['area_m2'] < 100

_, gydesc = gdr.gld_overyears(exclude_condition) # or
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
########################################################################################################
# working with subsamples of data
########################################################################################################
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

###################################################
# exclude some groups from the data and run plots for entire data again
######################################################
# %% Define groups to exclude
excgroups = ['dauerkulturen', 'others'] #'ffc', 'environmental', 'dauergrünland'

# Call the function
_, gydesc_exc = gdr.gld_overyears_filt('category3', excgroups)
# %%
# total area
plt.figure(figsize=(8, 6))

plt.plot(gydesc_exc['year'], gydesc_exc['area_sum'], marker='o', color = 'purple')
plt.xlabel('Year')
plt.ylabel('Area Sum (ha)')
plt.title('Total Agricultural Area (ha) in Data Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()

#  total area change
plt.figure(figsize=(8, 5))

plt.plot(gydesc_exc['year'], gydesc_exc['area_sum_percdiff_to_y1'], marker='o', color = 'purple') # or area_sum_diff_from_y1

plt.xlabel('Year')
plt.ylabel('Perc Change from 2012 in Area Sum (ha)') #retitle as needed
plt.title('Perc Change from 2012 in Total Area Over Time')
plt.grid(True)
plt.tight_layout()
# save plot as png
plt.savefig('reports/figures/ToF/totarech12_perc.png')

plt.show()











# %% idenitfy the top 5 groups with the highest average area sum
# from combined df, extract gruppe, year and area_sum columns into another df
areasum_df = ss_combined[['Gruppe', 'year', 'area_sum']]
# Pivot the DataFrame to get years as columns and area_sum as values
areasum = areasum_df.pivot(index='Gruppe', columns='year', values='area_sum')
# Calculate the average of each Gruppe across the years
areasum['average'] = areasum.mean(axis=1)
areasum.reset_index(inplace=True)

# from area_sum, extract value of Gruppe column for top 5 rows of average
areasum.sort_values('average', ascending=False, inplace=True)
top5_gruppe = areasum['Gruppe'].head().unique()
print(top5_gruppe)

# %% bar plot to show yearly area sum for each gruppe
# Melt the DataFrame to long format for plotting
melted_df = areasum.melt(id_vars='Gruppe', var_name='year', value_name='area_sum')
# Filter out the 'average' rows
filtered_df = melted_df[melted_df['year'] != 'average']

# Create the bar plot
fig = px.bar(filtered_df, x='Gruppe', y='area_sum', color='year', barmode='group',
             title='Yearly Area Sum for Each Gruppe')

# Show the plot
fig.show()

