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
# I always want to load gridgdf and process clean gridgdf separately so I can have uncleeaned data for comparison or sensitivity analysis
gridgdf_cl, _ = gd.clean_gridgdf(gridgdf)
# %%
_, grid_yearly_cl = gd.silence_prints(gd.desc_grid,gridgdf_cl)
_, grid_yearly = gd.silence_prints(gd.desc_grid,gridgdf)

# %% define objects for plotting
multiline_df = grid_yearly
#correlation_df = gridgdf_cl
#corration_wtoutlier = gridgdf

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
        'MeanPAR': 'medianPAR_percdiff_to_y1'
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

# %% examine the top 5 groups with the highest average area sum
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


###################################################
# exclude some groups from the data and run plots for entire data
# to see effect of excluding groups on total area
######################################################
# %% load data without grid
from src.analysis.desc import gld_desc_raw as gdr

# Define groups to exclude
excgroups = ['dauerkulturen', 'others'] #'ffc', 'environmental', 'dauergrünland'
# Call the function
_, gydesc_exc = gdr.gld_overyears_filt('category3', excgroups)

# %% plot total area for entire data
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

###################################################
# work with dicctionaries of dataframes
###################################################
'''e.g., dictionaries created in datamani or''' 
from src.analysis import subsampling_mod as ssm
gridgdf_dict, combined_gridgdf_ss = ssm.create_gridgdf_ss(gld_base, 'Gruppe')
result = ssm.ss_desc(gridgdf_dict)

#gld_dict, combined_gld_ss = ssm.create_gld_ss(gld_base, 'category3')
# %%
# Iterate over the dictionary returned by create_gridgdf_ss
for subsample_name, subsample in gridgdf_dict.items():
    unique_years = sorted(subsample['year'].unique())
    pm.stack_plots_in_grid(
        subsample, 
        unique_years, 
        pm.scatterplot_mpar_marea,
        col1 = "fields",
        col2 = "medpar",
        ncols=4, 
        figsize=(25, 15), 
        grid_title=f"Scatterplots for Subsample {subsample_name}"  # Use the subsample name for naming
    )


# %%
# Iterate over the dictionary returned by create_gld_ss
for subsample_name, subsample in gld_dict.items():
    unique_years = sorted(subsample['year'].unique())
    pm.stack_plots_in_grid(
        subsample, 
        unique_years, 
        pm.scatterplot_par_area,
        col1 = "",
        col2 = "",
        ncols=4, 
        figsize=(25, 15), 
        grid_title=f"{subsample_name}"  # Use the subsample name for naming
    )
# %%
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=result['combined_grid_yearly'], x='year', y='mfs_ha_mean', hue='subsample',
             marker='o')

# Add titles and labels
plt.title('Trend of Average MFS (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average MFS (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Show the plot
plt.show()


# %%
for key, gdf in gridgdf_dict.items():
    # Convert the GeoDataFrame to EPSG 4326
    geoData = gdf.to_crs(epsg=4326)
    
    # Plot the choropleth, including the key as part of the title
    pm.plot_facet_choropleth_with_geoplot(
        geoData, 
        column='mpar', 
        cmap='plasma', 
        year_col='year', 
        ncols=4, 
        title=f"Choropleth for Subsample: {key}"
    )

############################################
# maps
############################################
geoData = gdf.to_crs(epsg=4326)
# %% simple plot
geoData.plot()

# %% subset data to plot single data
g_23 = geoData[geoData['year'] == 2023]

# %% or loop through years
for year in geoData['year'].unique():
    g_year = geoData[geoData['year'] == year]
    g_year.plot()
    plt.title(f'Year {year}')
    plt.show()
    
 # %%   
# Assuming `geoData` is your GeoDataFrame with 'year' and 'medpar' columns
geoData = gridgdf_cl.to_crs(epsg=4326)
pm.plot_facet_choropleth_with_geoplot(geoData,
                                      column='medpar_percdiff_to_y1',
                                      cmap='plasma', year_col='year', ncols=4)
# %% very many lines
# Set the plot style
sns.set(style="whitegrid")

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the line plots on the same axis

#sns.lineplot(data=grid_yearly, x='year', y='mperi_apercdiffy1', ax=ax, label='MPeri', marker='o')
#sns.lineplot(data=grid_yearly_cl, x='year', y='mfs_ha_apercdiffy1', ax=ax, label='MFSc', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='mfs_ha_apercdiffy1', ax=ax, label='MFS', marker='o')
#sns.lineplot(data=gydesc, x='year', y='area_mean_percdiff_to_y1', ax=ax, label='MFSgy', marker='o')
sns.lineplot(data=gydesc, x='year', y='medianPAR_percdiff_to_y1', ax=ax, label='Medpargy', marker='o')
sns.lineplot(data=grid_yearly_cl, x='year', y='medpar_percdiffy1_med', ax=ax, label='Medparc', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='med_fsha_percdiffy1_med', ax=ax, label='MedFS', marker='o')
sns.lineplot(data=grid_yearly_cl, x='year', y='medpar_apercdiffy1', ax=ax, label='Medparac', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='med_fsha_apercdiffy1', ax=ax, label='MedFSa', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='fields_ha_apercdiffy1', ax=ax, label='Fields/Ha', marker='o')
#sns.lineplot(data=grid_yearly_cl, x='year', y='mpar_apercdiffy1', ax=ax, label='MPAR', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='medpar_percdiffy1_med', ax=ax, label='MedPAR', marker='o')
#sns.lineplot(data=grid_yearly, x='year', y='medpar_apercdiffy1', ax=ax, label='MedPARa', marker='o')

# Set the labels and title
plt.xlabel('Year')
plt.ylabel('Relative Diff to Year 1 (%)')
plt.title('Trend of FiSC Over Time')

# Set the legend
plt.legend(title='FiSC', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

# %% density plot with mean median markers# %%
data = gridgdf_cl[gridgdf_cl['year'] == 2018]
# Calculate mean and median
mean = np.mean(data['medfs_ha'])
median = np.median(data['medfs_ha'])

# Plot density plot
plt.figure(figsize=(8, 6))
sns.kdeplot(data, x = 'medfs_ha', fill=True, color='skyblue', alpha=0.5, label='Density')

# Add vertical lines for mean and median
plt.axvline(mean, color='red', linestyle='--', label=f'Mean: {mean:.2f}')
plt.axvline(median, color='green', linestyle='-', label=f'Median: {median:.2f}')

# Add legend and labels
plt.title('2018 Density Plot for medfs_ha with Mean and Median')
plt.xlabel('Values')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()