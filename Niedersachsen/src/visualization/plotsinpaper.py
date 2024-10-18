# %%
import os
import pandas as pd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis_and_models import describe_single as ds

#from src.visualization import heatmaps

gld, griddf, griddf_ext, grid_year_average, landkreis_average, category2_average, gridgdf = ds.process_descriptives()

# %% Single line plots
####################################
# for data and descriptives section (Figure 1)
####################################
# Line plot of yearly change in count of fields
sns.lineplot(data=grid_year_average, x='year', y='sum_fsha_sum', color='teal')
# Set the plot title and labels
#plt.title('Trend of Total Agricultural Land Area (ha)')
plt.xlabel('Year')
plt.ylabel('Total Agricultural Land Area (ha)')
# Show the plot
plt.legend()
plt.show()
#save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))

# %% Line plot of yearly change in total land area
sns.lineplot(data=desc, x='year', y='fs_sumsum_diff_2012', color='purple')
# Set the plot title and labels
plt.title('Change in total agricultural land area (ha) from 2012')
plt.xlabel('Year')
plt.ylabel('Difference from 2012 of total agricultural land area (ha)')
# Show the plot
plt.legend()
plt.show()
#save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))

# %% Line plot of yearly change in mean field size
sns.lineplot(data=desc, x='year', y='mfshm_diff_2012', color='purple')
# Set the plot title and labels
#plt.title('Change in total agricultural land area (ha) from 2012')
plt.xlabel('Year')
plt.ylabel('Difference from 2012 of mean field size (ha)')
# Show the plot
plt.legend()
plt.show()
#save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))

# %% Line plot of yearly change in mean field size
sns.lineplot(data=desc, x='year', y='mean_fields_ha_diff12', color='purple')
# Set the plot title and labels
#plt.title('Change in total agricultural land area (ha) from 2012')
plt.xlabel('Year')
plt.ylabel('Difference from 2012 of average field/ha')
# Show the plot
plt.legend()
plt.show()
#save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))


# %% Mean MCPAR circle and square (Figure 2)
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

sns.lineplot(data=grid_year_average, x='year', y='mean_mean_cpar', label='Mean MCPAR_circle', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_mean_cpar2', label='Mean MCPAR_square', marker='o')

# Add titles and labels
plt.title('Trend of Yearly Mean of Grid MeanCPAR_circle and Grid MeanCPAR_square')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()


# %% Figure 3
# Single plot of target grid cell with certain metric value for one year
########################################################################
def plot_gridcell_single(gridgdf, gridcell, year):
    """
    Plots a single grid cell for a given year.

    Parameters:
    gridgdf (DataFrame): The DataFrame containing grid data.
    gridcell (str): The CELLCODE of the grid cell to plot.
    year (int): The year to plot.
    """
    # Plot the grid cell
    fig_single, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 500x500 pixels
    gld[(gld['CELLCODE'] == gridcell) & (gld['year'] == year)].plot(ax=ax)
    
    # overlay the grid cell with the bounding boxes from column 'bbox'
    # gld[(gld['CELLCODE'] == gridcell) & (gld['year'] == year)].bbox.plot(ax=ax, edgecolor='red', facecolor='none')
    
    # Disable the grid
    ax.grid(False)
    
    # Annotate the plot with the metric value 'CELLCODE', 'year', 'fields', 'mfs_ha', 'mean_cpar',
    metrics = ['lsi', 'grid_polspy']
    
    # Get metric values
    gridcell_data = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)]
    
    if not gridcell_data.empty:
        # Annotate each metric under the subplot
        for j, metric in enumerate(metrics):
            metric_value = gridcell_data[metric].values[0] if metric in gridcell_data else 'N/A'
            ax.annotate(f'{metric}: {metric_value}', xy=(0.5, -0.20 - j*0.1), 
                        xycoords='axes fraction', ha='center', fontsize=10)
    
    # Save plot to directory
    # fig_single.savefig(f'reports/figures/gridcell_{gridcell}_{year}.png', dpi=100)
    plt.show()

# Example usage
plot_gridcell_single(gridgdf, '10kmE412N339', 2019)
plot_gridcell_single(gridgdf, '10kmE441N330', 2020)


# %% Figure 4: 
########################################################################
# %%
fields2 = griddf_ext[griddf_ext['fields'] == 2]
merged = gld.merge(fields2, on=['CELLCODE', 'year'], how='left', indicator=True)
subsample_df = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])

def plot_subsample(gridgdf):
    # List of metrics for annotation
    metrics = ['CELLCODE', 'year', 'fields', 'mfs_ha', 'mean_cpar', 'lsi', 'mean_polspy', 'grid_polspy']

    # Get unique combinations of 'year' and 'CELLCODE'
    unique_combinations = subsample_df[['year', 'CELLCODE']].drop_duplicates()
    
    # Loop through each unique combination
    for i, (year, gridcell) in enumerate(unique_combinations.itertuples(index=False)):
        # Filter data for the current year and grid cell
        yearly_data = subsample_df[(subsample_df['CELLCODE'] == gridcell) & (subsample_df['year'] == year)]
        
        # Create a new figure and axis for each plot
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Plotting, assuming 'geometry' column or other relevant data is present for plotting
        yearly_data.plot(ax=ax, legend=False)  # Customize based on how you want to plot the fields
        
        # Set the plot title
        ax.set_title(f'Gridcell {gridcell} in {year}')

        # Get metric values for annotation
        gridcell_data = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)]
        
        if not gridcell_data.empty:
            # Annotate each metric under the plot
            for j, metric in enumerate(metrics):
                metric_value = gridcell_data[metric].values[0] if metric in gridcell_data else 'N/A'
                ax.annotate(f'{metric}: {metric_value}', xy=(0.5, -0.20 - j*0.1), 
                            xycoords='axes fraction', ha='center', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Show the individual plot
        plt.show()

# Call the function with your GeoDataFrame
plot_subsample(gridgdf)

# %% Figure 5: 
########################################################################
def plot_gridcell_single(gridgdf):
    # Define metrics
    metrics = ['lsi', 'grid_par']
    
    # Find grid cells with min and max values for each metric
    min_max_gridcells = {}
    for metric in metrics:
        min_value = gridgdf[metric].min()
        max_value = gridgdf[metric].max()
        min_gridcell = gridgdf.loc[gridgdf[metric] == min_value, 'CELLCODE'].values[0]
        max_gridcell = gridgdf.loc[gridgdf[metric] == max_value, 'CELLCODE'].values[0]
        min_year = gridgdf.loc[gridgdf[metric] == min_value, 'year'].values[0]
        max_year = gridgdf.loc[gridgdf[metric] == max_value, 'year'].values[0]
        min_max_gridcells[metric] = {'min': (min_gridcell, min_year), 'max': (max_gridcell, max_year)}
    
    # Plot each grid cell with min and max values
    for metric, values in min_max_gridcells.items():
        for value_type, (gridcell, year) in values.items():
            fig_single, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 500x500 pixels
            gld[(gld['CELLCODE'] == gridcell) & (gld['year'] == year)].plot(ax=ax)
            # overlay the grid cell with the bounding boxes from column 'bbox'
            # gld[(gld['CELLCODE'] == gridcell) & (gld['year'] == year)].bbox.plot(ax=ax, edgecolor='red', facecolor='none')
            # Disable the grid
            ax.grid(False)
            # Annotate the plot with the metric value 'CELLCODE', 'year', 'fields', 'mfs_ha', 'mean_cpar',
            gridcell_data = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)]
            if not gridcell_data.empty:
                # Annotate each metric under the subplot
                for j, metric in enumerate(metrics):
                    metric_value = gridcell_data[metric].values[0] if metric in gridcell_data else 'N/A'
                    ax.annotate(f'{metric}: {metric_value}', xy=(0.5, -0.20 - j*0.05), 
                                xycoords='axes fraction', ha='center', fontsize=10)
            # Save plot to directory
            # fig_single.savefig(f'reports/figures/gridcell_{gridcell}_{year}_{value_type}.png', dpi=100)
            plt.show()

# Example usage
plot_gridcell_single(gridgdf)

# %% Figure 6: 
########################################################################
sns.scatterplot(data=gridgdf, x='grid_polspy', y='lsi', color='purple')
# Add line of best fit
sns.regplot(data=gridgdf, x='grid_polspy', y='lsi', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between grid_polspy and LSI')
plt.xlabel('grid_polspy')
plt.ylabel('lsi')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
sns.scatterplot(data=gridgdf, x='mean_polspy', y='mean_cpar2', color='purple')
# Add line of best fit
sns.regplot(data=gridgdf, x='mean_polspy', y='mean_cpar2', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between mean_polspy and mean_cpar')
plt.xlabel('mean_polspy')
plt.ylabel('mean_cpar2')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %% Figure 7: 
########################################################################
sns.scatterplot(data=gridgdf, x='fields_ha_diff_from_2012', y='mfs_ha_diff_from_2012', color='purple')
# Add line of best fit
sns.regplot(data=gridgdf, x='fields_ha_diff_from_2012', y='mfs_ha_diff_from_2012', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between Change in fields_ha and mfs_ha')
plt.xlabel('fields_ha_diff_from_2012')
plt.ylabel('mfs_ha_diff_from_2012')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# %% Figure 8: Field level Multi-line plot
########################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

#plot metrics
sns.lineplot(data=grid_year_average, x='year', y='mean_fields_ha_diff12', label='mean Fields/Ha change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_mfs_ha_diff12', label='mean MFS change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_mean_cpar2_diff12', label='mean MCPAR change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_mean_polsby_diff12', label='mean Compactness change from 2012', marker='o')

# Add titles and labels
plt.title('Trend of Yearly Average of FiSC Metrics from 2012 (Field level)')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()

# %% Figure 9 
########################################################################
sns.scatterplot(data=griddf_ext, x='mfs_ha_diff_from_2012', y='lsi_diff_from_2012', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='mfs_ha_diff_from_2012', y='lsi_diff_from_2012', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between Change in Grid Mean Field Size and LSI')
plt.xlabel('mfs_ha_diff_from_2012')
plt.ylabel('lsi_diff_from_2012')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


sns.scatterplot(data=gridgdf, x='fields_ha_diff_from_2012', y='lsi_diff_from_2012', color='purple')
# Add line of best fit
sns.regplot(data=gridgdf, x='fields_ha_diff_from_2012', y='lsi_diff_from_2012', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between Change in fields_ha and LSI')
plt.xlabel('fields_ha_diff_from_2012')
plt.ylabel('lsi_diff_from_2012')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %% Figure 10: Grid level Multi-line plot
######################################################################## 
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

#plot metrics
sns.lineplot(data=grid_year_average, x='year', y='mean_fields_ha_diff12', label='mean Fields/Ha change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_mfs_ha_diff12', label='mean MFS change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_lsi_diff12', label='mean LSI change from 2012', marker='o')
sns.lineplot(data=grid_year_average, x='year', y='mean_grid_polspy_diff12', label='mean Compactness change from 2012', marker='o')

# Add titles and labels
plt.title('Trend of Yearly Average of FiSC Metrics from 2012 (Grid level)')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()

#Heatmaps
########################################################################
griddf_ext = heatmaps.process_dataframe(griddf_ext)
heatmaps.plot_heatmap(griddf_ext, 'mfs_fields_ha_diff_group', 'lsi_polspy_diff_group', 
             'Heatmap of Change in Size and Count vs Grid Shape Complexity', 
             'lsi_polspy_diff_group', 'mfs_fields_ha_diff_group')

heatmaps.plot_heatmap(griddf_ext, 'mfs_fields_ha_diff_group', 'MCPAR_polspy_diff_group', 
             'Heatmap of Change in Size and Count vs Mean Shape Complexity', 
             'MCPAR_polspy_diff_group', 'mfs_fields_ha_diff_group')


# next steps: 
# kulturcode gruppe: change in Fisc for each gruppe

