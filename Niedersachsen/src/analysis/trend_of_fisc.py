# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import csv
import pickle

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis import gridgdf_desc2 as gd

''' This script is used to analyze the trend of field metrics over time without subsampling.'''

output_dir = 'reports/figures/ToF'

# %%
gld_trimmed, gridgdf = gd.create_gridgdf()
grid_allyears_stats, grid_yearly_stats = gd.desc_grid(gridgdf)

# %% to load data with outlier, uncomment the following lines
#gld_ext, gridgdf_wtoutlier = gd.create_gridgdf_wtoutlier()
#grid_allyears_stats__wtoutlier, grid_yearly_stats_wtoutlier = gd.desc_grid(gridgdf_wtoutlier)

# %%
# Global dictionary to store colors for each label after first plot
label_color_dict = {}

# %% create multimetric plot and save the colors for each label for consistency
def multimetric_plot(df, title, metrics):
    
    global label_color_dict  # Access the global color dictionary
    
    # Set the plot style
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))  # Use `ax` to manage the axes
    
    # Plot each metric, checking if a color is already assigned
    for label, column in metrics.items():
        if label in label_color_dict:
            # Use the stored color for the label if it exists
            color = label_color_dict[label]
            sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
        else:
            # Plot without specifying color to allow Seaborn to assign it
            line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
            
            # Retrieve the color from the Line2D object in the Axes' lines
            color = line.get_lines()[-1].get_color()
            label_color_dict[label] = color  # Save color to dictionary

    # Add titles and labels
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Absolute Change in Field Metric Value from 2012')
    plt.legend(title='Metrics')

    # Show the plot
    plt.show()
    
    # Return the color dictionary for external use
    #return label_color_dict

# %% 1a. First plot with one set of variables
multimetric_plot(
    df=grid_yearly_stats, # or grid_yearly_stats_wtoutlier
    title='Trend of Percentage Change in Field Metric Value Over Time',
    metrics={
        'mperi': 'mperi_apercdiff_y1',
        'MFS': 'mfs_ha_apercdiff_y1',
        'Fields/Ha': 'fields_ha_apercdiff_y1',
        'MeanPAR': 'mean_par_apercdiff_y1'
        }
)
# %% Save label_color_dict to a file
def save_color_dict(color_dict, filename='reports/figures/ToF/label_color_dict.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(color_dict, f)
save_color_dict(label_color_dict)    


# %%
# 1b. multiline plot of actual change in metrics over time
multimetric_plot(
    df=grid_yearly_stats,
    title='Trend of Average Absolute Change in Field Metric Value Over Time',
    metrics={
        'MFS': 'mfs_ha_adiff_y1',
        'mperi': 'mperi_adiff_y1',
        'MeanPAR': 'mean_par_adiff_y1',
        'Fields/Ha': 'fields_ha_adiff_y1'
    }
)
# with absolute, we don't see the trend in GridPAR or F/ha as much as with percentage change because of the scale of the values

#############################################################################
''' The following code is for additional analysis that may be useful'''
# %% single correlation matrix without outlier
def test_correlation(df):
    # Select target columns
    target_columns = ['mfs_ha_percdiff_to_y1', 'fields_ha_percdiff_to_y1', 'mean_par_percdiff_to_y1', 'mperi_percdiff_to_y1']
    #target_columns = ['mfs_ha_adiff_y1', 'fields_ha_adiff_y1', 'grid_par_adiff_y1', 'grid_par2_adiff_y1', 'mean_par_apercdiff_y1', 'mperi_adiff_y1'] #for absolute change
    new_column_names = ['Δ_mfs', 'Δ_f/ha', 'Δ_MeanPAR', 'Δ_mperi']
    
    # Calculate correlation matrix
    correlation_matrix = df[target_columns].corr()
    
    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = new_column_names
    correlation_matrix.index = new_column_names
    
    return correlation_matrix

def plot_correlation_matrix(df, title):
    # Get the correlation matrix
    corr_matrix = test_correlation(df)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_matrix(gridgdf, 'Correlation Matrix without Outliers')

# %% Correlation matrix for metrics with and without outlier
''' requires data with outlier to be loaded'''
def test_correlation(df):
    # Select target columns
    target_columns = ['mfs_ha_percdiff_to_y1', 'fields_ha_percdiff_to_y1', 'mean_par_percdiff_to_y1', 'mperi_percdiff_to_y1']
    #target_columns = ['mfs_ha_adiff_y1', 'fields_ha_adiff_y1', 'grid_par_adiff_y1', 'grid_par2_adiff_y1', 'mean_par_apercdiff_y1', 'mperi_adiff_y1'] #for absolute change
    new_column_names = ['Δ_mfs', 'Δ_f/ha', 'Δ_MeanPAR', 'Δ_mperi']
    
    # Calculate correlation matrix
    correlation_matrix = df[target_columns].corr()
    
    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = new_column_names
    correlation_matrix.index = new_column_names
    
    return correlation_matrix

def plot_correlation_matrices(df1, df2, title1, title2):
    # Get the correlation matrices
    corr_matrix1 = test_correlation(df1)
    corr_matrix2 = test_correlation(df2)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first correlation matrix
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[0])
    axes[0].set_title(title1)
    
    # Plot the second correlation matrix
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[1])
    axes[1].set_title(title2)
    
    plt.tight_layout()
    plt.show()

# Example usage
plot_correlation_matrices(gridgdf, gridgdf_wtoutlier, 
                          'Correlation Matrix without Outliers', 
                          'Correlation Matrix with Outliers')

# in case it makes sense to use yearly aggregated data
    # Select target columns
    #target_columns = ['mfs_ha_apercdiff_y1', 'fields_ha_apercdiff_y1', 'grid_par_apercdiff_y1', 'grid_par2_apercdiff_y1', 'mean_par_apercdiff_y1', 'mperi_apercdiff_y1', 'edges_apercdiff_y1']
    #target_columns = ['mfs_ha_adiff_y1', 'fields_ha_adiff_y1', 'grid_par_adiff_y1', 'grid_par2_adiff_y1', 'mean_par_apercdiff_y1', 'mperi_adiff_y1'] #for absolute change
    #new_column_names = ['Δ_mfs', 'Δ_f/ha', 'Δ_GridPAR', 'Δ_GridPAR2', 'Δ_MeanPAR', 'Δ_mperi', 'Δ_edges']

# %%
# 2a. multiline plot of percentage change in size, count and shape metrics in aggregate area over time
def multimetric_plot(df, title):
    # Set the plot style
    sns.set(style="whitegrid")

    # Create a line plot
    plt.figure(figsize=(12, 6))

    #plot metrics
    sns.lineplot(df, x='year', y='fsha_sum_apercdiff_y1', label='Area', marker='o')
    sns.lineplot(df, x='year', y='totuperi_apercdiff_y1', label='Peri', marker='o')
    sns.lineplot(df, x='year', y='totuedges_apercdiff_y1', label='edges', marker='o')
    sns.lineplot(df, x='year', y='grid_par_apercdiff_y1', label='GridPAR', marker='o')
    sns.lineplot(df, x='year', y='fields_ha_apercdiff_y1', label='Fields/Ha', marker='o')
        
    # Add titles and labels
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Percentage Change in Aggregate Area Metric from 2012')
    plt.legend(title='Metrics')

    # Show the plot
    plt.show()

multimetric_plot(df = grid_yearly_stats, title = 'Trend of Average Percentage Change in Aggregate Area Metric Over Time')
# or multimetric_plot(df = grid_yearly_stats_wtoutlier, title = 'Trend of Average Percentage Change in Aggregate Area Metric  Over Time (with Outlier)')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# %%    
# Load color dictionary from JSON
def load_color_dict(filename='reports/figures/ToF/label_color_dict.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Load in another script
label_color_dict = load_color_dict()   
    
    
