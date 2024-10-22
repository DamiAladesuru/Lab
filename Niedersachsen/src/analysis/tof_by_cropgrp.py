# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis import gridgdf_desc2 as gd

'''
script plots all metrics like trend_of_fisc script but disaggregates by crop group
'''

# %% dfs for subsamples
# Load or create gld_trimmed for subsample loop
output_dir = 'data/interim/gridgdf'
gld_trimmed_filename = os.path.join(output_dir, 'gld_trimmed.pkl')

if os.path.exists(gld_trimmed_filename):
    gld_trimmed = pd.read_pickle(gld_trimmed_filename)
    print(f"Loaded gld_trimmed from {gld_trimmed_filename}")
else:
    gld_trimmed = gd.adjust_trim_gld()
    gld_trimmed.to_pickle(gld_trimmed_filename)
    print(f"Saved gld_trimmed to {gld_trimmed_filename}")
    
#%%    
# List of different crop values you want to explore
cropss_list = ['others', 'ffc', 'environmental', 'dauergrünland', 'dauerkulturen']

gld_dict = {}
gridgdf_dict = {}
allyears_dict = {}
yearly_dict = {}

# Loop through the list of cropss values
for cropss in cropss_list:
    # Pass gld_trimmed as an argument
    gld_dict[f'{cropss}'], gridgdf_dict[f'{cropss}'] = gd.silence_prints(gd.create_gridgdf_subsample, cropsubsample=cropss, col2='category3', gld_data=gld_trimmed)
    allyears_dict[f'{cropss}'], yearly_dict[f'{cropss}'] = gd.silence_prints(gd.desc_grid, gridgdf_dict[f'{cropss}'])
      
print(allyears_dict.keys())
   
# %%
cropss_to_plot = 'environmental'  # Replace 'ffc' with the crop group you want to plot
# %%
# 1a. multiline plot of percentage change in size, count and shape metrics of average fields over time
def multimetric_plot(df, title):
    # Set the plot style
    sns.set(style="whitegrid")

    # Create a line plot
    plt.figure(figsize=(12, 6))

    #plot metrics
    sns.lineplot(df, x='year', y='mean_edges_apercdiff_y1', label='medges', marker='o')
    sns.lineplot(df, x='year', y='mfs_ha_apercdiff_y1', label='MFS', marker='o')
    sns.lineplot(df, x='year', y='mperi_apercdiff_y1', label='mperi', marker='o')
    sns.lineplot(df, x='year', y='mean_par_apercdiff_y1', label='MeanPAR', marker='o')

        
    # Add titles and labels
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Percentage Change in Field Metric Value from 2012')
    plt.legend(title='Metrics')

    # Show the plot
    plt.show()
    
multimetric_plot(df = yearly_dict[f'{cropss_to_plot}'], title = cropss_to_plot)
# or multimetric_plot(df = grid_yearly_stats_wtoutlier, title = 'Trend of Average Percentage Change in Field Metric Value  Over Time (with Outlier)')

# %%
# 1b. multiline plot of actual change in size, count and shape metrics of average fields over time
def multimetric_plot(df, title):
    # Set the plot style
    sns.set(style="whitegrid")

    # Create a line plot
    plt.figure(figsize=(12, 6))

    #plot metrics
    #sns.lineplot(df, x='year', y='mean_edges_adiff_y1', label='medges', marker='o')
    sns.lineplot(df, x='year', y='mfs_ha_adiff_y1', label='MFS', marker='o')
    sns.lineplot(df, x='year', y='mperi_adiff_y1', label='mperi', marker='o')
    sns.lineplot(df, x='year', y='mean_par_adiff_y1', label='MeanPAR', marker='o')

        
    # Add titles and labels
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Average Absolute Change in Field Metric Value from 2012')
    plt.legend(title='Metrics')

    # Show the plot
    plt.show()
    
multimetric_plot(df = yearly_dict[f'{cropss_to_plot}'], title = cropss_to_plot)
# or multimetric_plot(df = grid_yearly_stats_wtoutlier, title = 'Trend of Average Percentage Change in Field Metric Value  Over Time (with Outlier)')


# %% 2. Correlation matrix without outlier
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
plot_correlation_matrix(gridgdf_dict[f'{cropss_to_plot}'], 'Correlation Matrix without Outliers')
# %%
