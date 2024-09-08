# %%
import os
import pandas as pd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# %%

os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis_and_models import describe

gld_subsets, yearly_desc_dict, griddfs, griddfs_ext, mean_median, gridgdf = describe.process_descriptives() # or

# %% for working with the entire dataset ungrouped into envi and others
#from src.analysis_and_models import describe_single
#gld, griddf, griddf_ext, mean_median, gridgdf = describe_single.process_descriptives()
   
# %% check info
for key, df in griddfs_ext.items():
    print(f"Info for griddfs_ext_{key}:")
    print(df.info())

##########################
#works for multiple keys
##########################   
#%%
for key, df in mean_median.items():
    # Create line plot of yearly average change in count of fields
    sns.lineplot(data=df, x='year', y='mean_fields_diff12', color='purple', label=key)
    # Set the plot title and labels
    plt.title('Trend in Average Change in Count of Fields')
    plt.xlabel('Year')
    plt.ylabel('Average Change in Count of Fields within GridCells')
    # Show the plot
    plt.legend()
    plt.show()
    #save plot
    
    output_dir = 'reports/figures/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))

#%%
#for key, df in mean_median.items():
# Create line plot of yearly average change in grid par
sns.lineplot(data=mean_median, x='year', y='fields_ha', color='purple')
# Set the plot title and labels
#plt.title('Trend in Average Change in PercentageSum-AreaSum Ratio of GridCells')
plt.xlabel('Year')
plt.ylabel('fields_ha')
# Show the plot
plt.legend()
plt.show()
#save plot

output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'gridpar_diff12{key}.png'))
    
#%%
for key, df in yearly_desc_dict.items():
    # Create line plot of yearly sum of all field areas
    sns.lineplot(data=df, x='year', y='area_ha_sum', color='purple', label=key)
    # Set the plot title and labels
    plt.title('Total Agricultural Area in Data (ha) per Year')
    plt.xlabel('Year')
    plt.ylabel('Sum of Field Size (ha)')
    # Show the plot
    plt.legend()
    plt.show()
    #save plot
    
    output_dir = 'reports/figures/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f'area_ha_sum_{key}.png'))
            
#%%
#for key, df in mean_median.items():
    # Create line plot of yearly average change in mean field size
    meanplot = sns.lineplot(data=df, x='year', y='mean_mfs_ha_diff12', color='purple', label=key)
    for line in range(0, df.shape[0]):
        meanplot.text(df.year[line]+0.2, df.mean_mfs_ha_diff12[line], 
        round(df.mean_mfs_ha_diff12[line], 2), horizontalalignment='left', 
        size='medium', color='black', weight='semibold')
    # Set the plot title and labels
    plt.title('Trend in Mean Field Size(ha) Change over Years')
    plt.xlabel('Year')
    plt.ylabel('Change in Mean of Field Size (ha)')
    # Show the plot
    plt.legend()
    plt.show()
    #save plot
    
    output_dir = 'reports/figures/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #meanplot.get_figure().savefig(os.path.join(output_dir, f'meanfsdiffplot_{key}.png'))
                

##################################################
#correlation plots of cellcodes mfs and shape index
##################################################           
#%%
#for key, df in griddf_ext.items():
# Create scatter plot of grid mfs_ha and mean shp index
sns.scatterplot(data=griddf_ext, x='grid_par', y='PARsq', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='grid_par', y='PARsq', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between grid_par and PARsq')
plt.xlabel('grid_par')
plt.ylabel('PARsq')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f'mfs_ha_cpar_{key}.png'))
# here, when shape index is CPA, we see for envi, the correlation is negative
# while for other, it is positive. But when shape index is grid_par,
# the correlation is negative for both
#%%
#for key, df in griddfs_ext.items():
# Create scatter plot of chnage in grid mfs_ha and grid shp index
sns.scatterplot(data=gridgdf, x='fields_ha_diff_from_2012', y='lsi_diff_from_2012', color='purple')
# Add line of best fit
sns.regplot(data=gridgdf, x='fields_ha_diff_from_2012', y='lsi_diff_from_2012', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between Change in fields_ha and lsi')
plt.xlabel('fields_ha_diff_from_2012')
plt.ylabel('lsi_diff_from_2012')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'mfs_ha_cpar_diff_{key}.png'))
# here, correlation is negative for both envi and other

# %%
#for key, df in griddfs_ext.items():
# Create scatter plot of chnage in grid mfs_ha and grid shp index
sns.scatterplot(data=griddf_ext, x='mfs_ha_diff_from_2012', y='grid_polspy_diff_from_2012', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='mfs_ha_diff_from_2012', y='grid_polspy_diff_from_2012', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between Change in Grid Mean Field Size and Grid Compactness')
plt.xlabel('mfs_ha_diff_from_2012')
plt.ylabel('grid_polspy_diff_from_2012')
# Show the plot
plt.legend()
plt.show()
# Save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'mfs_ha_gpar_diff_{key}.png'))
# here, correlation is negative for both envi and other

#########################################################################

#%%
for key, df in griddfs_ext.items():
    # Filter the DataFrame for the year 2013
    df_2013 = df[df['year'] == 2023]
    
    # Create scatter plot of grid mfs_ha and mean shp index for the year 2023
    sns.scatterplot(data=df_2013, x='mfs_ha', y='grid_par', color='purple', label=key)
    # Add line of best fit
    sns.regplot(data=df_2013, x='mfs_ha', y='grid_par', scatter=False, color='blue')
    # Set the plot title and labels
    plt.title('Correlation between Grid Mean Field Size and Mean CPA ratio (2013)')
    plt.xlabel('mfs_ha')
    plt.ylabel('mean_cpar ratio')
    # Show the plot
    plt.legend()
    plt.show()
    # Save plot
    output_dir = 'reports/figures/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #plt.savefig(os.path.join(output_dir, f'mfs_ha_cpar_2023_{key}.png'))

#########################################################################


# %%
def test_inverse_relationship(griddfs_ext, col1, col2):
    # Dictionary to store the correlation results
    correlation_results = {}
    
    for key, df in griddfs_ext.items():
        if col1 in df.columns and col2 in df.columns:
            # Calculate the Pearson correlation coefficient
            correlation = df[[col1, col2]].corr().iloc[0, 1]
            correlation_results[key] = correlation
            
            # Print the result
            print(f"Correlation between {col1} and {col2} in {key}: {correlation:.3f}")
            
            # Check if the correlation suggests an inverse relationship
            if correlation < 0:
                print(f"Inverse relationship detected in {key}.")
            else:
                print(f"No inverse relationship detected in {key}.")
        else:
            print(f"Columns {col1} or {col2} not found in {key}.")

    return correlation_results

# Example usage
# Assuming 'col1' and 'col2' are the names of the columns you want to test
correlation_results = test_inverse_relationship(griddfs_ext, 'mfs_ha', 'grid_par')


# %%
# Dictionary to store DataFrames with outliers
outliers_dict = {}

# Iterate through each DataFrame in griddfs_ext
for key, df in griddfs_ext.items():
    # Filter rows where mfs_ha is greater than 10
    outliers_df = df[df['fields'] < 10]
    # Store the filtered DataFrame in the outliers_dict
    outliers_dict[key] = outliers_df

# Example usage: print the outliers for a specific key
for key, outliers_df in outliers_dict.items():
    # save the outliers to a csv file
    output_dir = 'reports/outliers'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outliers_df.to_csv(os.path.join(output_dir, f'outliers_{key}.csv'), encoding='windows-1252', index=False)
   
    
# %% 



# Dictionary to store DataFrames without outliers
griddfs_ext_no_outliers = {}

# Iterate through each DataFrame in griddfs_ext
for key, df in griddfs_ext.items():
    # Calculate the 75th percentile for mfs_ha
    percentile_75 = np.percentile(df['mfs_ha'], 75)
    # Filter out rows where mfs_ha is greater than the 75th percentile
    df_no_outliers = df[df['mfs_ha'] <= percentile_75]
    # Store the filtered DataFrame in the new dictionary
    griddfs_ext_no_outliers[key] = df_no_outliers


#%%
for key, df in griddfs_ext_no_outliers.items():
    # Create scatter plot of grid mfs_ha and mean shp index
    sns.scatterplot(data=df, x='mfs_ha', y='mean_cpar', color='purple', label=key)
    # Add line of best fit
    sns.regplot(data=df, x='mfs_ha', y='mean_cpar', scatter=False, color='blue')
    # Set the plot title and labels
    plt.title('Correlation between Grid Mean Field Size and Mean CPA ratio')
    plt.xlabel('Mean Field Size')
    plt.ylabel('Mean CPA ratio')
    # Show the plot
    plt.legend()
    plt.show()
    # Save plot
    output_dir = 'reports/figures/subsets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, f'mfs_ha_cpar_nooutliers{key}.png'))


# %%
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

#plot metrics
sns.lineplot(data=griddf_ext, x='peri_sum', y='grid_par', label='grid_par', marker='o')


# Add titles and labels
#plt.title('Trend of Yearly Average of FiSC Metrics from 2012 (Grid level)')
plt.xlabel('peri')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()

# %%
# Create scatter plot of grid mfs_ha and mean shp index
sns.scatterplot(data=griddf_ext, x='peri_sum', y='PARsq', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='peri_sum', y='PARsq', scatter=False, lowess=True, color='blue')
# Set the plot title and labels
plt.title('Correlation between peri_sum and PARsq')
plt.xlabel('peri_sum')
plt.ylabel('PARsq')
# Show the plot
plt.legend()
plt.show()
# %%
# Create scatter plot of grid mfs_ha and mean shp index
sns.scatterplot(data=griddf_ext, x='peri_sum', y='lsi', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='peri_sum', y='lsi', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between peri_sum and lsi')
plt.xlabel('peri_sum')
plt.ylabel('lsi')
# Show the plot
plt.legend()
plt.show()
# %%
# Create scatter plot of grid mfs_ha and mean shp index
sns.scatterplot(data=griddf_ext, x='grid_par', y='fsm2_sum', color='purple')
# Add line of best fit
sns.regplot(data=griddf_ext, x='grid_par', y='fsm2_sum', scatter=False, color='blue')
# Set the plot title and labels
plt.title('Correlation between grid_par and fsm2_sum')
plt.xlabel('grid_par')
plt.ylabel('fsm2_sum')
# Show the plot
plt.legend()
plt.show()
# %%
