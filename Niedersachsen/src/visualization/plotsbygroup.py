# %%
import os
import pandas as pd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
import math

# %%
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis_and_models import describe_single as ds

gld, griddf, griddf_ext, grid_year_average, landkreis_average, category2_average, gridgdf = ds.process_descriptives()

# %%
# Define the output directory
output_dir = 'reports/figures/plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Category2 dictionary color mapping
hue_colors = {
    'getreide': '#bcbd22',
    'ackerfutter': '#8c564b',
    'dauergrünland': '#006D5B',
    'gemüse': '#d62728',
    'hackfrüchte': '#9467bd',
    'ölsaaten': '#e377c2',
    'leguminosen': '#ff7f0e',
    'mischkultur': '#17becf',
    'dauerkulturen': '#2ca02c',
    'sonstige flächen': '#7f7f7f',
    'environmental': '#1f77b4',
}

# Translation dictionary
translation_dict = {
    'ackerfutter': 'forage crops',
    'gemüse': 'vegetables',
    'environmental': 'environmental areas',
    'leguminosen': 'legumes',
    'mischkultur': 'mixed culture',
    'hackfrüchte': 'root crops',
    'dauergrünland': 'permanent grassland',
    'ölsaaten': 'oilseeds',
    'dauerkulturen': 'perennial crops',
    'getreide': 'cereals',
    'sonstige flächen': 'other areas'
}

# %% Show the colors for each category in category2
plt.figure(figsize=(12, 6))
for category, color in hue_colors.items():
    plt.plot([], [], color=color, label=translation_dict[category], marker='o', linestyle='')

# Set the legend to be in two rows with a custom background color and increased border padding
ncol = math.ceil(len(hue_colors) / 2)
legend = plt.legend(title='Category2', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=ncol, borderpad=2)
plt.axis('off')

# Adjust layout to prevent cut-off
plt.tight_layout()

# Save plot with bbox_inches='tight' to include all elements
plt.savefig(os.path.join(output_dir, 'category2_legend.png'), bbox_inches='tight')

plt.show()

# %% 
# fig 1: multi-line plot showing average mean field size for each crop group over time
#################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=means, x='year', y='fsha_sum_sum', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average MFS (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average MFS (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot
plt.savefig(os.path.join(output_dir, 'mfs_category2.svg'))

# Show the plot
plt.show()

# %% 
# fig 2: multi-line plot showing average diff from base year of total agricultural land for each crop group over time
#######################################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data=gr, x='year', y='fsha_sum_mean_diff_from_y1', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average Relative Difference from Year One of TL (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average R. Diff of TL (ha) from Year One')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'adiff_y1_TL_cat2.svg'))

# Show the plot
plt.show()

# %% 
# fig 3: multi-line plot showing average diff from base year of mean field size for each crop group over time
############################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data=category2_average, x='year', y='mfs_ha_apercdiff_y1', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average Relative Difference from Year One of MFS (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average R. Diff of MFS (ha) from Year One')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'adiff_y1_MFS_cat2.svg'))

# Show the plot
plt.show()

# %% fig 4: multi-line plot showing av. diff from base year of mean par for each crop group over time
########################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=category2_average, x='year', y='mean_par_apercdiff_y1', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average Relative Difference from Year One of Mean PAR for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Averahe R. Diff of MeanPAR from Year One')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'adiff_y1_mPAR_cat2.svg'))

# Show the plot
plt.show()


# %% fig 5: multi-line plot showing av diff from base year of GRID par for each crop group over time
########################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data=category2_average, x='year', y='grid_par_apercdiff_y1', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average Relative Difference from Year One of Grid PAR for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average R. Difference of Grid PAR from Year One')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'adiff_y1_GPAR_cat2.svg'))

# Show the plot
plt.show()

# %% fig 6: multi-line plot showing av diff from base year of fields_ha for each crop group over time
########################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data=category2_average, x='year', y='fields_ha_apercdiff_y1', hue='category2',
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Trend of Average Relative Difference from Year One of Fields/ha for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Average R. Diff of Fields_ha from Year One')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'adiff_y1_fieldsha_cat2.svg'))

# Show the plot
plt.show()


# %% single line plot for grid par diff from year one
sns.lineplot(data=grid_year_average, x='year', y='mean_grid_par_diff_y1', color='teal')
# Set the plot title and labels
#plt.title('Trend of Total Agricultural Land Area (ha)')
plt.xlabel('Year')
plt.ylabel('Grid PAR Difference from Year One')
# Remove the top and right spines
sns.despine(left=True, bottom=True)
# Show the plot
plt.legend()
plt.show()
#save plot
output_dir = 'reports/figures/subsets'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
#plt.savefig(os.path.join(output_dir, f'count_diff12{key}.png'))













