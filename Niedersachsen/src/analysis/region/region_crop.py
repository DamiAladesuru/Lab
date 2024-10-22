# %%
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis import gridgdf_desc2 as gd


# %%
# filter gld_trimmed for target landkreis
def load_gld(landkreis): 
    input_dir = 'data/interim/gridgdf'    
        
    gld_trimmed_filename = os.path.join(input_dir, 'gld_trimmed.pkl')

    # Load or create gld_trimmed
    if os.path.exists(gld_trimmed_filename):
        gld_trimmed = pd.read_pickle(gld_trimmed_filename)
        print(f"Loaded gld_trimmed from {gld_trimmed_filename}")
    else:
        gld_trimmed = gd.adjust_trim_gld()
    region_gld = gld_trimmed[gld_trimmed['LANDKREIS'] == landkreis]

    return region_gld


# %%
def create_catdf(gld, cropgroup):
    columns = [cropgroup, 'year']

    # Extract the specified columns and remove duplicates
    catdf = gld[columns].drop_duplicates().copy()
    
    # Number of fields in each crop category per year
    fields = gld.groupby([cropgroup, 'year'])['geometry'].count().reset_index()
    fields.columns = [cropgroup, 'year', 'fields']
    catdf = pd.merge(catdf, fields, on=[cropgroup, 'year'])
    
    # Sum of field size per crop category per year
    fsha_sum = gld.groupby([cropgroup, 'year'])['area_ha'].sum().reset_index()
    fsha_sum.columns = [cropgroup, 'year', 'fsha_sum']
    catdf = pd.merge(catdf, fsha_sum, on=[cropgroup, 'year'])
    
    # Sum of field perimeter per crop category per year
    peri_sum = gld.groupby([cropgroup, 'year'])['peri_m'].sum().reset_index()
    peri_sum.columns = [cropgroup, 'year', 'peri_sum']
    catdf = pd.merge(catdf, peri_sum, on=[cropgroup, 'year'])

    # Mean field size per crop category per year
    catdf['mfs_ha'] = (catdf['fsha_sum'] / catdf['fields'])

    # Rate of fields per hectare of land per crop category per year
    catdf['fields_ha'] = (catdf['fields'] / catdf['fsha_sum'])
    
    #Shape 
    # Sum of par per grid
    par_sum = gld.groupby([cropgroup, 'year'])['par'].sum().reset_index()
    par_sum.columns = [cropgroup, 'year', 'par_sum']
    catdf = pd.merge(catdf, par_sum, on=[cropgroup, 'year'])

    # Mean par per grid
    catdf['mean_par'] = (catdf['par_sum'] / catdf['fields'])
    
    return catdf    
# %%
def calculate_yearlydiff(df, cropgroup): #yearly differences
    # Create a copy of the original DataFrame to avoid altering the original data
    df_ext = df.copy()
    
    # Sort the DataFrame by year
    df_ext.sort_values(by=[cropgroup, 'year'], inplace=True)
    numeric_columns = df_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate yearly difference for numeric columns and store in the dictionary
    for col in numeric_columns:
        new_columns[f'{col}_yearly_diff'] = df_ext.groupby(cropgroup)[col].diff().fillna(0)
    # Calculate yearly relative difference for numeric columns and store in the dictionary
        new_columns[f'{col}_yearly_percdiff'] = (df_ext.groupby(cropgroup)[col].diff() / df_ext.groupby(cropgroup)[col].shift(1)).fillna(0) * 100
        
     # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    df_ext = pd.concat([df_ext, new_columns_df], axis=1)

    return df_ext          
        
def calculate_diff_fromy1(df, cropgroup): #difference from first year
    # Create a copy of the original DataFrame to avoid altering the original data
    df_ext = df.copy()
    
    # Sort the DataFrame by year
    df_ext.sort_values(by=[cropgroup, 'year'], inplace=True)
    numeric_columns = df_ext.select_dtypes(include='number').columns

    # Create a dictionary to store the new columns
    new_columns = {}

    # Calculate difference relative to the first year
    y1_df = df_ext.groupby(cropgroup).first().reset_index()
    
    # Rename the numeric columns to indicate the first year
    y1_df = y1_df[[cropgroup] + list(numeric_columns)]
    y1_df = y1_df.rename(columns={col: f'{col}_y1' for col in numeric_columns})

    # Merge the first year values back into the original DataFrame
    df_ext = pd.merge(df_ext, y1_df, on=cropgroup, how='left')

    # Calculate the difference from the first year for each numeric column (excluding yearly differences)
    for col in numeric_columns:
        new_columns[f'{col}_diff_from_y1'] = df_ext[col] - df_ext[f'{col}_y1']
        new_columns[f'{col}_percdiff_to_y1'] = ((df_ext[col] - df_ext[f'{col}_y1']) / df_ext[f'{col}_y1'])*100

    # Drop the temporary first year columns
    df_ext.drop(columns=[f'{col}_y1' for col in numeric_columns], inplace=True)

    # Concatenate the new columns to the original DataFrame all at once
    new_columns_df = pd.DataFrame(new_columns)
    df_exty1 = pd.concat([df_ext, new_columns_df], axis=1)
    
    return df_exty1

def combine_dfs(df_ext, df_exty1):
    # Ensure the merge is based on cropgroup and 'year'
    # Select columns from df_exty1 that are not in df_ext (excluding cropgroup and 'year')
    columns_to_add = [col for col in df_exty1.columns if col not in df_ext.columns or col in [cropgroup, 'year']]

    # Merge the DataFrames on cropgroup and 'year', keeping the existing columns in df_ext
    combined_df = pd.merge(df_ext, df_exty1[columns_to_add], on=[cropgroup, 'year'], how='left')
    
    return combined_df

# %%
def df_regioncrop():
    #cropgroup = 'category2'
    region_gld = load_gld('Göttingen')
    catdf = create_catdf(region_gld, cropgroup)
    # Calculate yearly differences
    catdf_ydiff = calculate_yearlydiff(catdf, cropgroup)
    # Calculate differences from the first year
    catdf_exty1 = calculate_diff_fromy1(catdf, cropgroup)
    # Combine the two DataFrames
    catdf_ext = combine_dfs(catdf_ydiff, catdf_exty1)
    
    return catdf_ext

##########################################################################

# %%
cropgroup = 'category2'
catdf = df_regioncrop()

# Remove rows belonging to 'sonstige flächen'
#cat_averages = cat_averages[cat_averages[cropgroup] != 'sonstige flächen']

# %%
# Define the output directory
output_dir = 'reports/figures/regionplots/cropcats'
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
    'environmental': '#1f77b4'
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
legend = plt.legend(title=cropgroup, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=ncol, borderpad=2)
plt.axis('off')

# Adjust layout to prevent cut-off
plt.tight_layout()

# Save plot with bbox_inches='tight' to include all elements
plt.savefig(os.path.join(output_dir, 'category2_legend.png'), bbox_inches='tight')

plt.show()

# %% 
# fig 1a: multi-line plot showing total land area for each crop group in region over time
########################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=catdf, x='year', y='fsha_sum_yearly_percdiff', hue=cropgroup,
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Yearly Total Land Area Change for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Yearly R. Diff of Total Land Area (%)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot
plt.savefig(os.path.join(output_dir, 'tla_gottingen_category2.svg'))

# Show the plot
plt.show()

# %%
# fig 1b: multi-line plot showing diff from base year of total agricultural land for each crop group over time
#######################################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data = catdf, x='year', y='fsha_sum_percdiff_to_y1', hue=cropgroup,
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title(' elative Difference from Year One of TL for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('R. Diff of TL from Year One (%)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'diffy1_TLGot_cat2.svg'))

# Show the plot
plt.show()
# %% 
# fig 2a: multi-line plot showing absolute total land area change for each crop group in region over time
#########################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=catdf, x='year', y='fsha_sum_yearly_diff', hue=cropgroup,
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Yearly Total Land Area Change for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Yearly Diff of Total Land Area (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot
plt.savefig(os.path.join(output_dir, 'tla_gottingen_cat2_abs.svg'))

# Show the plot
plt.show()

# %%
# fig 2b: multi-line plot showing abs diff from base year of total agricultural land for each crop group over time
#######################################################################################################################
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Set the background color
#plt.gca().set_facecolor('#e6e6e6')
#plt.gcf().set_facecolor('#e6e6e6')

# Create a line plot for each category with custom colors
sns.lineplot(data = catdf, x='year', y='fsha_sum_diff_from_y1', hue=cropgroup,
             marker='o', palette=hue_colors, legend=False)

# Add titles and labels
plt.title('Absolute Difference from Year One of TL for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Abs. Diff of TL from Year One (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot before showing it
plt.savefig(os.path.join(output_dir, 'diffy1_TLGot_cat2_abs.svg'))

# Show the plot
plt.show()
