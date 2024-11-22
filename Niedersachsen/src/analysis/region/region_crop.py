# %%
import os
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import plotly.express as px

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis.raw import gld_desc_raw as gdr

# %%
# filter gld for target landkreis
def load_gld(landkreis): 
    gld = gdr.adjust_gld()
    region_gld = gld[gld['LANDKREIS'] == landkreis]

    return region_gld


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

def df_regioncrop():
    #cropgroup = 'category2'
    region_gld = load_gld(region)
    regioncropdf = create_catdf(region_gld, cropgroup)
    # Calculate yearly differences
    regioncropdf_ydiff = calculate_yearlydiff(regioncropdf, cropgroup)
    # Calculate differences from the first year
    regioncropdf_exty1 = calculate_diff_fromy1(regioncropdf, cropgroup)
    # Combine the two DataFrames
    regioncropdf_ext = combine_dfs(regioncropdf_ydiff, regioncropdf_exty1)
    
    return region_gld, regioncropdf_ext

##########################################################################

# %%
# Define the output directory
output_dir = 'reports/figures/regionplots/cropcats'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

region = 'Vechta'
cropgroup = 'Gruppe'
region, regioncrpdf = df_regioncrop()

#groups = regioncrpdf['Gruppe'].unique()

# Remove rows belonging to 'sonstige flächen'
#regioncrop_averages = regioncrop_averages[regioncrop_averages[cropgroup] != 'sonstige flächen']

# %%
# Read colour and translation spreadsheet data into a DataFrame
df = pd.read_excel('reports/figures/grp_hue_translation.xlsx')

# Convert the DataFrame into dictionaries
hue_colors = dict(zip(df[cropgroup], df['Hue']))
translation_dict = dict(zip(df[cropgroup], df['Translation']))

# Show the colors for each group using the imported data
plt.figure(figsize=(12, 6))
for group, color in hue_colors.items():
    plt.plot([], [], color=color, label=translation_dict[group], marker='o', linestyle='')

# Set the legend to be in two rows with a custom background color and increased border padding
ncol = math.ceil(len(hue_colors) / 2)
legend = plt.legend(title='Crop Group', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=ncol, borderpad=2)
plt.axis('off')

# Adjust layout to prevent cut-off
plt.tight_layout()

# Save plot with bbox_inches='tight' to include all elements
plt.savefig(os.path.join(output_dir, f'{cropgroup}_legend.png'), bbox_inches='tight')

plt.show()


# %% 
# fig 1a: multi-line plot showing total land area for each crop group in region over time
########################################################################################
fig = px.line(regioncrpdf, x='year', y='fsha_sum_yearly_percdiff', color=cropgroup,
              color_discrete_map=hue_colors)
fig.update_layout(
    #showlegend=False,
    legend_title= cropgroup,
    xaxis_title='Year',
    yaxis_title= 'Perc Yearly Change in Area Sum (ha)',
    title= f'Yearly Total Area Change for Each Crop Group in {region}',
    template='plotly_white'
)
# save plot as html
#fig.write_html('reports/figures/ToF/totarepch_cat12.html')

fig.show()

# %%
# fig 1b: multi-line plot showing diff from base year of total agricultural land
# for each crop group over time
##################################################################################
fig = px.line(regioncrpdf, x='year', y='fsha_sum_percdiff_to_y1', color=cropgroup,
              color_discrete_map=hue_colors)
fig.update_layout(
    #showlegend=False,
    legend_title= cropgroup,
    xaxis_title='Year',
    yaxis_title= 'Perc Change from 2012 in Area Sum (ha)',
    title= f'Perc Change from 2012 in Total Area for Each Crop Group in {region}',
    template='plotly_white'
)
# save plot as html
#fig.write_html('reports/figures/ToF/totarepch_cat12.html')

fig.show()

# %% 
# fig 2a: multi-line plot showing absolute total land area change for each crop group in region over time
#########################################################################################################
fig = px.line(regioncrpdf, x='year', y='fsha_sum_yearly_diff', color=cropgroup,
              color_discrete_map=hue_colors)
fig.update_layout(
    #showlegend=False,
    legend_title= cropgroup,
    xaxis_title='Year',
    yaxis_title= 'Yearly Change in Area Sum (ha)',
    title= f'Yearly Total Area Change for Each Crop Group in {region}',
    template='plotly_white'
)
# save plot as html
#fig.write_html('reports/figures/ToF/totarepch_cat12.html')

fig.show()

# %%
# fig 2b: multi-line plot showing abs diff from base year of total agricultural land for each crop group over time
#######################################################################################################################
fig = px.line(regioncrpdf, x='year', y='fsha_sum_diff_from_y1', color=cropgroup,
              color_discrete_map=hue_colors)
fig.update_layout(
    #showlegend=False,
    legend_title= cropgroup,
    xaxis_title='Year',
    yaxis_title= 'Abs Change from 2012 in Area Sum (ha)',
    title= f'Abs Change from 2012 in Total Area for Each Crop Group in {region}',
    template='plotly_white'
)
# save plot as html
#fig.write_html('reports/figures/ToF/totarepch_cat12.html')

fig.show()


'''
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
'''
# %%
