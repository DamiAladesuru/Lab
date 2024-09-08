
# %%
gridgdf[(gridgdf['mfs_ha_yearly_diff'] < 1) & (gridgdf['mean_cpar_yearly_diff'] > 1)]

# %%
# Define the cellcode and the years of interest
cell = ['10kmE413N340', ' 10kmE411N340']
years = [2020]

# Create a subplot grid with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))

# Plot data for each year in a separate subplot
for i, year in enumerate(years):
    # Filter the data for the specific cellcode and year
    ax = gld[(gld['CELLCODE'] == cell) & (gld['year'] == year)].plot(ax=axes[i])

    # Get metric values for the current cellcode and year
    filtered_row = gridgdf[(gridgdf['CELLCODE'] == cell) & (gridgdf['year'] == year)]
    if not filtered_row.empty:
        metric_value1 = filtered_row['mfs_ha'].values[0]
        metric_value2 = filtered_row['mean_cpar'].values[0]
        metric_value3 = filtered_row['grid_par'].values[0]

        # Add text annotations below the plot
        ax.annotate(f'mfs_ha: {metric_value1}', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=10)
        ax.annotate(f'mean_cpar: {metric_value2}', xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)
        ax.annotate(f'grid_par: {metric_value3}', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=10)
    else:
        ax.annotate('No data available', xy=(0.5, -0.1), xycoords='axes fraction', ha='center', fontsize=10)

    # Set title for each subplot
    ax.set_title(f'Year: {cell}')

# Adjust layout to ensure titles and annotations don't overlap
plt.tight_layout()

# Display the plot
plt.show()

# %%
# Check for duplicate mfs_ha values in gridgdf
duplicate_mfs_ha = gridgdf[gridgdf.duplicated('mfs_ha', keep=False)]

# Print the duplicate rows
if not duplicate_mfs_ha.empty:
    print("Duplicate mfs_ha values found:")
    print(duplicate_mfs_ha)
else:
    print("No duplicate mfs_ha values found.")



# %%
import matplotlib.pyplot as plt

# Define the cellcodes and the years of interest
cells = ['10kmE413N340', '10kmE411N340']
years = [2020]

# Create a subplot grid with 1 row and 2 columns
fig, axes = plt.subplots(nrows=1, ncols=len(cells), figsize=(14, 7))

# Plot data for each cell in a separate subplot
for i, cell in enumerate(cells):
    for year in years:
        # Filter the data for the specific cellcode and year
        ax = gld[(gld['CELLCODE'] == cell) & (gld['year'] == year)].plot(ax=axes[i])

        # Get metric values for the current cellcode and year
        filtered_row = gridgdf[(gridgdf['CELLCODE'] == cell) & (gridgdf['year'] == year)]
        if not filtered_row.empty:
            metric_value1 = filtered_row['mfs_ha'].values[0]
            metric_value2 = filtered_row['mean_cpar'].values[0]
            metric_value3 = filtered_row['grid_par'].values[0]

            # Add text annotations below the plot
            ax.annotate(f'mfs_ha: {metric_value1}', xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)
            ax.annotate(f'mean_cpar: {metric_value2}', xy=(0.5, -0.25), xycoords='axes fraction', ha='center', fontsize=10)
            ax.annotate(f'grid_par: {metric_value3}', xy=(0.5, -0.35), xycoords='axes fraction', ha='center', fontsize=10)
        else:
            ax.annotate('No data available', xy=(0.5, -0.15), xycoords='axes fraction', ha='center', fontsize=10)

        # Set title for each subplot
        ax.set_title(f'Cell: {cell}, Year: {year}')

# Adjust layout to ensure titles and annotations don't overlap
plt.tight_layout()

# Display the plot
plt.show()
plt.show()

# %%
# Create a new column with mfs_ha values rounded to zero decimal places
gridgdf['mfs_ha_rounded'] = gridgdf['mfs_ha'].round(0)

# Group by the rounded mfs_ha values and filter groups with more than one unique CELLCODE
duplicate_mfs_ha_rounded_diff_cellcode = gridgdf.groupby('mfs_ha_rounded').filter(lambda x: x['CELLCODE'].nunique() > 1)

# Print the rows with the same rounded mfs_ha but different CELLCODE
if not duplicate_mfs_ha_rounded_diff_cellcode.empty:
    print("Rows with the same rounded mfs_ha but different CELLCODE found:")
    print(duplicate_mfs_ha_rounded_diff_cellcode)
else:
    print("No rows with the same rounded mfs_ha but different CELLCODE found.")

# %%
# Group by the rounded mfs_ha values and other fields, and filter groups with more than one unique CELLCODE
duplicate_mfs_ha_rounded_same_fields_diff_cellcode = gridgdf.groupby(['mfs_ha_rounded', 'fields']).filter(lambda x: x['CELLCODE'].nunique() > 1)

# Print the rows with the same rounded mfs_ha and same values of other fields but different CELLCODE
if not duplicate_mfs_ha_rounded_same_fields_diff_cellcode.empty:
    print("Rows with the same rounded mfs_ha and same values of other fields but different CELLCODE found:")
    print(duplicate_mfs_ha_rounded_same_fields_diff_cellcode)
else:
    print("No rows with the same rounded mfs_ha and same values of other fields but different CELLCODE found.")
    
# %% save to csv as grid_sample.csv
gridgdf.to_csv('grid_sample.csv', index=False)

# %%
from PIL import Image
import visualcheck

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return resized_image

# Load images
image1_path = 'reports/figures/test2014Goslar.png'
image2_path = 'reports/figures/test2016Goslar.png'

image1 = Image.open(image1_path)
image2 = Image.open(image2_path)

# Check dimensions
if image1.size != image2.size:
    print(f"Resizing images to match dimensions: {image1.size} -> {image2.size}")
    image1 = resize_image(image1_path, image2.size)
    image1.save('resized_image1.png')
    image1_path = 'resized_image1.png'

# Compare images
ssim_score, difference_image = visualcheck.compare_images(image1_path, image2_path)
print(f"SSIM Score: {ssim_score}")

















# %%

# 1. obtain cellcode of grid with maximum mean fract
CELLCODEMX = gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'CELLCODE'].values[0]
# 2. obtain year of grid with maximum mean fract
gridgdf.loc[gridgdf['mfs_ha'] == gridgdf.mfs_ha.max(), 'year'].values[0]
# %% 3. plot fields with target cellcode and year using gld data
gld[(gld['CELLCODE'] == '10kmE412N339') & (gld['year'] == 2019)].plot()
# %% 4. get list of mean_fract for all years for the gridcell with maximum mean_fract
# this allows to see how this value changed over the years
gridgdf[gridgdf['CELLCODE'] == CELLCODEMX].groupby('year')['mean_fract'].apply(list)
# %%
# obtain more stats of this grid cell
# a. fract sum of grid and year with maximum mean fract
mxa = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'fract_sum'].values[0]
# b. number of field in grid with maximum mean fract
mxb = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'fields'].values[0]
# c. mean field size in grid with maximum mean fract
mxc = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'mfs_ha'].values[0]
print(mxa, mxb, mxc)
# d. mean shape index in grid with maximum mean fract
mxd = gridgdf.loc[(gridgdf['CELLCODE'] == CELLCODEMX) & (gridgdf['year'] == 2018), 'mean_shp'].values[0]
print(mxa, mxb, mxc, mxd)



# %% fields 100
# subset griddf_ext for where fields is less than or equal 100
fields100 = griddf_ext[griddf_ext['fields'] <= 100]
# %% get the mode of fields for fields100
fields100['fields'].mode()
# %%
fields2 = gridgdf[gridgdf['fields'] == 2]
fields2 = fields2.sort_values(by='grid_polspy')
print(fields2[['year', 'CELLCODE', 'mfs_ha', 'peri_sum', 'mean_cpar', 'mean_cpar2', 'grid_par', 'lsi', 'mean_polspy', 'grid_polspy']])  

#%% from fields2, create df rows with certain combination of cellcode and year values
# List of tuples containing CELLCODE and year combinations to filter out
filter_conditions = [('10kmE442N331', 2018), ('10kmE429N339', 2016), ('10kmE440N334', 2021),
                     ('10kmE440N334', 2022)]

# Filter out rows that match any of the conditions
fields2_uniq = fields2[fields2[['CELLCODE', 'year']].apply(tuple, axis=1).isin(filter_conditions)]


# %%
fields2 = fields2.sort_values(by='grid_polspy')

# %%
import pandas as pd
desc = pd.read_csv('reports/statistics/grid/grid_desc_stats.csv')
# Assuming desc is your DataFrame

# Reset index and use old index as a column named 'index'
desc.reset_index(inplace=True, drop=False)

#%% Filter the DataFrame to include only rows from the year 2012
desc_2012 = desc[desc['year'] == 2012]

# Compute the difference between rows relative to the year 2012
# Assuming 'fieldcsum' is the column to compute the difference for
desc['fieldcsum_diff_2012'] = desc['fieldcsum'] - desc_2012['fieldcsum'].values[0]
desc['fs_sumsum_diff_2012'] = desc['fs_sumsum'] - desc_2012['fs_sumsum'].values[0]
# %%
desc['mfshm_diff_2012'] = desc['mfshm'] - desc_2012['mfshm'].values[0]



# %% Get unique (Year, Value) combinations
nodes = griddf_ext[['year', 'mfs_ha_diff_from_2012_bins']].drop_duplicates().reset_index(drop=True)
nodes['index'] = nodes.index

# %% Create a lookup for node indices
node_lookup = {(row['year'], row['mfs_ha_diff_from_2012_bins']): row['index'] for _, row in nodes.iterrows()}

# %% Initialize source, target, and values lists
source = []
target = []
values = []

# %% Iterate over each unique CELLCODE to calculate transitions
for unique_CELLCODE in griddf_ext['CELLCODE'].unique():
    # Subset data for each CELLCODE
    CELLCODE_data = griddf_ext[griddf_ext['CELLCODE'] == unique_CELLCODE].sort_values('year')
    
    # Get transitions for the CELLCODE
    for i in range(len(CELLCODE_data) - 1):
        src = node_lookup[(CELLCODE_data.iloc[i]['year'], CELLCODE_data.iloc[i]['mfs_ha_diff_from_2012_bins'])]
        tgt = node_lookup[(CELLCODE_data.iloc[i+1]['year'], CELLCODE_data.iloc[i+1]['mfs_ha_diff_from_2012_bins'])]
        source.append(src)
        target.append(tgt)
        values.append(1)
# %%

# Create the Sankey diagram using Plotly
fig = go.Figure(go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes.apply(lambda row: f"{row['year']} {row['mfs_ha_diff_from_2012_bins']}", axis=1)
    ),
    link=dict(
        source=source,
        target=target,
        value=values
    )
))

fig.update_layout(title_text="Sankey Diagram of 'mfs_ha_diff_from_2012' Transitions by Year", font_size=10)
fig.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.sankey import Sankey

# Assuming griddf_ext is already defined and loaded
griddf_ext = pd.DataFrame({
    'year': [2012, 2013, 2014, 2012, 2013, 2014],
    'mfs_ha_diff_from_2012_bins': ['low', 'medium', 'high', 'low', 'medium', 'high'],
    'CELLCODE': [1, 1, 1, 2, 2, 2]
})

# Get unique (Year, Value) combinations
nodes = griddf_ext[['year', 'mfs_ha_diff_from_2012_bins']].drop_duplicates().reset_index(drop=True)
nodes['index'] = nodes.index

# Create a lookup for node indices
node_lookup = {(row['year'], row['mfs_ha_diff_from_2012_bins']): row['index'] for _, row in nodes.iterrows()}

# Initialize source, target, and values lists
source = []
target = []
values = []

# Iterate over each unique CELLCODE to calculate transitions
for unique_CELLCODE in griddf_ext['CELLCODE'].unique():
    # Subset data for each CELLCODE
    CELLCODE_data = griddf_ext[griddf_ext['CELLCODE'] == unique_CELLCODE].sort_values('year')
    
    # Get transitions for the CELLCODE
    for i in range(len(CELLCODE_data) - 1):
        src = node_lookup[(CELLCODE_data.iloc[i]['year'], CELLCODE_data.iloc[i]['mfs_ha_diff_from_2012_bins'])]
        tgt = node_lookup[(CELLCODE_data.iloc[i+1]['year'], CELLCODE_data.iloc[i+1]['mfs_ha_diff_from_2012_bins'])]
        source.append(src)
        target.append(tgt)
        values.append(1)  # Assume weight of 1 for each transition

# Convert source-target pairs to flow data
flows = []
labels = []
orientations = []

# Ensure balanced flows for Sankey diagram
for s, t in zip(source, target):
    flows.append(1)
    flows.append(-1)
    labels.append(f"{nodes.iloc[s]['year']} {nodes.iloc[s]['mfs_ha_diff_from_2012_bins']}")
    labels.append(f"{nodes.iloc[t]['year']} {nodes.iloc[t]['mfs_ha_diff_from_2012_bins']}")
    orientations.append(0)
    orientations.append(0)

# Create the Sankey diagram
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[], title="Sankey Diagram of 'mfs_ha_diff_from_2012' Transitions by Year")
sankey = Sankey(ax=ax, unit=None)

# Add the flows and labels
sankey.add(flows=flows, labels=labels, orientations=orientations)

# Finish the diagram
diagrams = sankey.finish()

# Show the plot
plt.show()

# %%
# Filter the DataFrame for the specific CELLCODE
filtered_df = gridgdf[gridgdf['CELLCODE'] == '10kmE438N336']

# Group by year and extract the 'fsha_sum' value for each year
fsha_sum_by_year = filtered_df.groupby('year')['fsha_sum'].first().reset_index()

# Convert the result to a dictionary for easier access
fsha_sum_dict = fsha_sum_by_year.set_index('year')['fsha_sum'].to_dict()

# Print the result
print(fsha_sum_dict)
# %%
# Filter the DataFrame for the specific CELLCODE
filtered_df = gridgdf[gridgdf['CELLCODE'] == '10kmE438N336']

# Group by year and extract the 'groups' value for each year
groups_by_year = filtered_df.groupby('year')['groups'].first().reset_index()

# Convert the result to a dictionary for easier access
groups_dict = groups_by_year.set_index('year')['groups'].to_dict()

# Print the result
print(groups_dict)

# %%
# Filter the DataFrame for the specific CELLCODE
filtered_df = gridgdf[gridgdf['CELLCODE'] == '10kmE438N336']

# Group by year and extract the 'fsha_sum' value for each year
fields_by_year = filtered_df.groupby('year')['fields'].first().reset_index()

# Convert the result to a dictionary for easier access
fields_sum_dict = fields_by_year.set_index('year')['fields'].to_dict()

# Print the result
print(fields_sum_dict)

# %%
# Filter the DataFrame for the specific CELLCODE
filtered_df = gridgdf[gridgdf['CELLCODE'] == '10kmE438N336']

# Group by year and extract the 'peri_sum', 'mean_cpar2', and 'lsi' values for each year
peri_sum_by_year = filtered_df.groupby('year')['peri_sum'].first().reset_index()
mean_cpar2_by_year = filtered_df.groupby('year')['mean_cpar2'].first().reset_index()
lsi_by_year = filtered_df.groupby('year')['lsi'].first().reset_index()
lsidiff_by_year = filtered_df.groupby('year')['lsi_diff_from_2012'].first().reset_index()

# Convert the results to dictionaries for easier access
peri_sum_dict = peri_sum_by_year.set_index('year')['peri_sum'].to_dict()
mean_cpar2_dict = mean_cpar2_by_year.set_index('year')['mean_cpar2'].to_dict()
lsi_dict = lsi_by_year.set_index('year')['lsi'].to_dict()
lsidiff_dict = lsidiff_by_year.set_index('year')['lsi_diff_from_2012'].to_dict()

# Print the results
print("peri_sum by year:", peri_sum_dict)
print("mean_cpar2 by year:", mean_cpar2_dict)
print("lsi by year:", lsi_dict)
print("lsidiff by year:", lsidiff_dict)
# %%
gridgdf

# %% clipping and plotting polygons within their grid cells

# Ensure each polygon appears within its grid cell
clipped_polygons = []

# Iterate over each grid cell
for idx, grid_cell in grid_landkreise.iterrows():
    # Get the grid cell code
    cell_code = grid_cell['CELLCODE']
    
    # Filter polygons that belong to this grid cell
    polygons_in_cell = gld[gld['CELLCODE'] == cell_code]
    
    # Clip polygons to the grid cell
    clipped = gpd.clip(polygons_in_cell, grid_cell.geometry)
    clipped_polygons.append(clipped)

# Combine all clipped polygons into a single GeoDataFrame
clipped_polygons_gdf = gpd.GeoDataFrame(pd.concat(clipped_polygons, ignore_index=True), crs=gld.crs)

# save the clipped polygons to pickle
#clipped_polygons_gdf.to_pickle('data/interim/clipped_polygons.pkl')

# Load the clipped polygons from pickle
#clipped_polygons_gdf = pd.read_pickle('data/interim/clipped_polygons.pkl')


# %%
# Plot grid cells
fig, ax = plt.subplots(figsize=(10, 10))
grid_landkreise.boundary.plot(ax=ax, color='black', linewidth=1)  # Plot grid boundaries

# Plot clipped polygons
clipped_polygons_gdf.plot(ax=ax, column='CELLCODE', cmap='Set3', edgecolor='black', alpha=0.7)

plt.title('Polygons within their 10km Grid Cells')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


######################################################################
#calculate diff from 2012 function but I don't use that anymore
# %%
def calculate_differences(griddfs):
    # Create a copy of the original dictionary to avoid altering the original data
    griddfs_ext = {key: df.copy() for key, df in griddfs.items()}
    
    for key, df in griddfs_ext.items():
        # Ensure the data is sorted by 'CELLCODE' and 'year'
        df.sort_values(by=['CELLCODE', 'year'], inplace=True)
        numeric_columns = df.select_dtypes(include='number').columns

        # Calculate yearly difference for numeric columns
        for col in numeric_columns:
            df[f'{col}_yearly_diff'] = df.groupby('CELLCODE')[col].diff().fillna(0)

        # Calculate difference relative to the year 2012
        if 2012 in df['year'].values:
            # Create a DataFrame for 2012 values
            df_2012 = df[df['year'] == 2012][['CELLCODE'] + list(numeric_columns)]
            df_2012 = df_2012.rename(columns={col: f'{col}_2012' for col in numeric_columns})
            
            # Merge the 2012 values back to the original DataFrame
            df = pd.merge(df, df_2012, on='CELLCODE', how='left')

            # Calculate the difference from 2012 for each numeric column
            for col in numeric_columns:
                df[f'{col}_diff_from_2012'] = df[col] - df[f'{col}_2012']

            # Drop the temporary 2012 columns
            df.drop(columns=[f'{col}_2012' for col in numeric_columns], inplace=True)
        
        # Update the dictionary with the modified DataFrame
        griddfs_ext[key] = df

    return griddfs_ext

griddfs_ext = calculate_differences(griddfs)
for key, df in griddfs_ext.items():
    print(f"Info for griddf_{key}:")
    print(df.info())
    griddf_ext_filename = os.path.join(output_dir, f'griddf_{key}_extended.csv')#_{current_date}
    if not os.path.exists(griddf_ext_filename):
        df.to_csv(griddf_ext_filename, encoding='windows-1252', index=False)
        print(f"Saved griddf_{key}_extended to {griddf_ext_filename}")
        
        
######################################################################################
# %%
import os
import pandas as pd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
# %%
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis_and_models import describe_new

gld, griddf, griddf_ext, grid_year_average, gridgdf = describe_new.process_descriptives()


# Load the clipped polygons from pickle
#clipped_polygons_gdf = pd.read_pickle('data/interim/clipped_polygons.pkl')

# %%
griddf_ext['PARsq'] = griddf_ext['grid_par']/((griddf_ext['lsi'])**2)

# %%
griddf_ext['lsisq'] = griddf_ext['lsi']**2

 # %%
# Filter rows where there is less than 100 fields
outliers_df = griddf_ext[griddf_ext['fields'] < 100]

#remove the outliers
griddf_ext = griddf_ext[~griddf_ext['fields'].isin(outliers_df['fields'])]

# fields 100 - 200 - subset for where fields is less than or equal 200
fields200 = griddf_ext[griddf_ext['fields'] <= 200]

# %% 
def get_extreme_rows_by_year(df, metrics):
    """
    Groups the DataFrame by year and gets the rows with the minimum and maximum values for the specified metrics.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    metrics (list): A list of column names to find the minimum and maximum values for each year.

    Returns:
    dict: A dictionary containing DataFrames with the minimum and maximum rows for each metric.
    """
    result = {}
    
    for metric in metrics:
        # Find the minimum and maximum values for the metric for each year
        min_metric = df.groupby('year')[metric].transform('min')
        max_metric = df.groupby('year')[metric].transform('max')
        
        # Filter the DataFrame to get the rows with the minimum and maximum values for the metric for each year
        min_metric_rows = df[df[metric] == min_metric]
        max_metric_rows = df[df[metric] == max_metric]
        
        # Store the results in the dictionary
        result[f'min_{metric}_rows'] = min_metric_rows
        result[f'max_{metric}_rows'] = max_metric_rows
    
    return result

# Usage
metrics = ['mean_cpar2', 'lsi', 'mean_par', 'grid_par']
extreme_rows = get_extreme_rows_by_year(fields200, metrics)

# Accessing the results
min_mean_cpar_rows = extreme_rows['min_mean_cpar2_rows']
max_mean_cpar_rows = extreme_rows['max_mean_cpar2_rows']
min_lsi_rows = extreme_rows['min_lsi_rows']
max_lsi_rows = extreme_rows['max_lsi_rows']
min_mean_par_rows = extreme_rows['min_mean_par_rows']
max_mean_par_rows = extreme_rows['max_mean_par_rows']
min_grid_par_rows = extreme_rows['min_grid_par_rows']
max_grid_par_rows = extreme_rows['max_grid_par_rows']

# %% print year. cellcode and Parsq for the max_lsi_rows
min_lsi_rows[['year', 'CELLCODE', 'PARsq']]
 


# %%
fields2 = gridgdf[gridgdf['fields'] == 2]
merged = gld.merge(fields2, on=['CELLCODE', 'year'], how='left', indicator=True)
subsample_df = merged[merged['_merge'] == 'both'].drop(columns=['_merge'])

########################################
# Plotting the grid cells with polygons
########################################
# %%
# Specify the grid cell code you want to plot
specific_cell_code = '10kmE422N326'
year = 2022

# Filter the grid GeoDataFrame for the specific cell code
specific_grid_cell = gridgdf[(gridgdf['CELLCODE'] == specific_cell_code) & (gridgdf['year'] == year)]
# Filter the polygons GeoDataFrame for polygons in this grid cell
polygons_in_specific_cell = gld[(gld['CELLCODE'] == specific_cell_code) & (gld['year'] == year)]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the grid cell boundary
specific_grid_cell.boundary.plot(ax=ax, color='black', linewidth=2, label='Grid Cell Boundary')
# Plot the polygons within the grid cell
polygons_in_specific_cell.plot(ax=ax, color='skyblue', edgecolor='black', alpha=0.7, label='Polygons')

# Add a title and legend
plt.title(f'Grid Cell {specific_cell_code} with Polygons')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()

# Show the plot
plt.show()

################################################
# Plotting the grid cells with polygons for a df
################################################
# %%
def plot_grid_cells_with_polygons_from_df(gridgdf, gld, cell_year_df):
    """
    Plots the specified grid cells and their polygons for given years from a DataFrame.

    Parameters:
    gridgdf (GeoDataFrame): The GeoDataFrame containing the grid cells.
    gld (GeoDataFrame): The GeoDataFrame containing the polygons.
    cell_year_df (DataFrame): A DataFrame containing 'CELLCODE' and 'year' columns.
    """
    for _, row in cell_year_df.iterrows():
        cell_code = row['CELLCODE']
        year = row['year']
        
        # Filter the grid GeoDataFrame for the specific cell code and year
        specific_grid_cell = gridgdf[(gridgdf['CELLCODE'] == cell_code) & (gridgdf['year'] == year)]
        # Filter the polygons GeoDataFrame for polygons in this grid cell and year
        polygons_in_specific_cell = gld[(gld['CELLCODE'] == cell_code) & (gld['year'] == year)]

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        # Plot the grid cell boundary
        specific_grid_cell.boundary.plot(ax=ax, color='black', linewidth=2, label='Grid Cell Boundary')
        # Plot the polygons within the grid cell
        polygons_in_specific_cell.plot(ax=ax, color='skyblue', edgecolor='black', alpha=0.7, label='Polygons')

        # Add a title and legend
        plt.title(f'Grid Cell {cell_code} with Polygons for Year {year}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()

        # Show the plot
        plt.show()

# Usage

plot_grid_cells_with_polygons_from_df(gridgdf, gld, max_lsi_rows)
# %%
def print_and_export_metrics(df, metrics, df_name):
    """
    Prints the values for the specified metrics for each combination of year and CELLCODE in each row of the DataFrame as a table,
    and exports the table to CSV and HTML files.

    Parameters:
    df (DataFrame): The DataFrame containing the data.
    metrics (list): A list of column names to print and export.
    df_name (str): The name of the DataFrame.
    """
    # Create a new DataFrame with the specified metrics
    metrics_df = df[metrics]
    
    # Print the DataFrame as a table
    print(metrics_df.to_string(index=False))
    
    output_dir = 'reports/UnderstandingShape/'
    
    # Construct filenames with the DataFrame name included
    csv_filename = f"{output_dir}{df_name}.csv"
    
    # Export the DataFrame to a CSV file
    metrics_df.to_csv(csv_filename, index=False)
    print(f"Table exported to {csv_filename}")


# Usage
metrics = ['CELLCODE', 'year', 'fields', 'fsha_sum', 'fields_ha', 'mfs_ha', 'peri_sum', 'grid_par', 'lsi', 'mean_cpar2', 'mean_par']
print_and_export_metrics(min_lsi_rows, metrics, 'min_lsi_rows')
print_and_export_metrics(max_lsi_rows, metrics, 'max_lsi_rows')
print_and_export_metrics(min_grid_par_rows, metrics, 'min_grid_par_rows')
print_and_export_metrics(max_grid_par_rows, metrics, 'max_grid_par_rows')
print_and_export_metrics(min_mean_par_rows, metrics, 'min_mean_par_rows')
print_and_export_metrics(max_mean_par_rows, metrics, 'max_mean_par_rows')
print_and_export_metrics(min_mean_cpar_rows, metrics, 'min_mean_cpar2_rows')
print_and_export_metrics(max_mean_cpar_rows, metrics, 'max_mean_cpar2_rows')

# %%
gridgdf['lsi'].max()
# %%
griddf_ext.loc[griddf_ext['lsi'] == griddf_ext['lsi'].max(), 'CELLCODE'].values[0]
griddf_ext.loc[griddf_ext['lsi'] == griddf_ext['lsi'].max(), 'year'].values[0]

# %% Grid level Multi-line plot
######################################################################## 
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

#plot metrics

sns.lineplot(data=grid_year_average, x='year', y='mean_grid_par_diff12', label='mean grid PAR from 2012', marker='o')

# Add titles and labels
plt.title('Trend of Yearly Average of FiSC Metrics from 2012 (Grid level)')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()

# %%
######################################################################## 
# Set the plot style
sns.set(style="whitegrid")

# Create a line plot
plt.figure(figsize=(12, 6))

#plot metrics

sns.lineplot(data=grid_year_average, x='year', y='mean_grid_par', label='mean grid PAR', marker='o')

# Add titles and labels
plt.title('Trend of Yearly Average of FiSC Metrics from 2012 (Grid level)')
plt.xlabel('Year')
plt.ylabel('Values')
plt.legend(title='Metrics')

# Show the plot
plt.show()

# %% E438N336 Function to add missing year data

def add_missing_year_data(df, cellcode, from_year, to_year):
    # Filter the rows for the specified CELLCODE and from_year
    filtered_rows = df[(df['CELLCODE'] == cellcode) & (df['year'] == from_year)]
    
    # Create a copy of the filtered rows and update the year to to_year
    new_rows = filtered_rows.copy()
    new_rows['year'] = to_year
    
    # Concatenate the new rows to the original DataFrame
    df = pd.concat([df, new_rows], ignore_index=True)
    
    return df

# Filter the rows for CELLCODE '10kmE438N336' and year 2016
cellcode_2016 = '10kmE438N336'
year_2016 = 2016
year_2017 = 2017

# Update griddf, griddf_ext, and gridgdf
griddf = add_missing_year_data(griddf, cellcode_2016, year_2016, year_2017)
griddf_ext = add_missing_year_data(griddf_ext, cellcode_2016, year_2016, year_2017)
gridgdf = add_missing_year_data(gridgdf, cellcode_2016, year_2016, year_2017)



# %%
from src.analysis_and_models import describe_new



grid_year_average = describe_new.compute_grid_year_average(griddf_ext)
print(f"Info for grid_year_average:")
print(grid_year_average.info())
grid_year_average_filename = os.path.join(output_dir, f'grid_year_average.csv')
if not os.path.exists(grid_year_average_filename):
    grid_year_average.to_csv(grid_year_average_filename, index=False)
    print(f"Saved grid_year_average to {grid_year_average_filename}")
        

landkreis_average = describe_new.compute_landkreis_average(griddf_ext)
print(f"Info for landkreis_average:")
print(landkreis_average.info())
landkreis_average_filename = os.path.join(output_dir, f'landkreis_average.csv')
if not os.path.exists(landkreis_average_filename):
    landkreis_average.to_csv(landkreis_average_filename, index=False)
    print(f"Saved landkreis_average to {landkreis_average_filename}")
# %%
# get the unique values in gruppe column of gld
gld['Gruppe'].unique()
# %%
