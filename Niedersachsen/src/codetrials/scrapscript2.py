
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
