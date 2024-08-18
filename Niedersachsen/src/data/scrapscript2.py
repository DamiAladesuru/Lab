
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