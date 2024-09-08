# %%
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import math as m

# %% Change the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %% Load pickle file
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
gld.info()    
gld.head() 

######################################################################################################
# %% landscape visualization
landdesc = pd.read_csv('reports/statistics/ldscp/ldscp_desc.csv') 

# %%
# Create line plot of yearly count of all fields
sns.lineplot(data=landdesc, x='year', y='count', color='purple')
# Set the plot title and labels
plt.title('Count of Fields per Year')
plt.xlabel('Year')
plt.ylabel('Field Count')
# Show the plot
plt.show()

# %%
# Create line plot of yearly sum of all field areas 
sns.lineplot(data=landdesc, x='year', y='areahsum', color='purple')
# Set the plot title and labels
plt.title('Total Agricultural Area in Data (ha) per Year')
plt.xlabel('Year')
plt.ylabel('Sum of Field Size (ha)')
# Show the plot
plt.show()

# %%
# Create line plot of yearly sum of all field perimeter
sns.lineplot(data=landdesc, x='year', y='perimsum', color='purple')
# Set the plot title and labels
plt.title('Total Perimeter of Fields in Data (m) per Year')
plt.xlabel('Year')
plt.ylabel('Sum of Field Perimeter (m)')
# Show the plot
plt.show()

# %%
# Create line plot of yearly mean field size
meanplot = sns.lineplot(data=landdesc, x='year', y='areahmean', color='purple')
# Annotate each point on the regplot
for line in range(0, landdesc.shape[0]):
     meanplot.text(landdesc.year[line]+0.2, landdesc.areahmean[line], 
     round(landdesc.areahmean[line], 2), horizontalalignment='left', 
     size='medium', color='black', weight='semibold')
# Set the plot title and labels
plt.title('Mean Field Size (ha) per Year')
plt.xlabel('Year')
plt.ylabel('Mean of Field Size (ha)')
# Show the plot
plt.show()

# %%Create line plot of yearly mean shp index
sns.lineplot(data=landdesc, x='year', y='shpimean', color='purple')
# Set the plot title and labels
plt.title('Trend in Mean Shape Index Across Years')
plt.xlabel('Year')
plt.ylabel('Mean Shape Index')
# Show the plot
plt.show()

# %% Create line plot of yearly mean fractal dimension
sns.lineplot(data=landdesc, x='year', y='fractmean', color='purple')
# Set the plot title and labels
plt.title('Trend in Mean Fractal Dimension Across Years')
plt.xlabel('Year')
plt.ylabel('Mean Fractal Dimension (1 \< MFD \< 2)')
# Show the plot
plt.show()

######################################################################################################
# %% Load gridgdf 
with open('data/interim/gridgdf.pkl', 'rb') as f:
    gridgdf = pickle.load(f)
gridgdf.info()    
gridgdf.head()

# %% 
# Distribution of count of grids based on mean field size range 
# Define the bin edges for Mean Field Size (mfs_ha)
bins = [0, 2, 4, 8, 16, 32, 64, 150, float('inf')]
# Define the bin labels
labels = ['<2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-150', '>150']
# Create the range column
gridgdf['mfs_range'] = pd.cut(gridgdf['mfs_ha'], bins=bins, labels=labels)

# %% Create a FacetGrid with a count plot for each year in gridgdf
g = sns.FacetGrid(gridgdf, col="year", col_wrap=4, height=4)
g.map(sns.countplot, "mfs_range", color='purple')
g.set_titles("{col_name}")
g.set_xlabels('MFS Range (ha)')
g.set_ylabels('Grid Count')
g.show()
 
# %%
# Create correlation plot of grid mfs_ha and mean_fract 
sns.scatterplot(data=gridgdf, x='mfs_ha', y='mean_fract', color='purple')
# Set the plot title and labels
plt.title('Correlation between Mean Field Size and Mean Fractal Dimension')
plt.xlabel('Mean Field Size')
plt.ylabel('Mean Fractal Dimension')
# Show the plot
plt.show()

# %%
# Create correlation plot of grid mfs_ha and field count 
sns.scatterplot(data=gridgdf, x='mfs_ha', y='fields', color='purple')
# Set the plot title and labels
plt.title('Correlation between Mean Field Size and Field Count')
plt.xlabel('Mean Field Size')
plt.ylabel('Field Count')
# Show the plot
plt.show()

######################################################################################################################
# %% load the grid descriptive stat csv file
griddesc = pd.read_csv('reports/statistics/grid/grid_desc_statssum.csv') 

# %%
# Create line plot of yearly mean field size
meanplot = sns.lineplot(data=griddesc, x='year', y='mfshm', color='purple')
# Annotate each point on the regplot
for line in range(0, griddesc.shape[0]):
     meanplot.text(griddesc.year[line]+0.2, griddesc.mfshm[line], 
     round(griddesc.mfshm[line], 2), horizontalalignment='left', 
     size='medium', color='black', weight='semibold')
# Set the plot title and labels
plt.title('Grid-level Mean Field Size (ha) per Year')
plt.xlabel('Year')
plt.ylabel('Mean of Field Size (ha)')
# Show the plot
plt.show()

# %%
# Create correlation plot of grid mfshm and mfractm 
sns.scatterplot(data=griddesc, x='mfshm', y='mfractm', color='purple')
# Set the plot title and labels
plt.title('Correlation between Grid Mean Field Size Mean and Mean Fractal Dimension Mean')
plt.xlabel('Mean Field Size Mean (ha)')
plt.ylabel('Mean Fractal Dimension Mean')
# Show the plot
plt.show()

# %% basic plotting of grid cell with cellcode
# gld[gld['CELLCODE'] == '10kmE442N331'].plot() or
# gld.loc[gld['CELLCODE'] == '10kmE442N331'].plot()


# %% visulaize grid cells with average range of field count
contains_value = (gridgdf['fields'] == 1600).any()

if contains_value:
    print("There is at least one gridcell with 1600 field")
else:
    print("There is no gridcell with 1600 field.")

filtered_gdf = gridgdf[gridgdf['fields'] == 1600]

# %% plot fields with cellcode in filtered_gdf and year using gld data
gld[(gld['CELLCODE'] == '10kmE429N329') & (gld['year'] == 2021)].plot()
gld[(gld['CELLCODE'] == '10kmE429N329') & (gld['year'] == 2022)].plot()
gld[(gld['CELLCODE'] == '10kmE438N323') & (gld['year'] == 2015)].plot()

# %% see data columns for grid cell E425N331
E425N331 = gridgdf[gridgdf['CELLCODE'] == '10kmE425N331']

# %%
# Calculate the 475th percentiles of 'mfs_ha' for each CELLCODE
percentile_75 = gridgdf['mfs_ha'].quantile(0.75)

# Filter CELLCODEs based on 'area_m2' being within the 40th and 75th percentiles
filtered_cellcodes = gridgdf[(gridgdf['mfs_ha'] >= percentile_75)]['CELLCODE'].unique()

# Apply this filter to the DataFrame to get only the rows with the filtered CELLCODEs
filtered = gridgdf[gridgdf['CELLCODE'].isin(filtered_cellcodes)]



# %%
# Assuming 'df' is your DataFrame, 'x' is the column of interest, and 'y' is the grouping column
# First, ensure the DataFrame is sorted by 'y' and then by 'year'
filtered_sorted = filtered.sort_values(by=['CELLCODE', 'year'])

# Adjusted function to apply to each group for multiple columns
def determine_change_multiple(group, columns):
    for col in columns:
        prev_value = group[col].shift(1)  # Shift column up to compare with the previous row
        conditions = [
            group[col] > prev_value,  # Value increased compared to the previous year
            group[col] < prev_value   # Value decreased compared to the previous year
        ]
        choices = ['increased', 'decreased']
        change_col_name = f'{col}_change'  # Dynamic column name for each column's change
        group[change_col_name] = np.select(conditions, choices, default=np.nan)  # Assign 'increased', 'decreased', or NaN
    return group

# List of columns to apply the change detection
columns_to_check = ['mfs_ha', 'mean_shp', 'mean_fract']  # Add your column names here

# Apply the function to each group of 'CELLCODE'
filtered_with_changes = filtered_sorted.groupby('CELLCODE').apply(lambda group: determine_change_multiple(group, columns_to_check))

filtered_with_changes.head(50).to_csv('reports/filtered_with_changes.csv')
# %%
# Assuming 'df' is your DataFrame, 'x' is the column of interest, and 'y' is the grouping column
# First, ensure the DataFrame is sorted by 'y' and then by 'year'
filtered_sorted = gridgdf.sort_values(by=['CELLCODE', 'year'])

# Adjusted function to apply to each group for multiple columns
def determine_change_multiple(group, columns):
    for col in columns:
        prev_value = group[col].shift(1)  # Shift column up to compare with the previous row
        conditions = [
            group[col] > prev_value,  # Value increased compared to the previous year
            group[col] < prev_value   # Value decreased compared to the previous year
        ]
        choices = ['increased', 'decreased']
        change_col_name = f'{col}_change'  # Dynamic column name for each column's change
        group[change_col_name] = np.select(conditions, choices, default=np.nan)  # Assign 'increased', 'decreased', or NaN
    return group

# List of columns to apply the change detection
columns_to_check = ['mfs_ha', 'mean_shp', 'mean_fract']  # Add your column names here

# Apply the function to each group of 'CELLCODE'
filtered_with_changes = filtered_sorted.groupby('CELLCODE').apply(lambda group: determine_change_multiple(group, columns_to_check))
