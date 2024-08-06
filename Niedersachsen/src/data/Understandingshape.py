# %%
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import pickle
import os


# %% Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify
print(os.getcwd())

# %% Load pickle file
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
gld.info()    
gld.head()

# %%
# print unique geometry types
gld['geometry'].geom_type.unique()

# %%
# count number of vertices and store in a new column
def count_vertices(geometry):
    if geometry.geom_type == 'Polygon':
        return len(geometry.exterior.coords)
    elif geometry.geom_type == 'MultiPolygon':
        return sum(len(poly.exterior.coords) for poly in geometry)
    else:
        return None  # Handle other geometry types if necessary

gld['n_vertices'] = gld['geometry'].apply(count_vertices)
gld.head()
# %%
print(gld['geometry'].head().apply(lambda geom: geom.geom_type))

# %%
# create dataframe with only polygons
gld_poly = gld[gld['geometry'].geom_type == 'Polygon'].copy()
# %%
gld_poly.info()
# %%
#count vertices
gld_poly['n_vertices'] = gld_poly['geometry'].apply(lambda geom: len(geom.exterior.coords))
# %%
gld_poly.head()
# %%
gld_poly['n_vertices'].describe()
# %%
# create dataframe with other geometry types
gld_other = gld[gld['geometry'].geom_type != 'Polygon'].copy()
# %%
gld_other.plot()
# %%
# count vertices
gld_other['n_vertices'] = gld_other['geometry'].apply(count_vertices)

# %%
import matplotlib.pyplot as plt

# Assuming gld_other is a GeoDataFrame and you're plotting it
fig, ax = plt.subplots(1, 1)
gld_other.plot(ax=ax, alpha=0.8)

# Set x and y limits
ax.set_xlim([350000, 650000])
ax.set_ylim([5.60*1e6, 5.95*1e6])

plt.show()
# %%
# Assuming 'all_years' is your DataFrame and it has columns 'year' and 'cellcode'

# Group by 'year' and 'cellcode' and filter
filtered_groups = gld.groupby(['year', 'CELLCODE']).filter(lambda x: len(x) <= 6)

# Now, 'filtered_groups' contains rows where the same 'cellcode' within a year does not exceed 6 entries
print(filtered_groups)

# %%
df = filtered_groups[['year', 'CELLCODE', 'LANDKREIS']].drop_duplicates().copy()
df.info()

# %%
# Number of fields per grid
fields = filtered_groups.groupby(['year', 'CELLCODE'])['area_m2'].count().reset_index()
fields.columns = ['year', 'CELLCODE', 'fields']
fields.head()
df = pd.merge(df, fields, on=['year', 'CELLCODE'])
# %%
# Sum of field size per grid
fs_sum = filtered_groups.groupby(['year', 'CELLCODE'])['area_m2'].sum().reset_index()
fs_sum.columns = ['year', 'CELLCODE', 'fs_sum']
fs_sum.head()
df = pd.merge(df, fs_sum, on=['year', 'CELLCODE'])

# Mean field size in the grid
df['mfs_ha'] = (df['fs_sum'] / df['fields'])*(1/10000)

# Sum of field peri per grid
peri_sum = filtered_groups.groupby(['year', 'CELLCODE'])['peri_m'].sum().reset_index()
peri_sum.columns = ['year', 'CELLCODE', 'peri_sum']
peri_sum.head()
df = pd.merge(df, peri_sum, on=['year', 'CELLCODE'])

# Mean perimeter in the grids
df['mperi'] = (df['peri_sum'] / df['fields'])

# Mean shape index in the grids
mean_shp = filtered_groups.groupby(['year', 'CELLCODE'])['shp_index'].mean().reset_index()
mean_shp.columns = ['year', 'CELLCODE', 'mean_shp']
df = pd.merge(df, mean_shp, on=['year', 'CELLCODE'])

# Mean fractal dimension in the grids
mean_fract = filtered_groups.groupby(['year', 'CELLCODE'])['fract'].mean().reset_index()
mean_fract.columns = ['year', 'CELLCODE', 'mean_fract']
df = pd.merge(df, mean_fract, on=['year', 'CELLCODE'])

df.head()


# %% Load pickle file
with open('data/interim/grid_landkreise.pkl', 'rb') as f:
    geom = pickle.load(f)
geom.info()    
geom.crs

# %% Join grid to df using cellcode
gdf = df.merge(geom, on='CELLCODE')
gdf.info()
gdf.head()

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

#%% Dropping the 'LANDKREIS_y' column and rename LANDKREIS_x
gridgdf.drop(columns=['LANDKREIS_y'], inplace=True)
gridgdf.rename(columns={'LANDKREIS_x': 'LANDKREIS'}, inplace=True)







# %%
gld[(gld['CELLCODE'] == '10kmE411N340') & (gld['year'] == 2012)].plot(figsize=(10, 6))

# %%
gld[(gld['CELLCODE'] == '10kmE411N340') & (gld['year'] == 2013)].plot(figsize=(10, 6))

# %%
gld[(gld['CELLCODE'] == '10kmE429N339') & (gld['year'] == 2013)].plot(figsize=(10, 6))

# %%
gld[(gld['CELLCODE'] == '10kmE429N339') & (gld['year'] == 2015)].plot(figsize=(10, 6))

# %%
gld[(gld['CELLCODE'] == '10kmE438N336') & (gld['year'] == 2016)].plot(figsize=(10, 6))


# %%
# Assuming 'df' is your DataFrame and it contains columns 'CELLCODE', 'years', and 'field'

# Group by 'CELLCODE' and then apply a custom function to check for unique values in 'years' and 'field'
def check_unique_values(group):
    return (group['year'].nunique() > 1) and (group['fields'].nunique() > 1)

# Apply the function to each group and filter groups that return True
cellcodes_with_multiple_unique_values = df.groupby('CELLCODE').filter(check_unique_values)

# Get unique CELLCODEs that meet the criteria
unique_cellcodes = cellcodes_with_multiple_unique_values['CELLCODE'].unique()

# Print or process the CELLCODEs as needed
print(unique_cellcodes)

# %%
# filter df for the rows with the unique cellcodes above
df_filtered = df[df['CELLCODE'].isin(unique_cellcodes)]
df_filtered.info()

# %%
# sort df_filtered by year and cellcode
df_filtered = df_filtered.sort_values(['CELLCODE', 'year']).reset_index(drop=True)
df_filtered.head()
# save to csv
df_filtered.to_csv('reports/df_filtered.csv', index=False)





# %%
gld[(gld['CELLCODE'] == '10kmE440N334') & (gld['year'] == 2016)].plot()











# %% 
# Filter the DataFrame based on your conditions
filtered_gld = gld[(gld['CELLCODE'] == '10kmE411N340') & (gld['year'] == 2017)]

# Create a figure and axes object for plotting
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the filtered data
filtered_gld.plot(ax=ax)

ax.set_xlim(620000, 650000)
ax.set_ylim(5780000, 5900000)


plt.show()
# %%

# Assuming 'gld' is your DataFrame

# Filter the DataFrame based on your conditions
filtered_data = gld[(gld['CELLCODE'] == '10kmE440N334') & (gld['year'] == 2016)]

# Create a figure and axes object
fig, ax = plt.subplots(figsize=(10, 6))

# Plot your data on the axes object
# Note: You need to specify what you're plotting. Here's an example using a generic 'x' and 'y' column from your DataFrame
ax.plot(filtered_data)

# Set the limits of the x and y axes
ax.set_xlim(620000, 650000)
ax.set_ylim(5780000, 5900000)

# %% Display the plot
plt.show()
# Get the x and y limits
x_lim = ax.get_xlim()
y_lim = ax.get_ylim()

# Print the x and y limits
print("X limits:", x_lim)
print("Y limits:", y_lim)

# %%
ax = gld[(gld['CELLCODE'] == '10kmE411N340') & (gld['year'] == 2017)].plot(figsize=(10, 6))
ax.set_xlim(620000, 650000)
ax.set_ylim(5780000, 5900000)


# %%

f = gld[(gld['CELLCODE'] == '10kmE429N339') & (gld['year'] == 2015)]



# %%
#filter df for 6 fields
df4 = df[df['fields'] == 4]
df4.info()

# %%
df4.head(16)

# %%
df4.to_csv('reports/df4.csv', index=False)






# %%
# First, ensure the DataFrame is sorted by 'y' and then by 'year'
df = df.sort_values(by=['CELLCODE', 'year'])

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




# %%
#sort df by year and cellcode
df = df.sort_values(['year', 'CELLCODE']).reset_index(drop=True)
df.head()