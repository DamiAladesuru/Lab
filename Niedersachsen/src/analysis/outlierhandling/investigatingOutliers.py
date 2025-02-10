''' Use data mani or trend_of_fisc to load data'''
# %% Importing modules
import os
import seaborn as sns
import joypy
from matplotlib import cm
import matplotlib.pyplot as plt

from src.visualization import plotting_module as pm

# %%
os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

########################################
# General subsetting and filtering codes
########################################
# %% or subsetting
gridclean = gridgdf_raw[~(gridgdf_raw['LANDKREIS'] == 'Küstenmeer Region Weser-Ems')| (gridgdf_raw['LANDKREIS'] == 'Lüneburg')]

# %% Filter the DataFrame for the specified CELLCODE values
# List of CELLCODE values to filter
cellcodes = ['10kmE433N332', '10kmE439N322', '10kmE442N331', '10kmE438N336', '10kmE417N341']
 
edgegrids = gridgdf_raw[gridgdf_raw['CELLCODE'].isin(cellcodes)]
print(edgegrids[['CELLCODE', 'LANDKREIS']])

# %% 
Niedergld = gld[gld['FLIK'].str.startswith('DENI')]


###########################
# Transform field data to log and obtain z scrore
###########################
# %%
# Reset index and rename for consistency
gld_ = gld.reset_index().rename(columns={'index': 'id'})

# Drop the geometry column if it exists (optional if needed)
#gld_ = gld_.drop(columns='geometry', errors='ignore')

# Select numerical columns except 'year' and 'kulturcode'
data = gld_.select_dtypes(include=[np.number]).drop(['year', 'kulturcode'], axis=1)

# Apply log transformation and create new columns with '_log' suffix
for column in data.columns:
    gld_[f'{column}_log'] = np.log1p(gld_[column])

# Calculate Z-scores for the log-transformed columns grouped by 'year'
log_columns = [f'{column}_log' for column in data.columns]  # List of log-transformed columns

# Group by 'year' and calculate Z-scores within each group
for column in log_columns:
    # Compute Z-score within each year group
    gld_[f'{column}_zscore'] = gld_.groupby('year')[column].transform(lambda x: (x - x.mean()) / x.std())

#%% Identify outliers (Z-score > 3 or Z-score < -3)
#outliers_par = gld_[gld_['par_log_zscore'] > 3]
outliears_area = gld_[gld_['area_m2_log_zscore'] < -3]
#outliears_area = outliears_area.drop(columns='geometry')
gld_noout = gld_[~(gld_['area_m2_log_zscore'] < -3)]

###########################
# calculate z score at grid level
###########################
# %% Reset index and rename for consistency
gridgdfout_ = gridgdfout.reset_index().rename(columns={'index': 'id'})

# Select numerical columns except 'year' and 'kulturcode'
data = gridgdfout_.select_dtypes(include=[np.number]).drop(['year'], axis=1)

# Group by 'year' and calculate Z-scores within each group
for column in data:
    # Compute Z-score within each year group
    gridgdfout_[f'{column}_zscore'] = gridgdfout_.groupby('year')[column].transform(lambda x: (x - x.mean()) / x.std())

# %% drop grids with z score less than -1.55
gridgdfout__ = gridgdfout_[~(gridgdfout_['fields_zscore'] < -1.55)]
''' go to datamani sheet to create new grid_yearlyout for plotting aggregate line plot'''
# see grids left out    
zscoredout = gridgdfout_[(gridgdfout_['fields_zscore'] < -1.55)]
zscoredoutdf = zscoredout.drop(columns='geometry')
# %%
###########################
# get polygons within a buffer
###########################
'''you can use gridgdf or land shp as polygons_gdf'''
# %% Combine all boundary geometries into a single geometry
polygons_gdf = gridgdf
boundary_geometry = polygons_gdf.unary_union

# Create a 10km inner buffer (negative buffer shrinks the boundary)
inner_buffer_10km = boundary_geometry.buffer(-10000)  # Shrink by 10,000 meters (10km)

# Ensure the inner buffer geometry is valid (fix if necessary)
if not inner_buffer_10km.is_valid:
    inner_buffer_10km = inner_buffer_10km.buffer(0)

# Select polygons that are within the inner buffer
polygons_within_10km_inside = polygons_gdf[polygons_gdf.geometry.intersects(inner_buffer_10km)]

# %% see grids left out
bufferedout = gridgdf[~(gridgdf['CELLCODE'].isin(polygons_within_10km_inside['CELLCODE']))]
bufferedoutdf = bufferedout.drop(columns='geometry')
# %%For grids in the upper left that are not included
import pandas as pd
add = gridgdf[gridgdf['CELLCODE'].isin(['10kmE412N339','10kmE412N338','10kmE412N337','10kmE412N336'])]
#For grid in the middle left that ought not to be
polygons_within_10km_inside = polygons_within_10km_inside[~(polygons_within_10km_inside['CELLCODE'] == '10kmE409N327')]
polygons_within_10km_inside = pd.concat([polygons_within_10km_inside, add], ignore_index=True)

###########################
# %% multi layer plot
###########################
boundary_gdf = gpd.read_file("N:/ds/data/Niedersachsen/verwaltungseinheiten/NDS_Landesflaeche.shp")
boundary_gdf = boundary_gdf.to_crs("EPSG:25832")
        
base = polygons_gdf.plot(color='white', edgecolor='black', figsize=(10, 10))

boundary_gdf.plot(ax=base, color='blue', alpha=0.5)

gpd.GeoSeries(inner_buffer_10km).plot(ax=base, color='red', alpha=0.2)

polygons_within_10km_inside.plot(ax=base, color='green', alpha=0.7)
plt.show()

# %%
# %%
g =gridgdf[(gridgdf['year'] == 2016)]
g.plot(color='white', edgecolor='black', figsize=(10, 10))
plt.show()
###########################
#scatter plots
###########################
# %% scatter gld
unique_years = sorted(gldno100['year'].unique())
pm.stack_plots_in_grid(gld_, unique_years, scatterplot_par_area, "area_ha_log", "par_log", ncols=4, figsize=(25, 15))

# %% scatter gridgdf
#d = gridgdfnoout_[~(gridgdfnoout_['fields_zscore'] < -1.55)]
unique_years = sorted(gridgdf['year'].unique())
pm.stack_plots_in_grid(gridgdf_cl, unique_years, pm.scatterplot_mpar_marea, "mfs_ha", "medpar", ncols=4, figsize=(25, 15))


###########################
#box plots
###########################
# %% box single
def box_plot(df, column):
    sns.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()
    
box_plot(gld, 'area_ha')

# %% boxplot for every year
data = gridgdf_cl
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=data, x='year', y='medfs_ha')
#save plot as svg
plt.savefig(f'reports/figures/distplots/cl_boxplot_medfs_ha.svg')

# %%
# Extract upper whiskers (upper limit) for each year
upper_limits = []

# Access the whiskers (ax.lines contains the whiskers data)
whiskers = ax.lines

# Number of unique years
unique_years = data['year'].unique()

# Loop through the whiskers and extract the upper limit for each year
for i, year in enumerate(unique_years):
    # Get the upper whisker for the current year
    upper_limit = whiskers[2 * i + 1].get_ydata()[1]  # The upper whisker y-value (second value)
    upper_limits.append((year, upper_limit))

# Print the upper limits for each year
for year, upper_limit in upper_limits:
    print(f"Year: {year}, Upper Limit: {upper_limit:.2f}")

# Show the plot
plt.tight_layout()
plt.show()

###########################
#kde plots
###########################
# %% kde all years
def kde_by_year(df, value_column):
    # Set up the FacetGrid to create separate plots for each year
    g = sns.FacetGrid(df, col="year", col_wrap=4, height=4)  # Adjust col_wrap and height as needed
    
    # Map the kdeplot function onto the facet grid for each year
    g.map(sns.kdeplot, value_column, fill=True)
    
    # Add titles and labels
    g.set_axis_labels(value_column, 'Density')
    g.set_titles("Year: {col_name}")
    
    # add a super title
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'KDE Plot of {value_column} by Year')
    
    #save plot as svg
    plt.savefig(f'reports/figures/kdes/kde_{value_column}_gridgdf.svg')
    
    # Show the plot
    #plt.show()

# %%
kde_by_year(gridgdf, 'fields_ha')

# %% KDE plot for a specific year
def kde_for_year(df, value_column, year):
    # Filter data for the specified year
    year_data = df[df['year'] == year]
    
    # Check if the year exists in the DataFrame
    if year_data.empty:
        print(f"No data available for year {year}")
        return
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Plot KDE for the specified year and value column
    sns.kdeplot(year_data[value_column], fill=True)
    
    # Add labels and title
    plt.title(f'KDE Plot of {value_column} for Year {year}')
    plt.xlabel(value_column)
    plt.ylabel('Density')
    
    # Show the plot
    plt.show()
    
# %%
kde_for_year(gld_, 'area_m2_log', 2017)

###########################
#joy plots
###########################
# %% simple joyplot
df = gridgdf_raw
labels = [y for y in list(df.year.unique())]
fig, axes = joypy.joyplot(df, by="year", column="mean_par", labels=labels, range_style='own', 
                          linewidth=1, legend=True, figsize=(6,5),
                          title="Mean PAR distribution of all CELLCODES",
                          colormap=cm.autumn)

# %% yearly joyplot
pm.create_yearly_joyplot(gld_no4, 'Gruppe', 'par', "PAR distribution in {year}")

###########################
# hist plots
###########################
# %% simple histogram for all years in one plot
def hist_all_years(df, value_column):
    plt.figure(figsize=(12, 6))
    sns.histplot(df, x=value_column, hue='year', kde=True, bins=42)
    plt.title(f'Histogram of {value_column} for all years')
    plt.show()
    
# %%
hist_all_years(gridgdfout__, 'medfs_ha')

###########################
# geoplot
###########################
# %%
geoData = polygons_within_10km_inside.to_crs(epsg=4326)
pm.plot_facet_choropleth_with_geoplot(geoData, column='medfs_ha_percdiff_to_y1', cmap='plasma', year_col='year', ncols=4)

###########################
# plot fields and grids
###########################
# %%
def plot_geometries_fitted(gdf):
    # Ensure indices are sequential
    gdf = gdf.reset_index(drop=True)
    
    for idx, row in gdf.iterrows():
        geometry = row.geometry
        bounds = geometry.bounds  # Get the bounding box (minx, miny, maxx, maxy)
        
        # Calculate the center of the bounding box
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        
        # Define a 3m x 3m box around the center
        buffer = 1.5  # Half of 10,000 meters
        x_min = center_x - buffer
        x_max = center_x + buffer
        y_min = center_y - buffer
        y_max = center_y + buffer
        
        # Plot the geometry
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.iloc[[idx]].plot(ax=ax, color="blue", edgecolor="black")
        
        # Set axis limits to ensure a 10km x 10km box
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        
        # Add grid and title
        ax.grid(True)
        ax.set_title(f"Geometry {idx}")
        
        plt.show()

# Example usage
# gdf = gpd.read_file("your_file.geojson")  # Load your GeoDataFrame
# plot_geometries_fitted(gdf)

plot_geometries_fitted(df_400)

# %% plot single field
def plot_single_row(gdf, col1_value, col2_value, col1_name, col2_name):
    # Filter the GeoDataFrame based on the values of columns 1 and 2
    row_to_plot = gdf[(gdf[col1_name] == col1_value) & (gdf[col2_name] == col2_value)]
    
    if row_to_plot.empty:
        print(f"No matching row found with {col1_name}={col1_value} and {col2_name}={col2_value}")
        return
    
    # Get the geometry
    geometry = row_to_plot.geometry.iloc[0]
    
    # Get the bounds of the geometry
    bounds = geometry.bounds  # minx, miny, maxx, maxy
    
    # Calculate the center of the bounding box
    center_x = (bounds[0] + bounds[2]) / 2
    center_y = (bounds[1] + bounds[3]) / 2
    
    # Define a 10km x 10km box around the center
    buffer = 300  # Half of 10,000 meters
    x_min = center_x - buffer
    x_max = center_x + buffer
    y_min = center_y - buffer
    y_max = center_y + buffer
    
    # Plot the geometry
    fig, ax = plt.subplots(figsize=(6, 6))
    row_to_plot.plot(ax=ax, color="blue", edgecolor="black")
    
    # Set axis limits to ensure a 10km x 10km box
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    
    # Add grid and title
    ax.grid(False)
    ax.set_title(f"Row with {col1_name}={col1_value} and {col2_name}={col2_value}")
    
    plt.show()

# Example usage
# gdf = gpd.read_file("your_file.geojson")  # Load your GeoDataFrame
plot_single_row(gld_ext, col1_value="DENILI1531570006", col2_value=2016, col1_name="FLIK", col2_name="year")

# %%
def plot_gridcell_single(gridgdf, gridcell, year):
    """
    Plot a single grid cell for a given year.
    """
    # Plot the grid cell
    fig_single, ax = plt.subplots(figsize=(5, 5))  
    df[(df['CELLCODE'] == gridcell) & (df['year'] == year)].plot(ax=ax)
    
    # Disable the grid
    ax.grid(False)
    
    # Annotate the plot with the metric value
    metrics = ['mean_par', 'mfs_ha', 'mperi']
    
    # Get metric values
    gridcell_data = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)]
    
    if not gridcell_data.empty:
        # Annotate each metric under the subplot
        for j, metric in enumerate(metrics):
            metric_value = gridcell_data[metric].values[0] if metric in gridcell_data else 'N/A'
            ax.annotate(f'{metric}: {metric_value}', xy=(0.5, -0.20 - j*0.1), 
                        xycoords='axes fraction', ha='center', fontsize=10)
    
    # Save plot to directory
    # fig_single.savefig(f'reports/figures/gridcell_{gridcell}_{year}.png', dpi=100)
    plt.show()

# Example usage
plot_gridcell_single(gridgdf_raw, '10kmE416N330', 2017)

################################
# grid outliers removal final
'''criteria: fields < 300'''
################################
# %%
outliers300 =gridgdf[(gridgdf['fields'] < 300)]
outliers300_ =outliers300.drop(columns='geometry')

# %%
gridgdf_ = gridgdf[~(gridgdf['fields'] < 300)]

# %% examine distribution of medpar after dropping grids with fields < 300
data = gridgdf_
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=data, x='year', y='medpar')

# %%
pm.stack_plots_in_grid(gridgdf_, unique_years, pm.scatterplot_mpar_marea, "fields", "medpar", ncols=4, figsize=(25, 15))


####################################################################
# verifying grids dropped for having fields < 300
####################################################################
'''1. see if each unique grid was dropped in all 12 years'''
# %% Get unique CELLCODE values and their count of occurrences
celcode_counts = outliers300_['CELLCODE'].value_counts()
# Convert the result to a DataFrame for better readability
celcode_counts_df = celcode_counts.reset_index()
celcode_counts_df.columns = ['CELLCODE', 'count']
''' 2. next goal is that: for some of the dropped grids,
there are some which have less than 12 years.
we need to check if the fact that these grids have less than 300 fields 
makes them outliers and if so, we should probably make sure that other years not
already dropped are also dropped. If not, we should add them back to the data.'''
# first step is to verify in gridgdf what the values of fields are for these grid
#in all years. This allows us to know why these grids were not identified as
# containing < 300 fields for 12 years
# before testing, filter gridgdf for a CELLCODE, remove geometry and check
filtered_df = gridgdf[gridgdf['CELLCODE'] == '10kmE417N340']
filtered_df_ = filtered_df.drop(columns='geometry')

# %% then filter out all rows of each target grid from the data
filtered = gridgdf[gridgdf['CELLCODE'] == '10kmE427N319']
# this is because we will add all back to see the effect of the grid
# %% aalso drop thm from cleaned data so that we don't have them duplicated
gir = gridgdf_[~(gridgdf_['CELLCODE'] == '10kmE427N319')]
# gir should be equal to gridgdf_ - number of rows of this grid that 
# was retained as having fields > 300
# %% add filtered back to this version of gridgdf
testgdf = pd.concat([gir, filtered], ignore_index=True)
# check number of rows in testgdf.
# it should be equal to gridgdf_ + number of rows that
# was cleaned out for this grid
# %% examine distribution of medpar after adding back filtered = 10kmE417N340
data = testgdf
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=data, x='year', y='medpar')

''' Learnings:
T1: adding back '10kmE417N340' to 2016 does not change the distribution of medpar and fields
we should add this back to the data

'10kmE427N319': box and scatterplots look the same as gridgdf_. this grid is indifferent
'10kmE430N325': box and scatterplots look the same as gridgdf_. this grid is indifferent
'10kmE414N324': box and scatterplots look the same as gridgdf_. this grid is indifferent
'10kmE442N331': we see a jump in medpar in 2018. this grid is an outlier
'10kmE429N339': we see a outliers in medpar in 2021, 22, 23. this grid is an outlier
'10kmE433N331': box and scatterplots look the same as gridgdf_. this grid is indifferent
'10kmE438N336': we see lower limit outliers in medpar in years except 2015 and 2017. this grid is an outlier
'''
# %% if indifferent i.e., cellcode is ['10kmE417N340', '10kmE427N319', '10kmE430N325', '10kmE414N324', '10kmE433N331']
# filter the cellcode from outliers300

# List of CELLCODE values to filter
cellcodes = ['10kmE417N340', '10kmE427N319', '10kmE430N325', '10kmE414N324', '10kmE433N331']
# Filter the DataFrame for the specified CELLCODE values
notoutlier = outliers300[outliers300['CELLCODE'].isin(cellcodes)]
notoutlier_ = notoutlier.drop(columns='geometry')

# %%
outliers300 = outliers300[~(outliers300['CELLCODE'].isin(cellcodes))]
# %% add notoutlier back to gridgdf_
gridgdffin = pd.concat([gridgdf_, notoutlier], ignore_index=True)

# %% examine distribution of medpar 
data = gridgdffin
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=data, x='year', y='medpar')

# %% examine scatter plot of medpar and fields
pm.stack_plots_in_grid(data, unique_years, pm.scatterplot_mpar_marea, "fields", "medpar", ncols=4, figsize=(25, 15))
# plot should look like plot for gridgdf_
