
# %% Importing modules
import os
import seaborn as sns
import joypy
from joypy import joyplot
from matplotlib import cm
import matplotlib.pyplot as plt

# %%
os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.raw import gridgdf_desc_raw as grdr
from src.analysis import gridgdf_desc2 as gd
from src.visualization import plotting_module as pm

''' in pm, we have intialize_plotting which we have to run first to set up the color dictionary and plot
 for the first time. After that we can use other functions directly and metric colors will be consistent across all plots.'''

# %% load data
gld_ext, gridgdf_raw = grdr.silence_prints(
    grdr.create_gridgdf_raw,
    gridfile_suf='nole100',
    apply_t=False,
    gld_file='data/interim/gldkc_min100.pkl'
    )
grid_allyears_raw, grid_yearly_raw = grdr.silence_prints(grdr.desc_grid,gridgdf_raw)

# %% all data - base
import pickle as pkl
gld_base = pkl.load(open('data/interim/gld_wtkc.pkl', 'rb'))

# %%
gld_clean =gld_ext[~(gld_ext['area_m2'] < 100)]
df16_10 = gld_clean[(gld_clean['year'] == 2016) & (gld_clean['par'] > 10)]
df16_10 = df16_10.drop(columns='geometry')
# field with FLIK DENILI1531570006 and year 2016 has par > 10

# %%
#df = gld_ext
# filter for rows where 'area_m2' is greater than or equal to 100.
# in germany, gardens, representing comparatively small fields have size between 100 and 800m2
# depending on whether private, schreber or larger rural area gardens.
# thus makes sense to throw out fields smaller than 100m2 in our data
gld_300 = gld_base[~(gld_base['area_m2'] < 300)]


# %% rows with area_m2 below 300
df_300 = gld_clean[(gld_clean['area_m2'] < 300)]
df_300 = df_300.drop(columns='geometry')
# over 36k rows have area_m2 below 300

# %% Initial call to set up the color dictionary and plot absolute change in field metrics
multiline_df = grid_yearly_raw
color_dict_path = 'reports/figures/ToF/label_color_dict.pkl'

pm.initialize_plotting(
    df=multiline_df,
    title='Trend of Absolute Change in Field Metric Value Over Time',
    ylabel='Average Absolute Change',
    metrics={
        'MFS': 'mfs_ha_adiff_y1',
        'mperi': 'mperi_adiff_y1',
        'MeanPAR': 'mean_par_adiff_y1',
        'Fields/Ha': 'fields_ha_adiff_y1'
    },
    color_dict_path=color_dict_path
)



# %%
df = gld_250
unique_years = df['year'].unique()

# %% Loop through each year and create scatterplot of median par and field size
for year in unique_years:
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create a new figure for each year
    plt.figure()
    
    # Create scatterplot 
    sns.scatterplot(data=df_year, x="medfs_ha", y="medpar")
    
    # Set plot title
    plt.title(f'Median PAR and Median FS for Year {year}')
    
    # Show the plot
    plt.show()


# %% Loop through each year and create scatterplot of par and field size
for year in unique_years:
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create a new figure for each year
    plt.figure(figsize=(10, 6))
    
    # Create scatterplot with color based on 'category3'
    sns.scatterplot(data=df_year, x="area_ha", y="par", hue="category3")
    
    # Set plot title and labels
    plt.title(f'PAR and Area for Year {year}', fontsize=16)
    plt.xlabel('Field Size (ha)', fontsize=14)
    plt.ylabel('PAR', fontsize=14)
    
    # Show legend
    plt.legend(title='Category 3', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# %% simple joyplot
df = gridgdf_raw
labels = [y for y in list(df.year.unique())]
fig, axes = joypy.joyplot(df, by="year", column="mean_par", labels=labels, range_style='own', 
                          linewidth=1, legend=True, figsize=(6,5),
                          title="Mean PAR distribution of all CELLCODES",
                          colormap=cm.autumn)


# %% loopityloop create yearly joyplot
unique_years = df['year'].unique()
for year in unique_years:
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create labels for the current year
    labels = [y for y in list(df_year.Gruppe.unique())]
    
    # Create the joyplot for the current year
    fig, axes = joypy.joyplot(
        df_year, 
        by="Gruppe", 
        column="area_ha", 
        labels=labels, 
        range_style='own', 
        linewidth=1, 
        legend=True, 
        figsize=(6, 5),
        title=f"Area distribution in {year}",
        colormap=cm.autumn
    )
    
plt.show()

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


########################################################################################
# examine grids with median area greater than 10 and median par greater than 0.10
########################################################################################
# %%
gridgdf_raw.info()

# %% subset data to exclude rows where median area is greater than 10
gridle10 = gridgdf_raw[~(gridgdf_raw['medfs_ha'] > 10)]
# keep these outlier rows in a df and check the characteristics
out_area = gridgdf_raw[(gridgdf_raw['medfs_ha'] > 10)]

# %% subset data to exclude rows where median area is greater than 6
gridle6 = gridgdf_raw[~(gridgdf_raw['medfs_ha'] > 6)]
# keep these outlier rows in a df and check the characteristics
out_area = gridgdf_raw[(gridgdf_raw['medfs_ha'] > 6)]

# %% subset data to exclude rows where mean area is greater than 8
gridlem8 = gridgdf_raw[~(gridgdf_raw['mfs_ha'] > 8)]
# keep these outlier rows in a df and check the characteristics
out_aream = gridgdf_raw[(gridgdf_raw['mfs_ha'] > 8)]
# %% drop geomertry column
out_areamdf = out_aream.drop(columns='geometry')

# %% subset data to exclude rows where median par is greater than 0.07
gridle07 = gridgdf_raw[~(gridgdf_raw['medpar'] > 0.07)]
# keep these outlier rows in a df and check the characteristics
out_par = gridgdf_raw[(gridgdf_raw['medpar'] > 0.07)]

# %% before checking the characteristics, plot scatterplot and joyplot to see if data changes
df = gridlem8
labels = [y for y in list(df.year.unique())]
fig, axes = joypy.joyplot(df, by="year", column="medpar", labels=labels, range_style='own', 
                          linewidth=1, legend=True, figsize=(6,5),
                          title="Median Par distribution of all CELLCODES",
                          colormap=cm.pink)
# %%
stack_plots_in_grid(gridle07, unique_years, scatterplot_mpar_marea, ncols=4, figsize=(25, 15))

# %%
stack_plots_in_grid(gridlem8, unique_years, scatterplot_mpar_marea, "medfs_ha", "medpar", ncols=4, figsize=(25, 15))


# %%
gridle7[(gridle7['year'] == 2014)]['medfs_ha'].max()

# %%
def box_plot(df, column):
    sns.boxplot(df[column])
    plt.title(f'Box Plot of {column}')
    plt.show()
    
box_plot(gridle7, 'medfs_ha')

# %% facet grid of KDE plots for each year
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def kde_by_year(df, value_column):
    # Set up the FacetGrid to create separate plots for each year
    g = sns.FacetGrid(df, col="year", col_wrap=4, height=4)  # Adjust col_wrap and height as needed
    
    # Map the kdeplot function onto the facet grid for each year
    g.map(sns.kdeplot, value_column, fill=True)
    
    # Add titles and labels
    g.set_axis_labels(value_column, 'Density')
    g.set_titles("Year: {col_name}")
    
    # Show the plot
    plt.show()

kde_by_year(gridle6, 'mean_par')


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
kde_for_year(gridle6, 'medpar', 2020)
kde_for_year(gridle6, 'mean_par', 2020)

# %%
kde_for_year(gld_ext, 'mean_par', 2013)

# %%
geoData = gridlemed5.to_crs(epsg=4326)
plot_facet_choropleth_with_geoplot(geoData, column='medfs_ha', cmap='plasma', year_col='year', ncols=4)
# %%
gridclean = gridgdf_raw[~(gridgdf_raw['LANDKREIS'] == 'Küstenmeer Region Weser-Ems')| (gridgdf_raw['LANDKREIS'] == 'Lüneburg')]
# %%
# List of CELLCODE values to filter
cellcodes = ['10kmE433N332', '10kmE439N322', '10kmE442N331', '10kmE438N336', '10kmE417N341']

# Filter the DataFrame for the specified CELLCODE values
edgegrids = gridgdf_raw[gridgdf_raw['CELLCODE'].isin(cellcodes)]
print(edgegrids[['CELLCODE', 'LANDKREIS']])
# %%
def box_plot(df, column):
    sns.boxplot(df[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
    
box_plot(gridlem8, 'mfs_ha')
# %%
plt.figure(figsize=(12,6))
ax = sns.boxplot(data = gridlem8, x='year', y='mfs_ha')

plt.show()


# %% Create the boxplot for every year
plt.figure(figsize=(12, 6))
ax = sns.boxplot(data=gridgdf_raw, x='year', y='mfs_ha')

# Extract upper whiskers (upper limit) for each year
upper_limits = []

# Access the whiskers (ax.lines contains the whiskers data)
whiskers = ax.lines

# Number of unique years
unique_years = gridgdf_raw['year'].unique()

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


# %%
# %% subset data to exclude rows where medfs_ha is greater than 5
gridlemed5 = gridlem8[~(gridlem8['medfs_ha'] > 5)]
# keep these outlier rows in a df and check the characteristics
out_areamed = gridlem8[(gridlem8['medfs_ha'] > 5)]

# %% drop geomertry column
out_areameddf = out_areamed.drop(columns='geometry')
# %%
stack_plots_in_grid(gridlem8, unique_years, scatterplot_mpar_marea, "mfs_ha", "medpar", ncols=4, figsize=(25, 15))

# %%
geoData = gld.to_crs(epsg=4326)
plot_facet_choropleth_with_geoplot(geoData, column='area_ha', cmap='plasma', year_col='year', ncols=4)


# %% checking for field with Par highr than 14 in 2016
m14_16 = gld[(gld['year'] == 2016) & (gld['par'] > 14) & (gld['area_m2'] > 100)]
# %%
m14_16 = m14_16.drop(columns='geometry')
# %%

# %% yearly explorations: scatterplot and joyplot
unique_years = sorted(gridgdf_raw['year'].unique())
stack_plots_in_grid(gridlem8, unique_years, scatterplot_mpar_marea, "mfs_ha", "mean_par", ncols=4, figsize=(25, 15))

# %% Example usage:
gld_300 = gld_base[~(gld_base['area_m2'] < 300)]
create_yearly_joyplot(gld_300, 'Gruppe', 'par', "PAR distribution in {year}")