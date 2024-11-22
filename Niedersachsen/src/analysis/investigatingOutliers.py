
# %% Importing modules
import os
import seaborn as sns
import joypy
from matplotlib import cm
import matplotlib.pyplot as plt
os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.raw import gridgdf_desc_raw as grdr
from src.analysis import gridgdf_desc2 as gd
from src.visualization import plotting_module as pm

''' in pm, we have intialize_plotting which we have to run first to set up the color dictionary and plot
 for the first time. After that we can use other functions directly and metric colors will be consistent across all plots.'''

# %% load data
#gld_ext, gridgdf_raw = grdr.silence_prints(grdr.create_gridgdf_raw, include_sonstige=False, filename_suffix='no_son')
gld_ext, gridgdf_raw = grdr.silence_prints(grdr.create_gridgdf_raw, include_sonstige=True)
#if include_sonstige = True, the data will include 'sonstige flächen' in the data and for that, you should not specify filename_suffix
grid_allyears_raw, grid_yearly_raw = grdr.silence_prints(grdr.desc_grid,gridgdf_raw)

# %%
gld_clean =gld_ext[~(gld_ext['area_m2'] < 100)]
df16_10 = gld_clean[(gld_clean['year'] == 2016) & (gld_clean['par'] > 10)]
df16_10 = df16_10.drop(columns='geometry')
# field with FLIK DENILI1531570006 and year 2016 has par > 10

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
#df = gld_ext
# filter for rows where 'area_m2' is greater than or equal to 100.
# in germany, gardens, representing comparatively small fields have size between 100 and 800m2
# depending on whether private, schreber or larger rural area gardens.
# thus makes sense to throw out fields smaller than 100m2 in our data
df_100 = gld_ext[~(gld_ext['area_m2'] < 100)]
unique_years = df['year'].unique()

# %% Loop through each year and create scatterplot of par and field size
for year in unique_years:
    # Subset the DataFrame for the current year
    df_year = df_100[df_100['year'] == year]
    
    # Create a new figure for each year
    plt.figure()
    
    # Create scatterplot 
    sns.scatterplot(data=df_year, x="area_ha", y="par", hue="category3")
    
    # Set plot title
    plt.title(f'Scatterplot of PAR and Field Size for Year {year}')
    
    # Show the plot
    plt.show()


# %% simple joyplot
labels = [y for y in list(df.year.unique())]
fig, axes = joypy.joyplot(df, by="year", column="par", labels=labels, range_style='own', 
                          linewidth=1, legend=True, figsize=(6,5),
                          title="PAR distribution for all groups in 2017",
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