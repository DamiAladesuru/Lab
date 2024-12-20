''' chloropleth mapping '''
# %%
import cartopy.crs as ccrs
import geopandas as gpd
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import mapclassify as mc
import plotly.express as px
import seaborn as sns

'''
for choropleth mapping, ensure that:
- the GeoDataFrame is epsg:4326
e.g., simply point geoData = gdf.to_crs(epsg=4326)
25832 is the default crs for the data

This module contains other choropleth functios to be further explored
'''

# %% single chloropleth map: sequential
fig, ax = plt.subplots(figsize=(12, 8))
geoData.plot(column='mean_par_percdiff_to_y1', cmap='viridis', linewidth=0.5, edgecolor='black', ax=ax, legend=True)
# Set aspect ratio if needed
ax.set_aspect('equal')
#plt.title()
plt.show()

# %% single chloropleth map: interval
scheme = mc.Quantiles(geoData['mfs_ha'], k=10)

# Map
gplt.choropleth(
    geoData,
    projection=gcrs.AlbersEqualArea(),
    hue="mfs_ha",
    scheme=scheme, cmap='inferno_r',
    linewidth=0.5,
    edgecolor='black',
    figsize=(12, 8),
    legend=True
)
plt.show()



# %% individual yearly chloropleth maps
"""
# Get unique years in the GeoDataFrame
unique_years = geoData['year'].unique()

# Loop through each year and create a choropleth map
for year in unique_years:
    # Subset the GeoDataFrame for the current year
    geoData_year = geoData[geoData['year'] == year]
    
    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the choropleth map for the current year
    geoData_year.plot(column='mfs_ha', cmap='inferno', linewidth=0.5, edgecolor='black', ax=ax, legend=True)
    
    # Set aspect ratio if needed
    ax.set_aspect('equal')
    
    # Set the title for the plot
    plt.title(f'Choropleth Map of mfs_ha for Year {year}')
    
    # Show the plot
    plt.show()
"""
##################################################
# facet grid of chloropleth maps: interval
##################################################
# %%
def plot_choropleth_by_year(gdf, hue_column='mfs_ha', n_quantiles=6, cmap='inferno_r', ncols=4):
    """
    Plots a facet grid of choropleth maps for each unique year with quantiles.

    Parameters:
    - gdf: GeoDataFrame containing the geometries and data.
    - hue_column: The column name to visualize in the choropleth map.
    - n_quantiles: Number of quantiles for classification.
    - cmap: Colormap to use for the choropleth.
    - ncols: Number of columns in the facet grid.
    """
    
    # Ensure the GeoDataFrame is in the correct CRS
    gdf = gdf.to_crs(epsg=4326)

    # Get unique years in the GeoDataFrame
    unique_years = gdf['year'].unique()

    # Create a figure and axes for the facet grid
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 5 * nrows), subplot_kw={'projection': gcrs.AlbersEqualArea()})

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each year and create a choropleth map
    for i, year in enumerate(unique_years):
        # Subset the GeoDataFrame for the current year
        gdf_year = gdf[gdf['year'] == year]
        
        # Apply quantile classification scheme
        scheme = mc.Quantiles(gdf_year[hue_column], k=n_quantiles)
        
        # Plot the choropleth map for the current year
        gplt.choropleth(
            gdf_year,
            projection=gcrs.AlbersEqualArea(),
            hue=hue_column,
            scheme=scheme,
            cmap=cmap,
            linewidth=0.5,
            edgecolor='black',
            ax=axes[i],
            legend=True,  # Include legend for each plot
            legend_kwargs={'loc': 'lower left'},  # Position the legend
        )
        
        # Set the title for the subplot
        axes[i].set_title(f'Year {year}', pad=0)
        axes[i].set_aspect('equal')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# %% Example usage:
plot_choropleth_by_year(geoData, hue_column='mean_par_diff_from_y1', n_quantiles=4, cmap='viridis', ncols=4)

# %% interactive plotly chloropleth. uses epsg:4326
# Create the choropleth map
fig = px.choropleth_mapbox(geoData,
                           geojson=geoData.geometry,
                           locations=geoData.index,
                           color='mfs_ha',
                           mapbox_style="carto-positron",
                           center={"lat": geoData.geometry.centroid.y.mean(), "lon": geoData.geometry.centroid.x.mean()},
                           zoom=6,
                           title="Choropleth Map of mfs_ha")

# Update layout
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Show the plot
fig.show()


# %% with user defined intervals
def plot_choropleth_by_year(gdf, hue_column='mfs_ha', cmap='inferno_r', ncols=4):
    """
    Plots a facet grid of choropleth maps for each unique year in the GeoDataFrame using custom intervals.

    Parameters:
    - gdf: GeoDataFrame containing the geometries and data.
    - hue_column: The column name to visualize in the choropleth map.
    - cmap: Colormap to use for the choropleth.
    - ncols: Number of columns in the facet grid.
    """
    # Custom intervals
    intervals_by_year = {
        'before_2018': [0.00, 0.05, 0.10, 0.15, float('inf')],
        '2018_2019': [0.00, 0.10, 0.20, 0.30, float('inf')],
        'after_2020': [0.00, 0.07, 0.14, 0.20, float('inf')]
    }

    # Ensure the GeoDataFrame is in the correct CRS
    gdf = gdf.to_crs(epsg=4326)

    # Get unique years in the GeoDataFrame
    unique_years = sorted(gdf['year'].unique())

    # Create a figure and axes for the facet grid
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 5 * nrows), subplot_kw={'projection':ccrs.AlbersEqualArea()})

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each year and create a choropleth map
    for i, year in enumerate(unique_years):
        # Subset the GeoDataFrame for the current year
        gdf_year = gdf[gdf['year'] == year]
        
        # Choose the intervals based on the year
        if year < 2018:
            intervals = intervals_by_year['before_2018']
        elif 2018 <= year <= 2019:
            intervals = intervals_by_year['2018_2019']
        else:
            intervals = intervals_by_year['after_2020']
        
        # Apply UswrDefined classification scheme
        scheme = mc.UserDefined(gdf_year[hue_column], bins=intervals)

    
        # Plot the choropleth map for the current year
        gplt.choropleth(
            gdf_year,
            hue=hue_column,
            scheme=scheme,
            cmap=cmap,
            linewidth=0.5,
            edgecolor='black',
            ax=axes[i],
            legend=True,  # Include legend for each plot
            legend_kwargs={'loc': 'lower left'},  # Position the legend
        )
        
        # Set the title for the subplot
        axes[i].set_title(f'Year {year}', pad=20)
        axes[i].set_aspect('equal')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    print(scheme.bins)  # Check the actual bins being applied

    # Show the plot
    plt.show()

# %% Example usage:
plot_choropleth_by_year(geoData, hue_column='mean_par', cmap='viridis_r', ncols=4)


# %%
# Define your custom colors for each interval
custom_colors = ['red', 'green', 'blue', 'cyan', 'magenta']  # Add more colors as needed

# %% working as I want for user defined intervals
def plot_choropleth_by_year(gdf, hue_column='mfs_ha', cmap = 'Set1', ncols=4):
    """
    Plots a facet grid of choropleth maps for each unique year in the GeoDataFrame using custom intervals.

    Parameters:
    - gdf: GeoDataFrame containing the geometries and data.
    - hue_column: The column name to visualize in the choropleth map.
    - cmap: Colormap to use for the choropleth.
    - ncols: Number of columns in the facet grid.
    """
    # Custom intervals
    intervals_by_year = {
        'before_2018': [0.00, 0.05, 0.10, 0.15, float('inf')],
        '2018_2019': [0.00, 0.10, 0.20, 0.30, float('inf')],
        'after_2020': [0.00, 0.07, 0.14, 0.20, float('inf')]
    }

    # Function to classify values based on the intervals
    def classify(value, intervals):
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                return f"{intervals[i]:.2f} - {intervals[i + 1]:.2f}"
        return f"> {intervals[-2]:.2f}"

    # Ensure the GeoDataFrame is in the correct CRS
    gdf = gdf.to_crs(epsg=4326)

    # Get unique years in the GeoDataFrame
    unique_years = sorted(gdf['year'].unique())

    # Create a figure and axes for the facet grid
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 5 * nrows), subplot_kw={'projection': ccrs.AlbersEqualArea()})

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Define color map
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Loop through each year and create a choropleth map
    for i, year in enumerate(unique_years):
        # Subset the GeoDataFrame for the current year
        gdf_year = gdf[gdf['year'] == year]
        
        # Choose the intervals based on the year
        if year < 2018:
            intervals = intervals_by_year['before_2018']
        elif 2018 <= year <= 2019:
            intervals = intervals_by_year['2018_2019']
        else:
            intervals = intervals_by_year['after_2020']
        
        # Classify data based on intervals
        gdf_year['classification'] = gdf_year[hue_column].apply(lambda x: classify(x, intervals))
        
        # Plot the choropleth map for the current year
        gplt.choropleth(
            gdf_year,
            hue='classification',
            cmap=cmap,
            linewidth=0.5,
            edgecolor='black',
            ax=axes[i],
            legend=True,  # Include legend for each plot
            legend_kwargs={'loc': 'lower left'},  # Position the legend
        )
        
        # Set the title for the subplot
        axes[i].set_title(f'Year {year}', pad=20)
        axes[i].set_aspect('equal')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# %% Example usage:
plot_choropleth_by_year(geoData, hue_column='mean_par', cmap = 'Dark2', ncols=4)

# %% for mfs
def plot_choropleth_by_year(gdf, hue_column='mfs_ha', cmap='Set1', ncols=4):
    """
    Plots a facet grid of choropleth maps for each unique year in the GeoDataFrame using fixed intervals.

    Parameters:
    - gdf: GeoDataFrame containing the geometries and data.
    - hue_column: The column name to visualize in the choropleth map.
    - cmap: Colormap to use for the choropleth.
    - ncols: Number of columns in the facet grid.
    """

    # Fixed intervals (ensure sorted order)
    intervals = sorted([0.00, 2.50, 5.00, 7.50, 10.00, float('inf')])  # Sorting to ensure ascending order

    # Function to classify values based on the intervals
    def classify(value):
        for i in range(len(intervals) - 1):
            if intervals[i] <= value < intervals[i + 1]:
                return f"{intervals[i]:.2f} - {intervals[i + 1]:.2f}"
        return f"> {intervals[-2]:.2f}"

    # Ensure the GeoDataFrame is in the correct CRS
    gdf = gdf.to_crs(epsg=4326)

    # Get unique years in the GeoDataFrame
    unique_years = sorted(gdf['year'].unique())

    # Create a figure and axes for the facet grid
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate the number of rows needed
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 5 * nrows), subplot_kw={'projection': ccrs.AlbersEqualArea()})

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop through each year and create a choropleth map
    for i, year in enumerate(unique_years):
        # Subset the GeoDataFrame for the current year
        gdf_year = gdf[gdf['year'] == year]
        
        # Classify data based on intervals
        gdf_year['classification'] = gdf_year[hue_column].apply(classify)
        
        # Plot the choropleth map for the current year
        gplt.choropleth(
            gdf_year,
            hue='classification',
            cmap=cmap,
            linewidth=0.5,
            edgecolor='black',
            ax=axes[i],
            legend=True,  # Include legend for each plot
            legend_kwargs={'loc': 'lower left'},  # Position the legend
        )
        
        # Set the title for the subplot
        axes[i].set_title(f'Year {year}', pad=20)
        axes[i].set_aspect('equal')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# Example usage
plot_choropleth_by_year(geoData, hue_column='mfs_ha', cmap='viridis', ncols=4)

# %% specify if main function
_