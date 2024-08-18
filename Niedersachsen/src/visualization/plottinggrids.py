# %%
import os
import pandas as pd
from shapely.geometry import Polygon
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2


os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')
# %%
from src.analysis_and_models import describe_single

from src.visualization import visualcheck

gld, griddf, griddf_ext, mean_median, gridgdf = describe_single.process_descriptives()


output_path = 'reports/figures/'

# %% ################################################################
# Simple plot of the grid cell with certain metric value for one year
#####################################################################
def plot_gridcell_simple(gridgdf):
    # plot the grid cell
    metric = 'mfs_ha'
    metricvalue = gridgdf.mfs_ha.max()
    gridcell = '10kmE409N326'  # gridgdf.loc[gridgdf[metric] == metricvalue, 'CELLCODE'].values[0]
    year = 2013  # a random year of interest
    # year = gridgdf.loc[gridgdf[metric] == metricvalue, 'year'].values[0]
    
    # Plot the grid cell
    fig_single, ax = plt.subplots(figsize=(5, 5))  # Set figure size to 500x500 pixels
    gld[(gld['CELLCODE'] == gridcell) & (gld['year'] == year)].plot(ax=ax)
    # plt.title(f'Gridcell {gridcell} in {year} with {metric} of {metricvalue}')
    
    # Save plot to directory
    #fig_single.savefig(f'reports/figures/gridcell_{gridcell}_{year}.png', dpi=100)
    plt.show()

plot_gridcell_simple(gridgdf)

# %% sample compare images

def resize_image(image_path, target_size):
    image = Image.open(image_path)
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    return resized_image

# Load images
image1_path = 'reports/figures/test2013Cloppenburg.png'
image2_path = 'reports/figures/test2018Cloppenburg.png'

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
ssim_score, difference_image = visualcheck.compare_images('reports/figures/gridcell_10kmE418N332_2013.png', 'reports/figures/gridcell_10kmE418N332_2018.png')



# %% #######################################################################
# Plot over all years of a target grid cell with annotation of metric values
############################################################################
def plot_gridcell(gridgdf):
    # list metrics for annotation
    metric1 = 'mfs_ha'
    metric2 = 'mean_cpar'
    metric3 = 'grid_par'
    # get value for specific metric
    metricvalue = gridgdf.mfs_ha.max()
    gridcell = gridgdf.loc[gridgdf[metric1] == metricvalue, 'CELLCODE'].values[0]

    # plot this grid cell for all years showing the metric value
    gridcell_df = gld[gld['CELLCODE'] == gridcell]
    # facetgrid for the plots
    fig, ax = plt.subplots(4, 3, figsize=(15, 10))
    # loop through the years and plot the grid cell in the facetgrid
    for i, year in enumerate(gridcell_df['year'].unique()):
        yearly_data = gridcell_df[gridcell_df['year'] == year]
        yearly_data.plot(ax=ax[i//3, i%3])
        ax[i//3, i%3].set_title(f'Gridcell {gridcell} in {year}')

        # get yearly metric value
        metric_value1 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric1].values[0]
        metric_value2 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric2].values[0]
        metric_value3 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric3].values[0]
        
        # Add text annotation below the plot
        ax[i//3, i%3].annotate(f'{metric1}: {metric_value1}', xy=(0.5, -0.5), xycoords='axes fraction', ha='center', fontsize=10)
        ax[i//3, i%3].annotate(f'{metric2}: {metric_value2}', xy=(0.5, -0.7), xycoords='axes fraction', ha='center', fontsize=10)
        ax[i//3, i%3].annotate(f'{metric3}: {metric_value3}', xy=(0.5, -0.9), xycoords='axes fraction', ha='center', fontsize=10)
        
    # Adjust the spacing between the plots
    plt.subplots_adjust(hspace=1.5, wspace=0.4)

    plt.show()


plot_gridcell(gridgdf)

# %% #######################################################################
# Plot for specific years of a target grid cell with annotation of metric values
############################################################################
def plot_gridcell_spec(gridgdf):
    # list metrics for annotation
    metric1 = 'mfs_ha'
    metric2 = 'mean_cpar'
    metric3 = 'grid_par'
    # get value for specific metric
    #metricvalue = gridgdf.mfs_ha.max()
    gridcell = '10kmE436N318' #gridgdf.loc[gridgdf[metric1] == metricvalue, 'CELLCODE'].values[0]
    
    # plot this grid cell for all years showing the metric value
    gridcell_df = gld[gld['CELLCODE'] == gridcell]
    
    # Filter for specific years
    years_to_plot = [2012, 2014, 2015, 2016]
    gridcell_df = gridcell_df[gridcell_df['year'].isin(years_to_plot)]
    # facetgrid for the plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    # loop through the years and plot the grid cell in the facetgrid
    for i, year in enumerate(years_to_plot):
        yearly_data = gridcell_df[gridcell_df['year'] == year]
        yearly_data.plot(ax=ax[i//2, i%2])
        ax[i//2, i%2].set_title(f'Gridcell {gridcell} in {year}')

        # get yearly metric value
        metric_value1 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric1].values[0]
        metric_value2 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric2].values[0]
        metric_value3 = gridgdf[(gridgdf['year'] == year) & (gridgdf['CELLCODE'] == gridcell)][metric3].values[0]
        
        # Add text annotation below the plot
        ax[i//2, i%2].annotate(f'{metric1}: {metric_value1}', xy=(0.5, -0.2), xycoords='axes fraction', ha='center', fontsize=10)
        ax[i//2, i%2].annotate(f'{metric2}: {metric_value2}', xy=(0.5, -0.3), xycoords='axes fraction', ha='center', fontsize=10)
        ax[i//2, i%2].annotate(f'{metric3}: {metric_value3}', xy=(0.5, -0.4), xycoords='axes fraction', ha='center', fontsize=10)
        
    # Adjust the spacing between the plots
    plt.subplots_adjust(hspace=0.5, wspace=0.4)

    plt.show()

plot_gridcell_spec(gridgdf)

# %%
