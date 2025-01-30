# %%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import mapclassify
from matplotlib import font_manager
#print(font_manager.findSystemFonts(fontpaths=None, fontext='ttf'))


# %% read data
geoData = gridgdf_cl.to_crs(epsg=4326)


# %% examine variable histogram plots so as to see distribution of values and decide on bins
gdf_oneyear = geoData[geoData['year'] == 2022]

# %%
gdf_oneyear["medperi_percdiff_to_y1"].hist(bins=20)
plt.xlabel("medperi_percdiff_to_y1")
plt.ylabel("Number of grid cells")
plt.title("Distribution of % change in perimeter relative to 2012")
plt.show()

# %%
gdf_oneyear["medfs_ha_percdiff_to_y1"].hist(bins=20)
plt.xlabel("medfs_ha_percdiff_to_y1")
plt.ylabel("Number of grid cells")
plt.title("Distribution of  % change in area relative to 2012")
plt.show()

# %%
gdf_oneyear["fields_ha_percdiff_to_y1"].hist(bins=20)
plt.xlabel("fields_ha_percdiff_to_y1")
plt.ylabel("Number of grid cells")
plt.title("Distribution of  % change in fields/ha relative to 2012")
plt.show()

# %%
gdf_oneyear["medpar_percdiff_to_y1"].hist(bins=20)
plt.xlabel("medpar_percdiff_to_y1")
plt.ylabel("Number of grid cells")
plt.title("Distribution of  % change in shape relative to 2012")
plt.show()

# %% examine quantiles
q = mapclassify.Quantiles(gdf.fields_ha, k=5)
q

# %% adjust color
# Define the color range
colors = plt.cm.YlGnBu(np.linspace(0.2, 1, 5))  # Start from 0.2 instead of 0
n_bins = 5  # Number of bins
# Create the colormap
cmap = colors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)

'''Step 1: Create difference and 2012 plots seperately'''
# %% difference or change plots
def plot_diff_maps(gdf, year_list, columns, bins, labels, cmap="coolwarm"):
    """
    Creates a multi-row plot where each row corresponds to a column, and each row
    contains choropleth maps for the specified years.

    Parameters:
    - gdf: GeoDataFrame containing the data.
    - year_list: List of years to include in the plots.
    - columns: List of columns to visualize.
    - cmap: Colormap to use for the choropleth.
    """
    # Number of rows is equal to the number of columns
    nrows = len(columns)
    ncols = len(year_list)

    # Create the figure and axes grid
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5, 7)
    )

    # Normalize the bins and set up the colormap
    norm = colors.BoundaryNorm(boundaries=bins, ncolors=256)
    scalar_mappable = cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])

    # Loop through each column and plot
    for row, column in enumerate(columns):
        for col, year in enumerate(year_list):
            ax = axes[row, col] if nrows > 1 else axes[col]  # Handle single-row case

            # Subset GeoDataFrame for the current year
            gdf_year = gdf[gdf['year'] == year]

            # Remove rows with NaN or Inf in the specified column
            gdf_year = gdf_year.replace([np.inf, -np.inf], np.nan).dropna(subset=[column])

            if gdf_year.empty:  # Skip if there's no valid data
                ax.set_axis_off()
                continue

            # Create binned data
            binned_column = f"{column}_bins"
            gdf_year[binned_column] = pd.cut(gdf_year[column], bins=bins, labels=labels, right=False)              
            
            # Plot the choropleth map for the current year and column
            gdf_year.plot(
                column=binned_column,
                cmap=cmap,
                legend=False,  # Disable individual subplot legends
                edgecolor="white",
                linewidth=0.1,
                ax=ax,
            )

            # Set the title for the current column
            if row == 0:
                ax.set_title(f"{year}", fontsize=11)

            # Remove x and y ticks for clarity
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.set_axis_off()  # To drop axes box

    # Add a shared legend below the subplots
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.1,
        aspect=40,
    )


    # Compute bin midpoints for tick placement
    bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    # Set tick positions and labels
    cbar.set_ticks(bin_midpoints)  # Use midpoints for tick placement
    cbar.set_ticklabels(labels)  # Ensure the labels match the intervals
    cbar.set_label("Percentage Change from Base Year (2012)")
   
    # Adjust layout using subplots_adjust
    fig.subplots_adjust(
        left=0.1,  # Adjust left margin
        right=0.9,  # Adjust right margin
        top=0.9,  # Adjust top margin
        bottom=0.15,  # Adjust bottom margin for the legend
        wspace=0,  # Adjust width between subplots
        hspace=0,  # Adjust height between subplots
    )

    # Show the plot
    plt.show()
    
# %%
columns = ["medfs_ha_percdiff_to_y1", "medperi_percdiff_to_y1", "fields_ha_percdiff_to_y1", "medpar_percdiff_to_y1"]
year = [2015, 2020, 2023]
bins = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labels = ["<-6", "-5", "-3", "-1.5", "-0.5", "0.5", "1.5", "3", "5", ">6"]

plot_diff_maps(geoData, year, columns, bins, labels, cmap="coolwarm")

# %% baseline maps i.e., 2012 plots
def create_2012bins_and_plot(gdf, column, bins, labels, cmap="Paired", ltitle="", plottitle="", ax=None):
    binned_column = f"{column}_bins"
    gdf.loc[:, binned_column] = pd.cut(gdf[column], bins=bins, labels=labels, right=False)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    gdf.plot(
        column=binned_column,
        cmap=cmap,
        legend=True,
        ax=ax,
        legend_kwds={
            "loc": "lower left",
            "fmt": "{:.3f}",
            "fontsize": "small",
            "frameon": False,
            "title": ltitle,
            "labelspacing": 0.3
        },
        edgecolor="white",
        linewidth=0.1
    )
    ax.set_ylabel(plottitle, rotation=90, va='center')
    ax.yaxis.set_label_coords(-0.1, 0.5)

    #ax.set_axis_off()
    return ax

def create_baseline_plot(gdf):
    fig, axs = plt.subplots(4, 1, figsize=(4.5, 8))  # 4 rows, 1 column
    
    # Median Field Size
    mfsbins = [0.5, 1.5, 2.0, 2.5, 3.5, 5.0]
    mfslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(mfsbins[:-1], mfsbins[1:])]
    create_2012bins_and_plot(gdf, column="medfs_ha", bins=mfsbins, labels=mfslabels, ltitle="Median\nField Size (ha)", plottitle="Average Field Size (ha)", ax=axs[1])    

    # Perimeter
    peribins = [440, 620, 670, 730, 780, 1300]
    perilabels = ["440 - 620", "620 - 670", "670 - 730", "730 - 780", "780 - 1300"]
    create_2012bins_and_plot(gdf, column="medperi", bins=peribins, labels=perilabels, ltitle="Median\nPerimeter (m)", plottitle="Average Perimeter", ax=axs[3])    
    
    # Fields/Ha
    fieldsbins = [0, 0.29, 0.33, 0.37, 0.43, 1.0]
    fieldslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(fieldsbins[:-1], fieldsbins[1:])]
    create_2012bins_and_plot(gdf, column="fields_ha", bins=fieldsbins, labels=fieldslabels, ltitle="Fields/Ha", plottitle="Count of Fields", ax=axs[2])
    
    # Shape Complexity
    parbins = [0.02, 0.03, 0.035, 0.04, 0.05, 0.07]
    parlabels = [f"{low * 10:.2f} - {high * 10:.2f}" for low, high in zip(parbins[:-1], parbins[1:])]
    create_2012bins_and_plot(gdf, column="medpar", bins=parbins, labels=parlabels, ltitle="Median PAR", plottitle="Average Shape Complexity", ax=axs[0])   
     
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.07, wspace=0)
    plt.show()

# %% Call the function to create the combined plot
gdf = geoData[geoData['year'] == 2012]
gdf = geoData[geoData['year'] == 2012].copy()
create_baseline_plot(gdf)

'''Step 21: Create the 2012 and difference plots together'''
# %% combined baseline and difference plots
def plot_choropleths_maps(gdf, year_list, columns, col_titles, bins_labels_dict, basecmap="Paired", diffcmap="coolwarm", suptitle=""):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Nimbus Sans", "Arial", "Helvetica", "DejaVu Sans"]
    nrows = 4
    ncols = 4

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))  # Added extra height for colorbar

    # First column: Baseline values for 2012
    gdf_2012 = gdf[gdf['year'] == 2012]
    axes[0, 0].set_title("2012", fontsize=12)

    for row, column in enumerate(columns[:4]):
        bins, labels = bins_labels_dict[column]
        
        binned_column = f"{column}_bins"
        gdf_2012[binned_column] = pd.cut(gdf_2012[column], bins=bins, labels=labels, right=False)
        
        gdf_2012.plot(
            column=binned_column,
            cmap=basecmap,
            legend=True,
            legend_kwds={
                "loc": "lower left",
                "fontsize": "x-small",
                "title": None,
                "title_fontsize": "x-small",
                "frameon": False,
                "labelspacing": 0.2,
                "alignment": "center",
                "borderpad": 0.5,
                "bbox_to_anchor": (0, -0.1),
            },
            edgecolor="white",
            linewidth=0.1,
            ax=axes[row, 0],
        )
        
        axes[row, 0].set_axis_off()
        axes[row, 0].text(-0.05, 0.5, col_titles[row], rotation=90, verticalalignment='center', 
                          horizontalalignment='center', transform=axes[row, 0].transAxes, fontsize=12)

    # Remaining columns: Percentage differences
    bins, labels = bins_labels_dict[columns[4]]  # Use the same bins and labels for all percentage difference columns
    norm = colors.BoundaryNorm(bins, plt.get_cmap(diffcmap).N)

    for col, year in enumerate(year_list, start=1):
        gdf_year = gdf[gdf['year'] == year]
        
        for row, column in enumerate(columns[4:]):
            binned_column = f"{column}_bins"
            gdf_year[binned_column] = pd.cut(gdf_year[column], bins=bins, labels=labels, right=False)
            
            gdf_year.plot(
                column=column,  # Use the actual values, not the binned ones
                cmap=diffcmap,
                norm=norm,
                legend=False,  # We'll add a shared legend later
                edgecolor="white",
                linewidth=0.1,
                ax=axes[row, col],
            )
            
            if row == 0:
                axes[row, col].set_title(f"{year}", fontsize=12)
            
            axes[row, col].set_axis_off()

    # Add a shared colorbar legend below the subplots
    scalar_mappable = plt.cm.ScalarMappable(cmap=diffcmap, norm=norm)
    scalar_mappable.set_array([])
    cbar = fig.colorbar(
        scalar_mappable,
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        pad=0.1,
        aspect=40,
        shrink=0.8,
    )

    # Compute bin midpoints for tick placement
    bin_midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

    # Set tick positions and labels
    cbar.set_ticks(bin_midpoints)
    cbar.set_ticklabels(labels)
    cbar.set_label("Percentage Change from Base Year (2012)")

    # Adjust layout using subplots_adjust
    fig.subplots_adjust(
        left=0.15,  # Adjust left margin
        right=0.85,  # Adjust right margin
        top=0.9,  # Adjust top margin
        bottom=0.15,  # Adjust bottom margin for the legend
        wspace=0,  # Adjust width between subplots
        hspace=0,  # Adjust height between subplots
    )
    
    # Add a super title
    fig.suptitle(suptitle, fontsize=16, y=0.925) # reduce y to move title nearer to plots
    
    #save plot as svg
    plt.savefig("reports/figures/results/FiSC_choropleth_mapBerlin_.svg", format="svg", bbox_inches='tight')
    plt.savefig("reports/figures/results/FiSC_choropleth_mapBerlinPN_.png", format="png", bbox_inches='tight')
    
    plt.show()
# %%
# Usage
columns = ["medfs_ha", "medperi", "fields_ha", "medpar", 
           "medfs_ha_percdiff_to_y1", "medperi_percdiff_to_y1", "fields_ha_percdiff_to_y1", "medpar_percdiff_to_y1"]
year_list = [2015, 2019, 2023]
col_titles = ["Median Field Size (ha)", "Median Perimeter (m)", "Number of Fields", "Median Shape Complexity"]

binsA = [-60, -6, -4, -2, -1, 0, 1, 2, 4, 6, 60]
labelsA = ["<-6%", "-5%", "-3%", "-1.5%", "-0.5%", "0.5%", "1.5%", "3%", "5%", ">6%"]

mfsbins = [0.5, 1.5, 2.0, 2.5, 3.5, 5.0]
mfslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(mfsbins[:-1], mfsbins[1:])]

peribins = [440, 620, 670, 730, 780, 1300]
perilabels = ["440 - 620", "620 - 670", "670 - 730", "730 - 780", "780 - 1300"]

fieldsbins = [0, 0.29, 0.33, 0.37, 0.43, 1.0]
fieldslabels = [f"{low:.2f} - {high:.2f}" for low, high in zip(fieldsbins[:-1], fieldsbins[1:])]

parbins = [0.02, 0.03, 0.035, 0.04, 0.05, 0.07]
parlabels = [f"{low * 10:.2f} - {high * 10:.2f}" for low, high in zip(parbins[:-1], parbins[1:])]

bins_labels_dict = {
    "medfs_ha": (mfsbins, mfslabels),
    "medperi": (peribins, perilabels),
    "fields_ha": (fieldsbins, fieldslabels),
    "medpar": (parbins, parlabels),
    "medfs_ha_percdiff_to_y1": (binsA, labelsA),
    "medperi_percdiff_to_y1": (binsA, labelsA),
    "fields_ha_percdiff_to_y1": (binsA, labelsA),
    "medpar_percdiff_to_y1": (binsA, labelsA)
}
# %%
plot_choropleths_maps(geoData, year_list, columns, col_titles, bins_labels_dict, basecmap="berlin", diffcmap="coolwarm", suptitle="Field Structural Change Over Time")


# %%
