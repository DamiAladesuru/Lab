# %%
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns


# %% Change the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %% Load pickle file
with open('data/interim/griddf.pkl', 'rb') as f:
    griddf = pickle.load(f)
griddf.info()    
griddf.head()
# Convert the DataFrame to a GeoDataFrame
gridgeodf = gpd.GeoDataFrame(griddf, geometry='geometry')

# %% ########################################
# Load Germany grid, join to main data and remore duplicates using largest intersection
grid = gpd.read_file('data/interim/eeagrid_25832')
grid.plot()
grid.info()
grid.crs

# %% Join grid to griddf using cellcode
griddf_ = gridgeodf.merge(grid, on='CELLCODE')
griddf_.info()
griddf_.head()

# %%
# Define the bin edges for Mean Field Size (mfs_ha)
bins = [0, 2, 4, 8, 16, 32, 64, 150, float('inf')]
# Define the bin labels
labels = ['<2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-150', '>150']
# Create the range column
griddf__['mfs_range'] = pd.cut(griddf__['mfs_ha'], bins=bins, labels=labels)

# %% Create a FacetGrid with a count plot for each year in griddf
g = sns.FacetGrid(griddf__, col="year", col_wrap=4, height=4)
g.map(sns.countplot, "mfs_range")
g.set_titles("{col_name}")
g.set_xlabels('MFS Range')
g.set_ylabels('Grid Count')
g.show()

#####################################################################################################################
# %% Create a count plot
sns.countplot(x='mfs_range', data=griddf__)
# Set the plot title and labels
plt.title('Count of Grid Cells in Each Range Category')
plt.xlabel('Range')
plt.ylabel('Count')
# Show the plot
plt.show()


#%%
sns.histplot(griddf__, x="year", y="mfs_ha", hue="year", multiple="stack")

