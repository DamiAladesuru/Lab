# %%
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

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
# %% Load grid 
with open('data/interim/griddf.pkl', 'rb') as f:
    griddf = pickle.load(f)
griddf.info()    
griddf.head()


# %% ########################################
# Load Germany grid to obtain the grid geometry
grid = gpd.read_file('data/interim/eeagrid_25832')
grid.plot()
grid.info()
grid.crs

# %% Join grid to griddf using cellcode
griddf_ = griddf.merge(grid, on='CELLCODE')
griddf_.info()
griddf_.head()

# %% Convert the DataFrame to a GeoDataFrame
griddf_ = gpd.GeoDataFrame(griddf_, geometry='geometry')

# %% 
# Distribution of count of grids based on mean field size range 
# Define the bin edges for Mean Field Size (mfs_ha)
bins = [0, 2, 4, 8, 16, 32, 64, 150, float('inf')]
# Define the bin labels
labels = ['<2', '2-4', '4-8', '8-16', '16-32', '32-64', '64-150', '>150']
# Create the range column
griddf_['mfs_range'] = pd.cut(griddf_['mfs_ha'], bins=bins, labels=labels)

# %% Create a FacetGrid with a count plot for each year in griddf
g = sns.FacetGrid(griddf_, col="year", col_wrap=4, height=4)
g.map(sns.countplot, "mfs_range", color='purple')
g.set_titles("{col_name}")
g.set_xlabels('MFS Range (ha)')
g.set_ylabels('Grid Count')
g.show()
 
# %%
# Create correlation plot of grid mfs_ha and mean_fract 
sns.scatterplot(data=griddf, x='mfs_ha', y='mean_fract', color='purple')
# Set the plot title and labels
plt.title('Correlation between Mean Field Size and Mean Fractal Dimension')
plt.xlabel('Mean Field Size')
plt.ylabel('Mean Fractal Dimension')
# Show the plot
plt.show()

# %%
# Create correlation plot of grid mfs_ha and field count 
sns.scatterplot(data=griddf, x='mfs_ha', y='fields', color='purple')
# Set the plot title and labels
plt.title('Correlation between Mean Field Size and Field Count')
plt.xlabel('Mean Field Size')
plt.ylabel('Field Count')
# Show the plot
plt.show()

######################################################################################################################
# %% load the grid descriptive stat csv file
griddesc = pd.read_csv('reports/statistics/grid_desc_statssum.csv') 

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

 
##########################################################################
# Visulaizing fields in select grid cells
##########################################################################
# %%
# 1. find in griddf the CELLCODE with random x (e.g., 200) number of fields
CELLCODEX = griddf.loc[griddf['MFSChng'] == 3.3129241246123406, 'CELLCODE'].values[0]
print(CELLCODEX)

# %% plot fields with target cellcode and selected year using gld data
gld[(gld['CELLCODE'] == CELLCODEX) & (gld['year'] == 2023)].plot()


# %% plot cell 10kmE419N331
gld[(gld['CELLCODE'] == '10kmE419N331') & (gld['year'] == 2023)].plot(edgecolor='red', facecolor='none', zoom=5)
# %%
