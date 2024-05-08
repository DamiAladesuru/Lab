# %%
import pickle
import geopandas as gpd
import pandas as pd
import os


# %% Change the current working directory
os.chdir('C:/Users/aladesuru/sciebo/StormLab/Research/Damilola/DataAnalysis/Lab/Niedersachsen')
# Print the current working directory to verify the change
print(os.getcwd())

# %% Load pickle file
with open('data/interim/gld.pkl', 'rb') as f:
    gld = pickle.load(f)
    
#########################################
# %% Field/ landscape level descriptive statistics
#########################################
    # total number of fields per year
    # min, max and mean value of field size, peri and shape index per year across landscape. We could have a box plot of these values across years.
gld.info()
ldscp1_desc_stats = gld.groupby('year')[['area_m2', 'peri_m', 'shp_index', 'fract']].describe()
ldscp1_desc_stats.to_csv('reports/statistics/ldscp1_desc_stats.csv') #save to csv

#######################################
# %% Grid level descriptive statistics
#######################################
    # total number of grids within the geographic boundaries of the study area
print("gridcount =", gld['CELLCODE'].nunique())
    # Area weight per grid for each year = area/ sum of area of fields in respective grid for each year
#all_years['area_wght'] = all_years.area / all_years.groupby('year')['area_m2'].transform('sum')
    # mean and sd distribution across grids over years. Create table of year, grid id, number of fields in grid, mean field size, sd_fs, mean peri, sd_peri, mean shape index, sd_shape index. Use line plot with shaded area to show the sd of each metric across grids over years.
    

############################################################
# Temporal analysis: change in number of grid as number of field increases or decreses over years within the grids
############################################################

# %%
