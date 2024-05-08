# %%
import geopandas as gpd
import shapely as sh
import matplotlib
import matplotlib.pyplot as plt
import fiona
import pyogrio
from datetime import datetime
import pandas as pd
import os

# %% Load the Tschlag data
tschlag = gpd.read_file("C://Users//aladesuru//Documents//Empirical//NRW//Data//Teilschläge_Shape.zip")
# %% Get information about the dataframe
tschlag.crs
tschlag.head()
tschlag.info()
tschlag.describe() #summary statistics of numeric variables 
#You can also use the attribute .columns to see just the column names in a dataframe,
#or the attribute .shape to just see the number of rows and columns.
# %% estimate additonal layer attributes like perimeter in metric and km. Area in m2 = 10000 * area in ha
tschlag['perimeter_m'] = tschlag.length
tschlag['perimeter_km'] = tschlag.length / 1000
tschlag['area_m2'] = tschlag.area
print(tschlag[['area_m2', 'AREA_HA', 'perimeter_m', 'perimeter_km']].head())
# %% Tschlaghist data
tschlaghist = gpd.read_file("C://Users//aladesuru//Documents//Empirical//NRW//Data//Teilschläge_Hist_Shape.zip")
tschlaghist.crs
tschlaghist.head()
tschlaghist.info()
tschlaghist.describe()
# %% estimate additonal layer attributes like perimeter in metric and km. Area in m2 = 10000 * area in ha
tschlaghist['perimeter_m'] = tschlaghist.length
tschlaghist['perimeter_km'] = tschlaghist.length / 1000
tschlaghist['area_m2'] = tschlaghist.area
print(tschlaghist[['area_m2', 'AREA_HA', 'perimeter_m', 'perimeter_km']].head())

# %% load the NUTS data for administrative boundaries
nuts = gpd.read_file("C://Users//aladesuru//Documents//Empirical//NRW//Data//NUTS_RG_20M_2021_3035.zip")
nuts.crs
# Reproject the NUTS data to match Tscombined
nuts = nuts.to_crs(tschlag.crs)
nuts.head()
nuts.info()
sorted_nuts = nuts.sort_values(by="NUTS_ID", ascending=True)
sorted_nuts.head()
#by viewing data in ArcGIS, I was able to identify the indices of the NUTS_3 data observations for NRW that I need for the next subsetting steps
# That is indices 499:557 which holds data for NUTS_ID DEA1 to DEA5C
nrw_nuts3 = sorted_nuts.iloc[499:557]
nrw_nuts3.info()
nrw_nuts3.plot()
#Save nuts to file
nrw_nuts3.to_file("C://Users//aladesuru//Documents//Empirical//NRW//Data//NRW_NUTS//NRW.shp")

# %% load grid file
grid = gpd.read_file("C:/Users/aladesuru/Documents/Empirical/NRW/Data/Grid.zip")
grid.crs
grid.head()
grid.plot()

# %% # Concatenate the Tschlag GeoDataFrames
TScombined = pd.concat([tschlag, tschlaghist], ignore_index=True)
TScombined.info()
TScombined.head()
TScombined.to_file("C://Users//aladesuru//Documents//Empirical//NRW//Data//TSCombined//TScombined.shp")
# %%
# Check for null values in the GeoDataFrame
print(TScombined.isnull().sum())
# %%
# add nuts columns to the dataframe
TScombined_nuts = gpd.sjoin(TScombined, nrw_nuts3, how="inner", op="intersects")
# %%
TScombined_nuts_grid = gpd.sjoin(TScombined_nuts, grid, how="inner", op="intersects")
# %% drop columns that aren't needed
#cols = ['Name', 'Ticket', 'Cabin']
#df = df.drop(cols, axis=1)
#drop rows with missing values
#df = df.dropna()
TScombined_nuts.info()

# %%
#summarize data using groupby
tschlag_by_USE_TXT=tschlag.groupby(["USE_TXT"])[["AREA_HA"]].describe()
tschlag_by_USE_TXT
# %%

# %%
### in tschlaghist





# %%




# %%

# %%
TScombined_kreis.head()
# %%

# %%
#drop columns that aren't needed
cols = ['INSPIRE_ID', 'CODE', 'CODE_TXT', 'USE_CODE', 'D_PG', 'CROPDIV', 'EFA', 'ELER', 'DAT_BEARB', 'perimeter_m', 'area_m2', 'index_right']
TScombined_kreis = TScombined_kreis.drop(cols, axis=1)

# %%
grid_de_nrw = grid[grid["CNTR_CODE"] == "DE", grid["NUTS_NAME"] == 3]
# %%
grid_de.head()
# %%
grid_de.plot()
# %%
grid_de.crs
# %%
TScombined_kreis.crs
# %%
# Reproject the NUTS data to match Tscombined
grid_de = grid_de.to_crs(TScombined_kreis.crs)
# %%
# Check the new geometry values
grid_de.head()
# %%
grid_de["NUTS_NAME"]
# %%
# print unique values in the NUTS_NAME column
print(grid_de["NUTS_NAME"].unique())
# %%
