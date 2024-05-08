#
# ----- Adding additional spatial information to plot frequences over
#       years provided by Christoph
#                    
# - Conda environment 24_Spatial, using python 3.12.1

# %%
# - Load relevant packages

print(dir())
import geopandas as gpd
import pandas as pd
import numpy as np
import pyarrow

print(dir())

# %%
# - Load data with crops over assessed years
#

df = gpd.read_parquet('joined-plots-2015-2023_2.parquet')

# %%
# - Load data with district and commune information

data_com = gpd.read_file("L:\\Data_Commune\\dvg2gem_nw.shp")
data_dist = gpd.read_file("L:\\Data_District\\dvg2krs_nw.shp")

#%%
# - Create fitting Coordinate Reference Systems (CRS)

df = df.set_crs('epsg:25832')

######################################################
#%%
# - Perform a spatial join to add COMMUNE information to the data
#   based on the geometry of the data.

joined = gpd.sjoin(df, data_com,how='left', op="intersects")

#%%
# --- Create a sample with all double assigned plots from joined, which are 
#     crossing commune borders and, therefore, are assigned to more than one
#     commune.

double_assigned = joined[joined.index.isin(joined[joined.index.duplicated()].index)]

#%%
# - Delete all double assigned plots, i.d. plots that are assigned to more
#   than one commune

joined = joined[~joined.index.isin(joined[joined.index.duplicated()].index)]

#%%
# --- Estimate the largest intersection for each plot with a commune in the
#     double assigned sample. Use the unit of ha.

double_assigned['intersection'] = [
    a.intersection(data_com[data_com.index == b].\
      geometry.values[0]).area/10000 for a, b in zip(
       double_assigned.geometry.values, double_assigned.index_right
    )]

#%%
# --- Sort by intersection area and keep only the  row with the largest intersection.

double_assigned = double_assigned.sort_values(by='intersection').\
         groupby('ID').last().reset_index()

#%%
# --- Add the data double_assigned to the joined data

joined_com = pd.concat([joined,double_assigned])

#%%
# --- Only keep needed columns

joined_com = joined_com[['ID','Jahr',"Ha_beantragt","Nutzartcode_2023","Nutzartcode_2022",
                  "Nutzartcode_2021","Nutzartcode_2020","Nutzartcode_2019",
                  "Nutzartcode_2018","Nutzartcode_2017",
                  "KN","GN","geometry"]]

# ---- Rename GN to Gemeinde_name and KN to Gemeinde_ID

joined_com.rename(columns={"GN":"Gemeinde_Name","KN":"Gemeinde_ID"},inplace=True)

######################################################
#%%
# --- Perform a spatial join to add DISTRICT information to the data

# --- Perform the spatial join

joined2 = gpd.sjoin(df, data_dist,how='left', op="intersects")

#%%
# --- Create a sample with all double assigned plots from joined2, which are
#     crossing district borders and, therefore, are assigned to more than one
#     district.

double_assigned2 = joined2[joined2.index.isin(joined2[joined2.index.duplicated()]\
                      .index)]

#%%
# --- Delete all double assigned plots, i.d. plots that are assigned to more
#     than one district

joined2 = joined2[~joined2.index.isin(joined2[joined2.index.duplicated()].index)]

#%%
# --- Estimate the largest intersection for each plot with a district in the
#     double assigned sample. Use the unit of ha.

double_assigned2['intersection'] = [
    a.intersection(data_dist[data_dist.index == b].\
      geometry.values[0]).area/10000 for a, b in zip(
       double_assigned2.geometry.values, double_assigned2.index_right
    )]

#%%
# --- Sort by intersection area and keep only the  row with the largest intersection.

double_assigned2 = double_assigned2.sort_values(by='intersection').\
            groupby('ID').last().reset_index()

#%%
# --- Add the data double_assigned2 to the joined data

joined2 = pd.concat([joined2,double_assigned2])

#%%
# --- Only keep the columns ID and GN and KN

joined_dist = joined2[['ID',"KN","GN"]]

# --- Remande GN to Kreis_Name and KN to Kreis_ID

joined_dist.rename(columns={"GN":"Kreis_Name","KN":"Kreis_ID"},inplace=True)

#%%
# --- Merge the data with commune and district information

joined_complete = pd.merge(joined_com,joined_dist,how='left',on='ID')


# %%
# - Export data as geodataframe

joined_complete.to_parquet('joined-plots_2015-2023_2_inclAdminis.parquet')  

#%%
# - Export as csv without geometry

joined_complete.drop(columns=['geometry']).\
    to_csv('joined-plots_2015-2023_2_inclAdminis.csv')
