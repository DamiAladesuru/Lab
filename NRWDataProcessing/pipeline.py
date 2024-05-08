#%%
import os
import pandas as pd 
#%%
os.chdir("c://Code")
# %%
my_data = pd.read_csv("https://raw.githubusercontent.com/GeostatsGuy/GeoDataSets/master/unconv_MV.csv")
# %%
my_data[:7]   
# %%
my_data = my_data.iloc[:,1:8]                               # copy all rows and columns 1 through 8, note 0 column is removed
my_data.describe().transpose() 
# %%
