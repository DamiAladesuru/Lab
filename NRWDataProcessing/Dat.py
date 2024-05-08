# %%
import os
import geopandas as gpd
import shapely as sh
import matplotlib
import matplotlib.pyplot as plt
import fiona
import pyogrio
# %%
grid = gpd.read_file("C:\\Users\\aladesuru\\Documents\\coding\\NRW\\Data\\Grid\grid.shp")
# %%
grid.boundary.plot()
# %%
data = gpd.read_file("C:\\Users\\aladesuru\\Documents\\coding\\NRW\\Data\\EP_Kreis\\Simplified\\EP_Kreis25012023.shp")
data.head()
# %%
Splot = data.plot()
# %%
data.crs
# %%
data.to_crs(epsg=4326).plot()
# %%
#subset
southern = data.cx[80:, :110] #.cx[xmin:xmax, ymin:ymax]
southern.boundary.plot()
# %%
data_dauer = data[data["USE_TXT"] == "Dauergrünland"]
# %%
data[data["WJ"] == 2019].plot(kind="bar", x="USE_TXT", y="AREA_HA", alpha=0.4)
# %%
data_dauer.describe()
# %%
data_dauer.plot()
# %%
