'''obtain df containing the crop mostly cultivated
in each district of Lower Saxony, Germany'''

# %%
import geopandas as gpd
landkreise = gpd.read_file("N:/ds/data/Niedersachsen/verwaltungseinheiten/NDS_Landkreise.shp")
landkreise.info()
landkreise = landkreise.to_crs("EPSG:25832")
# %%
districts = landkreise.copy()
# %%
import pandas as pd
# Step 1: Create a DataFrame with unique LANDKREIS
unique_landkreis_df = gld[['LANDKREIS']].drop_duplicates()

# Step 2: Group gld by LANDKREIS and kulturart, and take the sum of values in the area_ha column
grouped_gld = gld.groupby(['LANDKREIS', 'kulturart'])['area_ha'].sum().reset_index()

# Step 3: Create a second DataFrame, sort total areas of landkreis' kulturart descending, and keep only the kulturart with the highest total area
sorted_gld = grouped_gld.sort_values(by=['LANDKREIS', 'area_ha'], ascending=[True, False])
df2 = sorted_gld.groupby('LANDKREIS').first().reset_index()

# Step 4: Extract Gruppe, category2, category3 columns from gld and merge with df2 on kulturart
category_columns = gld[['kulturart', 'Gruppe', 'category2', 'category3']].drop_duplicates()
df2 = pd.merge(df2, category_columns, on='kulturart', how='left')

# %%
# Translation dictionary
translation_cropdict = {
    'mähweiden': 'Pasture',
    'silomais': 'Silage maize',
    'winterweichweizen': 'Winter wheat'
}

# Add a new column 'crop' with the translated values
df2['crop'] = df2['kulturart'].map(translation_cropdict)
# %%
# Translation dictionary
translation_grpdict = {
    'dauergrünland': 'Grassland',
    'ackerfutter': 'Forage crops',
    'getreide': 'Cereals',
}

# Add a new column 'group' with the translated values
df2['group'] = df2['Gruppe'].map(translation_grpdict)

# %%
#merge districts with df2
districts = districts.merge(df2, on='LANDKREIS', how='left')
# %%
districts.head()
# %%
# find LANDKREIS with missing values
missing_values = districts[districts['crop'].isnull()]
missing_values['LANDKREIS'].unique()

# %%
#drop rows with missing values
districts = districts.dropna()

# %%
# drop Küstenmeer Region Weser-Ems and Küstenmeer Region Lüneburg
districts = districts[~districts['LANDKREIS'].isin(['Küstenmeer Region Weser-Ems', 'Küstenmeer Region Lüneburg'])]

# %%
# save districts to a shapefile
#districts.to_file("data/interim/districts/districts.shp")