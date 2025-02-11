# %%
import os
import pandas as pd
os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

# %%
# Load the farm file into a DataFrame
farmdata = pd.read_excel("data/raw/farmdata/farmNieder_corre.xlsx")
farmdata.info()
farmdata.head()

# %%
# count unique years in the dataset
farmdata['year'].nunique()
farmdata['LANDKREIS'].nunique()

# %%
#check for missing values
farmdata.isnull().sum()

#%%
#filter df for rows with missing values
missing = farmdata[farmdata.isnull().any(axis=1)]
missing

# %%
#subset the data for where klasse is Ingesamt
farmdata_ge = farmdata[farmdata['klasse'] == 'Insgesamt']
Niedertot = farmdata_ge[farmdata_ge['LK'] == 0]

# drop rows where LK = 0
farmdata_ge = farmdata_ge[farmdata_ge['LK'] != 0]
farmdata_ge.head()

# %%
# Group by 'year' and calculate descriptive statistics for 'anzahl' and 'LF_ha' columns
print(farmdata_ge.groupby('year')[['anzahl', 'LF_ha']].sum())

# %%
# Convert column dtypes to appropriate types
farmdata_ge['LK'] = farmdata_ge['LK'].astype(object)

# %%
# create a new column 'LF_mean' which is 'LF_ha' divided by 'anzahl'
farmdata_ge['LF_mean'] = farmdata_ge['LF_ha'] / farmdata_ge['anzahl']
farmdata_ge.head()

# %%
from src.analysis.region import regiongdf_df as reg
fdg_ext = reg.calculate_yearlydiff(farmdata_ge)
''' examine the columns e.g., using the year diff columns'''

# %%
# drop columns LK, klasse, year_yearly_diff, year_yearly_percdiff  
fdg_ext = fdg_ext.drop(columns=['LK', 'klasse', 'year_yearly_diff', 'year_yearly_percdiff'])

# sort by year
fdg_ext = fdg_ext.sort_values(by='year').reset_index(drop=True)
fdg_ext.head()

# %%
# Create a list to store all DataFrames
all_dfs = [fdg_ext]

# Define the years to copy data for
years_to_copy = {2010: range(2011, 2013), 2016: range(2013, 2016), 2020: list(range(2017, 2020)) + list(range(2021, 2024))}

# Loop through the years and create copies
for base_year, copy_years in years_to_copy.items():
    base_data = fdg_ext[fdg_ext['year'] == base_year].copy()
    for year in copy_years:
        df_copy = base_data.copy()
        df_copy['year'] = year
        all_dfs.append(df_copy)

# Concatenate all DataFrames
farmdata_geext = pd.concat(all_dfs, ignore_index=True)

# Sort the DataFrame by year
farmdata_geext = farmdata_geext.sort_values(by='year').reset_index(drop=True)

# %%
farmdata_geext = farmdata_geext[~farmdata_geext["LANDKREIS"].isin(["Stadt Oldenburg (Oldb) (kreisfrei)"])]

####################################################################################
# %% drop columns in gridgdf_cl
# Specify the columns to keep
columns_to_keep = [
    'geometry', 'LANDKREIS', 'year', 'mfs_ha', 'fsha_sum', 'fields', 'CELLCODE', 'medfs_ha',
    'fields_yearly_percdiff', 'fields_yearly_diff', 'medfs_ha_yearly_diff', 'medfs_ha_yearly_percdiff',
    'fsha_sum_yearly_diff', 'fsha_sum_yearly_percdiff', 'mfs_ha_yearly_diff', 'mfs_ha_yearly_percdiff'
]

# Drop all other columns
field_grids = gridgdf_cl[columns_to_keep]

field_grids = field_grids[~field_grids["LANDKREIS"].isin(["Küstenmeer Region Lüneburg", "Küstenmeer Region Weser-Ems"])]


# %%
df_farm = farmdata_geext.copy()
df_field = field_grids.copy()

# %%
# Standardize 'LANDKREIS' column in both DataFrames
df_field["LANDKREIS"] = df_field["LANDKREIS"].str.strip()
df_farm["LANDKREIS"] = df_farm["LANDKREIS"].str.strip()

# %%
# Display unique values after cleaning
print("Unique values in 'LANDKREIS' (df_field):", df_field["LANDKREIS"].unique())
print("Unique values in 'LANDKREIS' (df_farm):", df_farm["LANDKREIS"].unique())

# %%
df_farm["LANDKREIS"] = df_farm["LANDKREIS"].replace("Hannover,Region", "Region Hannover")

# %%
# Get unique LANDKREIS values from both DataFrames
farm_landkreise = set(df_farm["LANDKREIS"].unique())
field_landkreise = set(df_field["LANDKREIS"].unique())

# Check if they are the same
if farm_landkreise == field_landkreise:
    print("✅ LANDKREIS values in df_farm and df_field match exactly!")
else:
    print("❌ There are differences in LANDKREIS values between the two DataFrames.")
    
    # Find mismatched values
    only_in_farm = farm_landkreise - field_landkreise
    only_in_field = field_landkreise - farm_landkreise
    
    print(f"⚠️ Present in df_farm but missing in df_field: {only_in_farm}")
    print(f"⚠️ Present in df_field but missing in df_farm: {only_in_field}")

# %%
# drop rows for years 2010 and 2011 in df_farm
df_farm = df_farm[df_farm["year"] > 2011]

# drop unnecessary columns
df_farm = df_farm.drop(columns=['LK', 'klasse', 'year_yearly_diff', 'year_yearly_percdiff'])


#%%
# merge df_farm and df_field using the LANDKREIS and year columns, inner join
farm_field = df_farm.merge(df_field, on=["LANDKREIS", "year"])
farm_field.head()
farm_fielddf = pd.DataFrame(farm_field)

# Check for missing values
farm_field.isnull().sum()

# %%
# save farm_field to pickle
farm_field.to_pickle("data/interim/farm_field.pkl")

####################################################################################
# %% simple correlation tests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
selected_columns = farm_field[['anzahl_yearly_percdiff', 'LF_mean_yearly_percdiff', 'fields_yearly_percdiff', 'medfs_ha_yearly_percdiff', 'mfs_ha_yearly_percdiff']]
selected_columns.columns = ['Δ total farms', 'Δ mean farm size', 'Δ total fields', 'Δ median field size', 'Δ mean field size']

# Calculate the correlation matrix
correlation_matrix = selected_columns.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Yearly Percentage Differences')
plt.show()







# %%
import geopandas as gpd
# Load Landkreis file for regional boundaries
base_dir = "N:/ds/data/Niedersachsen/verwaltungseinheiten"

landkreise = gpd.read_file(os.path.join(base_dir, "NDS_Landkreise.shp"))
#landkreise.info()
# %%
landkreise_df = pd.DataFrame(landkreise)
# %%
del year
# %%
