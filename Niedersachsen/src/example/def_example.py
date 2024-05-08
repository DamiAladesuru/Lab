# %%
import math as m
# %%
def greet ():
    print("Hello, World!")
    print("How are you today?")

   
# %%
greet() 



# %%
def custom_function(row, all_data):
    p = row['peri_m']
    a = row['area_m2']
    SI = p/(2*m.sqrt(m.pi*a))
    return SI

    log_value1 = np.log(row['value1'])  # Calculate the natural logarithm of value1
    sqrt_value2 = np.sqrt(row['value2'])  # Calculate the square root of value2
    return log_value1 * sqrt_value2  # Multiply the two results

# Apply the custom function to each DataFrame and create a new column with the result
df1['result'] = df1.apply(custom_function, args=(df1,), axis=1)
df2['result'] = df2.apply(custom_function, args=(df2,), axis=1)







# %%
def meanshapeindex():
    MSI = mean
    

# %%
def fractaldimension(a, p):
    FD = (2*m.log(p))/m.log(a)
    return FD

# %%
import pandas as pd
import numpy as np

# Define the number of years and observations
num_years = 3
num_observations = 5

# Create a list to store data for each year
data_years = []

# Generate sample data for each year
for year in range(1, num_years + 1):
    # Generate random sample data for the area of patches for each observation
    areas = np.random.uniform(low=50, high=500, size=num_observations)  # Area in square meters
    peri = np.random.uniform(low=10, high=100, size=num_observations)  # Perimeter in meters
    
    # Create a DataFrame for the current year
    df_year = pd.DataFrame({'Year': [year] * num_observations,
                            'Patch_Area_m2': areas, 'Perimeter_m': peri})
    
    # Append the DataFrame to the list
    data_years.append(df_year)

# Concatenate DataFrames for all years into a single DataFrame
df_all_years = pd.concat(data_years, ignore_index=True)

# %% Print the generated sample data
print(df_all_years)


# %% defining a function that calculates shape index and mean shape index
def shapeindex(a, p):
    SI = p/(2*m.sqrt(m.pi*a))
    return SI

df_all_years['shp'] = df_all_years.apply(lambda row: shapeindex(row['Patch_Area_m2'], row['Perimeter_m']), axis=1)

# %% Print the DataFrame with the calculated shape index
sfg = 72/ (2*m.sqrt(m.pi*188.77))
print(sfg)



# %%
df_all_years['mean'] = df_all_years['Patch_Area_m2'].mean()
# %%
df_all_years['areaweighted_mean2'] = df_all_years['Patch_Area_m2'] / df_all_years.groupby('Year')['Patch_Area_m2'].transform('sum')
# %%
df_all_years['Patch_Area_sumbyyear'] = df_all_years.groupby('Year')['Patch_Area_m2'].transform('sum')

# %%
def AreaWeightedMean(row):
    Patch_Area_sumbyyear = row.groupby('Year')['Patch_Area_m2'].transform('sum')
    return row['Patch_Area_m2'] / Patch_Area_sumbyyear

df_all_years['areaweighted_mean2'] = df_all_years.apply(AreaWeightedMean, axis=1)
print(df_all_years)
# %%
df_all_years['areaweighted_mean'] = df_all_years['Patch_Area_m2'] / df_all_years.groupby('Year')['Patch_Area_m2'].transform('sum')
# %%
