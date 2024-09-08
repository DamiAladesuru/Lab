# %%
import geopandas as gpd
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import pickle


os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')


# %% Load the data
data = pd.read_pickle('data/interim/gld.pkl')
data.info()

# %% load all sheets from excel file containing actual unique values of kulturcode in data
# Define the path to the excel file
file_path = "reports/Kulturcode/kulturcode.xlsx"
# Load the data from all sheets
kulturcode_act = pd.read_excel(file_path, sheet_name=None)
# Print the keys
print(kulturcode_act.keys())
# Print the info for each sheet
for key in kulturcode_act:
    print(f"{key}: {kulturcode_act[key].info()}")
    
# %% drop column 'Unnamed: 0' from kulturcode_act dictionary
for key in kulturcode_act:
    kulturcode_act[key] = kulturcode_act[key].drop(columns='Unnamed: 0')
    print(f"{key}: {kulturcode_act[key].info()}")
    
# %% Load the multiple spreadsheets containing kulturart description and group
# Define the path to the excel file
file_path = "N:/ds/data/Niedersachsen/Niedersachsen/kulturcode/kulturart_allyears.xlsx"
# Load the data from all sheets
kulturart = pd.read_excel(file_path, sheet_name=None)
# Print the keys
print(kulturart.keys())
# Print the info for each sheet
for key in kulturart:
    print(f"{key}: {kulturart[key].info()}")
    
# %% keep only the columns 'Code', 'Kulturart' and 'Gruppe' for each year and rename 'Code'
for key in kulturart:
    kulturart[key] = kulturart[key][['Code', 'Kulturart', 'Gruppe']]

for key in kulturart:
    kulturart[key] = kulturart[key].rename(columns={'Code': 'kulturcode'})
    print(f"{key}: {kulturart[key].info()}")

# %% Iterate over each year to check if all unique kulturcode values are numbers \
    # or if there are characters
for key in kulturart:
    # Check for non-numeric kulturcode values
    non_numeric_kulturcodes = [code for code in kulturart[key]['kulturcode'] if not str(code).replace('.', '', 1).isdigit()]
    if non_numeric_kulturcodes:
        print(f"{key}: Non-numeric kulturcode values found: {non_numeric_kulturcodes}")
    else:
        print(f"{key}: All kulturcode values are numeric.")

# %% Convert kulturcode_act and kulturart keys to numeric values
kulturcode_act = {int(key): value for key, value in kulturcode_act.items()}
print(kulturcode_act.keys())
kulturart = {int(key): value for key, value in kulturart.items()}
print(kulturart.keys())

# Check if the keys in kulturcode_act are in kulturart. if yes, put in a new dictionary
# %% Initialize an empty dictionary to store the filtered datasets
codedatawithart = {}

# Iterate through the keys in the kulturcode_act
for key in kulturcode_act:
    # Check if the key is in the kulturart dictionary
    if key in kulturart:
        # Add the dataset to the new dictionary
        codedatawithart[key] = kulturcode_act[key]

# check the info for each year in the available_kulturbez dictionary
for key in codedatawithart:
    print(f"{key}: {codedatawithart[key].info()}")
    
    
# %% for each key in the codedatawithart dictionary, merge data with kulturart data of same key \
    # on 'kulturcode' column
for key in codedatawithart:
    codedatawithart[key] = pd.merge(codedatawithart[key], kulturart[key], on='kulturcode', how='left')
    print(f"{key}: {codedatawithart[key].info()}")
    
# %% for each key in the codedatawithart dictionary, create a year column and set it to the key
for key in codedatawithart:
    codedatawithart[key]['year'] = key
    print(f"{key}: {codedatawithart[key].info()}")
 

# %% append all dataframes in the available_kulturbez dictionary to a single dataframe
kulturcode_map = pd.concat(codedatawithart.values(), ignore_index=True)
kulturcode_map.info()
##############################################################################
# %% take the value of column 'gruppe' for each kulturcode in year 2020 and use it to fill empty \
    # values in 'gruppe' column for the same kulturcode in year 2021 and 2022
# Extract the values for year 2020
kulturcodemap_2020 = kulturcode_map[kulturcode_map['year'] == 2020]

# Extract the values for year 2021
kulturcodemap_2021 = kulturcode_map[kulturcode_map['year'] == 2021]
kulturcodemap_2021 = kulturcodemap_2021.drop(columns='Gruppe')

# Extract the values for year 2022
kulturcodemap_2022 = kulturcode_map[kulturcode_map['year'] == 2022]
kulturcodemap_2022 = kulturcodemap_2022.drop(columns='Gruppe')

# Merge the dataframes
kulturcodemap_2021 = pd.merge(kulturcodemap_2021, kulturcodemap_2020[['kulturcode', 'Gruppe']], on='kulturcode', how='left')
kulturcodemap_2022 = pd.merge(kulturcodemap_2022, kulturcodemap_2020[['kulturcode', 'Gruppe']], on='kulturcode', how='left')

#infor kulturcodemap_2020, kulturcodemap_2021 and kulturcodemap_2022
print(kulturcodemap_2020.info())
print(kulturcodemap_2021.info())
print(kulturcodemap_2022.info())

# %% from the kulturcode_map dataframe, drop the rows with year 2021 and 2022
kulturcode_map = kulturcode_map[kulturcode_map['year'] != 2021]
kulturcode_map = kulturcode_map[kulturcode_map['year'] != 2022]
kulturcode_map.info()


# %% concatenate the kulturcodemap_2021 and kulturcodemap_2022 dataframes to the kulturcode_map dataframe
kulturcode_map = pd.concat([kulturcode_map, kulturcodemap_2021, kulturcodemap_2022], ignore_index=True)
kulturcode_map.info()
############################################################################################
# %% rename 'Kulturart' column to 'kulturart' in kulturcode_map dataframe
kulturcode_map = kulturcode_map.rename(columns={'Kulturart': 'kulturart'})

# %% save the kulturcode_map dataframe to a csv file
#kulturcode_map.to_csv('reports/Kulturcode/kulturcode_map.csv', index=False)

# %% check if there are rows with empty values in 'kulturart' column
kulturcode_map[kulturcode_map['kulturart'].isnull()]

# %%
missing_kulturart = kulturcode_map[kulturcode_map['kulturart'].isnull()]
missing_kulturart.info()

# %% save the missing_kulturart to a csv file
#missing_kulturart.to_csv('reports/Kulturcode/Fixingkulturart/missing_kulturart.csv', encoding='windows-1252', index=False)

# %%
kulturcode_map = kulturcode_map[~kulturcode_map['kulturart'].isnull()]
kulturcode_map.info()

# %%
# Merge missing_kulturart with kulturcode_map on kulturcode and kulturart, allowing for multiple matches
kau_df = missing_kulturart.merge(kulturcode_map, on=['kulturcode'], how = 'left', suffixes=('_missing_kulturart', '_kulturcode_map'))
kau_df['year_diff'] = kau_df['year_missing_kulturart'] - kau_df['year_kulturcode_map']
# Drop rows with NaN values
kau_df = kau_df.dropna(subset=['year_diff'])
kau_df.info()
#kau_df.to_csv('reports/Kulturcode/Fixingkulturart/missingkulturart_df.csv', encoding='windows-1252', index=False)

# Split the DataFrame into positive and negative year_diff
positive_year_diff_df = kau_df[kau_df['year_diff'] > 0]
negative_year_diff_df = kau_df[kau_df['year_diff'] <= 0]

# Find the nearest following year within 2 years for each kulturcode
nearest_following_df = positive_year_diff_df[positive_year_diff_df['year_diff'] <= 2].sort_values('year_diff').drop_duplicates(['kulturcode', 'year_missing_kulturart'])

# Find the closest previous year for each kulturcode if no nearest following year is found within 2 years
closest_previous_df = negative_year_diff_df.sort_values('year_diff', ascending=False).drop_duplicates(['kulturcode', 'year_missing_kulturart'])

# Combine the results
combined_df = pd.concat([nearest_following_df, closest_previous_df]).drop_duplicates(['kulturcode', 'year_missing_kulturart'], keep='first')

# Update the kulturart in missing_kulturart based on the nearest year found
updated_missing_kulturart = missing_kulturart.copy()
updated_missing_kulturart = updated_missing_kulturart.merge(combined_df[['kulturcode', 'year_missing_kulturart', 'kulturart_kulturcode_map']], 
                                                            left_on=['kulturcode', 'year'], 
                                                            right_on=['kulturcode', 'year_missing_kulturart'], 
                                                            how='left')

# Fill the kulturart with the found values
updated_missing_kulturart['kulturart'] = updated_missing_kulturart['kulturart_kulturcode_map'].combine_first(updated_missing_kulturart['kulturart'])

# Drop the helper columns
updated_missing_kulturart = updated_missing_kulturart.drop(columns=['year_missing_kulturart', 'kulturart_kulturcode_map'])

# Save the updated DataFrame to a CSV file
updated_missing_kulturart.to_csv('reports/Kulturcode/Fixingkulturart/updated_missing_kulturart.csv', encoding='windows-1252', index=False)
# kulturart with missing values have been updated with the nearest year found in the kulturcode_map dataframe

# rejoin the updated_missing_kulturart with kulturcode_map dataframe
kulturcode_map = pd.concat([kulturcode_map, updated_missing_kulturart], ignore_index=True)
kulturcode_map.info()

# %% drop row still with missing kulturart. iT is code '0' which is not a valid kulturcode
kulturcode_map = kulturcode_map[~kulturcode_map['kulturart'].isnull()]
kulturcode_map.info()

################################################################################

# Apply similar strategy as above to fill missing values in 'Gruppe' column #
# %%
missing_gruppe = kulturcode_map[kulturcode_map['Gruppe'].isnull()]
missing_gruppe.info()
#
kulturcode_map = kulturcode_map[~kulturcode_map['Gruppe'].isnull()]
kulturcode_map.info()

# Merge missing_gruppe with kulturcode_map on kulturcode and kulturart, allowing for multiple matches
merged_df = missing_gruppe.merge(kulturcode_map, on=['kulturcode', 'kulturart'], how='left', suffixes=('_missing_gruppe', '_kulturcode_map'))
merged_df['year_diff'] = merged_df['year_missing_gruppe'] - merged_df['year_kulturcode_map']
# Drop rows with NaN values
merged_df = merged_df.dropna(subset=['year_diff'])
merged_df.info()
#merged_df.to_csv('reports/Kulturcode/FixingGruppe/merged_df.csv', encoding='windows-1252', index=False)

# Function to retain the row with the smallest positive year_diff or the single row
def retain_row(group):
    if len(group) > 1:
        positive_year_diff = group[group['year_diff'] > 0]
        if not positive_year_diff.empty:
            return positive_year_diff.loc[positive_year_diff['year_diff'].idxmin()]
    return group.iloc[0]

# Group by 'kulturcode', 'kulturart', and 'year_missing_gruppe' and apply the function
closest_year_df = merged_df.groupby(['kulturcode', 'kulturart', 'year_missing_gruppe']).apply(retain_row).reset_index(drop=True)

# save the closest_year_df to a csv file to view
#closest_year_df.to_csv('reports/Kulturcode/FixingGruppe/closest_year_missing_gr.csv', index=False)

# Create a dictionary for easy lookup
closest_year_dict = closest_year_df.set_index(['kulturcode', 'kulturart', 'year_missing_gruppe'])['Gruppe_kulturcode_map'].to_dict()

# Function to update the Gruppe value in missing_gruppe based on the dictionary
def update_Gruppe(row):
    return closest_year_dict.get((row['kulturcode'], row['kulturart'], row['year']), None)

# Apply the function to update missing_gruppe
missing_gruppe['Gruppe'] = missing_gruppe.apply(update_Gruppe, axis=1)
# save the missing_gruppe to a csv file
#missing_gruppe.to_csv('reports/Kulturcode/FixingGruppe/updated_missing_gruppe.csv', encoding='windows-1252', index=False)

# rejoin the updated missing_gruppe with kulturcode_map dataframe
kulturcode_map = pd.concat([kulturcode_map, missing_gruppe], ignore_index=True)
kulturcode_map.info()
# save the kulturcode_map dataframe to a csv file with encoding 'windows-1252'
# kulturcode_map.to_csv('reports/Kulturcode/kulturcode_map.csv', encoding='windows-1252', index=False)

# %% extract rows still with missing values in 'Gruppe' column to df and csv file
missing_gruppe = kulturcode_map[kulturcode_map['Gruppe'].isnull()]
missing_gruppe.info()
#missing_gruppe.to_csv('reports/Kulturcode/FixingGruppe/finalmissing_gruppe.csv', encoding='windows-1252', index=False)

# drop the rows from kulturcode_map dataframe
kulturcode_map = kulturcode_map[~kulturcode_map['Gruppe'].isnull()]
kulturcode_map.info()

# %% Update 'Gruppe' based on 'kulturcode' values
missing_gruppe.loc[missing_gruppe['kulturcode'].isin([48, 49, 866, 722, 764]), 'Gruppe'] = 'Sonstige Flächen'
missing_gruppe.loc[missing_gruppe['kulturcode'].isin([513, 514, 669, 687]), 'Gruppe'] = 'Küchenkräuter/Heil-und Gewürzpflanzen'
print(missing_gruppe)
missing_gruppe.info()

# %% Rejoin the updated missing_gruppe with kulturcode_map dataframe
kulturcode_map = pd.concat([kulturcode_map, missing_gruppe], ignore_index=True)
kulturcode_map.info()

###########################################################################
# Adjust group names so it's consistent
# %% get a list of all unique values in the 'Gruppe' column
unique_gruppe = kulturcode_map['Gruppe'].unique()
print(unique_gruppe)

# %% to csv file with encoding 'windows-1252'
unique_gruppe_df = pd.DataFrame(unique_gruppe, columns=['Gruppe'])
#unique_gruppe_df.to_csv('reports/Kulturcode/unique_gruppe.csv', encoding='windows-1252', index=False)


# %%
# Define the mapping for Gruppe values
gruppe_mapping = {
    'AUKM/GoG': 'AUKM',
    'AUKM/ unproduktive Fläche': 'AUKM',
    'unproduktive Fläche': 'AUKM',
    'Ackerfutter/GoG': 'Ackerfutter',
    'Sonstige  Flächen': 'Sonstige Flächen',
    'Sonstige LF auf AL': 'Sonstige Flächen',
    'Sonstige Flächen': 'Sonstige Flächen',
    'Aufforstung': 'Stilllegung/Aufforstung',
    'Stilllegung/Aufforstung': 'Stilllegung/Aufforstung',
    'Aus der Erzeugung genommen': 'Aus der Produktion genommen',
    'Aus der Produktion/Erzeugung genommen': 'Aus der Produktion genommen',
    'Leg-Mischung': 'Leguminosen'
}
# %% Replace the values in the 'Gruppe' column
kulturcode_map['Gruppe'] = kulturcode_map['Gruppe'].replace(gruppe_mapping)

# %% 
# Filter rows where 'Gruppe' starts with 'Küchenkräuter'
filtered_df = kulturcode_map[kulturcode_map['Gruppe'].str.startswith('Küchenkräuter', na=False)]
# Save the filtered DataFrame to a CSV file
#filtered_df.to_csv('reports/Kulturcode/filtered_küchen.csv', encoding='windows-1252', index=False)
# Remove the rows of the filtered DataFrame from the kulturcode_map
kulturcode_map = kulturcode_map.drop(filtered_df.index)

# For all of these rows in filtered_Df, let Gruppe value be 'Küchenkräuter'
filtered_df['Gruppe'] = 'Kräuter'

# Add the filtered DataFrame back to the kulturcode_map
kulturcode_map = pd.concat([kulturcode_map, filtered_df], ignore_index=True)
kulturcode_map.info()

# %% get a list of all unique values in the 'Gruppe' column
unique_gruppe = kulturcode_map['Gruppe'].unique()
print(unique_gruppe)
# Save the final kulturcode_map and unique group dataframe to a csv file
#kulturcode_map.to_csv('reports/Kulturcode/kulturcode_map.csv', encoding='windows-1252', index=False)
unique_gruppe_df = pd.DataFrame(unique_gruppe, columns=['Gruppe'])
#unique_gruppe_df.to_csv('reports/Kulturcode/unique_gruppe_final.csv', encoding='windows-1252', index=False)

######################################################################
# Check name pattern in kulturcode map
# %% 
# Group by 'kulturcode'
grouped1 = kulturcode_map.groupby('kulturcode')

# List to store the results
results = []

# Iterate through each group
for name, group in grouped1:
    if group['kulturart'].nunique() == 1:
        results.append({
            'kulturcode': name,
            'status': 'same',
            'years': 'all'
        })
    else:
        same_kulturart_years = group.groupby('kulturart')['year'].apply(list)
        for kulturart, years in same_kulturart_years.items():
            results.append({
                'kulturcode': name,
                'status': 'different',
                'kulturart': kulturart,
                'years': ', '.join(map(str, years))
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
# Save the results to a CSV file
#results_df.to_csv('reports/Kulturcode/kulturart_comparison.csv', encoding='windows-1252', index=False)

## we see that some kultuart identified as different only have space difference in the text.
# we can correct this by removing the extra spaces in the text and then run the comparison code again
# %%
# Correct multiple spaces in the 'text_column'
kulturcode_map['kulturart'] = kulturcode_map['kulturart'].str.replace(r'\s+', ' ', regex=True)
kulturcode_map.to_csv('reports/Kulturcode/kulturcode_map.csv', encoding='windows-1252', index=False)


# %%
# Group by 'year'
grouped_by_year = kulturcode_map.groupby('year')
results = [] # List to store the results
unique_kulturcodes = set(kulturcode_map['kulturcode'].unique()) #unique kulturcode values
# Iterate through each group
for year, group in grouped_by_year:
    # Get the unique kulturcode values for the current year
    year_kulturcodes = set(group['kulturcode'].unique())
    
    # Check if all unique_kulturcode values are present in the current year's kulturcodes
    if unique_kulturcodes.issubset(year_kulturcodes):
        results.append({
            'year': year,
            'status': 'complete'
        })
    else:
        results.append({
            'year': year,
            'status': 'incomplete',
           
        })
# Convert results to DataFrame
results_df = pd.DataFrame(results)


# %% Extract one row of 'kulturcode' such that the 'kulturart' is of the most recent year
grouped = kulturcode_map.groupby('kulturcode').apply(lambda x: x.sort_values('year', ascending=False)).reset_index(drop=True)
grouped = grouped.drop_duplicates('kulturcode')# %% keep only the first row for each kulturcode
grouped.info()
grouped.head(15)
# rename year column to 'most_recent_year'
grouped = grouped.rename(columns={'year': 'kulturart_sourceyear'})
kulturcode_mastermap = grouped.copy()


# %%
# Define the categories that should be labeled as 'Environmental'
environmental_categories = [
    'Stilllegung/Aufforstung', 
    'Greening / Landschaftselemente', 
    'AUKM', 
    'Aus der Produktion genommen'
]

# Create the new column 'category1' based on the conditions
kulturcode_mastermap['category1'] = kulturcode_mastermap['Gruppe'].apply(
    lambda x: 'Environmental' if x in environmental_categories else 'Others'
)

# Display the first few rows to verify the new column
kulturcode_mastermap.head()
# Save to a csv file
# kulturcode_mastermap.to_csv('reports/Kulturcode/kulturcode_mastermap.csv', encoding='windows-1252', index=False)


