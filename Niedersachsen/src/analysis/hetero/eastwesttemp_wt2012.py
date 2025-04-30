'''2012'''
# %% 1. create gld12 for 2012 data
gld12 = gld[gld['year'] == 2012]

gld12['category4'] = category4(gld12, 'Gruppe')['category4']

# %% 2. create grid df for categories of change (a.Less than 0%, b.0 – 4%
#c. >4%) in number of fields in east and west
# Create a dictionary to hold the DataFrames for each category
abcgrids_dfs = {}

# Loop through each region and apply the filters
for region, df in eastwest_dfs.items():
    abcgrids_dfs[f"{region}grids_a"] = df[df['fields_ha_percdiff_to_y1'] < 0]
    abcgrids_dfs[f"{region}grids_b"] = df[(df['fields_ha_percdiff_to_y1'] >= 0) & (df['fields_ha_percdiff_to_y1'] <= 4)]
    abcgrids_dfs[f"{region}grids_c"] = df[df['fields_ha_percdiff_to_y1'] > 4]


# %% 3. extract the rows from gld12 whose CELLCODE is in the fields_a, fields_b and fields_c
# Extract rows where CELLCODE is in fields_a['CELLCODE']
# Dictionary to store the filtered gld12 subsets
gld12_subsets = {}

#loop using abcgrids_dfs to create gld12_<region><gridcategory>
for key, grids_df in abcgrids_dfs.items():
    # Split at 'grids_' to get region and group
    # Example key: 'westgrids_a' → extract 'west' and 'a'
    region, group = key.rsplit('grids_', 1) 
    
    subset_name = f"gld12_{region}{group}"  # e.g., gld12_westa
    gld12_subsets[subset_name] = gld12[gld12['CELLCODE'].isin(grids_df['CELLCODE'])]

# access:
# gld12_subsets['gld12_westa']


# %% 4. create a new DataFrame with the total area for each unique kulturart
gld12_subsets_bycrop = {}
# Loop through each subset DataFrame in gld12_subsets
for subset, df in gld12_subsets.items():
    # Group by kulturart and calculate the total area
    ksum_df = df.groupby(['CELLCODE', 'kulturart'], as_index=False).agg({
        'area_ha': 'sum',
        'Gruppe': 'first',     
        'category4': 'first'
    })
    # reset index for grouped_df
    ksum_df.reset_index(drop=True, inplace=True)
    # Sort the DataFrame by area_ha in descending order
    ksum_df.sort_values(by='area_ha', ascending=False, inplace=True)
    
    #group by kulturart, and calculate the sum of area_ha and count of CELLCODE
    grouped_df = ksum_df.groupby(['kulturart'], as_index=False).agg({
    'area_ha': 'sum',
    'CELLCODE': 'count',  # This gives count of rows per kulturart
    'Gruppe': 'first',
    'category4': 'first'
    })
    grouped_df.rename(columns={'CELLCODE': 'count'}, inplace=True)

    # Calculate the proportion of area of each kulturart in relation
    # to the total area of all kulturarten
    area_allcrops = grouped_df['area_ha'].sum()
   
    grouped_df['kulturart_prop'] = grouped_df['area_ha'] / area_allcrops * 100
    grouped_df['kulturart_prop'] = grouped_df['kulturart_prop'].round(2)
    
    # Sort and reset index
    grouped_df.sort_values(by='kulturart_prop', ascending=False, inplace=True)
    grouped_df.reset_index(drop=True, inplace=True)

    gld12_subsets_bycrop[f"{subset}_bycrop"] = grouped_df

# access:
# gld12_subsets_bycrop['gld12_westa_bycrop']


# %% 5. extract the top 10 kulturart from the grouped DataFrames and add an "others" row
gld12_subsets_bycrop_top10 = {}

for subset, df in gld12_subsets_bycrop.items():
    # Separate top 10
    top10 = df.iloc[:10].copy()

    # Summarize the rest as "others"
    others = df.iloc[10:]
    if not others.empty:
        others_row = {
            'kulturart': 'others',
            'area_ha': others['area_ha'].sum(),
            'Gruppe': None,
            'category4': None,
            'count': others['count'].sum(),
            'kulturart_prop': others['kulturart_prop'].sum()
        }
    top10 = pd.concat([top10, pd.DataFrame([others_row])], ignore_index=True)

    gld12_subsets_bycrop_top10[f"{subset}_top10"] = top10


# %% 6. plot bar plot for top 10 kulturart in each subset
import matplotlib.pyplot as plt

def plot_top10_kulturart(df, criterion_label, region=None, color='skyblue'):
    """
    Plots the top 10 kulturart from a preprocessed DataFrame with kulturart percentages.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'kulturart' and 'kulturart_prop'.
        criterion_label (str): Description of the group criteria (e.g., '< 0%', '0–4%', '> 4%').
        region (str, optional): Region name to include in the title.
        color (str): Bar color.
    """
    kulturart = df['kulturart']
    total_area = df['kulturart_prop']

    title = "Top 10 Crops"
    if region:
        title += f" in {region.capitalize()}"
    title += f" Cells with {criterion_label} Change in Number of Fields"

    plt.figure(figsize=(10, 6))
    plt.bar(kulturart, total_area, color=color)

    plt.xlabel('Kulturart', fontsize=12)
    plt.ylabel('Kulturart Proportion (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

# %%
a = "Negative"
b = "0 - 4%"
c = "> 4%"  


# %%
plot_top10_kulturart(
    gld12_subsets_bycrop_top10['gld12_eastc_bycrop_top10'],
    criterion_label=c,
    region="Eastern"
)
