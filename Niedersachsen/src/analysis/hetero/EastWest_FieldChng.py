'''30/04/2025'''
#%%
# 1. create gld23
gld23 = gld[gld['year'] == 2023]

def category4(df, column_name):
    # categorization based on similar use of crops in reality
    environmental_categories = [
        'stilllegung/aufforstung', 
        'greening / landschaftselemente', 
        'aukm', 
        'aus der produktion genommen'
    ]
    other_arable = [
        'gemüse', 'leguminosen', 'eiweißpflanzen', 'hackfrüchte', 'ölsaaten', 'kräuter'
    ]
    others = [
        'sonstige flächen', 'andere handelsgewächse', 'zierpflanzen', 'mischkultur', 'energiepflanzen'
    ]

    # Create column 'category2'
    df['category4'] = df[column_name].apply(
        lambda x: 'environmental' if x in environmental_categories \
            else 'other_arable' if x in other_arable \
            else 'others' if x in others \
            else x
    )
    
    return df
gld23['category4'] = category4(gld23, 'Gruppe')['category4']

# %% 2. 
# Create east23, west23 and categories dataframes of change in number of fields
# which will always be used for conditioning
def create_eastwest_dfs(gridgdf_cluster):
    west23 = gridgdf_cluster[(gridgdf_cluster['year'] == 2023) & (gridgdf_cluster['eastwest'] == 1)]
    east23 = gridgdf_cluster[(gridgdf_cluster['year'] == 2023) & (gridgdf_cluster['eastwest'] == 0)]
    return {'west': west23, 'east': east23}

eastwest_dfs = create_eastwest_dfs(gridgdf_cluster)

# categories of change (a.Less than 0%, b.0 – 4% #c. >4%)
def create_abcgrids_dfs(eastwest_dfs):
    abcgrids_dfs = {}
    for region, df in eastwest_dfs.items():
        abcgrids_dfs[f"{region}grids_a"] = df[df['fields_ha_percdiff_to_y1'] < 0]
        abcgrids_dfs[f"{region}grids_b"] = df[(df['fields_ha_percdiff_to_y1'] >= 0) & (df['fields_ha_percdiff_to_y1'] <= 4)]
        abcgrids_dfs[f"{region}grids_c"] = df[df['fields_ha_percdiff_to_y1'] > 4]
    return abcgrids_dfs

abcgrids_dfs = create_abcgrids_dfs(eastwest_dfs)

# %% 3. extract the rows from gld23 whose CELLCODE is in the fields_a, fields_b and fields_c
def create_gld_subsets(abcgrids_dfs, gld, name='gld23'):
    subsets = {}
    for key, grids_df in abcgrids_dfs.items():
        region, group = key.rsplit('grids_', 1)
        subset_name = f"{name}_{region}{group}"  # dynamic name like gld23_westa
        subsets[subset_name] = gld[gld['CELLCODE'].isin(grids_df['CELLCODE'])]
    return subsets

# Usage
gld23_subsets = create_gld_subsets(abcgrids_dfs, gld23, name='gld23')

# access:
# gld23_subsets['gld23_westa']

# %% 4. create a new DataFrame with the total area for each unique kulturart
def summarize_by_kulturart(subsets_dict):
    grouped_dict = {}
    
    for subset_name, df in subsets_dict.items():
        # Sum area per CELLCODE and kulturart
        ksum_df = df.groupby(['CELLCODE', 'kulturart'], as_index=False).agg({
            'area_ha': 'sum',
            'Gruppe': 'first',     
            'category4': 'first'
        }).reset_index(drop=True)

        ksum_df.sort_values(by='area_ha', ascending=False, inplace=True)

        # Summarize by kulturart
        grouped_df = ksum_df.groupby('kulturart', as_index=False).agg({
            'area_ha': 'sum',
            'CELLCODE': 'count',
            'Gruppe': 'first',
            'category4': 'first'
        }).rename(columns={'CELLCODE': 'count'})

        # Proportion calculations
        total_area = grouped_df['area_ha'].sum()
        grouped_df['kulturart_prop'] = (grouped_df['area_ha'] / total_area * 100).round(2)

        grouped_df.sort_values(by='kulturart_prop', ascending=False, inplace=True)
        grouped_df.reset_index(drop=True, inplace=True)

        # Save with _bycrop suffix
        grouped_dict[f"{subset_name}_bycrop"] = grouped_df

    return grouped_dict
gld23_subsets_bycrop = summarize_by_kulturart(gld23_subsets)


# %% 5. extract the top 10 kulturart from the grouped DataFrames and add an "others" row
def extract_top10_kulturart(subsets_bycrop_dict, value_column='kulturart_prop', top_n=10):
    top10_dict = {}

    for subset_name, df in subsets_bycrop_dict.items():
        top_df = df.nlargest(top_n, value_column).copy()
        others_df = df.drop(top_df.index)

        if not others_df.empty:
            others_row = {
                'kulturart': 'others',
                'area_ha': others_df['area_ha'].sum(),
                'Gruppe': None,
                'category4': None,
                'count': others_df['count'].sum() if 'count' in others_df else None,
                'kulturart_prop': others_df['kulturart_prop'].sum()
            }
            top_df = pd.concat([top_df, pd.DataFrame([others_row])], ignore_index=True)

        top10_dict[f"{subset_name}_top10"] = top_df

    return top10_dict
gld23_subsets_bycrop_top10 = extract_top10_kulturart(gld23_subsets_bycrop)

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
    gld23_subsets_bycrop_top10['gld23_easta_bycrop_top10'],
    criterion_label=a,
    region="Eastern"
)


  
# %%
# 1. create gld12
gld12 = gld[gld['year'] == 2012]
gld12['category4'] = category4(gld12, 'Gruppe')['category4']

# 2. Use the same eastwest_dfs from gld23

# 3. Extract rows from gld12 based on CELLCODE
gld12_subsets = create_gld_subsets(abcgrids_dfs, gld12, name='gld12')

# 4.
gld12_subsets_bycrop = summarize_by_kulturart(gld12_subsets)

# 5. Extract top 10 kulturart and add "others" row
gld12_subsets_bycrop_top10 = extract_top10_kulturart(gld12_subsets_bycrop)

# 6. Plot
plot_top10_kulturart(
    gld12_subsets_bycrop_top10['gld12_easta_bycrop_top10'],
    criterion_label=a,
    region="Eastern"
)


###############################
# Merge, compute difference, make heatmap
#################################
# %% Merge kulturart_prop from 2012 into 2023
gld23_subsets_bycrop_merged = {}

for key in gld23_subsets_bycrop:
    # Match key in gld12 dict (assumes same naming convention like 'gld23_westa_bycrop' → 'gld12_westa_bycrop')
    gld12_key = key.replace('gld23', 'gld12')

    if gld12_key in gld12_subsets_bycrop:
        df23 = gld23_subsets_bycrop[key]
        df12 = gld12_subsets_bycrop[gld12_key]

        # Ensure kulturart_prop exists in 2012 df
        if 'kulturart_prop' in df12.columns:
            merged_df = df23.merge(
                df12[['kulturart', 'kulturart_prop']],
                on='kulturart',
                how='left',
                suffixes=('', '_2012')
            )
            gld23_subsets_bycrop_merged[key] = merged_df
        else:
            print(f"Warning: 'kulturart_prop' not found in {gld12_key}")
    else:
        print(f"Warning: Matching 2012 subset not found for {key}")

gld23_subsets_bycrop_merged['gld23_eastc_bycrop'].info()

# %% compute the difference between 2012 and 2023 values
# Loop through merged DataFrames and calculate the difference
for key, df in gld23_subsets_bycrop_merged.items():
    if 'kulturart_prop' in df.columns and 'kulturart_prop_2012' in df.columns:
        # Fill NaN values with 0
        df['kulturart_prop'] = df['kulturart_prop'].fillna(0)
        df['kulturart_prop_2012'] = df['kulturart_prop_2012'].fillna(0)

        # Calculate the difference
        df['kulturartprop_diff2312'] = df['kulturart_prop'] - df['kulturart_prop_2012']
    else:
        print(f"Warning: Missing kulturart_prop columns in {key}")

diff_columns_combined = {}

for key, df in gld23_subsets_bycrop_merged.items():
    # Extract subset name like 'easta', 'westb', etc. from 'gld23_easta_bycrop'
    subset_label = key.replace('gld23_', '').replace('_bycrop', '')
    
    # Set 'kulturart' as index and extract the diff column
    if 'kulturartprop_diff2312' in df.columns:
        diff_series = df.set_index('kulturart')['kulturartprop_diff2312']
        diff_columns_combined[subset_label] = diff_series
    else:
        print(f"Skipping {key}: 'kulturartprop_diff2312' column missing")

# Combine all into one DataFrame (outer join keeps all kulturart values)
diff_summary_df = pd.concat(diff_columns_combined, axis=1).fillna(0)

# Example preview
diff_summary_df.head()

# %% heatmap of how each kulturart changed across regions.
plt.figure(figsize=(12, 8))
sns.heatmap(diff_summary_df, cmap='coolwarm', center=0, annot=False, linewidths=0.5)

plt.title("Change in Kulturart Proportion (2023 vs 2012)", fontsize=14)
plt.xlabel("Region Group")
plt.ylabel("Kulturart")
plt.tight_layout()
plt.show()

# %% heatmap for the top 10 kulturart with the largest total change across all regions
# Step 1: Get top 10 kulturart by total absolute change across all regions
top10_kulturart = diff_summary_df.abs().sum(axis=1).sort_values(ascending=False).head(10).index

# Step 2: Subset the DataFrame to only those kulturart
top10_diff_df = diff_summary_df.loc[top10_kulturart]

# Step 3: Plot heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(top10_diff_df, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)

plt.title("Top 10 Kulturarten with Largest Proportion Change (2023 vs 2012)", fontsize=14)
plt.xlabel("Region Group")
plt.ylabel("Kulturart")
plt.tight_layout()
plt.show()

# %%
