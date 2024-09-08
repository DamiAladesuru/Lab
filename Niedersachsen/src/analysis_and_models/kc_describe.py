# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

# Set the current working directory
os.chdir('C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen')

from src.analysis_and_models import describe_subsets as ds

gld, category2_subsets, yearly_desc_dict, griddfs, griddfs_ext, grid_year_average, landkreis_average, gridgdf = ds.process_descriptives()

# %%
def kulturcode_years(gld):
    #Sort by 'kulturcode' and 'year'
    gld_sorted = gld.sort_values(by=['kulturcode', 'year'])
    
    #Group by 'kulturcode' and find the first (min) and last (max) occurrence of each 'kulturcode'
    first_occurrence = gld_sorted.groupby('kulturcode')['year'].min().reset_index(name='first_occurrence')
    last_occurrence = gld_sorted.groupby('kulturcode')['year'].max().reset_index(name='last_occurrence')
    
    #Create a DataFrame with unique 'kulturcode', first occurrence, and last occurrence
    kulturcode_occurrences = pd.merge(first_occurrence, last_occurrence, on='kulturcode')
    
    #Create additional columns for each unique year in the dataset
    unique_years = sorted(gld['year'].unique())
    
    #Add a column for each year and mark whether or not 'kulturcode' occurred in that year
    for year in unique_years:
        kulturcode_occurrences[year] = kulturcode_occurrences['kulturcode'].apply(
            lambda x: int(year in gld_sorted[gld_sorted['kulturcode'] == x]['year'].values)
        )

    #Add a column with a list of all the years in which each 'kulturcode' occurred
    year_occurrences = gld_sorted.groupby('kulturcode')['year'].apply(lambda x: list(sorted(x.unique()))).reset_index(name='year_occurrences')
    
    # Merge the year_occurrences column into the final DataFrame
    kulturcode_occurrences = pd.merge(kulturcode_occurrences, year_occurrences, on='kulturcode')
    
    return kulturcode_occurrences

    
def kulturcode_year_group(gld, group_by)
    #Select only the 'kulturcode' and group_by columns from gld
    grouped_data = gld[['kulturcode', group_by]].drop_duplicates()
    
    #get kulturecode_years
    kulturcode_occurrences = kulturcode_years(gld)

    #Perform a left merge of the 'kulturcode_occurrences' with the grouped data
    merged_data = pd.merge(kulturcode_occurrences, grouped_data, on='kulturcode', how='left')
    
    #Reorder columns to move group_by to the 3rd position
    cols = merged_data.columns.tolist()  # Get a list of column names
    cols.insert(2, cols.pop(cols.index(group_by)))  # Move group_by to position 2 (3rd column)
    
    # Apply the new column order
    merged_data = merged_data[cols]
    
    return merged_data



def calculate_group_stats(gld, group):
    #Group by 'year' and group and calculate the sum of 'area_ha'
    grouped = gld.groupby(['year', group])['area_ha'].sum().reset_index(name='group_area_sum')
    
    #Calculate the total sum of 'area_ha' for each year
    yearly_totals = gld.groupby('year')['area_ha'].sum().reset_index(name='total_area_sum')
    
    #Merge the yearly total sums back to the grouped data
    merged_data = pd.merge(grouped, yearly_totals, on='year')
    
    # Step 4: Calculate the percentage of the group's sum relative to the yearly total
    merged_data['percentage_of_total'] = (merged_data['group_area_sum'] / merged_data['total_area_sum']) * 100
    
    #Compute the yearly difference in 'group_area_sum' for each group
    datasorted = merged_data.sort_values(by=['year', 'percentage_of_total'], ascending=[True, False])
    datasorted['yearly_difference'] = datasorted.groupby(group)['group_area_sum'].diff()
    
    return merged_data

def barplot_kulturcode_data(data, columns, values, ylab, title):
    # Pivot the Data
    pivot_df = data.ppivot(index='year', columns=columns, values=values)

    # Plot the Data
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7))

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel(ylab)
    plt.title(title)
    # Move the legend to the right side
    plt.legend(title='Gruppe', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()

def lineplot_kulturcode_data(data, columns, values, ylab, title):
    # Pivot the Data
    pivot_df = data.pivot(index='year', columns=columns, values=values)

    # Plot the Data
    pivot_df.plot(kind='line', figsize=(10, 7))

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel(ylab)
    plt.title(title)
    # Move the legend to the right side
    plt.legend(title='Gruppe', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()

def yearlyplot_data(data, columns, values, ylab, title, plot_type='bar'):
    if plot_type == 'bar':
        barplot_kulturcode_data(data, columns, values, ylab, title)
    elif plot_type == 'line':
        lineplot_kulturcode_data(data, columns, values, ylab, title)
    else:
        raise ValueError("Invalid plot_type. Use 'bar' or 'line'.")
    
    
# %%
# Example usage

kc_occur_group = kulturcode_year_group(gld, 'category2')
category2_areasum_stat = calculate_group_stats(gld, 'category2')

ylab = 'Group Area Sum'
title = 'Stacked Bar Chart of Percentage of Total by Gruppe Across Years'
yearlyplot_data(category2_areasum_stat, 'category2', 'group_area_sum', ylab, title, plot_type='bar')  # or 'line' for line plot

# %% Obtain the first year for each DataFrame in griddfs
first_years = {key: df['year'].min() for key, df in griddfs.items()}
print("First year for each DataFrame in griddfs:", first_years)
# %%
