# %%
import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis import gridgdf_desc2 as gd

gld_trimmed, gridgdf = gd.create_gridgdf()

# gld_trimmed contains polygon level data with computed metrics for shape, kulturcode, kulturart,\
    # categories and has been trimmed to remove fields of less than 20 ha.

# %% ##########################################################
# trend of kulturcode for groups/ categories over the years
###########################################################
gld = gld_trimmed

# %%
def kulturcode_year_group(gld, groupby_column):
    # Sort, group, and get the first and last occurrences of 'kulturcode' in each year
    gld_sorted = gld.sort_values(by=['kulturcode', 'year'])
    first_occurrence = gld_sorted.groupby('kulturcode')['year'].min().reset_index(name='first_occurrence')
    last_occurrence = gld_sorted.groupby('kulturcode')['year'].max().reset_index(name='last_occurrence')

    # Merge first and last occurrences
    kulturcode_occurrences = pd.merge(first_occurrence, last_occurrence, on='kulturcode')
    
    # Create columns for each unique year marking if 'kulturcode' occurred
    occurrence_pivot = pd.crosstab(gld_sorted['kulturcode'], gld_sorted['year']).reset_index()
    kulturcode_occurrences = pd.merge(kulturcode_occurrences, occurrence_pivot, on='kulturcode')

    # Add column with a list of all years each 'kulturcode' occurred in
    year_occurrences = gld_sorted.groupby('kulturcode')['year'].apply(
        lambda x: list(sorted(x.unique()))
    ).reset_index(name='year_occurrences')

    # Merge year occurrences into the main DataFrame
    kulturcode_occurrences = pd.merge(kulturcode_occurrences, year_occurrences, on='kulturcode')

    # Select 'kulturcode' and groupby_column and merge with occurrences
    grouped_data = gld[['kulturcode', groupby_column]].drop_duplicates()
    merged_data = pd.merge(kulturcode_occurrences, grouped_data, on='kulturcode', how='left')

    # Reorder columns: move groupby_column to the 3rd position
    cols = merged_data.columns.tolist()
    cols.insert(2, cols.pop(cols.index(groupby_column)))
    merged_data = merged_data[cols]
    
    return merged_data


def calculate_group_stats(gld, groupby_column):
    # Group by 'year' and groupby_column and calculate the sum of 'area_ha'
    grouped = gld.groupby(['year', groupby_column])['area_ha'].sum().reset_index(name=f'{groupby_column}_area_sum')
    
    # Calculate the total sum of 'area_ha' for each year
    yearly_totals = gld.groupby('year')['area_ha'].sum().reset_index(name='total_area_sum')
    
    # Merge the yearly total sums back to the grouped data
    merged_data = pd.merge(grouped, yearly_totals, on='year')
    
    # Calculate the percentage of the group's sum relative to the yearly total
    merged_data['percentage_of_total'] = (merged_data[f'{groupby_column}_area_sum'] / merged_data['total_area_sum']) * 100
    
    # Compute the yearly difference in 'group_area_sum' for each group, sorting only by 'year'
    merged_data = merged_data.sort_values(by=['year'])
    merged_data['yearly_difference'] = merged_data.groupby(groupby_column)[f'{groupby_column}_area_sum'].diff()
    
    return merged_data


def barplot_kulturcode_data(data, groupby_column, values, ylab, title):  # for plotting trends
    # Pivot the Data
    pivot_df = data.pivot(index='year', columns=groupby_column, values=values).fillna(0)

    # Plot the Data
    pivot_df.plot(kind='bar', stacked=True, figsize=(10, 7))

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel(ylab)
    plt.title(title)
    # Move the legend to the right side
    plt.legend(title='Category2', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def lineplot_kulturcode_data(data, groupby_column, values, ylab, title):
    # Pivot the Data
    pivot_df = data.pivot(index='year', columns=groupby_column, values=values).fillna(0)

    # Plot the Data
    pivot_df.plot(kind='line', figsize=(10, 7))

    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel(ylab)
    plt.title(title)
    # Move the legend to the right side
    plt.legend(title='Category2', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def yearlyplot_data(data, groupby_column, values, ylab, title, plot_type='bar'):
    if plot_type == 'bar':
        barplot_kulturcode_data(data, groupby_column, values, ylab, title)
    elif plot_type == 'line':
        lineplot_kulturcode_data(data, groupby_column, values, ylab, title)
    else:
        raise ValueError("Invalid plot_type. Use 'bar' or 'line'.")


# Example execution
groupby_column = 'category2'
kc_occur_group = kulturcode_year_group(gld, groupby_column)
category2_areasum_stat = calculate_group_stats(gld, groupby_column)

# %%
# Plotting
ylab = f'{groupby_column} Area Sum'
title = f'Total Area For each {groupby_column.capitalize()} Across Years'
yearlyplot_data(category2_areasum_stat, groupby_column, f'{groupby_column}_area_sum', ylab, title, plot_type='line')

################################################################
# %%
# how many unique kulturcode constitute each category2 value for each year?
# Step 1: Group by year and category2 and count unique 'kulturcode' values
unique_kulturcode_count_by_year_category = gld.groupby(['year', 'category2'])['kulturcode'].nunique().reset_index()

# Step 2: Pivot the DataFrame to get a format suitable for plotting multiple lines
pivot_df = unique_kulturcode_count_by_year_category.pivot(index='year', columns='category2', values='kulturcode')

# Step 3: Create a multiline plot to show the trend of change in unique 'kulturcode' values over the years
plt.figure(figsize=(12, 8))
for category in pivot_df.columns:
    plt.plot(pivot_df.index, pivot_df[category], marker='o', linestyle='-', label=category)

plt.xlabel('Year')
plt.ylabel('Count of Unique kulturcode')
plt.title('Trend of Unique kulturcode Count by Year for Each category2')
plt.legend(title='category2')
plt.grid(True)
plt.show()  

# %%
# How many unique codes constitute gemuse for each year?
# Step 1: Filter the DataFrame to include only rows where 'Gruppe' is 'gemüse'
gld_gemuse = gld[gld['Gruppe'] == 'gemüse']

# Step 2: Group by year and count unique 'kulturcode' values
unique_kulturcode_count_by_year = gld_gemuse.groupby('year')['kulturcode'].nunique().reset_index()

# Step 3: Create a bar plot to show the count for each year
plt.figure(figsize=(10, 6))
plt.bar(unique_kulturcode_count_by_year['year'], unique_kulturcode_count_by_year['kulturcode'])
plt.xlabel('Year')
plt.ylabel('Count of Unique kulturcode')
plt.title('Count of Unique kulturcode for "gemüse" by Year')
plt.grid(True)
plt.show()


# %%
# group gld by 'year' and 'Gruppe' and calculate the sum of 'area_ha'
grouped = gld.groupby(['year', 'Gruppe'])['area_ha'].sum().reset_index(name='group_area_sum')
grouped['yearly_sum'] = grouped.groupby('year')['group_area_sum'].transform('sum')
grouped['percentage_of_total'] = (grouped['group_area_sum'] / grouped['yearly_sum']) * 100
# on averge across yeears, what is the percentage of each group
grouped['average_percentage'] = grouped.groupby('Gruppe')['percentage_of_total'].transform('mean')
grouped.to_csv('reports/Kulturcode/grouped.csv', index=False)



# %% save the unique kulturcode by year to an excel file
unique_kulturcode_by_year = gld_gemuse.groupby('year')['kulturcode'].unique().reset_index()
unique_kulturcode_by_year.to_excel('unique_kulturcode_by_year.xlsx', index=False)


