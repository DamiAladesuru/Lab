# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %%
def process_dataframe(df):
    # Define a function to create bins and labels for a given column
    def create_bins_and_labels(df, column_name, labels):
        col_min = df[column_name].min()
        col_max = df[column_name].max()
        col_quantile_1 = df[column_name][df[column_name] > 0].quantile(1/3)
        col_quantile_2 = df[column_name][df[column_name] > 0].quantile(2/3)
        col_bins = [col_min, 0, col_quantile_1, col_quantile_2, col_max]
        df[f'{column_name}_group'] = pd.cut(df[column_name], bins=col_bins, labels=labels, include_lowest=True)

        print(f"Ranges for {column_name}:")
        for i in range(len(labels)):
            print(f"{labels[i]}: {col_bins[i]} to {col_bins[i+1]}")
        print()

    # Define labels for the bins
    mfs_labels = ['Decrease_MFS', 'S_Increase_MFS', 'M_Increase_MFS', 'H_Increase_MFS']
    fieldsha_labels = ['Decrease_F/ha', 'S_Increase_F/ha', 'M_Increase_F/ha', 'H_Increase_F/ha']
    lsi_labels = ['Decrease_LSI', 'S_Increase_LSI', 'M_Increase_LSI', 'H_Increase_LSI']
    grid_polspy_labels = ['Decrease_GPolspy', 'S_Increase_GPolspy', 'M_Increase_GPolspy', 'H_Increase_GPolspy']
    mean_cpar2_labels = ['Decrease_MCPAR', 'S_Increase_MCPAR', 'M_Increase_MCPAR', 'H_Increase_MCPAR']
    mean_polspy_labels = ['Decrease_MPolspy', 'S_Increase_MPolspy', 'M_Increase_MPolspy', 'H_Increase_MPolspy']

    # Create bins and labels for each column
    create_bins_and_labels(df, 'mfs_ha_diff_from_2012', mfs_labels)
    create_bins_and_labels(df, 'fields_ha_diff_from_2012', fieldsha_labels)
    create_bins_and_labels(df, 'lsi_diff_from_2012', lsi_labels)
    create_bins_and_labels(df, 'grid_polspy_diff_from_2012', grid_polspy_labels)
    create_bins_and_labels(df, 'mean_cpar2_diff_from_2012', mean_cpar2_labels)
    create_bins_and_labels(df, 'mean_polspy_diff_from_2012', mean_polspy_labels)

    # Combine the new grouped columns into a single column
    df['mfs_fields_ha_diff_group'] = griddf_ext['mfs_ha_diff_from_2012_group'].astype(str).str.cat(griddf_ext['fields_ha_diff_from_2012_group'].astype(str), sep='_')
    df['lsi_polspy_diff_group'] = df['lsi_diff_from_2012_group'].astype(str).str.cat(df['grid_polspy_diff_from_2012_group'].astype(str), sep=',')
    df['MCPAR_polspy_diff_group'] = df['mean_cpar2_diff_from_2012_group'].astype(str).str.cat(df['mean_polspy_diff_from_2012_group'].astype(str), sep=',')

    # Define the columns to clean
    columns_to_clean = ['lsi_polspy_diff_group', 'MCPAR_polspy_diff_group', 'mfs_fields_ha_diff_group']

    # Define the unwanted categories
    unwanted_categories = ['nan_nan', 'nan,nan']

    # Drop rows with unwanted categories in the specified columns
    for column in columns_to_clean:
        df = df[~df[column].isin(unwanted_categories)]

    # Reset index after dropping rows
    df.reset_index(drop=True, inplace=True)
    
    return df

# Usage
griddf_ext = process_dataframe(griddf_ext)

# %% Crostab and heatmap 
def plot_heatmap(dataframe, row_group, col_group, title, xlabel, ylabel, figsize=(10, 6), cmap='YlGnBu'):
    """
    Plots a heatmap for the given dataframe and specified row and column groups.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame containing the data.
    row_group (str): The column name to be used for the rows of the crosstab.
    col_group (str): The column name to be used for the columns of the crosstab.
    title (str): The title of the heatmap.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    figsize (tuple): The size of the figure (default is (10, 6)).
    cmap (str): The colormap to be used for the heatmap (default is 'YlGnBu').
    """
    # Create a crosstab of the two categorical columns
    crosstab = pd.crosstab(dataframe[row_group], dataframe[col_group])

    # Plot the crosstab using a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(crosstab, annot=True, fmt='d', cmap=cmap)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

# Usage
plot_heatmap(griddf_ext, 'mfs_fields_ha_diff_group', 'lsi_polspy_diff_group', 
             'Heatmap of Change in Size and Count vs Grid Shape Complexity', 
             'lsi_polspy_diff_group', 'mfs_fields_ha_diff_group')

plot_heatmap(griddf_ext, 'mfs_fields_ha_diff_group', 'MCPAR_polspy_diff_group', 
             'Heatmap of Change in Size and Count vs Mean Shape Complexity', 
             'MCPAR_polspy_diff_group', 'mfs_fields_ha_diff_group')



# %% Plot the crosstab using a stacked bar chart
crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart of Change in Size and Count vs Shape Complexity')
plt.xlabel('mfs_ha_diff_from_2012_bins')
plt.ylabel('Count')
plt.legend(title='mean_cpar2_group')
plt.show()