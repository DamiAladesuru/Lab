# %%
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from joypy import joyplot
import seaborn as sns
import geoplot as gplt
import matplotlib.pyplot as plt

# %%
# Global dictionary to store colors for each label after first plot
label_color_dict = {}

'''Plotting functions'''

#############################################
# multiline plots for all data and color initialization
#############################################
def initialize_plotting(df, title, ylabel, metrics, color_dict_path):
    global label_color_dict

    if os.path.exists(color_dict_path):
        with open(color_dict_path, 'rb') as f:
            label_color_dict = pickle.load(f)
    else:
        label_color_dict = {}

    multimetric_plot(df, title, ylabel, metrics)
    
    with open(color_dict_path, 'wb') as f:
        pickle.dump(label_color_dict, f)


def multimetric_plot(df, title, ylabel, metrics):
    global label_color_dict
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for label, column in metrics.items():
        if label in label_color_dict:
            color = label_color_dict[label]
            sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
        else:
            line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
            color = line.get_lines()[-1].get_color()
            label_color_dict[label] = color

    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.legend(title='Metrics')
    plt.show()
# %%
#############################################
# Correlation
#############################################
# 1. Correlation test
def test_correlation(df, target_columns, new_column_names):
    # Calculate correlation matrix
    correlation_matrix = df[target_columns].corr()
    
    # Rename the columns and index of the correlation matrix
    correlation_matrix.columns = new_column_names
    correlation_matrix.index = new_column_names
    
    return correlation_matrix

# 2. Single matrix plot
def plot_correlation_matrix(df, title, target_columns, new_column_names):
    # Get the correlation matrix
    corr_matrix = test_correlation(df, target_columns, new_column_names)
    
    # Create the plot
    plt.figure(figsize=(8, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8})
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
# 3. Correlation matrix for metrics with and without outlier
''' requires both data with and without outlier to be loaded'''

def plot_correlation_matrices(df1, df2, title1, title2):
    # Get the correlation matrices
    corr_matrix1 = test_correlation(df1)
    corr_matrix2 = test_correlation(df2)
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first correlation matrix
    sns.heatmap(corr_matrix1, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[0])
    axes[0].set_title(title1)
    
    # Plot the second correlation matrix
    sns.heatmap(corr_matrix2, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                annot_kws={"size": 8}, ax=axes[1])
    axes[1].set_title(title2)
    
    plt.tight_layout()
    plt.show()

############################################################
# multimetric extended for facet plot for subsamples of data
############################################################
def multimetric_ss_plot(dict, title, ylabel, metrics):
    global label_color_dict  # Access the global color dictionary
    
    # Set the plot style
    sns.set(style="whitegrid")
    
    # Determine the number of subplots based on the number of Gruppe values
    n_subplots = len(dict)
    n_cols = min(3, n_subplots)  # Maximum 3 columns
    n_rows = (n_subplots - 1) // n_cols + 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows), squeeze=False)
    fig.suptitle(title, fontsize=16)
    
    for idx, (gruppe, df) in enumerate(dict.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # Plot each metric for this Gruppe
        for label, column in metrics.items():
            if label in label_color_dict:
                color = label_color_dict[label]
                sns.lineplot(data=df, x='year', y=column, label=label, marker='o', color=color, ax=ax)
            else:
                line = sns.lineplot(data=df, x='year', y=column, label=label, marker='o', ax=ax)
                color = line.get_lines()[-1].get_color()
                label_color_dict[label] = color
        
        ax.set_title(f'Gruppe: {gruppe}')
        ax.set_xlabel('Year')
        ax.set_ylabel(ylabel)
        ax.legend(title='Metrics', loc='best')
    
    # Remove any unused subplots
    for idx in range(n_subplots, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    plt.show()


############################################
# Function to stack plots in a grid
############################################
# %%  
def stack_plots_in_grid(df, unique_values, plot_func, col1, col2, ncols=3, figsize=(18, 12), grid_title=None):
    nrows = (len(unique_values) + ncols - 1) // ncols  # Calculate number of rows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)  # Create grid
    axes = axes.flatten()  # Flatten axes for easy iteration

    for i, value in enumerate(unique_values):
        ax = axes[i]
        plot_func(df, value, col1, col2, ax)  # Call the user-provided function to generate a plot
        ax.set_title(f"Year: {value}", fontsize=16)
    
    # Remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add a suptitle for the entire grid
    if grid_title:
        fig.suptitle(grid_title, fontsize=20, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

# scatterplot of PAR and area for a given year
def scatterplot_par_area(df, year, col1, col2, ax):
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create scatterplot
    sns.scatterplot(
        data=df_year,
        x=col1,
        y=col2,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel(col1, fontsize=14)
    ax.set_ylabel(col2, fontsize=14)

#unique_years = sorted(gld['year'].unique())
#stack_plots_in_grid(gld, unique_years, scatterplot_par_area, "area_ha", "par", ncols=4, figsize=(25, 15))

# scatterplot of grid average PAR and area for a given year
def scatterplot_mpar_marea(df, year, col1, col2, ax):
    # Subset the DataFrame for the current year
    df_year = df[df['year'] == year]
    
    # Create scatterplot
    sns.scatterplot(
        data=df_year,
        x=col1,
        y=col2,
        ax=ax
    )
    
    # Customize the plot
    ax.set_xlabel(col1, fontsize=14)
    ax.set_ylabel(col2, fontsize=14)


############################################
# Function to create a joyplot for each year
############################################
# %%
def create_yearly_joyplot(df, by_column, plot_column, title_template):
    """
    Create a joyplot for each year in the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - by_column: Column name to group by (e.g., 'Gruppe').
    - plot_column: Column name to plot (e.g., 'area_ha').
    - title_template: Template for the plot title (e.g., "Area distribution in {year}").
    """
    unique_years = df['year'].unique()
    
    for year in unique_years:
        # Subset the DataFrame for the current year
        df_year = df[df['year'] == year]
        
        # Create labels for the current year
        labels = [y for y in list(df_year[by_column].unique())]
        
        # Create the joyplot for the current year
        fig, axes = joyplot(
            df_year, 
            by=by_column, 
            column=plot_column, 
            labels=labels, 
            range_style='own', 
            linewidth=1, 
            legend=True, 
            figsize=(6, 5),
            title=title_template.format(year=year),
            colormap=cm.autumn
        )
    
    plt.show()

# call
#create_yearly_joyplot(gld_no4, 'Gruppe', 'par', "PAR distribution in {year}")

########################
# choropleth map geoplot
########################
# %% facet grid of chloropleth maps: sequential
def plot_facet_choropleth_with_geoplot(gdf, column, cmap='viridis', year_col='year', ncols=4, title=""):
    # Get unique years
    unique_years = sorted(gdf[year_col].unique())
    nrows = (len(unique_years) + ncols - 1) // ncols  # Calculate rows based on ncols

    # Create the figure and axes
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), 
        subplot_kw={'projection': gplt.crs.AlbersEqualArea()}  # Use an appropriate CRS
    )
    axes = axes.flatten()  # Flatten axes for easy iteration

    # Plot each year's choropleth map
    for i, year in enumerate(unique_years):
        ax = axes[i]
        
        # Subset GeoDataFrame for the current year
        gdf_year = gdf[gdf[year_col] == year]
        
        # Plot the choropleth map
        gplt.choropleth(
            gdf_year,
            hue=column,
            cmap=cmap,
            edgecolor='black',
            linewidth=0.5,
            ax=ax,
            legend=True,
        )
        # Add title
        ax.set_title(f"Year: {year}", fontsize=12)
    
    # Turn off unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add overall title
    if title:
        fig.suptitle(title, fontsize=18, y=1.02)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Example Usage
# Assuming `geoData` is your GeoDataFrame with 'year' and 'medpar' columns
#plot_facet_choropleth_with_geoplot(geoData, column='medpar', cmap='plasma', year_col='year', ncols=4)
