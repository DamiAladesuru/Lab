# %%
import matplotlib.pyplot as plt
from joypy import joyplot
import seaborn as sns

# %%  Function to stack plots in a grid
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



