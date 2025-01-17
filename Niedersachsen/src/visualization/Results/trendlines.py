'''Goal is to create plot with grid trend lines and aggregate median line for each metric'''
# %% function for subplots of grid metrics value with aggregate median line
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

def create_fisc_trend_plots(fig, axs, gridgdf, grid_yearly, plot_configs, suptitle):
    """
    Create a 2x2 subplot of FiSC trend plots.
    
    Parameters:
    fig (Figure): Matplotlib Figure object
    axs (Array of Axes): 2x2 array of Matplotlib Axes objects
    gridgdf (DataFrame): DataFrame containing individual cell data
    grid_yearly (DataFrame): DataFrame containing aggregate yearly data
    plot_configs (list): List of dictionaries containing plot configurations
    suptitle (str): Super title for the entire figure
    """
    fig.suptitle(suptitle, fontsize=16)

    for config in plot_configs:
        ax = config['ax']
        
        # Remove border around plot
        [ax.spines[side].set_visible(False) for side in ax.spines]
        
        # Plot individual lines for each unique CELLCODE
        for cellcode in gridgdf['CELLCODE'].unique():
            data = gridgdf[gridgdf['CELLCODE'] == cellcode]
            ax.plot(data['year'], data[config['y_col']], color='grey', alpha=0.9, linewidth=0.5)
        
        # Plot the aggregate thick line and annotate
        line = ax.plot(grid_yearly['year'], grid_yearly[config['agg_col']], color='purple', linewidth=1.5)[0]
        
        # Annotate the end of the line
        last_year = grid_yearly['year'].iloc[-1]
        last_value = grid_yearly[config['agg_col']].iloc[-1]
        ax.annotate(f'{last_value:.2f}', 
                    xy=(last_year, last_value),
                    xytext=(5, 0),
                    textcoords='offset points',
                    color='black',
                    fontweight='bold',
                    ha='left',
                    va='center')
        
        ax.set_xlabel('Year', labelpad=12, fontsize=12, x=0.46)

        ax.set_title(config['title'])
        
        # Style the grid
        ax.grid(which='major', color='#EAEAF2', linewidth=1.2)
        ax.grid(which='minor', color='#EAEAF2', linewidth=0.6)
        ax.minorticks_on()
        ax.tick_params(which='minor', bottom=False, left=False)
        
        # Only show minor gridlines once in between major gridlines
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # Adjust layout
    #plt.tight_layout()

# %%
fig, axs = plt.subplots(1, 4, figsize=(20, 5)) #for two rows, that would be plt.subplots(2, 2, figsize=(10, 10))
plot_configs = [
    {'ax': axs[0], 'title': 'Median Field Size', 'y_col': 'medfs_ha_percdiff_to_y1', 'agg_col': 'med_fsha_percdiffy1_med'},
    {'ax': axs[1], 'title': 'Number of fields', 'y_col': 'fields_ha_percdiff_to_y1', 'agg_col': 'fields_ha_percdiffy1_med'},
    {'ax': axs[2], 'title': 'Median Shape Index', 'y_col': 'medpar_percdiff_to_y1', 'agg_col': 'medpar_percdiffy1_med'},
    {'ax': axs[3], 'title': 'Median Perimeter', 'y_col': 'medperi_percdiff_to_y1', 'agg_col': 'medperi_percdiffy1_med'}
    
]
create_fisc_trend_plots(fig, axs, gridgdf_cl, grid_yearly_cl, plot_configs, 'Trend of FiSC Over Time')

plt.subplots_adjust(left=0.09, wspace=0.2, hspace=0.2, top=0.80)
fig.text(0.05, 0.45, 'Relative Diff (%) to Base Year 2012', va='center', ha='center', rotation='vertical', fontdict={'fontsize': 12}) #, transform=fig.transFigure

#save plot as svg
plt.savefig("reports/figures/results/FiSC_trendlines.svg", format="svg", bbox_inches='tight')
plt.savefig("reports/figures/results/FiSC_trendlinesPNG_.png", format="png", bbox_inches='tight')

plt.show()
# %%