'''here, I want to plot trend line for
fields of cereals, grassland, forage and environmental because
main cultivated kulturart in our data set includes
mähweiden, silomais, winterweichweizen.

We are gonna plot change in metrics for these kulturart,
and then we are gonna plot change in metrics for environmental
to see their gruppe plot, mähweiden is dauergrünland, silomais is
ackerfutter and winterweichweizen is getreide
'''
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from src.analysis.desc import gridgdf_desc as gd
from src.analysis import subsampling_mod as ss
from src.visualization import plotting_module as pm

# %%
cropsubsample = 'environmental'
gld_ss, gridgdf = ss.griddf_speci_subsample(cropsubsample,
                                            col1='category3', gld_data = gld)

_, grid_yearlym = gd.silence_prints(gd.desc_grid,gridgdf)

metrics = {
    'Median Field Size': 'med_fsha_percdiffy1_med',
    'Median Perimeter': 'medperi_percdiffy1_med',
    'Median PAR': 'medpar_percdiffy1_med',
    'Fields/TotalArea': 'fields_ha_percdiffy1_med'
}

color_mapping = {
    #https://personal.sron.nl/~pault/
    'Median Field Size': '#004488',
    'Median Perimeter': 'grey',
    'Median PAR': '#228833',
    'Fields/TotalArea': '#CC3311',
}

pm.multiline_metrics(
    df=grid_yearlym,
    title="Trends in Field Metrics for Environmental Fields",
    ylabel="Aggregate Change (%) in Field Metric Value from 2012",
    metrics=metrics,
    format='svg',
    save_path="reports/figures/ToF/env_trends.svg",
    color_mapping=color_mapping
)
# %%
