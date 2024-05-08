# %% Plot distribution curve for 'area_ha'
#for year in years:
## sns.histplot(data[year]['area_ha'], kde = True,
             ## stat="density", kde_kws=dict(cut=3),
             ## alpha=.4, edgecolor=(1, 1, 1, .4)) OR
## sns.kdeplot(data[year]['area_ha']) OR
   # sns.displot(data[year]['area_ha'], kind="kde", alpha=.4, rug=True)
   
# %% Create a FacetGrid with a distribution plot for each year in all_data
g = sns.FacetGrid(all_data, col="year", col_wrap=4, height=4)
# Map a distribution plot to the FacetGrid
g.map(sns.histplot, "area_ha")
plt.show()