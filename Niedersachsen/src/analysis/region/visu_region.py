# %%
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import os
import csv


os.chdir("C:/Users/aladesuru/Documents/DataAnalysis/Lab/Niedersachsen")

from src.analysis.region import regiongdf_df as rd

''' change y to the metric you want to plot'''
########################################################################################
#regdf = rd.create_regiongdf_fg() #regiongdf
r = rd.create_regiondf_fg() # regiondf

output_dir = 'reports/figures/regionplots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

########################################################################################

# %% simple seaborn plots of (metric) trend across regions
# Set the plot style
sns.set(style="whitegrid")

# Create a figure
plt.figure(figsize=(12, 6))

# Create a line plot for each category with custom colors
sns.lineplot(data=regdf, x='year', y='totarea_percdiff_to_y1', hue='LANDKREIS',
             marker='o', legend=False)

# Add titles and labels
#plt.title('Trend of Average MFS (ha) for Each Crop Group Over Time')
plt.xlabel('Year')
plt.ylabel('Total Area (ha)')
#plt.legend(title='Crop Group', bbox_to_anchor=(1.05, 1), loc='right')

# Remove the top and right spines
sns.despine(left=True, bottom=True)

# Save plot
#plt.savefig(os.path.join(output_dir, 'mfs_category2.svg'))

# Show the plot
plt.show()

# %%
metric = 'MFS'

# %%


######################################################################################
#Plotly interactive plots
######################################################################################
# %%  1. Change in total land area per region over time: Diff from year 1
fig = px.line(r, x='year', y='meanmfs_percdiff_to_y1', color='LANDKREIS',
              title=f'Rel. Diff from Y1 of {metric} for Each Region Over Time',
              markers=True)

# Update layout to match the style
fig.update_layout(
    xaxis_title='Year',
    yaxis_title=f'Rel. Diff of {metric} from Y1 (%)',
    legend_title='Region',
    template='plotly_white',
    #showlegend=False
)

# Retrieve colors from the legend
legend_colors = {}
for trace in fig.data:
    legend_colors[trace.name] = trace.line.color

# Print the legend colors
#print("Legend colors:", legend_colors)

'''
# Save legend_colors to a CSV file
csv_file_path = os.path.join(output_dir, 'legend_colors.csv')

with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Region', 'Color'])
    for region, color in legend_colors.items():
        writer.writerow([region, color])

print(f"Legend colors saved to {csv_file_path}")

# Save plot as an HTML file
fig.write_html(os.path.join(output_dir, 'chngfrom_y1_totare_regions.html'))
'''
# Show the plot
fig.show()

# %% 2. Yearly Change in total land area per region over time
fig = px.line(r, x='year', y='totarea_yearly_percdiff', color='LANDKREIS',
              title='Rel. Yearly Diff of Total LandArea (ha) for Each Region Over Time',
              markers=True)

# Update layout to match the style
fig.update_layout(
    xaxis_title='Year',
    yaxis_title='Rel. Yearly Diff of TA (%)',
    legend_title='Region',
    template='plotly_white'
)

# Save plot as an HTML file
#fig.write_html(os.path.join(output_dir, 'yearlychng_totare_regions.html'))

# Show the plot
fig.show()

# %%
# extract allunique LANDKREIS in r and save to csv
landkreis = r['LANDKREIS'].unique()
landkreis = pd.DataFrame(landkreis, columns=['LANDKREIS'])
landkreis.to_csv('reports/figures/regionplots/landkreis_BA.csv', index=False, encoding='ANSI')

# %%
