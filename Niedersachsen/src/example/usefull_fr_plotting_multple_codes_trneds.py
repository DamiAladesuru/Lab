# %%
import pandas as pd

# Main DataFrame with counts of x over years
df_main = pd.DataFrame({
    'year': [2020, 2020, 2021, 2021, 2022, 2022],
    'x': [1, 2, 1, 3, 2, 3],
    'count': [100, 150, 200, 250, 300, 350]
})

# Mapping DataFrame with descriptions for x over different periods
df_mapping = pd.DataFrame({
    'year': [2020, 2020, 2021, 2021, 2022, 2022],
    'x': [1, 2, 1, 3, 2, 3],
    'description': ['desc_1_2020', 'desc_2_2020', 'desc_1_2021', 'desc_3_2021', 'desc_2_2022', 'desc_3_2022']
})

# %%
# Merge df_main with df_mapping to add descriptions
df_merged = pd.merge(df_main, df_mapping, on=['year', 'x'])
print(df_merged)

# %%
# Group by year and description to see the trend
df_trend = df_merged.groupby(['year', 'x', 'description']).agg({'count': 'sum'}).reset_index()
print(df_trend)

# %%
import matplotlib.pyplot as plt

# Pivot the DataFrame for easier plotting
df_pivot = df_trend.pivot(index='year', columns='description', values='count')

# Plot the trend
df_pivot.plot(kind='line', marker='o')
plt.title('Trend of x Values Over Time with Descriptions')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='Description')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt

# Plot the trend
fig, ax = plt.subplots()

for key, grp in df_trend.groupby(['x']):
    ax.plot(grp['year'], grp['count'], marker='o', label=f'x={key}')

# Annotate points with descriptions
for i in range(len(df_trend)):
    ax.annotate(df_trend['description'][i], (df_trend['year'][i], df_trend['count'][i]))

plt.title('Trend of x Values Over Time with Descriptions')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='x Value')
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt

# Convert year to integer if not already
df_trend['year'] = df_trend['year'].astype(int)

# Plot the trend
fig, ax = plt.subplots()

for key, grp in df_trend.groupby(['x']):
    ax.plot(grp['year'], grp['count'], marker='o', label=f'x={key}')

# Set x-ticks to be years
ax.set_xticks(df_trend['year'].unique())

plt.title('Trend of x Values Over Time with Descriptions')
plt.xlabel('Year')
plt.ylabel('Count')
plt.legend(title='x Value')
plt.grid(True)
plt.show()

# %%
